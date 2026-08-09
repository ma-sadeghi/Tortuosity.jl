# Portable sparse CSC matrix type that works with any array backend (CPU, CUDA, Metal, AMD).
using KernelAbstractions
using Atomix

# -----------------------------------------------------------------------------
# Shared function docstrings.
#
# `laplacian` and `zero_rows!` each have multiple methods across files. Per the
# Julia manual's "Functions and Methods" recommendation, attach a single
# docstring to a function stub and leave the individual method definitions
# undocumented — that way `?laplacian` shows one authoritative description
# rather than Julia concatenating duplicated docstrings.
# -----------------------------------------------------------------------------

"""
    laplacian(am)

Compute the graph Laplacian `L = D - A`, where `D = diag(row_sums(A))` is the
degree matrix and `A` is an adjacency matrix.

Dispatches on the type of `am`:

- Generic `AbstractMatrix` (including `SparseMatrixCSC`) → builds `D` via
  `spdiagm` and returns `D - A`.
- [`PortableSparseCSC`](@ref) → assembled directly with KA kernels; no
  intermediate matrices. A diagonal entry is added only to columns that lack
  one, so an adjacency matrix carrying self-loops is handled correctly.

The two dispatches agree on every *value*, but not always on the sparsity
pattern: `D - A` prunes entries that evaluate to zero, whereas the
`PortableSparseCSC` path keeps a structural diagonal in every column. They
coincide after [`dropzeros!`](@ref), which the solver path applies anyway.
"""
function laplacian end

"""
    zero_rows!(A, rows)

Zero out all entries in the specified `rows` of sparse matrix `A` in place,
then drop the resulting structural zeros. Used to enforce Dirichlet boundary
conditions in the transient operator.

Supports `SparseMatrixCSC` (CPU) and [`PortableSparseCSC`](@ref) (any backend).
"""
function zero_rows! end

"""
    PortableSparseCSC{T,Ti,V,Vi} <: AbstractMatrix{T}

Backend-agnostic sparse matrix in Compressed Sparse Column (CSC) format.
Works with any array backend — `Vector` (CPU), `CuVector` (CUDA),
`MtlVector` (Metal), `ROCVector` (AMD) — through duck typing.

Implements `mul!(y, A, x)` via a KA SpMV kernel, enabling use with
Krylov.jl and LinearSolve.jl solvers.

The `_cache` field is an opaque slot reserved for backend extensions to store
reusable artifacts (e.g. `TortuosityCUDAExt` caches a CUSPARSE wrapper there so
each `mul!` call does not rebuild it). A cached artifact references `A.colptr`,
`A.rowval`, and `A.nzval` directly.

`symmetric` records that the *builder* knew `A == transpose(A)` exactly. It is
not checked and never inferred: a matrix is symmetric only if it was
constructed that way. Its one consumer is the CUSPARSE fast path, which reads a
symmetric CSC as CSR — the same bytes, the same product, but a gather rather
than an atomic scatter.

Both are claims about the *current* contents, so any mutation invalidates them:
call [`_invalidate_cache!`](@ref) before changing anything. Every mutator in
`kernels/sparse.jl` does.
"""
mutable struct PortableSparseCSC{
    T,Ti<:Integer,V<:AbstractVector{T},Vi<:AbstractVector{Ti}
} <: AbstractMatrix{T}
    m::Int
    n::Int
    colptr::Vi
    rowval::Vi
    nzval::V
    symmetric::Bool
    _cache::Base.RefValue{Any}

    function PortableSparseCSC{T,Ti,V,Vi}(
        m::Integer, n::Integer, colptr::Vi, rowval::Vi, nzval::V, symmetric::Bool=false
    ) where {T,Ti<:Integer,V<:AbstractVector{T},Vi<:AbstractVector{Ti}}
        return new{T,Ti,V,Vi}(
            Int(m), Int(n), colptr, rowval, nzval, symmetric, Base.RefValue{Any}(nothing),
        )
    end
end

function PortableSparseCSC(
    m::Integer, n::Integer, colptr::Vi, rowval::Vi, nzval::V; symmetric::Bool=false
) where {T,V<:AbstractVector{T},Ti<:Integer,Vi<:AbstractVector{Ti}}
    return PortableSparseCSC{T,Ti,V,Vi}(m, n, colptr, rowval, nzval, symmetric)
end

"""
    _invalidate_cache!(A::PortableSparseCSC)

Drop everything `A` remembers about its own contents: the artifact a backend
extension left in `A._cache`, and the `symmetric` claim.

Call it before any mutation, whether that mutation reassigns `A.colptr`,
`A.rowval` and `A.nzval` or edits them in place. A reassignment leaves the
cached artifact pinning storage the matrix no longer uses, and reading freed
memory once that storage is released. An in-place edit leaves the artifact
valid but can make the symmetry claim false, and the CUSPARSE fast path would
then quietly compute `transpose(A) * x`.
"""
function _invalidate_cache!(A::PortableSparseCSC)
    A._cache[] = nothing
    A.symmetric = false
    return nothing
end

# Releasing a matrix means releasing its three arrays; the cached artifact has
# to go first or it is left describing freed storage.
function _free!(A::PortableSparseCSC)
    _invalidate_cache!(A)
    _free!(A.colptr)
    _free!(A.rowval)
    _free!(A.nzval)
    return nothing
end

Base.size(A::PortableSparseCSC) = (A.m, A.n)
SparseArrays.nnz(A::PortableSparseCSC) = length(A.nzval)
SparseArrays.nonzeros(A::PortableSparseCSC) = A.nzval
SparseArrays.rowvals(A::PortableSparseCSC) = A.rowval
SparseArrays.getcolptr(A::PortableSparseCSC) = A.colptr

# Override the AbstractMatrix fallback — scalar indexing isn't supported, and
# the default `show` path walks every entry. Print a concise summary instead.
function Base.show(io::IO, A::PortableSparseCSC{T}) where {T}
    storage = eltype(A.nzval) === T ? "$(typeof(A.nzval).name.name)" : ""
    return print(io, "PortableSparseCSC{$T}($(A.m)×$(A.n), nnz=$(nnz(A)), storage=$(storage))")
end
Base.show(io::IO, ::MIME"text/plain", A::PortableSparseCSC) = show(io, A)

function Base.getindex(::PortableSparseCSC, ::Integer, ::Integer)
    error("Scalar indexing not supported for PortableSparseCSC; use mul! for SpMV")
end

# --- SpMV kernel ---

@kernel function _spmv_kernel!(
    y, @Const(colptr), @Const(rowval), @Const(nzval), @Const(x), alpha, n
)
    j = @index(Global)
    if j <= n
        @inbounds xj = alpha * x[j]
        @inbounds for idx in colptr[j]:(colptr[j + 1] - 1)
            r = rowval[idx]
            v = nzval[idx] * xj
            Atomix.@atomic y[r] += v
        end
    end
end

# `A * ones(n)` without the vector of ones: the same scatter as `_spmv_kernel!`
# with `x` folded away. Row sums, not column sums — the two coincide for a
# symmetric adjacency matrix but `laplacian` accepts any matrix, and `D` is
# defined as the row sums.
@kernel function _row_sums_kernel!(
    sums, @Const(colptr), @Const(rowval), @Const(nzval), n
)
    j = @index(Global)
    if j <= n
        @inbounds for idx in colptr[j]:(colptr[j + 1] - 1)
            Atomix.@atomic sums[rowval[idx]] += nzval[idx]
        end
    end
end

function LinearAlgebra.mul!(
    y::AbstractVector, A::PortableSparseCSC, x::AbstractVector
)
    return mul!(y, A, x, one(eltype(A)), zero(eltype(A)))
end

# The 5-argument form some Krylov solvers reach for. Without it the fallback is
# `generic_matvecmul!`, which reads `A` element by element and dies on the
# scalar-indexing error above — latent on every backend but CUDA, which has its
# own method in the extension.
function LinearAlgebra.mul!(
    y::AbstractVector, A::PortableSparseCSC, x::AbstractVector,
    alpha::Number, beta::Number,
)
    if iszero(beta)
        fill!(y, zero(eltype(y)))
    elseif !isone(beta)
        y .*= beta
    end
    n = A.n
    if n > 0 && nnz(A) > 0 && !iszero(alpha)
        backend = get_backend(A.nzval)
        _spmv_kernel!(backend)(y, A.colptr, A.rowval, A.nzval, x, alpha, n; ndrange=n)
        KernelAbstractions.synchronize(backend)
    end
    return y
end

function Base.:*(A::PortableSparseCSC, x::AbstractVector)
    T = promote_type(eltype(A), eltype(x))
    y = fill!(similar(A.nzval, T, A.m), zero(T))
    return mul!(y, A, x)
end

# --- LinearSolve integration ---
#
# `init_cacheval` is called twice per solve: `init` asks for a placeholder with
# `zeroinit=true`, then `solve!` asks for the real workspace with
# `zeroinit=false`. LinearSolve's generic path costs a full solution vector on
# each call when the matrix is a `PortableSparseCSC`:
#
#  1. The placeholder is only built empty for `Matrix` and `SparseMatrixCSC`;
#     anything else falls through to `KS(A, b)`, so the full workspace — four
#     n-length vectors for CG — is allocated twice, both live at the moment the
#     second one is built. Building it at zero length costs nothing, taking the
#     storage type from `b` so it stays on the right device.
#  2. The real workspace's own solution vector is dead on arrival: LinearSolve
#     replaces it with `u` (`solver.x = u`) as soon as the constructor returns.
#     Releasing it afterwards is not enough — it is allocated *before* the
#     workspace's other vectors, so by then the device is already holding all
#     four and the peak has been reached. On a pooled allocator a release only
#     hands the block back to the pool, which the driver still counts as in use.
#     It has to not be allocated at all: worth 0.95 GiB at 800³.
#
# Doing (2) means knowing which of a workspace's vectors the algorithm actually
# uses, so it is done for CG — the algorithm this package ships — and every
# other algorithm keeps LinearSolve's generic path and only gets (1).
function LinearSolve.init_cacheval(
    alg::LinearSolve.KrylovJL, A::PortableSparseCSC, b, u, Pl, Pr, maxiters::Int,
    abstol, reltol, verbose::Union{LinearSolve.LinearVerbosity,Bool},
    assumptions::LinearSolve.OperatorAssumptions; zeroinit=true,
)
    if zeroinit
        KS = LinearSolve.get_KrylovJL_solver(alg.KrylovAlg)
        workspace = KS(0, 0, LinearSolve.Krylov.ktypeof(b))
        workspace.x = u
        return workspace
    end
    alg.KrylovAlg === LinearSolve.Krylov.cg! && return _cg_workspace(A, b, u)
    return @invoke LinearSolve.init_cacheval(
        alg::LinearSolve.KrylovJL, A::Any, b::Any, u::Any, Pl::Any, Pr::Any,
        maxiters::Int, abstol::Any, reltol::Any,
        verbose::Union{LinearSolve.LinearVerbosity,Bool},
        assumptions::LinearSolve.OperatorAssumptions; zeroinit=false,
    )
end

"""
    _cg_workspace(A, b, u)

Build a `Krylov.CgWorkspace` for `A` whose solution vector *is* `u`.

Identical to what `CgWorkspace(A, b)` produces except that `x` is aliased to `u`
rather than freshly allocated — which is what LinearSolve does to it anyway, one
line after the constructor returns. `Δx`, `z` and `npc_dir` are left empty
exactly as the constructor leaves them; Krylov grows those lazily, and only when
a preconditioner, a trust region or a line search is in play.

Reaching into the workspace's fields means this has to keep step with Krylov's
definition of `CgWorkspace`, which the test suite checks by comparing against a
constructor-built workspace field by field.
"""
function _cg_workspace(A, b, u)
    workspace = LinearSolve.Krylov.CgWorkspace(0, 0, LinearSolve.Krylov.ktypeof(b))
    workspace.m, workspace.n = size(A)
    workspace.x = u
    workspace.r = similar(u)
    workspace.p = similar(u)
    workspace.Ap = similar(u)
    return workspace
end

# --- Laplacian: L = D - A ---
# D is the degree matrix (diagonal of row sums), A is the adjacency matrix.
#
# A column of L holds the column of A plus a diagonal entry — but only when the
# column does not already carry one. Adjacency matrices built from
# `build_connectivity_list` never do (a voxel is not its own face-neighbour),
# but `laplacian` accepts any matrix, so the output size is counted rather than
# assumed. Assuming `nnz(A) + n` over-allocates by one slot per self-loop column
# and leaves it uninitialised, which surfaces downstream as a garbage row index
# in the `@inbounds` SpMV kernel.

@kernel function _laplacian_diag_missing_kernel!(
    diag_missing, @Const(A_rowval), @Const(A_colptr), n
)
    j = @index(Global)
    if j <= n
        @inbounds A_start = A_colptr[j]
        @inbounds A_end = A_colptr[j + 1] - 1
        found = false
        @inbounds for idx in A_start:A_end
            if A_rowval[idx] == j
                found = true
                break
            end
        end
        T = eltype(diag_missing)
        @inbounds diag_missing[j] = found ? zero(T) : one(T)
    end
end

@kernel function _laplacian_colptr_kernel!(
    L_colptr, @Const(A_colptr), @Const(extra_scan), n
)
    j = @index(Global)
    if j == 1
        @inbounds L_colptr[1] = A_colptr[1]
    end
    if j <= n
        # Column j of L holds column j of A plus however many diagonals were
        # added in columns 1..j — that running total is `extra_scan`.
        @inbounds L_colptr[j + 1] = A_colptr[j + 1] + extra_scan[j]
    end
end

@kernel function _laplacian_entries_kernel!(
    L_rowval, L_nzval, @Const(L_colptr),
    @Const(A_rowval), @Const(A_nzval), @Const(A_colptr),
    @Const(degrees), n,
)
    j = @index(Global)
    if j <= n
        @inbounds A_start = A_colptr[j]
        @inbounds A_end = A_colptr[j + 1] - 1
        @inbounds L_pos = L_colptr[j]
        # Column j of L is column j of A plus a diagonal, except where A already
        # carried one — so the gap between the two column lengths is the flag,
        # and no separate `diag_missing` array has to stay live this long.
        @inbounds diag_missing_j = (L_colptr[j + 1] - L_pos) != (A_end - A_start + 1)

        if !diag_missing_j
            # The column already carries A[j,j]. Rewrite it in place as
            # degree[j] - A[j,j] and negate everything else; the entry count is
            # unchanged, so no diagonal may be spliced in as well.
            offset = 0
            for idx in A_start:A_end
                @inbounds row = A_rowval[idx]
                @inbounds L_rowval[L_pos + offset] = row
                @inbounds L_nzval[L_pos + offset] =
                    row == j ? degrees[j] - A_nzval[idx] : -A_nzval[idx]
                offset += 1
            end
        else
            # No diagonal present: negate the off-diagonals and splice the
            # diagonal in at its sorted position, or append it when every row
            # index is below j.
            offset = 0
            diag_inserted = false
            for idx in A_start:A_end
                @inbounds row = A_rowval[idx]
                if !diag_inserted && row > j
                    @inbounds L_rowval[L_pos + offset] = j
                    @inbounds L_nzval[L_pos + offset] = degrees[j]
                    offset += 1
                    diag_inserted = true
                end
                @inbounds L_rowval[L_pos + offset] = row
                @inbounds L_nzval[L_pos + offset] = -A_nzval[idx]
                offset += 1
            end
            if !diag_inserted
                @inbounds L_rowval[L_pos + offset] = j
                @inbounds L_nzval[L_pos + offset] = degrees[j]
            end
        end
    end
end

function laplacian(am::PortableSparseCSC{T}) where {T}
    m, n = size(am)
    @assert m == n "Adjacency matrix must be square"

    Ti = eltype(am.colptr)
    if n == 0
        return PortableSparseCSC(
            m, n,
            fill!(similar(am.colptr, 1), one(Ti)),
            similar(am.rowval, 0),
            similar(am.nzval, 0),
        )
    end

    backend = get_backend(am.nzval)

    # Row sums (degrees) = A * ones(n), computed without the ones.
    degrees = fill!(similar(am.nzval, m), zero(T))
    _row_sums_kernel!(backend)(degrees, am.colptr, am.rowval, am.nzval, n; ndrange=n)
    KernelAbstractions.synchronize(backend)

    # Count the diagonals L will have to add, then size it from that count.
    # Both of these are n-element scratch arrays, so they are released before
    # L's entry arrays — which are `nnz` long — get allocated below.
    diag_missing = similar(am.colptr, n)
    _laplacian_diag_missing_kernel!(backend)(
        diag_missing, am.rowval, am.colptr, n; ndrange=n,
    )
    KernelAbstractions.synchronize(backend)

    extra_scan = accumulate(+, diag_missing)
    _free!(diag_missing)

    L_colptr = similar(am.colptr, n + 1)
    _laplacian_colptr_kernel!(backend)(L_colptr, am.colptr, extra_scan, n; ndrange=n)
    KernelAbstractions.synchronize(backend)
    _free!(extra_scan)

    # Size the entry arrays from the column pointers we just built, not from
    # `nnz(am) + total_extra`. Those agree only when the input's stored-value
    # array matches its colptr span, which `PortableSparseCSC` never validates —
    # and sizing the output by assumption is the exact failure this routine
    # exists to avoid. Two single-slot host reads, the idiom
    # `_build_connectivity_list_ka` already uses.
    nnz_L = Int(Array(@view L_colptr[end:end])[1]) - Int(Array(@view L_colptr[1:1])[1])

    L_rowval = similar(am.rowval, nnz_L)
    L_nzval = similar(am.nzval, nnz_L)

    _laplacian_entries_kernel!(backend)(
        L_rowval, L_nzval, L_colptr,
        am.rowval, am.nzval, am.colptr,
        degrees, n; ndrange=n,
    )
    KernelAbstractions.synchronize(backend)
    _free!(degrees)

    return PortableSparseCSC(m, n, L_colptr, L_rowval, L_nzval)
end
