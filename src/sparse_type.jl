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
reusable artifacts (e.g. `TortuosityCUDAExt` caches a `CuSparseMatrixCSC`
wrapper there so each `mul!` call does not rebuild it). Extensions must
validate the cache is fresh before using it — `A.colptr`, `A.rowval`, or
`A.nzval` may be reassigned by in-place mutators like [`dropzeros!`](@ref).
"""
mutable struct PortableSparseCSC{
    T,Ti<:Integer,V<:AbstractVector{T},Vi<:AbstractVector{Ti}
} <: AbstractMatrix{T}
    m::Int
    n::Int
    colptr::Vi
    rowval::Vi
    nzval::V
    _cache::Base.RefValue{Any}

    function PortableSparseCSC{T,Ti,V,Vi}(
        m::Integer, n::Integer, colptr::Vi, rowval::Vi, nzval::V
    ) where {T,Ti<:Integer,V<:AbstractVector{T},Vi<:AbstractVector{Ti}}
        return new{T,Ti,V,Vi}(Int(m), Int(n), colptr, rowval, nzval, Base.RefValue{Any}(nothing))
    end
end

function PortableSparseCSC(
    m::Integer, n::Integer, colptr::Vi, rowval::Vi, nzval::V
) where {T,V<:AbstractVector{T},Ti<:Integer,Vi<:AbstractVector{Ti}}
    return PortableSparseCSC{T,Ti,V,Vi}(m, n, colptr, rowval, nzval)
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
    y, @Const(colptr), @Const(rowval), @Const(nzval), @Const(x), n
)
    j = @index(Global)
    if j <= n
        @inbounds for idx in colptr[j]:(colptr[j + 1] - 1)
            r = rowval[idx]
            v = nzval[idx] * x[j]
            Atomix.@atomic y[r] += v
        end
    end
end

function LinearAlgebra.mul!(
    y::AbstractVector, A::PortableSparseCSC, x::AbstractVector
)
    fill!(y, zero(eltype(y)))
    n = A.n
    if n > 0 && nnz(A) > 0
        backend = get_backend(A.nzval)
        _spmv_kernel!(backend)(y, A.colptr, A.rowval, A.nzval, x, n; ndrange=n)
        KernelAbstractions.synchronize(backend)
    end
    return y
end

function Base.:*(A::PortableSparseCSC, x::AbstractVector)
    T = promote_type(eltype(A), eltype(x))
    y = fill!(similar(A.nzval, T, A.m), zero(T))
    return mul!(y, A, x)
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
    @Const(degrees), @Const(diag_missing), n,
)
    j = @index(Global)
    if j <= n
        @inbounds A_start = A_colptr[j]
        @inbounds A_end = A_colptr[j + 1] - 1
        @inbounds L_pos = L_colptr[j]

        if iszero(@inbounds diag_missing[j])
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

    # Row sums (degrees) via SpMV: degrees = A * ones(n)
    ones_v = fill!(similar(am.nzval, n), one(T))
    degrees = fill!(similar(am.nzval, m), zero(T))
    mul!(degrees, am, ones_v)

    # Count the diagonals L will have to add, then size it from that count.
    diag_missing = similar(am.colptr, n)
    _laplacian_diag_missing_kernel!(backend)(
        diag_missing, am.rowval, am.colptr, n; ndrange=n,
    )
    KernelAbstractions.synchronize(backend)

    extra_scan = accumulate(+, diag_missing)

    L_colptr = similar(am.colptr, n + 1)
    _laplacian_colptr_kernel!(backend)(L_colptr, am.colptr, extra_scan, n; ndrange=n)
    KernelAbstractions.synchronize(backend)

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
        degrees, diag_missing, n; ndrange=n,
    )
    KernelAbstractions.synchronize(backend)

    return PortableSparseCSC(m, n, L_colptr, L_rowval, L_nzval)
end
