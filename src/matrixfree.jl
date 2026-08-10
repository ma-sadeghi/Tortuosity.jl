# Matrix-free form of the steady diffusion system: the same Dirichlet-eliminated
# Laplacian `build_steady_system` assembles, recomputed from the pore mask on
# every apply instead of stored as CSC arrays.

"""
    MaskedLaplacian{T,Ti,A,DT} <: AbstractMatrix{T}

The Dirichlet-eliminated steady diffusion operator, applied straight from the
grid with nothing about the matrix stored.

`idx` maps a grid position to its compact pore ordinal and holds `0` at solids,
so it doubles as the pore/solid test. It is built by the same `cumsum!`-and-mask
idiom [`build_steady_system`](@ref) uses and is numbering-identical to it, which
is what makes `sol.u` the same pore-ordered vector either way — every
postprocessing helper, [`reconstruct_field`](@ref) included, works unmodified.

The seven values of a row are recovered in O(1) from that array: the six face
weights come from the `_edge_weight` harmonic mean the assembler calls, the
diagonal is their sum, and Dirichlet membership is the two-comparison coordinate
test `_is_bc`. So there is no `colptr`, no `rowval`, no `nzval`, no edge list, no
boundary-node list, no stored diagonal and no cache — one full-grid index array
is the whole state.

Implements `mul!(y, A, x)` and its 5-argument form through a KA stencil kernel,
which is all `LinearProblem(A, b)` with `KrylovJL_CG()` needs.

# Fields
- `idx`: grid position → pore ordinal, `0` at solids.
- `nnodes`: number of pore voxels, i.e. the order of the operator.
- `bcdim`: transport-axis dimension, `axis_dim(axis)`.
- `nbc`: extent of the image along `bcdim`. A node is Dirichlet when its
  coordinate along that axis is `1` (inlet) or `nbc` (outlet).
- `D`: full-grid node diffusivity, or `nothing` for uniform diffusivity.
- `D0`: the unit diffusivity used when `D === nothing`.
"""
struct MaskedLaplacian{T,Ti<:Integer,A<:AbstractArray{Ti,3},DT} <: AbstractMatrix{T}
    idx::A
    nnodes::Int
    bcdim::Int
    nbc::Int
    D::DT
    D0::T

    function MaskedLaplacian{T,Ti,A,DT}(
        idx::A, nnodes::Integer, bcdim::Integer, nbc::Integer, D::DT, D0::T
    ) where {T,Ti<:Integer,A<:AbstractArray{Ti,3},DT}
        return new{T,Ti,A,DT}(idx, Int(nnodes), Int(bcdim), Int(nbc), D, D0)
    end
end

function MaskedLaplacian(
    idx::AbstractArray{Ti,3}, nnodes::Integer, bcdim::Integer, nbc::Integer, D, D0::T
) where {T,Ti<:Integer}
    return MaskedLaplacian{T,Ti,typeof(idx),typeof(D)}(idx, nnodes, bcdim, nbc, D, D0)
end

Base.size(A::MaskedLaplacian) = (A.nnodes, A.nnodes)

# The index array is the operator's whole state and is the only thing it owns —
# `D` belongs to whoever passed it in.
_free!(A::MaskedLaplacian) = (_free!(A.idx); nothing)

# Override the AbstractMatrix fallback — scalar indexing isn't supported, and
# the default `show` path walks every entry. Print a concise summary instead.
function Base.show(io::IO, A::MaskedLaplacian{T}) where {T}
    dims = "$(A.nnodes)×$(A.nnodes)"
    grid = join(size(A.idx), "×")
    storage = "$(typeof(A.idx).name.name)"
    weights = isnothing(A.D) ? "uniform" : "variable"
    msg = "MaskedLaplacian{$T}($dims, grid=$grid, D=$weights, storage=$storage)"
    return print(io, msg)
end
Base.show(io::IO, ::MIME"text/plain", A::MaskedLaplacian) = show(io, A)

function Base.getindex(::MaskedLaplacian, ::Integer, ::Integer)
    error("Scalar indexing not supported for MaskedLaplacian; use mul! for the apply")
end

# --- Apply kernel ---

"""
    _steady_apply_kernel!(y, x, idx, D, nx, ny, nz, bcdim, nbc, D0, alpha, beta)

KA kernel: one thread per grid voxel writes `y[p] = alpha * (A*x)[p] + beta*y[p]`
for the pore ordinal `p` it owns.

A solid voxel writes nothing and every pore ordinal is owned by exactly one
thread, so `y` is covered in full without a zeroing pass and without an atomic.

The six neighbours are walked in `_NEIGHBOURS` order — the lower three, then the
upper three — which is ascending pore ordinal, the order a CSC `mul!` reaches
the same row's entries in. The degree sums **every** pore neighbour, Dirichlet
or not: eliminating a boundary column empties it but leaves the diagonal it
already had, which is what [`_steady_fill_kernel!`](@ref) writes.
"""
@kernel function _steady_apply_kernel!(
    y, @Const(x), @Const(idx), @Const(D), nx, ny, nz, bcdim, nbc, D0, alpha, beta,
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        c0 = idx[i, j, k]
        if c0 > 0
            Tv = typeof(D0)
            Ty = eltype(y)
            self_bc = _is_bc(_face_coord(i, j, k, bcdim), nbc)
            da = _node_diffusivity(D, D0, i, j, k)

            # One degree accumulator spans both halves, because the assembled
            # diagonal is a single running sum over `_NEIGHBOURS`. The
            # off-diagonal action needs two, because the diagonal sits between
            # the halves and its value is not known until both have been walked
            # — so the upper half is summed on its own and folded in at the end
            # rather than term by term, which is the one place the association
            # departs from the assembled CSC `mul!`.
            deg = zero(Tv)
            acc_lo = zero(Ty)
            acc_hi = zero(Ty)

            for (di, dj, dk) in _LOWER_NEIGHBOURS
                ii, jj, kk = i + di, j + dj, k + dk
                (1 <= ii <= nx && 1 <= jj <= ny && 1 <= kk <= nz) || continue
                q = idx[ii, jj, kk]
                q > 0 || continue
                w = _edge_weight(D, D0, da, _node_diffusivity(D, D0, ii, jj, kk))
                deg += w
                (self_bc || _is_bc(_face_coord(ii, jj, kk, bcdim), nbc)) && continue
                acc_lo += Ty(-w * x[q])
            end

            for (di, dj, dk) in _UPPER_NEIGHBOURS
                ii, jj, kk = i + di, j + dj, k + dk
                (1 <= ii <= nx && 1 <= jj <= ny && 1 <= kk <= nz) || continue
                q = idx[ii, jj, kk]
                q > 0 || continue
                w = _edge_weight(D, D0, da, _node_diffusivity(D, D0, ii, jj, kk))
                deg += w
                (self_bc || _is_bc(_face_coord(ii, jj, kk, bcdim), nbc)) && continue
                acc_hi += Ty(-w * x[q])
            end

            # A free node with no neighbours is an empty column in the assembled
            # matrix, and by symmetry an empty row — the initial zero is its
            # whole answer, and neither branch below claims it.
            val = zero(Ty)
            if self_bc
                # A zero-degree boundary node has nothing to scale, so it is
                # pinned with a unit diagonal instead — see `_unit_where_zero`.
                d = iszero(deg) ? one(Tv) : deg
                val = Ty(d * x[c0])
            elseif !iszero(deg)
                val = Ty((acc_lo + deg * x[c0]) + acc_hi)
            end
            y[c0] = iszero(beta) ? alpha * val : alpha * val + beta * y[c0]
        end
    end
end

function LinearAlgebra.mul!(
    y::AbstractVector, A::MaskedLaplacian, x::AbstractVector
)
    return mul!(y, A, x, one(eltype(A)), zero(eltype(A)))
end

# The 5-argument form some Krylov solvers reach for. Without it the fallback is
# `generic_matvecmul!`, which reads `A` element by element and dies on the
# scalar-indexing error above. `alpha` and `beta` are converted up front so an
# `Int` pair does not widen the kernel's arithmetic away from `eltype(y)`.
function LinearAlgebra.mul!(
    y::AbstractVector, A::MaskedLaplacian, x::AbstractVector,
    alpha::Number, beta::Number,
)
    # No nodes means no output row to write, and `y` is empty, so `beta` has
    # nothing to scale either.
    A.nnodes == 0 && return y
    Ty = eltype(y)
    nx, ny, nz = size(A.idx)
    backend = get_backend(A.idx)
    # 256 threads laid out along the contiguous dimension, the same shape the
    # assembly kernels launch with.
    _steady_apply_kernel!(backend, (64, 4, 1))(
        y, x, A.idx, A.D, nx, ny, nz, A.bcdim, A.nbc, A.D0,
        convert(Ty, alpha), convert(Ty, beta); ndrange=(nx, ny, nz),
    )
    KernelAbstractions.synchronize(backend)
    return y
end

function Base.:*(A::MaskedLaplacian, x::AbstractVector)
    T = promote_type(eltype(A), eltype(x))
    y = fill!(similar(A.idx, T, A.nnodes), zero(T))
    return mul!(y, A, x)
end

# --- Right-hand side and construction ---

"""
    _steady_rhs_kernel!(b, idx, D, nx, ny, nz, bcdim, nbc, D0)

KA kernel: one thread per grid voxel writes that node's right-hand-side value.

The `b` half of [`_steady_count_kernel!`](@ref), with the entry counting only
the assembled path needs left out. A boundary node carries its diagonal on the
inlet face and zero on the outlet; a free node carries the load folded in by
eliminating its inlet-face neighbours, summed in `_NEIGHBOURS` order so the
result matches the assembler's to the last bit.
"""
@kernel function _steady_rhs_kernel!(
    b, @Const(idx), D, nx, ny, nz, bcdim, nbc, D0,
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        c0 = idx[i, j, k]
        if c0 > 0
            Tv = typeof(D0)
            Tb = eltype(b)
            fc = _face_coord(i, j, k, bcdim)
            self_bc = _is_bc(fc, nbc)
            da = _node_diffusivity(D, D0, i, j, k)

            deg = zero(Tv)
            rhs = zero(Tb)
            for (di, dj, dk) in _NEIGHBOURS
                ii, jj, kk = i + di, j + dj, k + dk
                (1 <= ii <= nx && 1 <= jj <= ny && 1 <= kk <= nz) || continue
                q = idx[ii, jj, kk]
                q > 0 || continue
                w = _edge_weight(D, D0, da, _node_diffusivity(D, D0, ii, jj, kk))
                deg += w
                self_bc && continue
                # Only the inlet face carries a nonzero value, so an outlet
                # neighbour contributes nothing to the folded-in load, and a
                # free neighbour contributes nothing to `b` at all.
                _face_coord(ii, jj, kk, bcdim) == 1 && (rhs += Tb(w))
            end

            if self_bc
                # A zero-degree boundary node has nothing to scale, so it is
                # pinned with a unit diagonal instead — see `_unit_where_zero`.
                d = iszero(deg) ? one(Tv) : deg
                b[c0] = fc == 1 ? Tb(d) : zero(Tb)
            else
                b[c0] = rhs
            end
        end
    end
end

"""
    build_steady_operator(img; nnodes, axis, D=nothing, T=Float64)

Build the steady diffusion system as `(A, b)` with `A` a matrix-free
[`MaskedLaplacian`](@ref) rather than an assembled sparse matrix.

`LinearProblem(A, b)` solved with `KrylovJL_CG()` returns what the assembled
path returns: the Dirichlet values are eliminated the same way (`c = 1` on the
low face along `axis`, `c = 0` on the high one), the pore numbering is the one
[`build_steady_system`](@ref) produces, and `b` matches its right-hand side
entry for entry — so `sol.u` feeds [`reconstruct_field`](@ref) unchanged.

Setup is an inclusive scan over the mask plus one kernel pass; no edge, row
index or stored value is ever materialised, and the weights are recomputed on
every apply.

# Keyword Arguments
- `nnodes`: number of pore voxels, i.e. `count(img)`.
- `axis`: transport direction (`:x`, `:y`, or `:z`).
- `D`: diffusivity array matching `img`, or `nothing` for uniform `D = 1`.
- `T`: element type of `b`. `A` follows `D`'s element type when one is given.
"""
function build_steady_operator(img; nnodes, axis, D=nothing, T=Float64)
    nx, ny, nz = size(img)
    bcdim = axis_dim(axis)
    nbc = size(img, bcdim)
    on_gpu = _on_gpu(img)
    # The same Int32-halves-the-index-traffic argument `build_steady_system`
    # makes, against a smaller bound: the only index the operator ever holds is
    # a pore ordinal, so `nnodes` bounds it rather than that path's `7 * nnodes`.
    Ti = (on_gpu || nnodes + 1 <= typemax(Int32)) ? Int32 : Int
    Tv = isnothing(D) ? T : eltype(D)
    D0 = one(Tv)

    # Pore numbering: an inclusive scan over the mask hands each pore voxel its
    # ordinal, and masking the solids back to zero lets `idx` double as the
    # pore/solid test — the same idiom `build_steady_system` uses, so the two
    # paths number the nodes identically.
    idx = similar(img, Ti)
    cumsum!(vec(idx), vec(img))
    idx .*= img
    backend = get_backend(idx)
    # 256 threads laid out along the contiguous dimension, so a warp reads one
    # run of `idx` and its two in-plane neighbour rows coalesced.
    wg = (64, 4, 1)

    b = similar(idx, T, nnodes)
    _steady_rhs_kernel!(backend, wg)(
        b, idx, D, nx, ny, nz, bcdim, nbc, D0; ndrange=(nx, ny, nz),
    )
    KernelAbstractions.synchronize(backend)

    # `idx` is not released here: it is the operator's state, not scratch.
    A = MaskedLaplacian(idx, nnodes, bcdim, nbc, D, D0)
    return A, b
end

# --- LinearSolve integration ---
#
# Deliberately the same method `PortableSparseCSC` has in `sparse_type.jl`, for
# the two reasons documented there: LinearSolve's placeholder costs a full
# workspace for any matrix type it does not recognise, and the real workspace's
# own solution vector is replaced by `u` the line after it is allocated. The two
# methods are one behaviour written twice — keep them in step.
function LinearSolve.init_cacheval(
    alg::LinearSolve.KrylovJL, A::MaskedLaplacian, b, u, Pl, Pr, maxiters::Int,
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
