# Matrix-free form of the steady diffusion system: the same Dirichlet-eliminated
# Laplacian `build_steady_system` assembles, recomputed from the pore mask on
# every apply instead of stored as CSC arrays.

"""
    MaskedLaplacian{T,Ti,A,DT} <: AbstractMatrix{T}

The Dirichlet-eliminated steady diffusion operator, applied straight from the
grid with nothing about the matrix stored.

`idx` maps a grid position to its compact pore ordinal and holds `0` at solids,
so it doubles as the pore/solid test. It is built by the same `cumsum!`-and-mask
idiom `build_steady_system` uses and is numbering-identical to it, which
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
- `owns_D`: whether `_free!` should release `D` as well as `idx`. True only when
  `D` is a device copy made on the operator's behalf.
"""
struct MaskedLaplacian{T,Ti<:Integer,A<:AbstractArray{Ti,3},DT} <: AbstractMatrix{T}
    idx::A
    nnodes::Int
    bcdim::Int
    nbc::Int
    D::DT
    D0::T
    owns_D::Bool

    function MaskedLaplacian{T,Ti,A,DT}(
        idx::A, nnodes::Integer, bcdim::Integer, nbc::Integer, D::DT, D0::T, owns_D::Bool
    ) where {T,Ti<:Integer,A<:AbstractArray{Ti,3},DT}
        return new{T,Ti,A,DT}(idx, Int(nnodes), Int(bcdim), Int(nbc), D, D0, owns_D)
    end
end

function MaskedLaplacian(
    idx::AbstractArray{Ti,3}, nnodes::Integer, bcdim::Integer, nbc::Integer, D, D0::T,
    owns_D::Bool=false,
) where {T,Ti<:Integer}
    return MaskedLaplacian{T,Ti,typeof(idx),typeof(D)}(idx, nnodes, bcdim, nbc, D, D0, owns_D)
end

Base.size(A::MaskedLaplacian) = (A.nnodes, A.nnodes)

# The index array is always the operator's to release. `D` usually belongs to
# the caller, but when the device copy was made on the operator's behalf nobody
# else holds a reference to it, so `owns_D` records that and it is released here
# too. Getting this wrong either leaks a grid-sized array or frees the caller's.
function _free!(A::MaskedLaplacian)
    _free!(A.idx)
    A.owns_D && _free!(A.D)
    return nothing
end

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

@inline function _cpu_apply_range!(
    y, x, idx, D, nx, ny, nz, nbc, D0, alpha, beta, j, k, ilo, ihi,
    ::Val{B}, ::Val{Z}, ::Val{Q},
) where {B,Z,Q}
    Ty = eltype(y)
    Tv = typeof(D0)
    total = zero(Ty)

    @inbounds for i in ilo:ihi
        c0 = idx[i, j, k]
        c0 > 0 || continue

        self_bc = _is_bc(_face_coord(i, j, k, B), nbc)
        da = _node_diffusivity(D, D0, i, j, k)
        deg = zero(Tv)
        acc_lo = zero(Ty)
        acc_hi = zero(Ty)

        if k > 1
            q = idx[i, j, k - 1]
            if q > 0
                w = _edge_weight(D, D0, da, _node_diffusivity(D, D0, i, j, k - 1))
                deg += w
                (self_bc || _is_bc(_face_coord(i, j, k - 1, B), nbc)) ||
                    (acc_lo += Ty(-w * x[q]))
            end
        end
        if j > 1
            q = idx[i, j - 1, k]
            if q > 0
                w = _edge_weight(D, D0, da, _node_diffusivity(D, D0, i, j - 1, k))
                deg += w
                (self_bc || _is_bc(_face_coord(i, j - 1, k, B), nbc)) ||
                    (acc_lo += Ty(-w * x[q]))
            end
        end
        if i > 1
            q = idx[i - 1, j, k]
            if q > 0
                w = _edge_weight(D, D0, da, _node_diffusivity(D, D0, i - 1, j, k))
                deg += w
                (self_bc || _is_bc(_face_coord(i - 1, j, k, B), nbc)) ||
                    (acc_lo += Ty(-w * x[q]))
            end
        end
        if i < nx
            q = idx[i + 1, j, k]
            if q > 0
                w = _edge_weight(D, D0, da, _node_diffusivity(D, D0, i + 1, j, k))
                deg += w
                (self_bc || _is_bc(_face_coord(i + 1, j, k, B), nbc)) ||
                    (acc_hi += Ty(-w * x[q]))
            end
        end
        if j < ny
            q = idx[i, j + 1, k]
            if q > 0
                w = _edge_weight(D, D0, da, _node_diffusivity(D, D0, i, j + 1, k))
                deg += w
                (self_bc || _is_bc(_face_coord(i, j + 1, k, B), nbc)) ||
                    (acc_hi += Ty(-w * x[q]))
            end
        end
        if k < nz
            q = idx[i, j, k + 1]
            if q > 0
                w = _edge_weight(D, D0, da, _node_diffusivity(D, D0, i, j, k + 1))
                deg += w
                (self_bc || _is_bc(_face_coord(i, j, k + 1, B), nbc)) ||
                    (acc_hi += Ty(-w * x[q]))
            end
        end

        val = zero(Ty)
        if self_bc
            d = iszero(deg) ? one(Tv) : deg
            val = Ty(d * x[c0])
        elseif !iszero(deg)
            val = Ty((acc_lo + deg * x[c0]) + acc_hi)
        end
        result = Z ? alpha * val : alpha * val + beta * y[c0]
        y[c0] = result
        Q && (total += x[c0] * result)
    end
    return total
end

# Threads take contiguous chunks of the grid: whole z-slabs when there are
# enough of them, (j, k) lines otherwise, and x-ranges of lines when even the
# lines are too few. `@threads` splits its range statically, so a tier is taken
# only once it holds `_CHUNKS_PER_THREAD` items per thread — one slab over the
# thread count would otherwise hand a task double the work — and its items are
# grouped into at most `maxchunks` chunks. With `Q` the per-chunk `xᵀ(A x)` sums
# land in `partial` and their total is returned.
function _cpu_mul_chunked!(
    y, A, x, α, β, partial, ::Val{B}, ::Val{Z}, ::Val{Q},
) where {B,Z,Q}
    nx, ny, nz = size(A.idx)
    nthreads = Threads.nthreads()
    balanced = _CHUNKS_PER_THREAD * nthreads
    maxchunks = Q ? length(partial) : balanced
    if nz >= balanced
        nchunks = min(nz, maxchunks)
        Threads.@threads :dynamic for chunk in 1:nchunks
            klo, khi = _host_chunk_bounds(nz, nchunks, chunk)
            total = zero(eltype(y))
            for k in klo:khi, j in 1:ny
                total += _cpu_apply_range!(
                    y, x, A.idx, A.D, nx, ny, nz, A.nbc, A.D0, α, β,
                    j, k, 1, nx, Val(B), Val(Z), Val(Q),
                )
            end
            Q && (partial[chunk] = total)
        end
        return Q ? _cpu_partial_sum(partial, nchunks) : zero(eltype(y))
    end

    nlines = ny * nz
    if nlines >= balanced
        nchunks = min(nlines, maxchunks)
        Threads.@threads :dynamic for chunk in 1:nchunks
            line_lo, line_hi = _host_chunk_bounds(nlines, nchunks, chunk)
            total = zero(eltype(y))
            for line in line_lo:line_hi
                j = (line - 1) % ny + 1
                k = (line - 1) ÷ ny + 1
                total += _cpu_apply_range!(
                    y, x, A.idx, A.D, nx, ny, nz, A.nbc, A.D0, α, β,
                    j, k, 1, nx, Val(B), Val(Z), Val(Q),
                )
            end
            Q && (partial[chunk] = total)
        end
        return Q ? _cpu_partial_sum(partial, nchunks) : zero(eltype(y))
    end

    xchunks = min(nx, max(1, cld(2 * nthreads, nlines)))
    xchunk_size = cld(nx, xchunks)
    nwork = nlines * xchunks
    nchunks = min(nwork, maxchunks)
    Threads.@threads :dynamic for chunk in 1:nchunks
        work_lo, work_hi = _host_chunk_bounds(nwork, nchunks, chunk)
        total = zero(eltype(y))
        for work in work_lo:work_hi
            line = (work - 1) ÷ xchunks + 1
            xchunk = (work - 1) % xchunks
            ilo = xchunk * xchunk_size + 1
            ihi = min(ilo + xchunk_size - 1, nx)
            j = (line - 1) % ny + 1
            k = (line - 1) ÷ ny + 1
            total += _cpu_apply_range!(
                y, x, A.idx, A.D, nx, ny, nz, A.nbc, A.D0, α, β,
                j, k, ilo, ihi, Val(B), Val(Z), Val(Q),
            )
        end
        Q && (partial[chunk] = total)
    end
    return Q ? _cpu_partial_sum(partial, nchunks) : zero(eltype(y))
end

function _cpu_mul!(y, A, x, α, β, ::Val{B}, ::Val{Z}) where {B,Z}
    _cpu_mul_chunked!(y, A, x, α, β, nothing, Val(B), Val(Z), Val(false))
    return y
end

@inline function _cpu_partial_sum(partial, n)
    total = zero(eltype(partial))
    @inbounds for i in 1:n
        total += partial[i]
    end
    return total
end

function _cpu_mul_dot!(y, A, x, partial, ::Val{B}) where {B}
    return _cpu_mul_chunked!(
        y, A, x, one(eltype(y)), zero(eltype(y)), partial, Val(B), Val(true), Val(true),
    )
end

function LinearAlgebra.mul!(
    y::Vector, A::MaskedLaplacian{T,Ti,AI,DT}, x::Vector,
    alpha::Number, beta::Number,
) where {T,Ti,AI<:Array{Ti,3},DT<:Union{Nothing,Array}}
    if length(y) != A.nnodes || length(x) != A.nnodes
        throw(DimensionMismatch(
            "operator is $(A.nnodes)×$(A.nnodes) but y has length $(length(y)) \
             and x has length $(length(x))"
        ))
    end
    A.nnodes == 0 && return y

    Ty = eltype(y)
    α = convert(Ty, alpha)
    β = convert(Ty, beta)
    return _cpu_mul!(y, A, x, α, β, Val(A.bcdim), Val(iszero(β)))
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
    # The kernel body is `@inbounds`, so a mismatched vector is not a wrong
    # answer but a write past the end of `y`. The assembled path raises here and
    # so must this one.
    if length(y) != A.nnodes || length(x) != A.nnodes
        throw(DimensionMismatch(
            "operator is $(A.nnodes)×$(A.nnodes) but y has length $(length(y)) \
             and x has length $(length(x))"
        ))
    end
    # No nodes means no output row to write, and `y` is empty, so `beta` has
    # nothing to scale either.
    A.nnodes == 0 && return y
    Ty = eltype(y)
    nx, ny, nz = size(A.idx)
    backend = get_backend(A.idx)
    _steady_apply_kernel!(backend, _steady_workgroup(A.idx))(
        y, x, A.idx, A.D, nx, ny, nz, A.bcdim, A.nbc, A.D0,
        convert(Ty, alpha), convert(Ty, beta); ndrange=(nx, ny, nz),
    )
    _async_return_safe(A.idx) || KernelAbstractions.synchronize(backend)
    return y
end

function Base.:*(A::MaskedLaplacian, x::AbstractVector)
    T = promote_type(eltype(A), eltype(x))
    y = fill!(similar(A.idx, T, A.nnodes), zero(T))
    return mul!(y, A, x)
end

# --- Right-hand side and construction ---

"""
    _operator_index_type(on_gpu, nnodes)

The integer type the pore ordinals are stored in, or an error when no supported
type fits.

On GPU the ordinal is always `Int32`: `idx` is the operator's whole state and
halving its traffic is what the apply spends its time on. That holds only while
an ordinal fits in 32 bits, and it fails silently if it does not — `cumsum!`
into an `Int32` buffer wraps to `typemin` rather than saturating, so `idx` would
go negative, the kernel's `c0 > 0` test would drop most voxels, and `y` would
come back partly unwritten with no error raised anywhere. Refuse instead.

The bound is `nnodes`, not the assembled path's `7 * nnodes`, which puts it near
1625³ at half porosity — past this card, but reachable on an 80 GiB one, which
is the regime the operator exists for.
"""
function _operator_index_type(on_gpu::Bool, nnodes::Integer)
    nnodes + 1 <= typemax(Int32) && return Int32
    on_gpu && throw(ArgumentError(
        "image has $(nnodes) pore voxels, more than a 32-bit pore ordinal can address \
         ($(typemax(Int32))); the GPU operator has no 64-bit index path"
    ))
    return Int
end

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
`build_steady_system` produces, and `b` matches its right-hand side
entry for entry — so `sol.u` feeds [`reconstruct_field`](@ref) unchanged.

Setup is an inclusive scan over the mask plus one kernel pass; no edge, row
index or stored value is ever materialised, and the weights are recomputed on
every apply.

# Keyword Arguments
- `nnodes`: number of pore voxels, i.e. `count(img)`.
- `axis`: transport direction (`:x`, `:y`, or `:z`).
- `D`: diffusivity array matching `img`, a scalar for uniform diffusivity at that
  value, or `nothing` for uniform `D = 1`. A scalar takes the same path as
  `nothing` — the operator holds no diffusivity array, `D0` carries the weight.
- `T`: element type of `b`. `A` follows `D`'s element type when one is given.
- `owns_D`: hand the operator ownership of `D`, so that `_free!` releases it.
  Set this only when `D` is a copy made for the operator and held nowhere else.
"""
function build_steady_operator(
    img; nnodes, axis, D=nothing, T=Float64, owns_D::Bool=false,
    return_flux::Bool=false, checkpoint_readout::Bool=false,
)
    nx, ny, nz = size(img)
    bcdim = axis_dim(axis)
    nbc = size(img, bcdim)
    on_gpu = _on_gpu(img)
    Ti = _operator_index_type(on_gpu, nnodes)
    # A scalar `D` is the uniform case, which the operator already expresses as
    # `D === nothing` with `D0` carrying the weight — so it rides that path and
    # the operator holds no diffusivity array at all.
    D_scalar = D isa Number
    # `float` on the scalar's own type, so `D = 2` means the same thing as
    # `D = 2.0` — see the same step in `build_steady_system` for what an
    # integer `D0` would otherwise make of the operator's element type.
    Tv = isnothing(D) ? T : (D_scalar ? float(typeof(D)) : eltype(D))
    D0 = D_scalar ? Tv(D) : one(Tv)
    # `nothing`, not the scalar — see the same step in `build_steady_system` for
    # why keeping it would depend on `@inbounds` eliding a bounds check. Here it
    # also decides what the operator holds for its whole life.
    D = D_scalar ? nothing : D

    # Pore numbering: an inclusive scan over the mask hands each pore voxel its
    # ordinal, and masking the solids back to zero lets `idx` double as the
    # pore/solid test — the same idiom `build_steady_system` uses, so the two
    # paths number the nodes identically.
    idx = similar(img, Ti)
    _pore_index!(idx, img)
    backend = get_backend(idx)
    inlet_flux = return_flux ?
        _build_inlet_flux(idx, D, bcdim, D0; checkpoint_readout) : nothing
    # The backend-selected shape is shared with assembly and operator applies.
    wg = _steady_workgroup(idx)

    b = similar(idx, T, nnodes)
    _steady_rhs_kernel!(backend, wg)(
        b, idx, D, nx, ny, nz, bcdim, nbc, D0; ndrange=(nx, ny, nz),
    )
    KernelAbstractions.synchronize(backend)

    # `idx` is not released here: it is the operator's state, not scratch.
    A = MaskedLaplacian(idx, nnodes, bcdim, nbc, D, D0, owns_D)
    return return_flux ? (A, b, inlet_flux) : (A, b)
end

# --- LinearSolve integration ---
#
# One method for both of this package's operator types, the assembled
# `PortableSparseCSC` and the matrix-free `MaskedLaplacian`; it lives here
# because that is where both names are in scope.
#
# `init_cacheval` is called twice per solve: `init` asks for a placeholder with
# `zeroinit=true`, then `solve!` asks for the real workspace with
# `zeroinit=false`. LinearSolve's generic path costs a full solution vector on
# each call for either of these types:
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
    alg::LinearSolve.KrylovJL, A::Union{MaskedLaplacian,PortableSparseCSC},
    b, u, Pl, Pr, maxiters::Int, abstol, reltol,
    verbose::Union{LinearSolve.LinearVerbosity,Bool},
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
