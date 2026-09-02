# Two-level preconditioner for the assembled steady system: a coarse space of
# piecewise-constant indicators over cubic voxel blocks, plus a scaled identity
# for everything the coarse space cannot see. The coarse problem is solved
# directly when it is small, and otherwise by a V-cycle over a hierarchy of
# coarser grids ending in that same direct solve.

# Slots of the coarse stencil, in this order: the block itself, then its six
# face neighbours at block-index offsets -nbxy, -nbx, -1, +1, +nbx, +nbxy.
# Opposite directions sit at slots `s` and `9 - s`, which is what makes the
# host-side symmetrisation a single indexed lookup.
const _COARSE_SLOTS = 7

# Default CPU ceiling on the number of unknowns solved *directly*. The direct
# solve runs once per CG iteration, so its cost has to stay well under one fine SpMV:
# measured on a 3-D 7-point operator, a 25³ coarse grid factorises in 0.44 s and
# solves in 1.9 ms, where a 50³ one takes 1.7 s and 47 ms. A coarse grid larger
# than this gets a hierarchy of coarser grids built under it, ending in a direct
# solve that does fit — see [`_coarse_hierarchy`](@ref).
const DEFAULT_MAX_COARSE = 32_000

# GPU fine-grid applies are much cheaper relative to the host sparse triangular
# solve, so the base crossover is lower. At 200³ the old 32k ceiling left a
# 15.6k-cell coarse factor on the host; one solve cost 2.6-4.4 ms. Putting one
# more grid under it cut that to 0.5-1.3 ms and made the complete automatic
# solve 1.44× faster geometrically over all 15 benchmark images.
const DEFAULT_GPU_MAX_COARSE = 14_000
# As the fine problem grows, its device apply dominates a 16k-32k host solve.
# Retaining a stronger direct coarse correction then avoids a whole benchmark
# iteration rung: 16k recovered the previous target rung across the measured
# 800³ matrix, and 32k did the same at 1000³ and porosity 0.6. The node count
# expresses that cost ratio without tying the route to cubic images or a named
# domain size.
const MID_GPU_MAX_COARSE = 16_000
const LARGE_GPU_MAX_COARSE = DEFAULT_MAX_COARSE
const MID_GPU_FINE_NODES = 250_000_000
const LARGE_GPU_FINE_NODES = 500_000_000
# Thin coarse grids have banded factors whose host solve stays cheap; keep the
# CPU ceiling until all three directions are large enough to create 3-D fill.
const MIN_GPU_COARSE_EDGE = 8

# Edge length in voxels of a coarse block. Fixed, and deliberately so: the ratio
# between the fine and coarse grids is what decides whether the method is
# mesh-independent, and growing the block with the image is what used to cost
# this preconditioner an iteration count proportional to the image edge.
const DEFAULT_COARSE_BLOCK = 8

# Edge ratio between one coarse grid and the next. A two-level method is
# mesh-independent only while the ratio is bounded, and 2 is both the standard
# choice and the measured best: 152 iterations on a 256³ spot check against 187
# at ratio 3 and 194 at ratio 4.
const COARSE_RATIO = 2

# Damped-Jacobi weight for the coarse smoother.
#
# Every operator in the hierarchy is a weakly diagonally dominant M-matrix — the
# aggregated row sums inherit the fine operator's, and the shift keeps the
# diagonal strictly the larger — so `λmax(D⁻¹A) ≤ 2` and the symmetric cycle is
# SPD for any `ω < 1`. Measured on both coarse levels of a 256³ case: 1.996.
#
# That bound is not slack. At `ω = 1.2` CG rejects the preconditioner outright.
# 0.8 keeps a real margin (2/0.8 = 2.5 against the measured 1.996) and costs
# 152 iterations against 150 at an unsafe 1.0.
const COARSE_SMOOTH_OMEGA = 0.8

# Rows below which the coarse apply runs serially. Threading it costs a flat
# ~11 µs of startup whatever the size, against a serial pass that is ~10 µs at
# 4096 rows and ~40 µs at 15 625 — so this is the measured crossover, and the
# levels below it are the ones where the startup would be the whole cost.
const COARSE_MUL_MIN_THREADED = 4096
# Prolongation does less work per element than a sparse row product, so thread
# startup pays back later: 100k entries is the measured crossover. At 319k it
# is 4.6× faster, and at 49k it is still slower than the serial loop.
const PROLONG_MIN_THREADED = 100_000

# Relative diagonal shift applied to the coarse operator before factorisation.
# `WᵀAW` is only positive *semi*-definite — a pore cluster that reaches neither
# Dirichlet face spans blocks whose coarse rows sum to zero — so a shift is what
# makes the Cholesky exist at all. See `two_level_preconditioner` for why the
# size of the shift is a tradeoff rather than "as small as possible".
const DEFAULT_COARSE_SHIFT = 1.0e-3

"""
    CoarseLevel

One grid of the coarse hierarchy, holding what a V-cycle needs at that level:
the operator, the smoother, the map onto the next grid down, and the scratch the
cycle writes through.

Every level lives on the host in `Float64`, next to the direct solve the cycle
ends in. The first one *is* the coarse space — its `A` is the coarse operator
itself — and each one below is `COARSE_RATIO^3` smaller than the one above, so
everything under the coarse space costs a fixed few percent of one
preconditioner application whatever the image size — measured at 2.1% on a 256³
case.

# Fields
- `A`: this level's operator: the coarse operator at the first level, and the
  Galerkin product `WᵀA₊W` of the level above at every level under it.
- `dinv`: `COARSE_SMOOTH_OMEGA ./ diag(A)`, the damped-Jacobi smoother.
- `parent`: this level's cell → the next level's cell, `0` where dropped. Read by
  `_restrict!` and `_prolong!` exactly as `agg` is at the fine level.
- `t`, `rc`, `ec`: scratch for the residual here, and for the residual and
  correction one level down.
"""
struct CoarseLevel
    A::SparseMatrixCSC{Float64,Int}
    dinv::Vector{Float64}
    parent::Vector{Int32}
    t::Vector{Float64}
    rc::Vector{Float64}
    ec::Vector{Float64}
end

"""
    TwoLevelPreconditioner

Left preconditioner for the steady diffusion system, applied through
`ldiv!(y, P, x)`:

    y = W (WᵀAW + ρ·diag)⁻¹ Wᵀ x  +  x / λmax

`W` holds the indicator of each cubic block of voxels, so `Wᵀ x` sums the
residual over a block and `W xc` broadcasts a block's correction back to its
voxels. Build one with [`two_level_preconditioner`](@ref).

Two-level is the shape of the preconditioner, not the depth of its machinery.
The coarse inverse above is applied by a direct solve when the coarse space is
small, and by a V-cycle over [`CoarseLevel`](@ref) grids ending in that same
direct solve when it is not. Which one runs changes the cost of an application,
never the operator it stands for.

# Fields
- `agg`: coarse index of each pore voxel, `0` where the voxel's block was
  dropped for carrying no coarse unknown. On a device this is an
  [`Aggregation`](@ref), which carries the same map plus its inverse.
- `nc`: number of coarse unknowns.
- `levels`: the grids the V-cycle sweeps, outermost first. `levels[1]` is the
  coarse space itself and the rest are the coarser grids interposed between it
  and the direct solve. Empty when the coarse space is solved directly.
- `fact`: host Cholesky factorisation of the shifted operator of the *coarsest*
  level, always in `Float64` regardless of the fine precision. That is the
  coarse operator itself when `levels` is empty.
- `inv_lambda`: reciprocal of a Gershgorin bound on `λmax(A)`.
- `block`: edge length in voxels of one coarse block.
"""
struct TwoLevelPreconditioner{T,Vi,Vc<:AbstractVector{T},F}
    agg::Vi
    nc::Int
    levels::Vector{CoarseLevel}
    fact::F
    inv_lambda::T
    block::Int
    rc::Vc                  # nc-length device scratch, restriction result
    xc::Vc                  # nc-length device scratch, coarse correction
    rc_host::Vector{T}
    coarse_rhs::Vector{Float64}
    coarse_sol::Vector{Float64}
end

function Base.show(io::IO, P::TwoLevelPreconditioner)
    depth = isempty(P.levels) ? "" : ", levels=$(length(P.levels) + 2)"
    return print(
        io,
        "TwoLevelPreconditioner(block=$(P.block)^3, nc=$(P.nc)$(depth), \
         nnz(L)=$(nnz(P.fact)))",
    )
end

Base.size(P::TwoLevelPreconditioner) = (length(P.agg), length(P.agg))
Base.eltype(::TwoLevelPreconditioner{T}) where {T} = T

# --- Coarse-space construction ---------------------------------------------

@kernel function _aggregate_kernel!(agg, @Const(idx), bs, nbx, nby)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        c0 = idx[i, j, k]
        if c0 > 0
            bi = div(i - 1, bs)
            bj = div(j - 1, bs)
            bk = div(k - 1, bs)
            agg[c0] = (bi + 1) + nbx * bj + nbx * nby * bk
        end
    end
end

# Which stencil slot an off-diagonal belongs in, from the signed difference of
# the two block indices. Two face-neighbour voxels always land in blocks that
# differ by one of these six offsets, so `0` means "not a face coupling" and the
# entry is dropped rather than misfiled.
#
# When the block grid is one deep along a dimension two of the six offsets
# coincide, and a coupling can then be filed under a slot naming the wrong
# direction. That is harmless: the direction whose offset was taken over has no
# block neighbour to couple to, so its own slot stays empty, and both slots
# describe the same pair of blocks — `_coarse_operator` emits each once and the
# two halves add back up. `test_preconditioner.jl` pins it with a slab thinner
# than one block.
@inline function _coarse_slot(d, nbx, nbxy)
    d == -nbxy && return 2
    d == -nbx && return 3
    d == -1 && return 4
    d == 1 && return 5
    d == nbx && return 6
    d == nbxy && return 7
    return 0
end

# `WᵀAW` in one pass over the stored entries of A: entry `A[r, j]` belongs to
# coarse entry `(agg[r], agg[j])`. Everything a thread contributes to its own
# block accumulates in a register, so the common case — an interior voxel whose
# six neighbours share its block — costs a single atomic.
@kernel function _coarse_stencil_kernel!(
    stencil, @Const(colptr), @Const(rowval), @Const(nzval), @Const(agg), nbx, nbxy, n
)
    j = @index(Global)
    if j <= n
        @inbounds a = Int(agg[j])
        self = zero(eltype(stencil))
        @inbounds for k in colptr[j]:(colptr[j + 1] - 1)
            b = Int(agg[rowval[k]])
            v = eltype(stencil)(nzval[k])
            if b == a
                self += v
            else
                s = _coarse_slot(b - a, nbx, nbxy)
                s == 0 && continue
                Atomix.@atomic stencil[(a - 1) * _COARSE_SLOTS + s] += v
            end
        end
        if !iszero(self)
            @inbounds Atomix.@atomic stencil[(a - 1) * _COARSE_SLOTS + 1] += self
        end
    end
end

# `max` has no atomic instruction for floating-point values on the host backend,
# so the running maximum is taken by compare-and-swap. The ordinary read that
# seeds the loop is deliberate: a thread that cannot raise the maximum touches
# nothing, which is what keeps one shared slot from serialising the whole grid
# pass, and a read that loses the race only costs the loop one more turn.
@inline function _atomic_max!(dst, i, v)
    ref = Atomix.IndexableRef(dst, (Int(i),))
    @inbounds old = dst[i]
    while v > old
        replaced = Atomix.replace!(ref, old, v)
        replaced.success && return nothing
        old = replaced.old
    end
    return nothing
end

# The same `WᵀAW` the kernel above accumulates, taken over the grid instead of
# over stored entries, for an operator that stores none. A thread owns one pore
# voxel and recomputes the column [`_steady_fill_kernel!`](@ref) would have
# written for it, so the entry set has to be that kernel's exactly: a Dirichlet
# column holds only its diagonal, a free column holds its diagonal plus one
# entry per free neighbour, and a free column with no neighbour at all holds
# nothing.
#
# `dmax` collects the largest diagonal in the same pass. That is the largest
# stored value of the matrix this operator stands for, because the diagonals are
# its only positive entries.
@kernel function _coarse_grid_stencil_kernel!(
    stencil, dmax, @Const(idx), @Const(agg), D, nx, ny, nz, bcdim, nbc, D0, nbx, nbxy,
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        c0 = idx[i, j, k]
        if c0 > 0
            Tv = typeof(D0)
            Ts = eltype(stencil)
            a = Int(agg[c0])
            self_bc = _is_bc(_face_coord(i, j, k, bcdim), nbc)
            da = _node_diffusivity(D, D0, i, j, k)

            deg = zero(Tv)
            self = zero(Ts)
            for (di, dj, dk) in _NEIGHBOURS
                ii, jj, kk = i + di, j + dj, k + dk
                (1 <= ii <= nx && 1 <= jj <= ny && 1 <= kk <= nz) || continue
                q = idx[ii, jj, kk]
                q > 0 || continue
                w = _edge_weight(D, D0, da, _node_diffusivity(D, D0, ii, jj, kk))
                deg += w
                # The degree sums every pore neighbour, but only a free node
                # facing a free neighbour leaves an off-diagonal behind: a
                # Dirichlet column was emptied, and a Dirichlet neighbour's
                # entry was eliminated into the right-hand side.
                (self_bc || _is_bc(_face_coord(ii, jj, kk, bcdim), nbc)) && continue
                b = Int(agg[q])
                if b == a
                    self += Ts(-w)
                else
                    s = _coarse_slot(b - a, nbx, nbxy)
                    s == 0 && continue
                    Atomix.@atomic stencil[(a - 1) * _COARSE_SLOTS + s] += Ts(-w)
                end
            end

            # A free node with no neighbour has an empty column: no diagonal to
            # contribute, and nothing to fold into the running maximum either.
            if self_bc || !iszero(deg)
                d = (self_bc && iszero(deg)) ? one(Tv) : deg
                self += Ts(d)
                _atomic_max!(dmax, 1, d)
            end
            if !iszero(self)
                Atomix.@atomic stencil[(a - 1) * _COARSE_SLOTS + 1] += self
            end
        end
    end
end

"""
Accumulate `WᵀAW` into `stencil` and return the largest diagonal of `A`.

The two methods are the same coarse operator read out of the two
representations: one pass over the stored entries of an assembled matrix, or one
pass over the grid for a [`MaskedLaplacian`](@ref).
"""
function _coarse_stencil!(stencil, A, agg, nbx, nbxy)
    nzv = nonzeros(A)
    n = size(A, 1)
    backend = get_backend(nzv)
    _coarse_stencil_kernel!(backend)(
        stencil, SparseArrays.getcolptr(A), rowvals(A), nzv, agg,
        nbx, nbxy, n; ndrange=n,
    )
    KernelAbstractions.synchronize(backend)
    # The diagonal entries are the only positive ones, so the largest stored
    # value is the largest diagonal — the quantity the grid pass reports.
    return maximum(nzv)
end

function _coarse_stencil!(stencil, A::MaskedLaplacian, agg, nbx, nbxy)
    T = eltype(A)
    nx, ny, nz = size(A.idx)
    backend = get_backend(A.idx)
    dmax = fill!(similar(A.idx, T, 1), zero(T))
    _coarse_grid_stencil_kernel!(backend, _steady_workgroup(A.idx))(
        stencil, dmax, A.idx, agg, A.D, nx, ny, nz, A.bcdim, A.nbc, A.D0, nbx, nbxy;
        ndrange=(nx, ny, nz),
    )
    KernelAbstractions.synchronize(backend)
    maximum_diagonal = Array(dmax)[1]
    _free!(dmax)
    return maximum_diagonal
end

@kernel function _remap_aggregates_kernel!(agg, @Const(remap), n)
    i = @index(Global)
    if i <= n
        @inbounds agg[i] = remap[agg[i]]
    end
end

"""
    _coarse_diagonal_floor(bs, maximum_diagonal)

Below what value an accumulated coarse diagonal is round-off rather than a
coarse unknown.

A block whose pore voxels all belong to clusters contained inside it has a
coarse diagonal of **exactly** zero: the degrees and the couplings that make it
up cancel term for term. Summed in floating point they cancel to a residue
instead, whose sign is decided by the order the threads happened to arrive in.
So `> 0` keeps such a block roughly half the time, and what it keeps is a coarse
row with a ~1e-16 diagonal — a direction along which the coarse solve amplifies
by 1e16. Measured on a 32³ blob with variable `D` at `block=2`: the assembled
path reached `+2.22e-16` where the matrix-free path reached `0.0`, and keeping it
took `‖ldiv!‖∞` from 6.9 to **5.6e15**.

The floor is the round-off bound for that sum. A block accumulates at most
`_COARSE_SLOTS` terms per voxel, each no larger than the biggest diagonal of `A`,
so a cancelling sum cannot leave more than `n·eps` times that behind; the
constant is slack on top. Nothing legitimate is anywhere near it — the smallest
diagonal a real coarse unknown can carry is a single edge weight, and in the case
above the kept diagonals went straight from `2.2e-16` to `0.76`.
"""
_coarse_diagonal_floor(bs, maximum_diagonal) =
    64 * _COARSE_SLOTS * bs^3 * eps(Float64) * abs(maximum_diagonal)

"""
Assemble the coarse operator on the host from the accumulated stencil.

Returns `(Ac, remap)`: the shifted coarse matrix over the blocks that carry a
coarse unknown, and the `nc0`-length map from block index to coarse index with
`0` for the blocks that were dropped.

A block is dropped when its coarse diagonal is not positive. That happens
exactly when the block holds no pore voxel, or when every pore voxel it holds
belongs to a cluster contained entirely within the block and touching neither
Dirichlet face — in which case the block's coarse basis function lies in the
null space of `A` and carries no information to correct.

Such a block's diagonal is a sum that cancels to *exactly* zero, so `diag_floor`
is what decides it rather than a comparison against zero. See
[`_coarse_diagonal_floor`](@ref) for why testing against zero is not safe.

Opposite stencil slots hold the same sum accumulated in a different order, so
averaging them makes `Ac` symmetric to the last bit rather than merely close.
"""
function _coarse_operator(S::Vector{Float64}, nc0, nbx, nbxy, shift, diag_floor)
    offs = (0, -nbxy, -nbx, -1, 1, nbx, nbxy)
    remap = zeros(Int32, nc0)
    nc = 0
    for a in 1:nc0
        if S[(a - 1) * _COARSE_SLOTS + 1] > diag_floor
            nc += 1
            remap[a] = nc
        end
    end

    rows = Int[]
    cols = Int[]
    vals = Float64[]
    sizehint!(rows, _COARSE_SLOTS * nc)
    sizehint!(cols, _COARSE_SLOTS * nc)
    sizehint!(vals, _COARSE_SLOTS * nc)
    for a in 1:nc0
        anew = remap[a]
        anew == 0 && continue
        base = (a - 1) * _COARSE_SLOTS
        push!(rows, anew)
        push!(cols, anew)
        push!(vals, S[base + 1] * (1 + shift))
        for s in 2:_COARSE_SLOTS
            b = a + offs[s]
            (1 <= b <= nc0) || continue
            bnew = remap[b]
            bnew == 0 && continue
            v = 0.5 * (S[base + s] + S[(b - 1) * _COARSE_SLOTS + (9 - s)])
            iszero(v) && continue
            push!(rows, anew)
            push!(cols, bnew)
            push!(vals, v)
        end
    end
    return sparse(rows, cols, vals, nc, nc), remap
end

"""
Each cell's parent under a `COARSE_RATIO`-fold coarsening of the block grid.

`cell_block` is the block index of every cell of the current level, in a grid of
`dims` blocks. Returns the parent block index of each, and the coarsened grid.

Levels below the first are structured — one cell per surviving block of a
regular grid — so a parent is integer division of the block's coordinates, and
no geometry has to be carried down from the image.
"""
function _coarse_parents(cell_block, dims)
    nbx, nby = dims[1], dims[2]
    cdims = cld.(dims, COARSE_RATIO)
    cnx, cny = cdims[1], cdims[2]
    pblock = Vector{Int}(undef, length(cell_block))
    @inbounds for (c, b0) in enumerate(cell_block)
        b = b0 - 1
        bi = b % nbx
        bj = (b ÷ nbx) % nby
        bk = b ÷ (nbx * nby)
        pblock[c] = 1 + (bi ÷ COARSE_RATIO) + cnx * (bj ÷ COARSE_RATIO) +
                    cnx * cny * (bk ÷ COARSE_RATIO)
    end
    return pblock, cdims
end

"""
Build the grids the V-cycle sweeps between the coarse space and the direct solve.

Returns `(levels, Acoarsest)`: the hierarchy outermost first, and the operator
the caller is to factorise. `levels[1]` holds `Ac` itself, so the interposed
grids are `levels[2:end]`. `levels` is empty and `Acoarsest === Ac` when the
coarse space already fits under `max_coarse`, which is the two-level method
unchanged.

`Ac` is the coarse operator over the block grid `dims` and `remap` maps a block
index to its coarse index, so inverting `remap` recovers where each coarse cell
sits. From there each level is the Galerkin product `WᵀA W` over a
`COARSE_RATIO`-fold aggregation of the grid, which is again an operator on a
regular grid and can be coarsened the same way.

A coarse cell is dropped against the same `diag_floor` [`_coarse_operator`](@ref)
uses, so one rule decides it at every level. It is rare below the first one but
not impossible: an aggregate's diagonal is the sum of every entry of `A` it
covers, which for a cluster enclosed within the aggregate collapses to the
shift's share of those diagonals — `shift` times a diagonal that only had to
clear the floor itself. Such a cell is dropped and gets no correction from below,
which is correct rather than merely tolerated: it is the same null direction the
first-level rule exists to remove. Coarsening stops if it ever fails to shrink
the problem, so the loop terminates whatever `max_coarse` is asked for.
"""
function _coarse_hierarchy(Ac, remap, dims, max_coarse, diag_floor)
    levels = CoarseLevel[]
    A = Ac
    # Coarse index → block index, the inverse of the map `_coarse_operator` built.
    cell_block = zeros(Int, size(A, 1))
    for (b, c) in enumerate(remap)
        c > 0 && (cell_block[c] = b)
    end

    while size(A, 1) > max_coarse
        n = size(A, 1)
        pblock, cdims = _coarse_parents(cell_block, dims)
        nc0 = prod(cdims)
        W = sparse(1:n, pblock, ones(Float64, n), n, nc0)
        # `(B + Bᵀ)/2` rather than `B`: entry `(i, j)` of the Galerkin product
        # sums the same terms as `(j, i)` but in the transposed order, and a
        # float sum is not associative, so the product is symmetric only to
        # rounding. Averaging makes it symmetric to the last bit — float addition
        # is commutative and halving is exact — which is what lets
        # [`_coarse_mul!`](@ref) read a column as a row.
        B = W' * A * W
        Anext = (B + transpose(B)) / 2

        keep = findall(>(diag_floor), diag(Anext))
        # Nothing to correct on, or a coarsening that buys nothing: stop and let
        # the caller factorise what is already here.
        (isempty(keep) || length(keep) >= n) && break

        cmap = zeros(Int32, nc0)
        for (i, c) in enumerate(keep)
            cmap[c] = Int32(i)
        end
        parent = Int32[cmap[p] for p in pblock]

        push!(levels, CoarseLevel(
            A, COARSE_SMOOTH_OMEGA ./ diag(A), parent,
            zeros(n), zeros(length(keep)), zeros(length(keep)),
        ))
        A = Anext[keep, keep]
        cell_block = keep
        dims = cdims
    end
    return levels, A
end

"""
    _coarse_mul!(y, A, x)

`y = A x` for a coarse operator, read down its columns rather than across them.

Every operator in the hierarchy is symmetric to the last bit — `_coarse_operator`
averages opposite stencil slots and [`_coarse_hierarchy`](@ref) averages each
Galerkin product — so column `j` of `A` *is* row `j`, and the product can be a
gather of independent dot products instead of the scatter `mul!` performs. That
buys two things the scatter cannot give: the rows split across threads with no
atomic, and each output element is summed in one fixed order, so the result does
not depend on the schedule. It is also bit-identical to `mul!`, since both walk a
row's entries in ascending index order.

Two of these run per level per CG iteration and they are the bulk of the cycle's
cost — 0.609 ms of 0.888 ms at 49 642 rows before threading, 0.068 ms after.
"""
function _coarse_mul!(y, A, x)
    cp = SparseArrays.getcolptr(A)
    rv = rowvals(A)
    nz = nonzeros(A)
    n = size(A, 2)
    if n >= COARSE_MUL_MIN_THREADED
        Threads.@threads for j in 1:n
            acc = zero(eltype(y))
            @inbounds for k in cp[j]:(cp[j + 1] - 1)
                acc += nz[k] * x[rv[k]]
            end
            @inbounds y[j] = acc
        end
    else
        @inbounds for j in 1:n
            acc = zero(eltype(y))
            for k in cp[j]:(cp[j + 1] - 1)
                acc += nz[k] * x[rv[k]]
            end
            y[j] = acc
        end
    end
    return y
end

"""
Solve the coarsest system into `e`, without allocating where the backend allows
it.

`SparseArrays`' CHOLMOD carries a three-argument `ldiv!` for plain vectors from
Julia 1.12 on. Below that the generic `LinearAlgebra` fallback takes the call and
delegates to a two-argument method CHOLMOD never defines, so it raises rather
than solving, and the throwaway coarse-sized vector is unavoidable. Both forms
give the same answer bit for bit.
"""
@static if VERSION >= v"1.12"
    _coarse_solve!(e, fact, r) = ldiv!(e, fact, r)
else
    _coarse_solve!(e, fact, r) = copyto!(e, fact \ r)
end

"""
Apply one symmetric V(1,1) cycle at level `l`: `e ← B r`, with `B ≈ A⁻¹`.

Pre-smooth from a zero guess, correct from the level below, post-smooth. The
smoother is diagonal and so is its own transpose, and the coarsest solve is a
Cholesky, which makes `B` symmetric — the property CG needs and the reason the
cycle is not truncated to a cheaper one-sided sweep.

Below the last level the residual is solved exactly, so a preconditioner with no
levels at all takes that branch directly and is the plain direct coarse solve.
"""
function _vcycle!(e, levels, l, r, fact)
    if l > length(levels)
        # `_coarse_solve!` rather than `e .= fact \ r`: this runs once per CG
        # iteration, and the two are bit-identical while only one of them is
        # free to skip the coarse-sized vector thrown away on every call.
        _coarse_solve!(e, fact, r)
        return e
    end
    L = levels[l]
    @. e = L.dinv * r
    _coarse_mul!(L.t, L.A, e)
    @. L.t = r - L.t
    _restrict!(L.rc, L.parent, L.t)
    _vcycle!(L.ec, levels, l + 1, L.rc, fact)
    # `e` is both the correction being added to and the vector carrying the
    # identity term, which is what `_prolong!`'s last argument scales. Every
    # method of it reads and writes index `i` alone, so the two may alias.
    _prolong!(e, L.parent, L.ec, e, 1.0)
    _coarse_mul!(L.t, L.A, e)
    @. L.t = r - L.t
    @. e += L.dinv * L.t
    return e
end

# The device array every allocation here is modelled on, and whose backend the
# kernels launch on: the stored values of an assembled matrix, the index grid of
# a matrix-free one.
_precond_template(A) = nonzeros(A)
_precond_template(A::MaskedLaplacian) = A.idx

function _gpu_max_coarse(n)
    n >= LARGE_GPU_FINE_NODES && return LARGE_GPU_MAX_COARSE
    n >= MID_GPU_FINE_NODES && return MID_GPU_MAX_COARSE
    return DEFAULT_GPU_MAX_COARSE
end

function _resolve_max_coarse(A, max_coarse, dims)
    isnothing(max_coarse) || return max_coarse
    use_gpu_ceiling = _on_gpu(_precond_template(A)) && minimum(dims) >= MIN_GPU_COARSE_EDGE
    return use_gpu_ceiling ? _gpu_max_coarse(size(A, 1)) : DEFAULT_MAX_COARSE
end

"""
Coarse index of every pore voxel: the cubic block its grid position falls in.

`idx` is the pore numbering the caller already holds — the assembled path builds
one for this pass, the matrix-free operator is one — and stays the caller's to
release.
"""
function _aggregate(idx, n, nc0, bs, nbx, nby)
    # `agg` is one entry per pore voxel and stays alive for the whole solve, so
    # its element type is worth narrowing. The block edge is fixed, so `nc0`
    # grows with the image and leaves `Int16` at a 249³ image (32³ blocks);
    # everything above that pays 4 bytes per pore voxel here, plus the same
    # again for [`Aggregation`](@ref)'s `fine` on a device.
    Ta = nc0 <= typemax(Int16) ? Int16 : Int32
    agg = similar(idx, Ta, n)
    backend = get_backend(idx)
    _aggregate_kernel!(backend, _steady_workgroup(idx))(
        agg, idx, bs, nbx, nby; ndrange=size(idx),
    )
    KernelAbstractions.synchronize(backend)
    return agg
end

"""
Coarse-to-fine adjacency: the aggregation read the other way round, in CSR form.

`_restrict!` sums each coarse cell's fine values. Scattering that sum with
`Atomix.@atomic` leaves its order up to whichever thread blocks arrive first, and
a float sum is not associative, so the same solve returns a slightly different
answer on every launch. Gathering over a fixed adjacency needs no atomic and
fixes the order, and the ordering is paid once here rather than on every CG
iteration.

The pore nodes of coarse cell `a` are `fine[offsets[a]:(offsets[a + 1] - 1)]`,
ascending. Device restriction needs it to avoid nondeterministic atomics;
multithreaded host restriction uses the same map to gather coarse cells in
parallel without changing the serial summation order within any cell.
"""
struct Aggregation{Vf<:AbstractVector,Vo<:AbstractVector}
    fwd::Vf                 # fine -> coarse, 0 where the block was dropped
    offsets::Vo             # nc+1 CSR offsets into `fine`
    fine::Vo                # pore nodes by coarse cell, ascending within a cell
end

# The forward map is what a caller means by "the aggregation", so how long it is,
# what is in it and where it lives all read through to it.
Base.length(a::Aggregation) = length(a.fwd)
Base.Array(a::Aggregation) = Array(a.fwd)
_on_gpu(a::Aggregation) = _on_gpu(a.fwd)
_async_return_safe(a::Aggregation) = _async_return_safe(a.fwd)
KernelAbstractions.get_backend(a::Aggregation) = get_backend(a.fwd)
_free!(a::Aggregation) = (_free!(a.fwd); _free!(a.offsets); _free!(a.fine); nothing)

"""
Invert `agg` into the adjacency [`_restrict!`](@ref) gathers over.

A counting sort on the host, in chunks: the nodes are split into contiguous
ranges, each range counts how many nodes it gives every cell, and the counts are
then run into a position for each range within each cell. Filing is what costs —
it writes across the whole output — so both passes run one chunk per thread.

Reserving each chunk's positions before anything is written is what keeps the
result off the thread schedule: a chunk always files into its own slice, in
ascending node order, and the chunks are laid down in ascending order within each
cell. So every cell's slice comes out ascending, whichever thread got there
first, and no sort is needed to make it so.
"""
function _invert_aggregates(agg, nc; max_scratch_bytes=256 * 1024 * 1024)
    fwd = agg isa Array ? agg : Array(agg)
    n = length(fwd)
    # One entry per pore node, so the same index wall the fine numbering faces.
    Tf = n <= typemax(Int32) ? Int32 : Int64
    # `counts` and `cursor` are both `nc x nchunks`, so this scratch grows with the
    # image and the thread count at once. A 1000^3 image has ~1.95M coarse cells,
    # which on a 64-thread host is a gigabyte of host memory holding a table used
    # only to reserve positions — and it is invisible to the device-side memory
    # model the solver is sized against. Cap the chunk count to keep the two
    # matrices inside a fixed budget.
    #
    # This costs parallelism here and nothing else: the layout below files each
    # chunk into its own reserved slice in ascending node order, so a cell's slice
    # comes out ascending for any number of chunks. The result is identical
    # whether this runs on one chunk or sixty-four, which is what the two-pass
    # reservation exists to guarantee.
    ndivs = clamp(max_scratch_bytes ÷ (2 * max(nc, 1) * sizeof(Tf)), 1, Threads.nthreads())
    bounds = find_chunk_bounds(; nelems=n, ndivs=ndivs)
    nchunks = length(bounds)

    # A column per chunk: a thread walks its own column, and two threads' columns
    # are far enough apart not to share a cache line.
    counts = zeros(Tf, nc, nchunks)
    Threads.@threads for c in 1:nchunks
        lo, hi = bounds[c]
        @inbounds for i in lo:hi
            a = fwd[i]
            a > 0 && (counts[a, c] += 1)
        end
    end

    offsets = Vector{Tf}(undef, nc + 1)
    cursor = Matrix{Tf}(undef, nc, nchunks)
    pos = one(Tf)
    @inbounds for a in 1:nc
        offsets[a] = pos
        for c in 1:nchunks
            cursor[a, c] = pos
            pos += counts[a, c]
        end
    end
    offsets[nc + 1] = pos

    fine = Vector{Tf}(undef, pos - 1)
    Threads.@threads for c in 1:nchunks
        lo, hi = bounds[c]
        @inbounds for i in lo:hi
            a = fwd[i]
            if a > 0
                fine[cursor[a, c]] = i
                cursor[a, c] += 1
            end
        end
    end
    return offsets, fine
end

"""
Everything the two constructors share once the aggregates exist: the coarse
operator, its factorisation, the remap that renumbers `agg` over the blocks that
survived, and the preconditioner itself. `nothing` when no block carries a
coarse unknown or the factorisation fails.

Only [`_coarse_stencil!`](@ref) knows which representation `A` is.
"""
function _two_level_from_aggregates(A, agg, bs, nbx, nby, nbz, shift, max_coarse, verbose)
    n = size(A, 1)
    nc0 = nbx * nby * nbz
    nbxy = nbx * nby
    proto = _precond_template(A)
    on_gpu = _on_gpu(proto)
    backend = get_backend(proto)
    T = eltype(A)

    stencil = similar(proto, Float64, _COARSE_SLOTS * nc0)
    fill!(stencil, 0.0)
    maximum_diagonal = _coarse_stencil!(stencil, A, agg, nbx, nbxy)
    S = Array(stencil)
    _free!(stencil)

    diag_floor = _coarse_diagonal_floor(bs, maximum_diagonal)
    Ac, remap = _coarse_operator(S, nc0, nbx, nbxy, shift, diag_floor)
    nc = size(Ac, 1)
    if nc == 0
        _free!(agg)
        return nothing
    end

    t0 = time_ns()
    # The block edge is fixed, so a large image gives a large coarse space rather
    # than a coarser one. Grids below it carry that space down to a size the
    # direct solve can still afford.
    levels, Acoarsest =
        _coarse_hierarchy(Ac, remap, (nbx, nby, nbz), max_coarse, diag_floor)
    fact = try
        cholesky(Symmetric(Acoarsest))
    catch err
        # `Ac + shift·diag(Ac)` is provably definite, and every Galerkin product
        # below it inherits that, so this only fires if the assumption behind it
        # — that `A` itself is positive semi-definite — has been broken. Running
        # without a preconditioner is slow; running with a broken one is wrong.
        @warn "coarse factorisation failed; solving without a preconditioner" exception = err
        _free!(agg)
        return nothing
    end
    verbose && @info "two-level coarse space" block = bs nc = nc levels = length(levels) + 2 coarsest =
        size(Acoarsest, 1) nnz_L = nnz(fact) seconds = (time_ns() - t0) / 1e9

    # Same element type as `agg`, so the remap kernel never converts in device
    # code where a range check has nowhere to throw.
    remap = convert(Vector{eltype(agg)}, remap)
    remap_dev = on_gpu ? _gpu_adapt[](remap) : remap
    _remap_aggregates_kernel!(backend)(agg, remap_dev, n; ndrange=n)
    KernelAbstractions.synchronize(backend)
    on_gpu && _free!(remap_dev)

    # Inverted after the remap, so the adjacency is indexed by the coarse
    # numbering that survived rather than the one the blocks started with.
    # Device arrays are adapted back to their backend; multithreaded host arrays
    # use the same representation directly so restriction can gather in
    # parallel. A one-thread host keeps the serial scatter and its lower setup
    # and storage cost.
    use_host_gather = Threads.nthreads() > 1 && n >= PROLONG_MIN_THREADED
    if on_gpu || use_host_gather
        offsets, fine = _invert_aggregates(agg, nc)
        agg = on_gpu ?
            Aggregation(agg, _gpu_adapt[](offsets), _gpu_adapt[](fine)) :
            Aggregation(agg, offsets, fine)
    end

    # Gershgorin: every column of this Laplacian has |offdiagonals| summing to
    # its diagonal, so twice the largest diagonal bounds every eigenvalue.
    inv_lambda = T(1) / (2 * maximum_diagonal)

    return TwoLevelPreconditioner(
        agg, nc, levels, fact, inv_lambda, bs,
        fill!(similar(proto, T, nc), zero(T)), fill!(similar(proto, T, nc), zero(T)),
        zeros(T, nc), zeros(Float64, nc), zeros(Float64, nc),
    )
end

"""
    two_level_preconditioner(A, img; block=nothing, max_coarse, shift, verbose=false)
    two_level_preconditioner(sim::SteadyDiffusionProblem; kwargs...)

Build a [`TwoLevelPreconditioner`](@ref) for the steady system `A` on the pore
mask `img`, or `nothing` when no usable coarse space exists. `A` is either an
assembled matrix or a matrix-free [`MaskedLaplacian`](@ref); the coarse space is
the same either way, read out of the stored entries in one case and off the grid
in the other.

Pass it to the solver as a left preconditioner:

```julia
sim = SteadyDiffusionProblem(img; axis=:x)
Pl = two_level_preconditioner(sim)
sol = solve(sim.prob, KrylovJL_CG(); Pl=Pl, reltol=1e-6)
```

Unpreconditioned CG on this operator needs a number of iterations that grows
with the image edge length, because the error modes it converges on last are the
ones that span the whole domain. A coarse space of block indicators resolves
exactly those modes and removes that growth; the `x / λmax` term covers the
high-frequency error the coarse space is blind to.

Removing the growth needs the ratio between the two grids to stay bounded, which
is why the block edge is fixed rather than sized from the image. A large image
therefore gets a large coarse space, and that space is solved by a V-cycle over
coarser grids rather than directly. Measured across 64³–256³ at a coarsening
ratio matched to the released one, iterations go flat where they used to track
the image edge: at ε≈0.5, 222 / 362 / 798 before against 222 / 204 / 152 after,
and at ε≈0.2, 614 / 1203 / 2333 against 614 / 454 / 422.

The result is a genuine preconditioner, not an approximate solve: it is
symmetric positive definite by construction, so CG converges to the same
solution as it would without it. Only the iteration count changes.

Worth building only when the unpreconditioned iteration count is large, which
in practice means images of a few hundred voxels per side and up. Measured on
blob images (seed 42, porosity 0.5) with the default block size: 400³ takes
2094 iterations and 11.4 s without it against 99 and 1.01 s with it, and 600³
2983 and 50.2 s against 132 and 3.66 s. On a 12³ image it costs iterations
rather than saving them.

Note that CG's stopping test is taken in the `M⁻¹` norm once a preconditioner is
present, so at the same `reltol` the two runs stop at slightly different true
residuals. They agree on `tortuosity` to solver tolerance — verified to 1e-9 at
`reltol=1e-10` — but not bit for bit at a loose one.

# Keyword Arguments
- `block`: edge length in voxels of a coarse block. `nothing` (default) is
  `DEFAULT_COARSE_BLOCK`, the same edge at every image size — see above for why
  it does not grow with the image.
- `max_coarse`: ceiling on the number of unknowns solved *directly*. `nothing`
  (default) selects 14,000 for a genuinely three-dimensional GPU coarse grid,
  rising to 16,000 at 250 million fine nodes and 32,000 at 500 million; CPU and
  thin grids use 32,000. The direct solve runs once per iteration, so a larger
  one stops paying for itself on smaller fine grids; a coarse space above the
  ceiling gets coarser grids built under it instead of being made coarser
  itself.
- `shift`: relative diagonal shift applied before factorisation. `WᵀAW` is
  positive semi-definite, not definite — a pore cluster reaching neither
  Dirichlet face spans blocks whose coarse rows sum to zero — so the shift is
  what makes the factorisation exist. It also bounds how far the coarse solve
  can amplify a residual that lies along one of those null directions, which is
  why the default is not the smallest number that works.
- `verbose`: report the coarse size, the depth of the hierarchy under it, and the
  setup cost.
"""
function two_level_preconditioner(
    A, img;
    block=nothing, max_coarse=nothing, shift=DEFAULT_COARSE_SHIFT,
    verbose=false,
)
    @assert shift > 0 "`shift` must be positive; the coarse operator is only positive semi-definite"
    img = atleast_3d(img)
    n = size(A, 1)
    # `agg` is sized from `n` but filled at the pore ordinals `img` produces, so a
    # mask that disagrees with the matrix leaves entries unwritten — and `agg` is
    # read back as an unchecked index into `stencil`. Uninitialised memory there
    # is an out-of-bounds device write, not a wrong answer, so refuse up front.
    @assert count(img) == n "`img` must be the pore mask `A` was built from: it has $(count(img)) pore voxels against `A`'s $(n) rows"
    nnz(A) == 0 && return nothing

    nx, ny, nz = size(img)
    bs = isnothing(block) ? DEFAULT_COARSE_BLOCK : block
    nbx, nby, nbz = cld(nx, bs), cld(ny, bs), cld(nz, bs)
    max_coarse = _resolve_max_coarse(A, max_coarse, (nbx, nby, nbz))

    # Pore ordinals are the prefix sum of the mask, exactly as in
    # `build_steady_system`; both scratch arrays go before the coarse operator
    # is assembled, so this pass never coincides with the solve's peak.
    img_dev = _on_gpu(nonzeros(A)) ? _gpu_adapt[](img) : img
    idx = similar(img_dev, Int32)
    _pore_index!(idx, img_dev)
    agg = _aggregate(idx, n, nbx * nby * nbz, bs, nbx, nby)
    _free!(idx)
    img_dev === img || _free!(img_dev)

    return _two_level_from_aggregates(A, agg, bs, nbx, nby, nbz, shift, max_coarse, verbose)
end

function two_level_preconditioner(
    A::MaskedLaplacian, img;
    block=nothing, max_coarse=nothing, shift=DEFAULT_COARSE_SHIFT,
    verbose=false,
)
    @assert shift > 0 "`shift` must be positive; the coarse operator is only positive semi-definite"
    img = atleast_3d(img)
    @assert size(img) == size(A.idx) "`img` must be the pore mask `A` was built from"
    # An operator of order zero is the matrix-free reading of `nnz(A) == 0`.
    # Every other degenerate pore space — isolated voxels, whose columns are all
    # empty — leaves no block carrying a coarse unknown, and falls out below.
    A.nnodes == 0 && return nothing

    nx, ny, nz = size(img)
    bs = isnothing(block) ? DEFAULT_COARSE_BLOCK : block
    nbx, nby, nbz = cld(nx, bs), cld(ny, bs), cld(nz, bs)
    max_coarse = _resolve_max_coarse(A, max_coarse, (nbx, nby, nbz))

    # The operator's index array is the pore numbering the aggregation needs, so
    # it stands in for the scratch array the assembled path builds here. It is
    # the operator's state, not scratch, and is not released.
    agg = _aggregate(A.idx, A.nnodes, nbx * nby * nbz, bs, nbx, nby)

    return _two_level_from_aggregates(A, agg, bs, nbx, nby, nbz, shift, max_coarse, verbose)
end

function two_level_preconditioner(sim::SteadyDiffusionProblem; kwargs...)
    return two_level_preconditioner(sim.prob.A, sim.img; kwargs...)
end

# --- Application ------------------------------------------------------------

# There is one work item per coarse cell, and there are far fewer coarse cells
# than pore voxels, so the size the backend picks for a grid-sized launch leaves
# the device idle. Measured on the two spot-check cases: 0.083 ms against
# 0.028 ms at 200³ and 1.54 ms against 1.33 ms at 400³.
const _RESTRICT_GROUP = 128

# One thread per coarse cell, gathering the cell's own fine nodes. The sum
# accumulates in a register in the order the adjacency lists them, so it needs no
# atomic and does not depend on how the launch happened to be scheduled.
@kernel function _restrict_kernel!(rc, @Const(offsets), @Const(fine), @Const(x), nc)
    a = @index(Global)
    if a <= nc
        acc = zero(eltype(rc))
        @inbounds for p in offsets[a]:(offsets[a + 1] - 1)
            acc += x[fine[p]]
        end
        @inbounds rc[a] = acc
    end
end

@kernel function _prolong_kernel!(y, @Const(agg), @Const(xc), @Const(x), inv_lambda, n)
    i = @index(Global)
    if i <= n
        @inbounds a = agg[i]
        coarse = a > 0 ? (@inbounds xc[a]) : zero(eltype(y))
        @inbounds y[i] = coarse + inv_lambda * x[i]
    end
end

# Host arrays take a plain loop rather than the kernel above. A block gathers
# thousands of voxels, so the scatter is a contended read-modify-write, and on
# a CPU that is a compare-and-swap per pore voxel — measured at ~19 ms per
# application on a 160³ image, against ~23 ms for the whole SpMV it is meant to
# accelerate. The serial accumulation needs no atomic at all.
function _restrict!(rc::Vector, agg::Vector, x::Vector)
    fill!(rc, zero(eltype(rc)))
    @inbounds for i in eachindex(agg)
        a = agg[i]
        a > 0 && (rc[a] += x[i])
    end
    return rc
end

# Every coarse cell is written, so unlike the scatter this replaces there is
# nothing to zero first.
function _restrict!(rc, agg::Aggregation, x)
    backend = get_backend(agg)
    nc = length(agg.offsets) - 1
    _restrict_kernel!(backend, _RESTRICT_GROUP)(
        rc, agg.offsets, agg.fine, x, nc; ndrange=nc,
    )
    _async_return_safe(agg) || KernelAbstractions.synchronize(backend)
    return rc
end

function _prolong!(y::Vector, agg::Vector, xc::Vector, x::Vector, inv_lambda)
    if Threads.nthreads() > 1 && length(agg) >= PROLONG_MIN_THREADED
        Threads.@threads for i in eachindex(agg)
            @inbounds a = agg[i]
            @inbounds y[i] = (a > 0 ? xc[a] : zero(eltype(y))) + inv_lambda * x[i]
        end
    else
        @inbounds for i in eachindex(agg)
            a = agg[i]
            y[i] = (a > 0 ? xc[a] : zero(eltype(y))) + inv_lambda * x[i]
        end
    end
    return y
end

function _prolong!(y, agg, xc, x, inv_lambda)
    backend = get_backend(agg)
    n = length(agg)
    _prolong_kernel!(backend)(y, agg, xc, x, inv_lambda, n; ndrange=n)
    _async_return_safe(agg) || KernelAbstractions.synchronize(backend)
    return y
end

# Prolongation is already a gather over the forward map, so the adjacency has
# nothing to add to it.
function _prolong!(y, agg::Aggregation, xc, x, inv_lambda)
    return _prolong!(y, agg.fwd, xc, x, inv_lambda)
end

function LinearAlgebra.ldiv!(
    y::AbstractVector, P::TwoLevelPreconditioner, x::AbstractVector
)
    _restrict!(P.rc, P.agg, x)
    copyto!(P.rc_host, P.rc)
    # The coarse solve is always double precision: the fine problem may run in
    # Float32, but the coarse operator is the near-singular one and it is small
    # enough that the extra precision is free. It is also entirely on the host,
    # and every step of it — the cycle's gathers included — sums each output
    # element in one fixed order, so it cannot reintroduce the schedule-dependent
    # reduction the gather in `_restrict!` exists to avoid.
    P.coarse_rhs .= P.rc_host
    _vcycle!(P.coarse_sol, P.levels, 1, P.coarse_rhs, P.fact)
    P.rc_host .= P.coarse_sol
    copyto!(P.xc, P.rc_host)

    _prolong!(y, P.agg, P.xc, x, P.inv_lambda)
    return y
end

LinearAlgebra.ldiv!(P::TwoLevelPreconditioner, x::AbstractVector) = ldiv!(x, P, copy(x))
Base.:\(P::TwoLevelPreconditioner, x::AbstractVector) = ldiv!(similar(x), P, x)
