# Two-level preconditioner for the assembled steady system: a coarse space of
# piecewise-constant indicators over cubic voxel blocks, factorised once and
# applied directly, plus a scaled identity for everything the coarse space
# cannot see.

# Slots of the coarse stencil, in this order: the block itself, then its six
# face neighbours at block-index offsets -nbxy, -nbx, -1, +1, +nbx, +nbxy.
# Opposite directions sit at slots `s` and `9 - s`, which is what makes the
# host-side symmetrisation a single indexed lookup.
const _COARSE_SLOTS = 7

# Default ceiling on the number of coarse unknowns. The coarse solve runs once
# per CG iteration, so its cost has to stay well under one fine SpMV: measured
# on a 3-D 7-point operator, a 25³ coarse grid factorises in 0.44 s and solves
# in 1.9 ms, where a 50³ one takes 1.7 s and 47 ms. The block size is grown
# until the coarse grid fits under this bound.
const DEFAULT_MAX_COARSE = 32_000

# The smallest block worth aggregating over. Below this the coarse problem
# stops being much smaller than the fine one.
const MIN_COARSE_BLOCK = 8

# Relative diagonal shift applied to the coarse operator before factorisation.
# `WᵀAW` is only positive *semi*-definite — a pore cluster that reaches neither
# Dirichlet face spans blocks whose coarse rows sum to zero — so a shift is what
# makes the Cholesky exist at all. See `two_level_preconditioner` for why the
# size of the shift is a tradeoff rather than "as small as possible".
const DEFAULT_COARSE_SHIFT = 1.0e-3

"""
    TwoLevelPreconditioner

Left preconditioner for the steady diffusion system, applied through
`ldiv!(y, P, x)`:

    y = W (WᵀAW + ρ·diag)⁻¹ Wᵀ x  +  x / λmax

`W` holds the indicator of each cubic block of voxels, so `Wᵀ x` sums the
residual over a block and `W xc` broadcasts a block's correction back to its
voxels. Build one with [`two_level_preconditioner`](@ref).

# Fields
- `agg`: coarse index of each pore voxel, `0` where the voxel's block was
  dropped for carrying no coarse unknown.
- `nc`: number of coarse unknowns.
- `fact`: host Cholesky factorisation of the shifted coarse operator, always in
  `Float64` regardless of the fine precision.
- `inv_lambda`: reciprocal of a Gershgorin bound on `λmax(A)`.
- `block`: edge length in voxels of one coarse block.
"""
struct TwoLevelPreconditioner{T,Vi<:AbstractVector,Vc<:AbstractVector{T},F}
    agg::Vi
    nc::Int
    fact::F
    inv_lambda::T
    block::Int
    rc::Vc                  # nc-length device scratch, restriction result
    xc::Vc                  # nc-length device scratch, coarse correction
    rc_host::Vector{T}
    coarse_rhs::Vector{Float64}
end

function Base.show(io::IO, P::TwoLevelPreconditioner)
    return print(
        io,
        "TwoLevelPreconditioner(block=$(P.block)^3, nc=$(P.nc), nnz(L)=$(nnz(P.fact)))",
    )
end

Base.size(P::TwoLevelPreconditioner) = (length(P.agg), length(P.agg))
Base.eltype(::TwoLevelPreconditioner{T}) where {T} = T

# --- Coarse-space construction ---------------------------------------------

"""Smallest block edge that keeps the coarse grid under `max_coarse` unknowns."""
function _choose_block(nx, ny, nz, max_coarse)
    bs = MIN_COARSE_BLOCK
    while cld(nx, bs) * cld(ny, bs) * cld(nz, bs) > max_coarse
        bs += 1
    end
    return bs
end

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

@kernel function _remap_aggregates_kernel!(agg, @Const(remap), n)
    i = @index(Global)
    if i <= n
        @inbounds agg[i] = remap[agg[i]]
    end
end

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

Opposite stencil slots hold the same sum accumulated in a different order, so
averaging them makes `Ac` symmetric to the last bit rather than merely close.
"""
function _coarse_operator(S::Vector{Float64}, nc0, nbx, nbxy, shift)
    offs = (0, -nbxy, -nbx, -1, 1, nbx, nbxy)
    remap = zeros(Int32, nc0)
    nc = 0
    for a in 1:nc0
        if S[(a - 1) * _COARSE_SLOTS + 1] > 0
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
    two_level_preconditioner(A, img; block=nothing, max_coarse, shift, verbose=false)
    two_level_preconditioner(sim::SteadyDiffusionProblem; kwargs...)

Build a [`TwoLevelPreconditioner`](@ref) for the assembled steady system `A` on
the pore mask `img`, or `nothing` when no usable coarse space exists.

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

The result is a genuine preconditioner, not an approximate solve: it is
symmetric positive definite by construction, so CG converges to the same
solution as it would without it. Only the iteration count changes.

Worth building only when the unpreconditioned iteration count is large, which
in practice means images of a few hundred voxels per side and up. Measured on
blob images (seed 42, porosity 0.5) with the default block size: 400³ takes
2094 iterations and 13.4 s without it against 111 and 1.5 s with it, and 600³
2983 and 64.6 s against 223 and 8.1 s. On a 12³ image it costs iterations
rather than saving them.

Note that CG's stopping test is taken in the `M⁻¹` norm once a preconditioner is
present, so at the same `reltol` the two runs stop at slightly different true
residuals. They agree on `tortuosity` to solver tolerance — verified to 1e-9 at
`reltol=1e-10` — but not bit for bit at a loose one.

# Keyword Arguments
- `block`: edge length in voxels of a coarse block. `nothing` (default) grows it
  from 8 until the coarse grid fits under `max_coarse`.
- `max_coarse`: ceiling on the number of coarse unknowns. The coarse solve runs
  once per iteration, so a larger coarse space stops paying for itself.
- `shift`: relative diagonal shift applied before factorisation. `WᵀAW` is
  positive semi-definite, not definite — a pore cluster reaching neither
  Dirichlet face spans blocks whose coarse rows sum to zero — so the shift is
  what makes the factorisation exist. It also bounds how far the coarse solve
  can amplify a residual that lies along one of those null directions, which is
  why the default is not the smallest number that works.
- `verbose`: report the coarse size and factorisation cost.
"""
function two_level_preconditioner(
    A, img;
    block=nothing, max_coarse=DEFAULT_MAX_COARSE, shift=DEFAULT_COARSE_SHIFT,
    verbose=false,
)
    @assert shift > 0 "`shift` must be positive; the coarse operator is only positive semi-definite"
    img = atleast_3d(img)
    n = size(A, 1)
    nnz(A) == 0 && return nothing

    nx, ny, nz = size(img)
    bs = isnothing(block) ? _choose_block(nx, ny, nz, max_coarse) : block
    nbx, nby, nbz = cld(nx, bs), cld(ny, bs), cld(nz, bs)
    nc0 = nbx * nby * nbz

    nzv = nonzeros(A)
    on_gpu = _on_gpu(nzv)
    backend = get_backend(nzv)

    # Pore ordinals are the prefix sum of the mask, exactly as in
    # `build_steady_system`; both scratch arrays go before the coarse operator
    # is assembled, so this pass never coincides with the solve's peak.
    img_dev = on_gpu ? _gpu_adapt[](img) : img
    idx = similar(img_dev, Int32)
    cumsum!(vec(idx), vec(img_dev))
    idx .*= img_dev
    # `nc0` is capped well inside `Int16`, and `agg` is one entry per pore voxel
    # — the largest array this preconditioner keeps alive during the solve.
    Ta = nc0 <= typemax(Int16) ? Int16 : Int32
    agg = similar(idx, Ta, n)
    _aggregate_kernel!(backend, (64, 4, 1))(
        agg, idx, bs, nbx, nby; ndrange=(nx, ny, nz),
    )
    KernelAbstractions.synchronize(backend)
    _free!(idx)
    img_dev === img || _free!(img_dev)

    stencil = similar(nzv, Float64, _COARSE_SLOTS * nc0)
    fill!(stencil, 0.0)
    _coarse_stencil_kernel!(backend)(
        stencil, SparseArrays.getcolptr(A), rowvals(A), nzv, agg,
        nbx, nbx * nby, n; ndrange=n,
    )
    KernelAbstractions.synchronize(backend)
    S = Array(stencil)
    _free!(stencil)

    Ac, remap = _coarse_operator(S, nc0, nbx, nbx * nby, shift)
    nc = size(Ac, 1)
    if nc == 0
        _free!(agg)
        return nothing
    end

    t0 = time_ns()
    fact = try
        cholesky(Symmetric(Ac))
    catch err
        # `Ac + shift·diag(Ac)` is provably definite, so this only fires if the
        # assumption behind that — that `A` itself is positive semi-definite —
        # has been broken. Running without a preconditioner is slow; running
        # with a broken one is wrong.
        @warn "coarse factorisation failed; solving without a preconditioner" exception = err
        _free!(agg)
        return nothing
    end
    verbose && @info "two-level coarse space" block = bs nc = nc nnz_L = nnz(fact) seconds =
        (time_ns() - t0) / 1e9

    # Same element type as `agg`, so the remap kernel never converts in device
    # code where a range check has nowhere to throw.
    remap = convert(Vector{Ta}, remap)
    remap_dev = on_gpu ? _gpu_adapt[](remap) : remap
    _remap_aggregates_kernel!(backend)(agg, remap_dev, n; ndrange=n)
    KernelAbstractions.synchronize(backend)
    on_gpu && _free!(remap_dev)

    T = eltype(nzv)
    # Gershgorin: every column of this Laplacian has |offdiagonals| summing to
    # its diagonal, so twice the largest stored value bounds every eigenvalue.
    # The diagonal entries are the only positive ones.
    inv_lambda = T(1) / (2 * maximum(nzv))

    return TwoLevelPreconditioner(
        agg, nc, fact, inv_lambda, bs,
        fill!(similar(nzv, nc), zero(T)), fill!(similar(nzv, nc), zero(T)),
        zeros(T, nc), zeros(Float64, nc),
    )
end

function two_level_preconditioner(sim::SteadyDiffusionProblem; kwargs...)
    return two_level_preconditioner(sim.prob.A, sim.img; kwargs...)
end

# --- Application ------------------------------------------------------------

@kernel function _restrict_kernel!(rc, @Const(agg), @Const(x), n)
    i = @index(Global)
    if i <= n
        @inbounds a = agg[i]
        if a > 0
            @inbounds Atomix.@atomic rc[a] += x[i]
        end
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

function _restrict!(rc, agg, x)
    backend = get_backend(agg)
    n = length(agg)
    fill!(rc, zero(eltype(rc)))
    _restrict_kernel!(backend)(rc, agg, x, n; ndrange=n)
    KernelAbstractions.synchronize(backend)
    return rc
end

function _prolong!(y::Vector, agg::Vector, xc::Vector, x::Vector, inv_lambda)
    @inbounds for i in eachindex(agg)
        a = agg[i]
        y[i] = (a > 0 ? xc[a] : zero(eltype(y))) + inv_lambda * x[i]
    end
    return y
end

function _prolong!(y, agg, xc, x, inv_lambda)
    backend = get_backend(agg)
    n = length(agg)
    _prolong_kernel!(backend)(y, agg, xc, x, inv_lambda, n; ndrange=n)
    KernelAbstractions.synchronize(backend)
    return y
end

function LinearAlgebra.ldiv!(
    y::AbstractVector, P::TwoLevelPreconditioner, x::AbstractVector
)
    _restrict!(P.rc, P.agg, x)
    copyto!(P.rc_host, P.rc)
    # The coarse solve is always double precision: the fine problem may run in
    # Float32, but the coarse operator is the near-singular one and it is small
    # enough that the extra precision is free.
    P.coarse_rhs .= P.rc_host
    P.rc_host .= P.fact \ P.coarse_rhs
    copyto!(P.xc, P.rc_host)

    _prolong!(y, P.agg, P.xc, x, P.inv_lambda)
    return y
end

LinearAlgebra.ldiv!(P::TwoLevelPreconditioner, x::AbstractVector) = ldiv!(x, P, copy(x))
Base.:\(P::TwoLevelPreconditioner, x::AbstractVector) = ldiv!(similar(x), P, x)
