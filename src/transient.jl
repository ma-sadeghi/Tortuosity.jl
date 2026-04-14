# Transient diffusion solver for 3D porous materials.
#
# The solver is a thin wrapper around the SciML ODE machinery:
# `TransientDiffusionProblem` captures the PDE discretisation (geometry,
# boundary conditions, sparse FD operator) and `solve(prob, alg; ...)` dispatches
# to OrdinaryDiffEq via `CommonSolve.solve`, returning a `TransientSolution` with
# CPU-resident snapshots even when the solver runs on GPU. Stop conditions are
# `DiscreteCallback`/`ContinuousCallback` factories that compose with SciML's
# callback ecosystem (see `StopAtSteadyState`, `StopAtFluxBalance`,
# `StopAtSaturation`, `StopAtPeriodicState`).

import CommonSolve: solve
using DiffEqCallbacks: SavingCallback, SavedValues, TerminateSteadyState

"""Type alias for boundary condition values: `Nothing` (insulated), `Number`
(constant Dirichlet), or `Function` (time-dependent Dirichlet)."""
const BoundValue = Union{Nothing,Number,Function}

"""
    TransientDiffusionProblem

Holds the finite-difference discretisation of a transient diffusion PDE over a
3D voxel image. The struct captures the geometry, boundary conditions, and
sparse operator; time-integration parameters (`saveat`, `tspan`, tolerances,
callbacks) are supplied at `solve` time.

# Fields
- `voxel_size::Float64`: physical spacing between adjacent voxel centers.
- `D`: diffusivity — either a scalar or a 3D field the shape of `img`.
- `img::BitArray{3}`: boolean pore mask (`true` = pore).
- `pore_index::Array{Int,3}`: lookup built by `build_pore_index`;
  used by slice-based observables to avoid walking the full image.
- `axis::Symbol`: transport direction (`:x`, `:y`, or `:z`).
- `bc_inlet`, `bc_outlet`: boundary condition values — `Nothing` (insulated),
  `Number` (constant Dirichlet), or `Function` (time-dependent Dirichlet).
- `A`: sparse finite-difference operator so that `dc/dt = A * c`.
"""
struct TransientDiffusionProblem{T,DType,MatType<:AbstractMatrix{T}}
    voxel_size::Float64
    D::DType
    img::BitArray{3}
    pore_index::Array{Int,3}
    axis::Symbol
    bc_inlet::BoundValue
    bc_outlet::BoundValue
    A::MatType
end

function Base.show(io::IO, prob::TransientDiffusionProblem)
    gpu = prob.A isa PortableSparseCSC
    nnodes = count(prob.img)
    bc_str(b::Nothing) = "insulated"
    bc_str(b::Number)  = string(b)
    bc_str(b::Function) = "f(t)"
    msg = "TransientDiffusionProblem(shape=$(size(prob.img)), nnodes=$(nnodes), " *
          "axis=$(prob.axis), bc=($(bc_str(prob.bc_inlet)) → $(bc_str(prob.bc_outlet))), " *
          "gpu=$(gpu))"
    return print(io, msg)
end

"""
    TransientDiffusionProblem(img; axis=:z, bc_inlet=1, bc_outlet=0,
                              D=1.0, voxel_size=nothing, dtype=Float32, gpu=nothing)

Construct a `TransientDiffusionProblem` describing transient diffusion through a
3D voxelized porous material. The input `img` defines which voxels are pore
space (nonzero) and which are solid (zero). A finite-difference operator is
built on this mask, with Dirichlet or insulated boundary conditions applied
along one axis.

# Arguments
- `img`: a 3D array whose nonzero entries indicate pore voxels.

# Keyword Arguments
- `axis`: `:x`, `:y`, or `:z`. Specifies which axis has non-insulated boundary
  faces. Default: `:z`.
- `bc_inlet`: Dirichlet concentration at the inlet face along `axis`. Use
  `nothing` for an insulated (Neumann) boundary, or a function `f(t)` for a
  time-varying boundary. Default: `1`.
- `bc_outlet`: Dirichlet concentration at the outlet face along `axis`. Use
  `nothing` for an insulated (Neumann) boundary, or a function `f(t)` for a
  time-varying boundary. Default: `0`.
- `D`: scalar diffusion coefficient or a 3D scalar field of diffusivity with
  the same shape as `img`. Default: `1.0`.
- `voxel_size`: physical spacing between adjacent voxel centers. If `nothing`,
  it is set to `1/(N_axis - 1)` so that the domain spans `[0, 1]` along the
  chosen axis.
- `dtype`: numeric type used for the operator and solution arrays (e.g.,
  `Float32` or `Float64`). Default: `Float32`.
- `gpu`: whether to run the solver on the GPU. If `nothing` (default), uses
  GPU when a backend package is loaded *and* the image has ≥ 100 000 pore
  voxels. See [GPU backends](@ref) for how to activate CUDA, Metal, or AMDGPU.
"""
function TransientDiffusionProblem(
    img;
    axis::Symbol=:z,
    bc_inlet::BoundValue=1,
    bc_outlet::BoundValue=0,
    D=1.0,
    voxel_size=nothing,
    dtype=Float32,
    gpu=nothing,
)
    img = atleast_3d(img)
    # Struct holds `img` on CPU; copy back from GPU if the caller passed a
    # device array (same convention as SteadyDiffusionProblem).
    if _on_gpu(img)
        @warn "`img` was passed on GPU; copying to CPU so the struct holds a CPU mask. \
               Pass `gpu=true` if you want the solver kernels to run on GPU." maxlog = 1
        img = Array(img)
    end
    img = BitArray(img .!= 0)
    @assert D isa Number || size(img) == size(D) "For scalar field D, size should match img size"
    D = D isa Number ? dtype(D) : dtype.(D)

    bc_inlet = _validate_bc(bc_inlet, dtype, "bc_inlet")
    bc_outlet = _validate_bc(bc_outlet, dtype, "bc_outlet")

    nnodes = count(img)
    @assert nnodes > 0 "Image must contain at least one pore voxel (got all-solid)"
    # Auto-detect GPU: see the matching block in SteadyDiffusionProblem for the
    # rationale behind the one-time warning on silent CPU fallback.
    if isnothing(gpu)
        has_backend = !isnothing(_preferred_gpu_backend[])
        if !has_backend && nnodes >= 100_000
            @warn "Image has $(nnodes) pore voxels but no GPU backend is loaded; \
                   running on CPU. To enable GPU kernels, load a backend package \
                   (`using CUDA`, `using Metal`, or `using AMDGPU`) before \
                   constructing the problem. Pass `gpu=false` explicitly to \
                   silence this message." maxlog = 1
        end
        gpu = has_backend && nnodes >= 100_000
    elseif gpu && isnothing(_preferred_gpu_backend[])
        error("`gpu=true` was requested but no GPU backend is registered. \
               Load a GPU package first (e.g. `using CUDA`, `using Metal`, or `using AMDGPU`).")
    end

    @assert size(img, axis_dim(axis)) > 1 "Image must have at least 2 voxels along the chosen axis"
    isnothing(voxel_size) && (voxel_size = 1 / (size(img, axis_dim(axis)) - 1))

    pidx = build_pore_index(img)
    A = build_transient_operator(img, D, bc_inlet, bc_outlet; axis=axis, voxel_size=voxel_size, gpu=gpu)

    return TransientDiffusionProblem(voxel_size, D, img, pidx, axis, bc_inlet, bc_outlet, A)
end

function _validate_bc(bc, dtype, name)
    if bc isa Function
        val = bc(0.0)
        @assert val isa Number "$name(t) must return a Number"
        return bc
    elseif bc isa Number
        return dtype(bc)
    else
        return nothing
    end
end

"""
    TransientSolution

The result of `solve(::TransientDiffusionProblem, alg; ...)`. A thin wrapper
around the underlying `SciMLBase.ODESolution` that materialises snapshots on
CPU via a `SavingCallback`, so `sol.t` and `sol.u` are always CPU-resident
regardless of whether the solver ran on GPU.

# Fields
- `t::Vector{Float64}`: snapshot times at the requested `saveat` intervals.
- `u::Vector{Vector{T}}`: concentration snapshots (CPU). Each entry is a 1D
  pore-only vector.
- `retcode::Symbol`: `:Success` if the solver reached the end of `tspan`,
  `:Terminated` if a stop-condition callback fired first. Other values are
  possible for failure modes (e.g. `:Unstable`, `:MaxIters`) and come
  verbatim from `Symbol(ode_sol.retcode)`. Users who need the raw
  `SciMLBase.ReturnCode.T` enum (for `successful_retcode` etc.) can reach
  through `sol.ode_sol.retcode`.
- `prob::TransientDiffusionProblem`: the originating problem.
- `alg`: the algorithm passed to `solve`.
- `ode_sol`: the raw `SciMLBase.ODESolution` the solve produced. Exposed for
  power users who need solver diagnostics like `ode_sol.destats`; the common
  path only needs `t`, `u`, and `retcode`.
"""
struct TransientSolution{T,P<:TransientDiffusionProblem,A,S}
    t::Vector{Float64}
    u::Vector{Vector{T}}
    retcode::Symbol
    prob::P
    alg::A
    ode_sol::S
end

function Base.show(io::IO, sol::TransientSolution)
    tmin = isempty(sol.t) ? 0.0 : first(sol.t)
    tmax = isempty(sol.t) ? 0.0 : last(sol.t)
    msg = "TransientSolution(snapshots=$(length(sol.t)), t ∈ [$(tmin), $(tmax)], retcode=$(sol.retcode))"
    return print(io, msg)
end

# Transient-specific method for CommonSolve.solve. The user-facing docs live
# in `docs/src/tutorials/transient.md` so we keep this comment-only — a
# docstring here would conflict with LinearSolve's `solve` under Documenter's
# `checkdocs=:exports` pass.
function solve(
    prob::TransientDiffusionProblem, alg=ROCK4();
    u0=nothing,
    tspan=(0.0, Inf),
    saveat,
    callback=nothing,
    reltol=1e-3,
    abstol=1e-6,
    kwargs...,
)
    T = eltype(nonzeros(prob.A))
    u0_dev = _initial_state(prob, u0, T)

    rhs! = build_rhs(prob)
    ode_prob = ODEProblem(rhs!, u0_dev, tspan)

    # CPU-materialising snapshot store. `Array(u)` forces a D2H copy at each
    # saveat boundary so sol.u never holds GPU-resident vectors, preventing
    # VRAM exhaustion on long runs.
    saved = SavedValues(Float64, Vector{T})
    save_cb = SavingCallback((u, t, integ) -> Array(u), saved; saveat=saveat)

    cbset = if callback === nothing
        save_cb
    else
        CallbackSet(save_cb, callback)
    end

    ode_sol = solve(
        ode_prob, alg;
        callback=cbset,
        save_everystep=false,
        save_start=false,
        save_end=false,
        reltol=reltol,
        abstol=abstol,
        kwargs...,
    )

    retcode = Symbol(ode_sol.retcode)
    return TransientSolution(saved.t, saved.saveval, retcode, prob, alg, ode_sol)
end

function _initial_state(prob::TransientDiffusionProblem, u0, ::Type{T}) where {T}
    if u0 === nothing
        c0 = zeros(T, size(prob.img))
    else
        c0 = atleast_3d(u0)
        @assert size(c0) == size(prob.img) "u0 dims must match img"
        c0 = T.(c0)
    end

    apply_boundaries!(c0, prob)
    c0 = c0[prob.img]

    if prob.A isa PortableSparseCSC
        c0 = _gpu_adapt[](c0)
    end
    return c0
end

"""
    build_transient_operator(img, D, bc_inlet, bc_outlet; axis, voxel_size, gpu)

Build the sparse finite-difference operator `A` such that `dc/dt = A * c` for
the pore-voxel concentration vector. Dirichlet boundary rows are zeroed so
that boundary values remain constant during integration.
"""
function build_transient_operator(img, D, bc_inlet, bc_outlet; axis, voxel_size, gpu)
    # Compute boundary nodes BEFORE GPU transfer (cheap CPU operation)
    inlet_face, outlet_face = axis_faces(axis)
    bc_nodes = Int[]
    if !isnothing(bc_inlet)
        append!(bc_nodes, find_boundary_nodes(img, inlet_face))
    end
    if !isnothing(bc_outlet)
        append!(bc_nodes, find_boundary_nodes(img, outlet_face))
    end

    # Keep the CPU `img` available for any downstream CPU-only work. `img_dev`
    # is the copy actually handed to the GPU kernels.
    img_dev = gpu ? _gpu_adapt[](img) : img

    nnodes = sum(img_dev)

    conns = build_connectivity_list(img_dev)

    D_dev = D
    if !(D isa Number)
        D_local = atleast_3d(D)
        D_dev = gpu ? _gpu_adapt[](D_local) : D_local
    end

    gd = D isa Number ? D : interpolate_edge_values(D_dev[img_dev], conns)

    am = build_adjacency_matrix(conns; n=nnodes, weights=gd)

    A = laplacian(am)

    nonzeros(A) .= nonzeros(A) ./ (-voxel_size^2)

    # Zero rows so Dirichlet values remain constant during integration
    zero_rows!(A, bc_nodes)

    return A
end

# Docstring lives on the stub in sparse_type.jl (shared with the PortableSparseCSC method).
function zero_rows!(A::SparseMatrixCSC, rows)
    target = Set(rows)
    @inbounds for i in eachindex(A.rowval)
        if A.rowval[i] in target
            A.nzval[i] = 0
        end
    end
    dropzeros!(A)
    return nothing
end

"""
    apply_boundaries!(c0, prob::TransientDiffusionProblem)

Set Dirichlet boundary values on the faces of `c0` along `prob.axis`. Each
face whose boundary condition is not `nothing` is overwritten with that value.
For time-varying (Function) boundaries, the function is evaluated at `t = 0`.
"""
function apply_boundaries!(c0, prob)
    ax = axis_dim(prob.axis)
    N = size(c0, ax)

    if prob.bc_inlet isa Function
        selectdim(c0, ax, 1) .= prob.bc_inlet(0.0)
    elseif !isnothing(prob.bc_inlet)
        selectdim(c0, ax, 1) .= prob.bc_inlet
    end
    if prob.bc_outlet isa Function
        selectdim(c0, ax, N) .= prob.bc_outlet(0.0)
    elseif !isnothing(prob.bc_outlet)
        selectdim(c0, ax, N) .= prob.bc_outlet
    end
end

"""
    build_rhs(prob::TransientDiffusionProblem)

Build the ODE right-hand side closure `(dc, c, p, t) -> …` implementing
`dc/dt = A * c` with the problem's boundary-condition handling. For
time-dependent (Function) boundaries, the closure updates boundary node
concentrations at each timestep before computing `A * c`. For constant
boundaries, the fast path is a simple matrix-vector multiply.
"""
function build_rhs(prob)
    in_is_func = prob.bc_inlet isa Function
    out_is_func = prob.bc_outlet isa Function

    inlet_inds_host = slice_indices(prob, 1)
    N_axis = size(prob.img, axis_dim(prob.axis))
    outlet_inds_host = slice_indices(prob, N_axis)

    # When A is on GPU and the BC is time-varying, the closure scatter
    # `c[inds] .= value` mixes a GPU `c` with CPU `inds`, which forces an
    # implicit H2D copy of the index vector on every ODE step. Move the
    # indices onto the device once at construction time instead. The
    # constant-BC fast path never touches the indices and is unaffected.
    gpu = prob.A isa PortableSparseCSC
    to_device = v -> begin
        d = similar(prob.A.rowval, eltype(v), length(v))
        copyto!(d, v)
        d
    end
    inlet_inds = (gpu && in_is_func) ? to_device(inlet_inds_host) : inlet_inds_host
    outlet_inds = (gpu && out_is_func) ? to_device(outlet_inds_host) : outlet_inds_host

    T = eltype(nonzeros(prob.A))

    if in_is_func && out_is_func
        return (dc, c, p, t) -> begin
            c[inlet_inds] .= convert(T, prob.bc_inlet(t))
            c[outlet_inds] .= convert(T, prob.bc_outlet(t))
            mul!(dc, prob.A, c)
        end
    elseif in_is_func
        return (dc, c, p, t) -> begin
            c[inlet_inds] .= convert(T, prob.bc_inlet(t))
            mul!(dc, prob.A, c)
        end
    elseif out_is_func
        return (dc, c, p, t) -> begin
            c[outlet_inds] .= convert(T, prob.bc_outlet(t))
            mul!(dc, prob.A, c)
        end
    else
        return (dc, c, p, t) -> mul!(dc, prob.A, c)
    end
end

# --- Stop conditions ---

"""
    StopAtSteadyState(; abstol=1e-8, reltol=1e-6, min_t=nothing)

Terminate the solve when the ODE state `c` satisfies
`|dc[i]/dt| ≤ max(abstol, reltol·|c[i]|)` for every component. This is a thin
forwarder to `DiffEqCallbacks.TerminateSteadyState` — the check reads the
integrator's cached `dc/dt` via `SciMLBase.get_du!`, so it works on CPU and
GPU states without any backend-specific code.

For diffusion problems with constant Dirichlet boundaries, this is the
strictest "near steady state" check available: zero `dc/dt` everywhere is
equivalent to exact steady state, and the boundary-row zeroing in
`build_transient_operator` ensures Dirichlet voxels always contribute
`0` to the norm. Use `min_t` to require a minimum elapsed time before the
callback starts testing (useful to ignore transient startup artefacts).

See also [`StopAtFluxBalance`](@ref) for a boundary-flux convergence criterion
that's more interpretable for porous-media work.
"""
StopAtSteadyState(; abstol=1e-8, reltol=1e-6, min_t=nothing) =
    TerminateSteadyState(abstol, reltol; min_t=min_t)

"""
    StopAtFluxBalance(prob::TransientDiffusionProblem; abstol=1e-4, reltol=1e-3)

Terminate the solve when the absolute difference between inlet and outlet
diffusive flux falls below the tolerance
`max(abstol, reltol · max(|flux_inlet|, |flux_outlet|))`. This is a porous-
media-native convergence check: flux balance is the physical quantity
experimenters measure to decide whether steady-state transport has been
reached.

Compared to [`StopAtSteadyState`](@ref), this is **less strict** — interior
voxels may still be equilibrating while boundary fluxes already agree — but
more interpretable because the tolerance is expressed in the same flux units
used to derive effective diffusivity. The flux value this check uses is the
same one the `flux` observable returns, so the threshold is directly in the
units users care about.

The callback materialises the integrator state to CPU on each fire via
`Array(u)`. For a GPU solve that's a full D2H copy per ODE step — typically a
small fraction of the per-step ODE work at realistic problem sizes, but
worth knowing about if you're running `StopAtFluxBalance` on a large grid
with a very cheap RHS. Benchmark your use case before assuming it's free.
"""
function StopAtFluxBalance(prob::TransientDiffusionProblem; abstol=1e-4, reltol=1e-3)
    D, voxel_size, img, axis, pidx = prob.D, prob.voxel_size, prob.img, prob.axis, prob.pore_index

    condition = function (c, t, integrator)
        c_cpu = Array(c)
        flux_in = flux(c_cpu, D, voxel_size, img, axis; ind=1, pore_index=pidx)
        flux_out = flux(c_cpu, D, voxel_size, img, axis; ind=:end, pore_index=pidx)
        tol = max(abstol, reltol * max(abs(flux_in), abs(flux_out)))
        return abs(flux_in - flux_out) <= tol
    end
    affect! = integrator -> terminate!(integrator)
    return DiscreteCallback(condition, affect!; save_positions=(false, false))
end

"""
    StopAtSaturation(target::Real; abstol=1e-4, reltol=1e-3)

Terminate the solve when the mean concentration over pore voxels reaches
`target - max(abstol, reltol·|target|)`. Implemented as a `ContinuousCallback`
with rootfinding, so the callback fires at the exact crossing time rather than
at the next integrator step.

The offset `- max(abstol, reltol·|target|)` handles the asymptotic-approach
footgun: for a monotonically-saturating system, picking a `target` equal to
the asymptotic limit (e.g. `1.0` for `bc_inlet = 1, bc_outlet = nothing`)
would otherwise never fire, because the mean concentration approaches `target`
from below but never crosses it. With the default tolerance, the callback
fires when the system is within `0.001 · target` of the target in finite time.
Users who want the strict crossing can set `abstol = 0, reltol = 0`.

The check is `sum(u)/length(u)` on the integrator state `u` directly — no
geometry lookup needed because `u` is already the pore-only concentration
vector.
"""
function StopAtSaturation(target::Real; abstol::Real=1e-4, reltol::Real=1e-3)
    tol = max(abstol, reltol * abs(target))
    threshold = target - tol
    condition = (u, t, integrator) -> sum(u) / length(u) - threshold
    affect! = integrator -> terminate!(integrator)
    return ContinuousCallback(
        condition, affect!; save_positions=(false, false),
    )
end

"""
    StopAtPeriodicState(freq, prob::TransientDiffusionProblem;
                        reltol=1e-2, samples_per_period=4,
                        compare_window=0.3, depth=1.0)

Detect periodic steady state under an oscillating boundary condition and
terminate the solve. On every ODE step, the callback records the slice
concentration at the specified `depth` along `prob.axis`; once at least one
full period of history is available, it compares the trailing `compare_window`
fraction of the current period against the same fraction of the previous
period at `samples_per_period` phase-aligned points. When every pair agrees
within `reltol · amplitude` (with `amplitude` measured over the recorded
window), the callback terminates.

# Keyword Arguments
- `reltol`: relative tolerance for period-to-period agreement. Scaled by the
  observed oscillation amplitude, so it's dimensionless regardless of the
  driving BC's magnitude. Default: `1e-2`.
- `samples_per_period`: number of phase-aligned comparison points. Higher
  means more conservative detection. Default: `4`.
- `compare_window`: fraction of a period used for the comparison, in
  `(0, 1]`. A smaller window lets long-running simulations terminate earlier
  (startup history required is `(1 + compare_window) · period`). Default:
  `0.3`. See issue #57 for the tradeoff discussion.
- `depth`: normalized position in `(0, 1]` along `prob.axis` at which the
  slice concentration is sampled. Default: `1.0` (outlet face).

!!! note
    The callback records one sample per ODE step, not per `saveat` boundary,
    so its resolution is governed by the integrator's adaptive step control
    (`reltol` / `abstol` / `dtmax`) rather than the `saveat` schedule. If the
    solver is taking very large steps relative to the driving period, tighten
    the ODE tolerances or set `dtmax ≤ 1 / (freq · samples_per_period)`.
    Closure-captured history grows unbounded at step granularity — see issue
    #69 for the trimming follow-up.
"""
function StopAtPeriodicState(
    freq, prob::TransientDiffusionProblem;
    reltol::Real=1e-2,
    samples_per_period::Int=4,
    compare_window::Real=0.3,
    depth::Real=1.0,
)
    @assert 0 < compare_window <= 1 "compare_window must be in (0, 1]."
    @assert 0 < depth <= 1 "depth must be in (0, 1]."

    period = 1 / freq
    phases = range(0, compare_window; length=samples_per_period)

    ax = axis_dim(prob.axis)
    N = size(prob.img, ax)
    depth_ind = round(Int, depth * (N - 1) + 1)

    # Precompute the pore-vector indices on the target slice and promote them
    # to the same backend as the integrator state. Each callback fire then pulls
    # just those entries to CPU rather than materialising the full state — on
    # a 200³ GPU problem this is ~180× faster and ~380× less memory per fire.
    slice_pore_inds_host = slice_indices(prob.pore_index, prob.axis, depth_ind)
    # Normalisation denominator: `slice_concentration(...; pore_only=false)`
    # divides `nansum(slice_2D)` by the full 2D slice area (solid cells
    # contribute 0 via nansum). Match that convention.
    slice_full_size = prod(size(selectdim(prob.img, ax, depth_ind)))

    gpu = prob.A isa PortableSparseCSC
    slice_inds = if gpu
        d = similar(prob.A.rowval, eltype(slice_pore_inds_host), length(slice_pore_inds_host))
        copyto!(d, slice_pore_inds_host)
        d
    else
        slice_pore_inds_host
    end

    # The callback maintains its own step-granular history in closure state so
    # `interp(t_current)` always has a bracketing entry at the right edge.
    # Issue #69 tracks trimming the head of the history to the active window.
    t_hist = Float64[]
    slice_hist = Float64[]

    function interp(t, ts, vs)
        i1 = findlast(ti -> ti <= t, ts)
        i2 = findfirst(ti -> ti >= t, ts)
        (isnothing(i1) || isnothing(i2)) && return nothing
        i1 == i2 && return vs[i1]
        w = (t - ts[i1]) / (ts[i2] - ts[i1])
        return (1 - w) * vs[i1] + w * vs[i2]
    end

    condition = function (u, t, integrator)
        # Pull only the pore values on the target slice — small per-fire D2H
        # regardless of the total state size.
        slice_vals = Array(@view u[slice_inds])
        push!(t_hist, t)
        push!(slice_hist, sum(slice_vals) / slice_full_size)

        tmin = t - (1 + compare_window) * period
        tmin < 0 && return false

        i0 = searchsortedfirst(t_hist, tmin)
        amplitude = maximum(@view slice_hist[i0:end]) - minimum(@view slice_hist[i0:end])

        for phi in phases
            c1 = interp(t - phi * period, t_hist, slice_hist)
            c2 = interp(t - period - phi * period, t_hist, slice_hist)
            (c1 === nothing || c2 === nothing) && return false
            abs(c1 - c2) > reltol * amplitude && return false
        end

        return true
    end
    affect! = integrator -> terminate!(integrator)
    return DiscreteCallback(condition, affect!; save_positions=(false, false))
end

# Convenience wrappers that unpack TransientDiffusionProblem fields
function slice_indices(prob::TransientDiffusionProblem, idx::Int)
    return slice_indices(prob.pore_index, prob.axis, idx)
end
function reconstruct_slice(u, prob::TransientDiffusionProblem, idx::Int)
    return reconstruct_slice(u, prob.pore_index, prob.axis, idx)
end
