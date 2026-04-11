# Transient diffusion solver for 3D porous materials

"""Type alias for boundary condition values: `Nothing` (insulated), `Number`
(constant Dirichlet), or `Function` (time-dependent Dirichlet)."""
const BoundValue = Union{Nothing, Number, Function}

struct TransientProblem{T,DType,MatType<:AbstractMatrix{T}}
    dx::Float64
    dt::Float64
    D::DType
    img::BitArray{3}
    grid_to_vec::Array{Int,3}
    axis::Symbol
    bc_inlet::BoundValue
    bc_outlet::BoundValue
    A::MatType
end

"""
    TransientState{T, I}

Holds the evolving state of a transient simulation. Created by [`init_state`](@ref).
Concentration history is stored on CPU to avoid GPU memory exhaustion.

# Fields
- `integrator::I`: ODE integrator instance (from OrdinaryDiffEq.jl).
- `t::Vector{Float64}`: recorded simulation times.
- `C::Vector{Vector{T}}`: concentration snapshots. Each entry is a 1D pore-only
  vector (CPU), one per recorded time step.
"""
struct TransientState{T,I}
    integrator::I
    t::Vector{Float64}
    C::Vector{Vector{T}}
end

"""
    TransientProblem(img, dt; axis=:z, bc_inlet=1, bc_outlet=0,
                     D=1.0, dx=nothing, dtype=Float32, gpu=nothing)

Construct a `TransientProblem` describing transient diffusion through a
3D voxelized porous material. The input `img` defines which voxels are
pore space (nonzero) and which are solid (zero). A finite‑difference
operator is built on this mask, with Dirichlet or insulated boundary
conditions applied along one axis.

# Arguments
- `img`: a 3D array whose nonzero entries indicate pore voxels.
- `dt`: the interval (in physical time units) between saved solution
        snapshots and between evaluations of the `stop_condition`.
        This is **not** the internal timestep used by the ODE solver.

# Keyword Arguments
- `axis`: `:x`, `:y`, or `:z`. Specifies which axis has non‑insulated
          boundary faces. Defaults to `:z`.
- `bc_inlet`: Dirichlet concentration at the inlet face along `axis`.
              Use `nothing` for an insulated (Neumann) boundary, or a
              function `f(t)` for a time-varying boundary. Defaults to `1`.
- `bc_outlet`: Dirichlet concentration at the outlet face along `axis`.
               Use `nothing` for an insulated (Neumann) boundary, or a
               function `f(t)` for a time-varying boundary. Defaults to `0`.
- `D`: scalar diffusion coefficient or scalar field of diffusivity
       at each pixel with shape img used inside pore voxels.
       Defaults to `1.0` for easy comparison.
- `dx`: physical spacing between adjacent voxel centers. If `nothing`,
        it is set to `1/(N_axis - 1)` so that the domain spans `[0,1]`
        along the chosen axis for easy comparison.
- `dtype`: numeric type used for the operator and solution arrays
           (e.g., `Float32` or `Float64`). Defaults to `Float32`.
- `gpu`: whether to run solver on the GPU. If `nothing`, uses GPU when the
         image has ≥100,000 pore voxels and a GPU backend is available. Defaults to `nothing`.
"""
function TransientProblem(
    img,
    dt;
    axis::Symbol=:z,
    bc_inlet::BoundValue=1,
    bc_outlet::BoundValue=0,
    D=1.0,
    dx=nothing,
    dtype=Float32,
    gpu=nothing,
)
    img = atleast_3d(img)
    img = BitArray(img .!= 0)
    @assert D isa Number || size(img) == size(D) "For scalar field D, size should match img size"
    D = D isa Number ? dtype(D) : dtype.(D)

    # Validate and convert boundary conditions
    bc_inlet = _validate_bc(bc_inlet, dtype, "bc_inlet")
    bc_outlet = _validate_bc(bc_outlet, dtype, "bc_outlet")

    nnodes = count(img)
    if isnothing(gpu)
        gpu = !isnothing(_preferred_gpu_backend[]) && nnodes >= 100_000
    elseif gpu && isnothing(_preferred_gpu_backend[])
        error("`gpu=true` was requested but no GPU backend is registered. \
               Load a GPU package first (e.g. `using CUDA`, `using Metal`, or `using AMDGPU`).")
    end

    @assert size(img, axis_dim(axis)) > 1 "Image must have at least 2 voxels along the chosen axis"
    isnothing(dx) && (dx = 1 / (size(img, axis_dim(axis)) - 1))

    g2v = grid_to_vec(img)
    A = build_transient_operator(img, D, bc_inlet, bc_outlet; axis=axis, dx=dx, gpu=gpu)

    return TransientProblem(dx, dt, D, img, g2v, axis, bc_inlet, bc_outlet, A)
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
    init_state(prob::TransientProblem; C0=nothing, alg=ROCK4(),
               reltol=1e-3, abstol=1e-6)

Initialize a `TransientState` for the given `TransientProblem`, applying boundary
conditions to `C0` and setting up the ODE integrator.

# Keyword Arguments
- `C0`: initial concentration distribution with dimensions matching `prob.img`.
  Defaults to zero everywhere.
- `alg`: ODE solver algorithm. Defaults to `ROCK4()`. Only explicit methods supported.
- `reltol`: relative tolerance for the ODE solver. Defaults to `1e-3`.
- `abstol`: absolute tolerance for the ODE solver. Defaults to `1e-6`.
"""
function init_state(
    prob::TransientProblem{T}; C0=nothing, alg=ROCK4(), reltol=1e-3, abstol=1e-6
) where {T}
    if C0 === nothing
        C0 = zeros(T, size(prob.img))
    else
        C0 = atleast_3d(C0)
        @assert size(C0) == size(prob.img) "C0 dims must match img"
        C0 = T.(C0)
    end

    apply_boundaries!(C0, prob)
    C0 = C0[prob.img]

    gpu = prob.A isa PortableSparseCSC
    if gpu
        C0 = _gpu_adapt[](C0)
    end

    dC! = make_dC_function(prob)

    prob_ode = ODEProblem(dC!, C0, (0.0, Inf))
    integrator = init(prob_ode, alg; save_everystep=false, reltol=reltol, abstol=abstol)

    C_hist = Vector{Vector{T}}()
    t_hist = Float64[]
    push!(t_hist, 0.0)
    push!(C_hist, Array(C0))

    return TransientState(integrator, t_hist, C_hist)
end

"""
    solve!(state::TransientState, prob::TransientProblem, stop_condition;
           max_iter=500, verbose=false)

Step the simulation forward by increments of `prob.dt` until `stop_condition(t_hist, C_hist)`
returns `true`. The concentration distribution is stored at every `dt` step, on CPU regardless
of whether the solver runs on GPU.

# Arguments
- `state`: a `TransientState` from [`init_state`](@ref).
- `prob`: the `TransientProblem` matching `state`.
- `stop_condition`: a function `(t_hist, C_hist) -> Bool`. Built-in options include
  [`stop_at_time`](@ref), [`stop_at_avg_concentration`](@ref), and
  [`stop_at_delta_flux`](@ref). Note that `C_hist` entries are 1D pore-voxel vectors.

# Keyword Arguments
- `max_iter`: maximum number of `dt`-sized steps before stopping. Defaults to `500`.
- `verbose`: print simulation time at each step. Defaults to `false`.
"""
function solve!(
    state::TransientState,
    prob::TransientProblem,
    stop_condition;
    max_iter=500,
    verbose=false,
)
    for _ in 1:max_iter
        step!(state.integrator, prob.dt, true)
        push!(state.t, state.integrator.t)
        push!(state.C, Array(state.integrator.u))

        verbose && @info "reached simulation time $(state.t[end])"

        if stop_condition(state.t, state.C)
            return
        end
    end
    @warn "Reached max_iter=$max_iter at t=$(state.t[end]) without satisfying stop_condition."
end

"""
    build_transient_operator(img, D, bc_inlet, bc_outlet; axis, dx, gpu)

Build the sparse finite-difference operator `A` such that `dC/dt = A * C` for the
pore-voxel concentration vector. Dirichlet boundary rows are zeroed so that
boundary values remain constant during integration.
"""
function build_transient_operator(img, D, bc_inlet, bc_outlet; axis, dx, gpu)
    # Compute boundary nodes BEFORE GPU transfer (cheap CPU operation)
    inlet_face, outlet_face = axis_faces(axis)
    bc_nodes = Int[]
    if !isnothing(bc_inlet)
        append!(bc_nodes, find_boundary_nodes(img, inlet_face))
    end
    if !isnothing(bc_outlet)
        append!(bc_nodes, find_boundary_nodes(img, outlet_face))
    end

    gpu && (img = _gpu_adapt[](img))

    nnodes = sum(img)

    conns = create_connectivity_list(img)

    if !(D isa Number)
        D_local = atleast_3d(D)
        gpu && (D_local = _gpu_adapt[](D_local))
    end

    gd = D isa Number ? D : interpolate_edge_values(D_local[img], conns)

    am = create_adjacency_matrix(conns; n=nnodes, weights=gd)

    A = laplacian(am)

    nonzeros(A) .= nonzeros(A) ./ (-dx^2)

    # Zero rows so Dirichlet values remain constant during integration
    zero_rows!(A, bc_nodes)

    return A
end

"""
    zero_rows!(A::SparseMatrixCSC, rows)
    zero_rows!(A::PortableSparseCSC, rows)

Zero out all entries in the specified `rows` of sparse matrix `A`, then drop
the resulting structural zeros. Used to enforce Dirichlet boundary conditions
in the transient operator.
"""
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
    apply_boundaries!(C0, prob::TransientProblem)

Set Dirichlet boundary values on the faces of `C0` along `prob.axis`. Each face
whose boundary condition is not `nothing` is overwritten with that value. For
time-varying (Function) boundaries, the function is evaluated at `t=0`.
"""
function apply_boundaries!(C0, prob)
    ax = axis_dim(prob.axis)
    N = size(C0, ax)

    if prob.bc_inlet isa Function
        selectdim(C0, ax, 1) .= prob.bc_inlet(0.0)
    elseif !isnothing(prob.bc_inlet)
        selectdim(C0, ax, 1) .= prob.bc_inlet
    end
    if prob.bc_outlet isa Function
        selectdim(C0, ax, N) .= prob.bc_outlet(0.0)
    elseif !isnothing(prob.bc_outlet)
        selectdim(C0, ax, N) .= prob.bc_outlet
    end
end

"""
    make_dC_function(prob::TransientProblem)

Build the ODE right-hand side `dC!(dC, C, p, t)`. For problems with
time-dependent (Function) boundaries, the closure updates boundary node
concentrations at each timestep before computing `A * C`. For constant
boundaries, the fast path is a simple matrix-vector multiply.
"""
function make_dC_function(prob)
    in_is_func = prob.bc_inlet isa Function
    out_is_func = prob.bc_outlet isa Function

    inlet_inds = slice_vec_indices(prob, 1)
    N_axis = size(prob.img, axis_dim(prob.axis))
    outlet_inds = slice_vec_indices(prob, N_axis)

    T = eltype(nonzeros(prob.A))

    if in_is_func && out_is_func
        return (dC, C, p, t) -> begin
            C[inlet_inds] .= convert(T, prob.bc_inlet(t))
            C[outlet_inds] .= convert(T, prob.bc_outlet(t))
            mul!(dC, prob.A, C)
        end
    elseif in_is_func
        return (dC, C, p, t) -> begin
            C[inlet_inds] .= convert(T, prob.bc_inlet(t))
            mul!(dC, prob.A, C)
        end
    elseif out_is_func
        return (dC, C, p, t) -> begin
            C[outlet_inds] .= convert(T, prob.bc_outlet(t))
            mul!(dC, prob.A, C)
        end
    else
        return (dC, C, p, t) -> mul!(dC, prob.A, C)
    end
end

# --- Stop conditions ---

"""
    stop_at_time(t_final)

Create a stop condition that returns `true` when the simulation time reaches
or exceeds `t_final`.
"""
function stop_at_time(t_final)
    return (t_hist, C_hist) -> t_hist[end] >= t_final
end

"""
    stop_at_avg_concentration(C_final, img)

Create a stop condition that returns `true` when the average concentration across
pore voxels reaches `C_final`.
"""
function stop_at_avg_concentration(C_final, img)
    num_active = sum(img)
    return (t_hist, C_hist) -> sum(C_hist[end]) / num_active >= C_final
end
function stop_at_avg_concentration(C_final, problem::TransientProblem)
    return stop_at_avg_concentration(C_final, problem.img)
end

"""
    stop_at_delta_flux(delta, prob::TransientProblem)

Create a stop condition that returns `true` when the absolute difference between
inlet and outlet flux falls at or below `delta`. A smaller `delta` drives the
simulation closer to steady state before stopping.
"""
function stop_at_delta_flux(delta, prob::TransientProblem)
    D, dx, img, axis, g2v = prob.D, prob.dx, prob.img, prob.axis, prob.grid_to_vec
    return (t_hist, C_hist) ->
        abs(compute_flux(C_hist[end], D, dx, img, axis; ind=:end, grid_to_vec=g2v) -
            compute_flux(C_hist[end], D, dx, img, axis; ind=1, grid_to_vec=g2v)) <= delta
end

"""
    stop_at_periodic(freq, prob; reltol=1e-2, Nphase=4, frac_period=0.3, depth=1.0)

Stop condition for detecting periodic steady state under oscillating boundary
conditions. Compares slice concentrations at several phase points across the
trailing `frac_period` fraction of one period, checking whether the current
period matches the previous period within `reltol` scaled by the amplitude.

!!! warning
    The returned closure tracks an internal history that must stay aligned with
    `state.t`. Do not reuse the same closure across non-consecutive `solve!`
    calls with a different stop condition in between — create a fresh one instead.

# Arguments
- `freq`: driving frequency (Hz).
- `prob`: `TransientProblem` defining geometry and slice axis.

# Keyword Arguments
- `reltol`: relative tolerance for periodicity (default `1e-2`).
- `Nphase`: number of phase points to test across the trailing window.
- `frac_period`: fraction of the period to test (`0 < frac_period ≤ 1`).
- `depth`: normalized depth in `(0,1]` at which concentration is evaluated.
"""
function stop_at_periodic(
    freq, prob::TransientProblem;
    reltol=1e-2, Nphase::Int=4, frac_period=0.3, depth=1.0,
)
    @assert 0 < frac_period <= 1 "frac_period must be in (0, 1]."
    @assert 0 < depth <= 1 "depth must be in (0,1]."

    period = 1 / freq
    phases = range(0, frac_period; length=Nphase)

    N = size(prob.img, axis_dim(prob.axis))
    depth_ind = round(Int, depth * (N - 1) + 1)

    slice_hist = Float64[]
    img, axis, g2v = prob.img, prob.axis, prob.grid_to_vec

    function interp(t, t_vals, C_vals)
        i1 = findlast(ti -> ti <= t, t_vals)
        i2 = findfirst(ti -> ti >= t, t_vals)
        if isnothing(i1) || isnothing(i2)
            return nothing
        end
        if i1 == i2
            return C_vals[i1]
        end
        w = (t - t_vals[i1]) / (t_vals[i2] - t_vals[i1])
        return (1 - w) * C_vals[i1] + w * C_vals[i2]
    end

    return (t_hist, C_hist) -> begin
        t_end = t_hist[end]
        push!(slice_hist, get_slice_conc(C_hist[end], img, axis, depth_ind; grid_to_vec=g2v))

        tmin = t_end - (1 + frac_period) * period
        if tmin < 0
            return false
        end

        ts = @view t_hist[(end - length(slice_hist) + 1):end]

        i0 = searchsortedfirst(ts, tmin)
        amplitude = maximum(@view slice_hist[i0:end]) - minimum(@view slice_hist[i0:end])

        for phi in phases
            C1 = interp(t_end - phi * period, ts, slice_hist)
            C2 = interp(t_end - period - phi * period, ts, slice_hist)

            if C1 === nothing || C2 === nothing
                return false
            end
            if abs(C1 - C2) > reltol * amplitude
                return false
            end
        end

        return true
    end
end

# Convenience wrappers that unpack TransientProblem fields
function slice_vec_indices(prob::TransientProblem, idx::Int)
    return slice_vec_indices(prob.img, prob.grid_to_vec, prob.axis, idx)
end
function vec_to_slice(u, prob::TransientProblem, idx::Int)
    return vec_to_slice(u, prob.img, prob.grid_to_vec, prob.axis, idx)
end
