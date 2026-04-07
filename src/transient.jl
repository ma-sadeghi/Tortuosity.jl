##------ define structs for convenience of only passing all parameters once ------
const BoundValue = Union{Number, Function}

#struct for problem definition, not solution 
struct TransientProblem{T, DType}
    dims::NTuple{3,Int}
    dx::Float64
    dt::Float64
    D::DType
    img::BitArray{3}
    grid_to_vec::Array{Int,3}
    axis::Symbol
    bound_mode::NTuple{2, BoundValue}
    A::Union{
        SparseMatrixCSC{T,Int},
        CUDA.CUSPARSE.CuSparseMatrixCSC{T,Int32}
    }
    gpu::Bool
    eltype::Type{T}
end


#stores integrator and datapoints
#the reason this struct exists instead of just using ODE integrator (the type that 'integrator' is), which could store t and C history in itself
# is that 'integrator.u' (C), would be stored on the same device as is being used
# for the GPU case this would mean huge GPU memory usage and seriously limit the size that can be run
# so instead the data for each timestep is moved off GPU
struct TransientState{T}
    integrator          
    t::Vector{Float64}
    C::Vector{Vector{T}}
end

##----- define constructors for structs ------
"""
    TransientProblem(img, dt; axis=:z, bound_mode=(1,0),
                     D=1.0, dx=nothing, dtype=Float32, gpu=true)

Construct a `TransientProblem` describing transient diffusion through a
3D voxelized porous material. The input `img` defines which voxels are
pore space (nonzero) and which are solid (zero). A finite‑difference
operator is built on this mask, with Dirichlet or insulated boundary
conditions applied along one axis.

The resulting problem can be passed to a transient diffusion solver.

# Arguments
- `img`: a 3D array whose nonzero entries indicate pore voxels.
- `dt`: the interval (in physical time units) between saved solution
        snapshots and between evaluations of the `stop_condition`.
        This is **not** the internal timestep used by the ODE solver.

# Keyword Arguments
- `axis`: `:x`, `:y`, or `:z`. Specifies which axis has non‑insulated
          boundary faces. Defaults to `:z`.
- `bound_mode`: a 2‑tuple `(C_inlet, C_outlet)` giving Dirichlet boundary
                values at the two faces along `axis`. Use `NaN` for an
                insulated (Neumann) boundary, or a function f(t) = C
                for a time varying boundary. Defaults to `(1, 0)`.
- `D`: scalar diffusion coefficient or scalar field of diffusivity 
            at each pixel with shape img used inside pore voxels.
            Defaults to `1.0` for easy comparison.
- `dx`: physical spacing between adjacent voxel centers. If `nothing`,
        it is set to `1/(N_axis - 1)` so that the domain spans `[0,1]`
        along the chosen axis for easy comparison.
- `dtype`: numeric type used for the operator and solution arrays
           (e.g., `Float32` or `Float64`). Defaults to `Float32`.
- `gpu`: whether run solver on the GPU. Defaults to `true`.

# Returns
A `TransientProblem` struct containing:
- grid size,
- spatial spacing `dx`,
- output timestep `dt`,
- diffusion coefficient or scalar field,
- pore mask image,
- axis and boundary mode,
- sparse operator `A` (CPU or GPU),
- numeric type.

"""
function TransientProblem(img, dt; 
    axis::Symbol = :z, bound_mode::NTuple{2, BoundValue}=(1,0),
    D = 1.0, dx = nothing,
    dtype=Float32, gpu=nothing
)

    # --- make sure the types are as expected ---
    img = BitArray(img .!= 0)
    img = atleast_3d(img)

    @assert D isa Number || size(img) == size(D) "For scalar field D, size should match img size"
    D = dtype.(D)

    nnodes = count(img)
    # If gpu is not specified, use GPU if the image is large enough
    gpu = !isnothing(gpu) ? gpu : (nnodes >= 100_000) && CUDA.functional()

    #check bound_mode has expected types
    for (i, b) in enumerate(bound_mode)
        if b isa Number
            continue
        elseif b isa Function
            # test-call at t=0 to ensure it returns a number
            val = b(0.0)
            @assert val isa Number "bound_mode[$i](t) must return a Number"
        else
            error("bound_mode[$i] must be a Number, Function, or NaN")
        end
    end
    #make sure bound_mode number types match data type
    bound_mode = ntuple(i -> begin
        b = bound_mode[i]
        b isa Number ? dtype(b) : b
        end, 2
    )

    !(axis == :x || axis == :y || axis == :z) && (error("axis must be :x, :y, or :z"))

    @assert size(img, AXIS_DEFINITION[axis]) > 1 "Image must have at least 2 voxels along the chosen axis"
    #default dx for dimensionless distance of 1 between bounds
    isnothing(dx) && (dx = 1/(size(img, AXIS_DEFINITION[axis])-1)) 

    # map from 3D image coords to pore_vector coords, for use in getting observables
    grid_to_vec = build_grid_to_vec(img)
    # finite difference matrix for dC = A*C
    A = build_transient_operator(img, D, bound_mode; axis=axis, dx=dx, gpu=gpu)

    return TransientProblem(size(img),dx,dt,D, img, grid_to_vec, axis, bound_mode,A,gpu,dtype)
end


"""
init_state(prob::TransientProblem)

returns a `TransientState` for the given `TransientProblem`

# Arguments
prob: A `TransientProblem` defining the problem

#kwargs
C0: an array with dimensions of the problem image representing initial concentration distribution.
    represents the initial concentration profile, defaults to zero everywhere
alg: the differential equation numerical algorithm, defaults to ROCK2, currently only supports explicit methods
reltol: relative tolerance for the differential equation solver, default 1e-3 *to improve accuracy, prioritize solver choice
abstol: absolute tolerance for the differential equation solver, default 1e-6
"""
function init_state(prob::TransientProblem; C0=nothing, alg=ROCK2(), reltol=1e-3, abstol=1e-6)

    if C0 === nothing 
        C0 = zeros(prob.eltype, prob.dims)
    else 
        C0 = atleast_3d(C0)
        @assert size(C0) == size(prob.img) "C0 dims must match img"
        C0 = prob.eltype.(C0)
    end
    
    apply_boundaries!(C0, prob) #sets dirichlet bounds along axis
    C0 = C0[prob.img] #compress C0 to vector of pore voxels

    if prob.gpu
        C0 = cu(C0)
    end

    dC! = make_appropriate_dC_function(prob)

    prob_ode = ODEProblem(dC!, C0, (0,1)) #tspan = (0,1) is arbitrary
    integrator = init(prob_ode, alg; save_everystep=false, reltol=reltol, abstol=abstol) #in terms of tolerance, polynomial max/min terms for ROCK algorithm should be adjustable?

    #create history and input initial values
    C_hist = Vector{Vector{prob.eltype}}() #stored as vector of pore voxels
    t_hist = Float64[]
    push!(t_hist, 0.0)  #push the initial conditions to the data arrays
    push!(C_hist, Array(C0))

    return TransientState(integrator, t_hist,C_hist)
end


##----- function for running the solver on the structs ------
"""
solve!(state::TransientState, prob::TransientProblem, stop_condition)

steps the state forward by steps of prob.dt until stop_condition(t, C) returns true, storing C distribution
at every dt step, on CPU regardless of whether running on GPU

#Arguments
    state: a TransientState
    prob: a TransientProblem matching the TransientState
    stop_condition: function, (t_hist, C_hist) -> bool. Based on the time and distribution vector up to a certain timestep
        returns true if condition for stopping integration and returning results is met.
        short hands include stop_at_time(time), stop_at_avg_concentration(conc, problem), stop_at_delta_flux(delta_flux, problem)
        if writing custom stop_condition, note that C_hist is in 1D pore voxel only form

kwargs: 
    max_iter: integer, steps of length prob.dt after which to stop integration if stop condition still unmet
    verbose: bool, if true, displays an every `prob.dt` time units time a data point is saved
"""
function solve!(state::TransientState, prob::TransientProblem, stop_condition; max_iter=500, verbose = false)

    for _ in 1:max_iter
        step!(state.integrator, prob.dt, true) #false means don't force timesteps to land at exactly t+dt, true means precisely t+dt
        push!(state.t, state.integrator.t)
        push!(state.C, Array(state.integrator.u)) #move from wherever u is to CPU memory

        verbose && @info "reached simulation time $(state.t[end])"

        if stop_condition(state.t, state.C)
            return
        end
    end
    @warn "Reached max_iter=$max_iter at t=$(state.t[end]) without satisfying stop_condition."
end


##--- functions for local use in construction of problem---

"""
returns a sparse matrix for finding the finite-difference based time derivative operator (Laplacian) of a voxel based
    concentration array

#Arguments
    img: 3D binary array representing porous material
    D: diffusion constant of substance in pores, or scalar field with dims of img
    bound_mode: Tuple{Number, Number}
        the values of dirichlet bounds for the two faces at either end of 'axis'
        a NaN value corresponds to an insulated boundary ex. (1,NaN) 
    axis: Symbol, the axis :x :y or :z which will have non-insulated bounds on the associated perpendicular faces
        defaults to :z
    dx: distance between nodes
    gpu: build a CUDA sparse matrix or normal sparse matrix

"""
function build_transient_operator(img, D, bound_mode; axis, dx, gpu)

    gpu && (img = cu(img))

    nnodes = sum(img)

    # Connectivity (CPU or GPU)
    conns = create_connectivity_list(img)

    # Edge diffusivity (scalar or field)
    if !(D isa Number)
        D_local = atleast_3d(D)
        gpu && (D_local = cu(D_local))
    end

    gd = D isa Number ? D : interpolate_edge_values(D_local[img], conns)

    # Adjacency matrix (CPU or GPU)
    am = create_adjacency_matrix(conns; n=nnodes, weights=gd)

    # Laplacian
    A = laplacian(am)

    # Scale by 1/dx^2, and inverted from the laplacian
    nonzeros(A) .= nonzeros(A) ./ (-dx^2)

    # Dirichlet nodes
    axis_to_boundaries = Dict(
        :x => (:left, :right), :y => (:front, :back), :z => (:bottom, :top)
    )
    inlet_face, outlet_face = axis_to_boundaries[axis]
    #bc_nodes are only zeroed if they are dirichlet (including time dependent function), otherwise face nodes are insulated by default
    bc_nodes = Int[]
    if bound_mode[1] isa Function || !isnan(bound_mode[1]) append!(bc_nodes, find_boundary_nodes(img, inlet_face)) end
    if bound_mode[2] isa Function || !isnan(bound_mode[2]) append!(bc_nodes, find_boundary_nodes(img, outlet_face)) end
    bc_nodes = gpu ? Int32.(bc_nodes) : bc_nodes

    # Transient BC: zero rows only -> asymmetric matrix, explicit only
    zero_rows!(A, bc_nodes)

    return A
end

#CPU version of zero_rows! from kernels/sparse.jl is needed
function zero_rows!(A::SparseMatrixCSC, rows)
    I, _, _ = findnz(A)
    row_inds = overlap_indices🚀(I, rows)
    A.nzval[row_inds] .= 0
    dropzeros!(A)
    return nothing
end


"""
apply_boundaries!(C0, problem::TransientProblem)

overwrites any faces of C0 corresponding to a non-NaN value of problem.bound_mode with bound_mode value
if the bound_mode is a function of time, it will write in the function evaluated at t=0
"""
function apply_boundaries!(C0, prob)

    ax = AXIS_DEFINITION[prob.axis]   # 1, 2, or 3
    N  = size(C0, ax)

    #dirichlet boundary handling for 2 faces of C0 perpendicular to axis, allow for C=f(t) case as well
    if prob.bound_mode[1] isa Function || !isnan(prob.bound_mode[1]) 
        selectdim(C0, ax, 1) .= 
        prob.bound_mode[1] isa Number ? prob.bound_mode[1] : prob.bound_mode[1](prob.eltype(0.0))
    end
    if prob.bound_mode[2] isa Function || !isnan(prob.bound_mode[2]) 
        selectdim(C0, ax, N) .= 
        prob.bound_mode[2] isa Number ? prob.bound_mode[2] : prob.bound_mode[2](prob.eltype(0.0))
    end

end

#build a dC! function that handles time-dependent boundaries only if necessary
function make_appropriate_dC_function(prob)
    in_is_function  = prob.bound_mode[1] isa Function
    out_is_function = prob.bound_mode[2] isa Function

    inlet_inds  = slice_vec_indices(prob, 1)
    outlet_inds = slice_vec_indices(prob, prob.dims[AXIS_DEFINITION[prob.axis]])

    return let inlet_inds=inlet_inds, outlet_inds=outlet_inds, prob=prob

        if in_is_function && out_is_function
            (dC, C, p, t) -> begin
                C[inlet_inds]  .= convert(prob.eltype, prob.bound_mode[1](t))
                C[outlet_inds] .= convert(prob.eltype, prob.bound_mode[2](t))
                mul!(dC, prob.A, C)
            end

        elseif in_is_function
            (dC, C, p, t) -> begin
                C[inlet_inds] .= convert(prob.eltype, prob.bound_mode[1](t))
                mul!(dC, prob.A, C)
            end

        elseif out_is_function
            (dC, C, p, t) -> begin
                C[outlet_inds] .= convert(prob.eltype, prob.bound_mode[2](t))
                mul!(dC, prob.A, C)
            end

        else
            # Fast path
            (dC, C, p, t) -> mul!(dC, prob.A, C)
        end
    end
end


##--- define prebuilt stop conditions for convenience---

"""
stop_at_time(t_final)

creates a stop_condition for stopping at a certain solver time
"""
function stop_at_time(t_final)
    return (t_hist, C_hist) -> t_hist[end] >= t_final
end

"""
stop_at_avg_concentration(C_final, img::Array)
stop_at_avg_concentration(C_final, problem::TransientProblem)

creates a stop_condition for the average concentration across all non-obstacle voxels reaching a given concentration

requires img to average only non-obstacle voxels
assumes obstacle_voxels remain at 0
"""
function stop_at_avg_concentration(C_final, img)
    num_active = sum(img)
    return (t_hist, C_hist) -> sum(C_hist[end])/num_active >= C_final
end
stop_at_avg_concentration(C_final, problem::TransientProblem) = stop_at_avg_concentration(C_final, problem.img)


"""
stop_at_delta_flux(delta, problem::TransientProblem)

creates a function to evaluate if the incoming flux and outgoing flux have a difference at or below delta
The flux at the two faces on 'prob.axis' should approach 0 as the distribution goes to equilibrium
put a smaller delta value to go closer to equilibrium before stopping run

#Arguments
    delta: the scalar for which function will return true when the flux difference is at or below it
    problem: relevant TransientProblem
"""
function stop_at_delta_flux(delta, prob::TransientProblem)
    return (t_hist, C_hist) -> abs(get_flux(C_hist[end], prob; ind = :end)-get_flux(C_hist[end], prob; ind=1)) <= delta
end



"""
    stop_at_periodic(freq, prob; reltol=1e-3, Nphase=4, frac_period=0.3, depth = 1.0)

Constructs a stop-condition function for `solve!` that detects when the
solution has reached periodic steady state for a frequency 'freq'.
Specifically for use with a TransientProblem with a periodic function 
boundary at a depth expected to reach a periodic state.

The condition checks whether the slice concentration at several phase
points (obtained by linear interpolation) within the *last `frac_period`
fraction of a period* matches the concentration one period earlier,
within relative tolerance `reltol`*A where A is amplitude of the latest period.

Do not use this stop condition if prob.dt is not many times smaller than the period.

Arguments
---------
- `freq` : driving frequency (Hz)
- `prob` : `TransientProblem` defining geometry and slice axis

Keyword Arguments
-----------
- `reltol`  : relative tolerance for periodicity (default 1e-3)
- `Nphase` : number of phase points to test across the trailing window
- `frac_period` : fraction of the period to test (0 < frac_period ≤ 1).
                  Smaller values require less history; default 0.3.
- `depth` : depth fraction in (0,1], how far along the main axis is the 
            the concentration will be evaluated for periodicity; default 1.0,
            associated with a insulated boundary at x=L face


Returns
-------
A function `(t_hist, C_hist) -> Bool` suitable for use as a stop condition
in `solve!`.
"""
function stop_at_periodic(freq, prob::TransientProblem;
                          reltol=1e-2, Nphase::Int=4, frac_period=0.3, depth = 1.0)

    @assert 0 < frac_period "frac_period must be greater than 0."
    @assert 0 < depth ≤ 1 "depth must be in (0,1]."

    T = 1/freq
    phases = range(0, frac_period; length=Nphase)

    N = prob.dims[AXIS_DEFINITION[prob.axis]]
    depth_ind = round(Int, depth * (N-1) +1)

    #keep track of the average concentration in the relevant slice each timestep w/o extra computation
    slice_hist = Float64[]

    # --- helper: linear interpolation ---
    function interp_slice_conc(t, t_vals, C_vals)
        i1 = findlast(ti -> ti ≤ t, t_vals)
        i2 = findfirst(ti -> ti ≥ t, t_vals)

        # not enough history yet
        if isnothing(i1) || isnothing(i2)
            return nothing
        end

        # exact hit
        if i1 == i2
            return C_vals[i1]
        end

        t1, t2 = t_vals[i1], t_vals[i2]
        C1 = C_vals[i1]
        C2 = C_vals[i2]

        w = (t - t1) / (t2 - t1)
        return (1 - w)*C1 + w*C2
    end

    # --- returned stop condition ---
    return (t_hist, C_hist) -> begin
        t_end = t_hist[end]
        push!(slice_hist, get_slice_conc(C_hist[end], prob, depth_ind))

        tmin = t_end - (1 + frac_period)*T
        # need at least (1 + frac_period)*T of history
        if tmin < 0
            return false
        end

        #in-case there were pre-existing ts before slice_hist
        ts = @view t_hist[end-length(slice_hist)+1:end]

        #get approx previous period amplitude to normalize error
        i0 = searchsortedfirst(ts, tmin)
        A = maximum(@view slice_hist[i0:end]) - minimum(@view slice_hist[i0:end])

        for ϕ in phases
            t1 = t_end - ϕ*T
            t2 = t_end - T - ϕ*T
            
            C1 = interp_slice_conc(t1, ts, slice_hist)
            C2 = interp_slice_conc(t2, ts, slice_hist)

            # insufficient history or interpolation failed
            if C1 === nothing || C2 === nothing
                return false
            end

            if abs(C1 - C2) > reltol *A #error scaled by amplitude and avoid 0 division
                return false
            end
        end

        return true
    end
end