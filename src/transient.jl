# Written by Sawyer Hossfeld
# Supervised by Prof. Jeff Gostick
# Winter Term, 2026
# Transient diffusion solver for cubic array representation of
# 3D porous material and related analysis tools


## definition of axis symbols and related shorthand

const AXIS_DEFINITION = Dict(
    :x => 1,
    :y => 2,
    :z => 3
)

#used for kwargs in functions to act on dims other than primary dim
const AXIS_COMPLEMENT = Dict(
    1 => (2,3),
    2 => (1,3),
    3 => (1,2),
    :x => (2,3),
    :y => (1,3),
    :z => (1,2)
)


##------ define structs for convenience of only passing all parameters once ------

#struct for problem definition, not solution 
struct TransientProblem{T}
    dims::NTuple{3, Int}
    dx::Float64
    dt::Float64
    D_free::T
    img::BitArray{3}
    axis::Symbol
    bound_mode::NTuple{2,T}
    A::Union{
    SparseMatrixCSC{T,Int},
    CUDA.CUSPARSE.CuSparseMatrixCSC{T,Int32}
    }
    gpu::Bool
    eltype::Type{T}
end

#stores integrator and datapoints
#the reason this struct exists instead of just using ODE integrator (the type that 'integrator' is), which could store t and C history in itself
# is that 'integrator.u' or C, would be stored on the same device as is being used
# for the GPU case this would be mean huge GPU memory usage and seriously limit the size that can be run
struct TransientState{T}
    integrator          
    t::Vector{Float64}
    C::Vector{Array{T,3}}
end

##----- define constructors for structs ------
@doc """
TransientProblem(img, D_free, dx, dt)

Constructs a TransientProblem for solving transient diffusion through a N^3 array representing a porous material

# Arguments
D: a NxbyNybyNz array of positive scalars representing the diffusion constant in each voxel of the problem
dx: the distance between adjacent nodes in each dimension
dt: the size of timesteps between saving the concentration profile and checking the stop_condition
    this is **not** the internal timestep used by the differential equation solver

# kwargs
axis: Symbol, the axis :x :y or :z which will have non-insulated bounds on the associated perpendicular faces
        defaults to :z
bound_mode: Tuple{Number, Number}
        the values of dirichlet bounds for the two faces at either end of 'axis'
        a NaN value corresponds to an insulated boundary ex. (1,NaN) 
        defaults to (1,0)
dtype: the data type of numbers used for the solution, ex. Float32, Float64
gpu: a bool for whether the solver is run on the GPU. defaults to true
"""
function TransientProblem(img, D_free, dx, dt; 
    axis::Symbol = :z, bound_mode::NTuple=(1,0),
    dtype=Float32, gpu=true
)

    #make sure the types are as expected
    mask = BitArray(img .!= 0)
    D_free = dtype(D_free)
    bound_mode = dtype.(bound_mode)

    # finite difference matrix for dC = A*C
    A = build_operator(mask, D_free, dx, axis, bound_mode, dtype)

    if gpu
        A = cu(A)
    end

    return TransientProblem(size(img),dx,dt,D_free, mask, axis, bound_mode,A,gpu,dtype)
end


"""
init_state(prob::TransientProblem)

returns a TransientState for the given TransientProblem

# Arguments
prob: A TransientProblem defining the problem

#kwargs
C0: an array with dimensions of porosity mask representing initial concentration distribution.
    represents the initial concentration profile, defaults to zero
alg: the differential equation numerical algorithm, defaults to ROCK2, currently only supports explicit methods
reltol: relative tolerance for the differential equation solver, default 1e-3 *to improve accuracy, prioritize solver choice
abstol: absolute tolerance for the differential equation solver, default 1e-6
"""
function init_state(prob::TransientProblem; C0=nothing, alg=ROCK2(), reltol=1e-3, abstol=1e-6)

    if C0 === nothing
        C0 = zeros(prob.eltype, prob.dims)
    end

    apply_boundaries!(C0, prob)
    C0 .*= prob.img #give C a zero concentration at 'obstacle' voxels
    C0 = vec(C0)

    if prob.gpu
        C0 = cu(C0)
    end

    function dC!(dC, C, p, t)
        mul!(dC, prob.A, C)
    end

    prob_ode = ODEProblem(dC!, C0, (0,1)) #tspan = (0,1) is arbitrary
    integrator = init(prob_ode, alg; save_everystep=false, reltol=reltol, abstol=abstol) #in terms of tolerance, polynomial terms of ROCK alg play a roll?

    #create history and input initial values
    C_hist = Vector{Array{prob.eltype,3}}() #stored in 3D shape
    t_hist = Float64[]
    push!(t_hist, 0.0)  #push the initial conditions to the data arrays
    push!(C_hist, reshape(C0, prob.dims))

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
    stop_condition: function, (t, C) -> bool. Based on the time and distribution vector up to a certain timestep
        returns true if condition for stopping integration and returning results is met.
        short hands include stop_at_time(time), stop_at_avg_concentration(conc, problem), stop_at_delta_flux(delta_flux, problem)

kwargs: 
    max_iter: integer, steps of length prob.dt after which to stop integration if stop condition still unmet
"""
function solve!(state::TransientState, prob::TransientProblem, stop_condition; max_iter=500, verbose = false)

    for _ in 1:max_iter
        step!(state.integrator, prob.dt, true) #false means don't force timesteps to land at exactly t+dt, true means precisely t+dt
        push!(state.t, state.integrator.t)
        push!(state.C, reshape(Array(state.integrator.u), prob.dims))

        verbose && @info "reached simulation time $(state.t[end])"

        if stop_condition(state.t, state.C)
            break
        end
    end
end


##--- functions for local use in construction of problem---

#= pre chatGPT feedback speed-up constructor, kept around incase I need something from it for now
# this also could have been improved by using bool arrays for indexing maybe
function build_operator(img, D_free, dx, bound_types, dtype)
    
    (Nx,Ny,Nz) = size(img)
    Nxy = Nx*Ny
    Nxyz = Nx*Ny*Nz
    
    #RHS matrix construction starting with off-diagonals
    #note multiplication by mask which prevents end of row/column from interaction with the start of the next row/column
    A = Int8.(spdiagm(
        1=>ones(Nxyz-1).*((1:Nxyz-1).%Nx .!= 0),
        -1=>ones(Nxyz-1).*((1:Nxyz-1).%Nx .!= 0),
        Nx=>ones(Nxyz-Nx).*((0:Nxyz-Nx-1).%Nxy .< Nxy-Nx),
        -Nx=>ones(Nxyz-Nx).*((0:Nxyz-Nx-1).%Nxy .< Nxy-Nx),
        Nxy => ones(Nxyz-Nxy),
        -Nxy=> ones(Nxyz-Nxy)
    ));

    #zero the columns and rows containing 'obstacle' for no interaction
    mask = spdiagm(img[:])
    A = mask * A
    A = A * mask #also zero columns

    # center diagonal for generic nodes equals -sum of off-diagonals
    A .-= spdiagm(0=> sum(A,dims=2)[:])

    #mask for zeroing dirichlet/constant boundary faces, a bit harder to read because its flattened to 1D
    mask =spdiagm( 1 .-( 
                ((1:Nxyz).<=(Nxy))          .&&!isnan(bound_types[1][1])
            .||((1:Nxyz).>(Nxyz-Nxy))       .&&!isnan(bound_types[1][2])
            .||((1:Nxyz).%Nx.==1)           .&&!isnan(bound_types[2][1])
            .||((1:Nxyz).%Nx.==0)           .&&!isnan(bound_types[2][2])
            .||((0:Nxyz-1).%Nxy.<=Nxy-Nx)   .&&!isnan(bound_types[3][1])
            .||((0:Nxyz-1).%Nxy.<=Nxy-Nx)   .&&!isnan(bound_types[3][2])
    ))
    A = mask * A
    dropzeros!(A) #there is now a lot of 'stored' zeros in the sparse matrix, get rid of them for the loop, probably worth it
    
    return D_free/dx^2 * dtype.(A)
end
=#

"""
returns a sparse matrix for finding the finite-difference based time derivative operator (Laplacian) of a voxel based
    concentration array

#Arguments
    mask: 3D binary array representing porous material
    D_free: diffusion constant of substance in pores
    axis: Symbol, the axis :x :y or :z which will have non-insulated bounds on the associated perpendicular faces
        defaults to :z
    bound_mode: Tuple{Number, Number}
        the values of dirichlet bounds for the two faces at either end of 'axis'
        a NaN value corresponds to an insulated boundary ex. (1,NaN) 
    dtype: datatype of output operator, to match datatype of problem
"""
function build_operator(mask, D_free, dx, axis, bound_mode, dtype)
    ax= AXIS_DEFINITION[axis] #from symbol to int
    Nx, Ny, Nz = size(mask)
    Nxyz = Nx*Ny*Nz

    # Dirichlet mask: true for each face corresponding to non NaN boundary type
    is_dirichlet = falses(Nx, Ny, Nz)

    #apply bounds to end faces perpendicular to axis
    if !isnan(bound_mode[1]) selectdim(is_dirichlet, ax, 1) .= true end
    if !isnan(bound_mode[2]) selectdim(is_dirichlet, ax, size(is_dirichlet)[ax]) .= true end

    rows = Int[]
    cols = Int[]
    vals = Int8[] # values only need to range from -4 to 1 before scaling with D/dx^2

    # helper to push entries
    function add(i,j,v)
        push!(rows, i)
        push!(cols, j)
        push!(vals, v)
    end

    # loop over all voxels
    @inbounds for j in 1:Ny, i in 1:Nx , k in 1:Nz
        if mask[i,j,k] == 0 || is_dirichlet[i,j,k] #mask and dirichlet rows are 0 -> constant concentration
            continue
        end

        idx = i + (j-1)*Nx + (k-1)*Nx*Ny #find flattened index of node

        diag = 0
        
        for (di,dj,dk) in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))# neighbors
            ii = i+di; jj = j+dj; kk = k+dk #flattened index of neighbor node
            if 1 ≤ ii ≤ Nx && 1 ≤ jj ≤ Ny && 1 ≤ kk ≤ Nz && mask[ii,jj,kk] == 1
                jdx = ii + (jj-1)*Nx + (kk-1)*Nx*Ny
                add(idx, jdx, 1)
                diag -= 1   #for flux balance, diagonal is negative sum of rest of row
            end
        end

        add(idx, idx, diag)
    end

    A = sparse(rows, cols, vals, Nxyz, Nxyz)
    return (D_free/dx^2) * dtype.(A)
end


"""
apply_boundaries!(C0, bound_types)

overwrites any faces of C0 corresponding to a non-NaN value of bound_types with bound_types value
this will include face voxels that correspond to obstacles (D = 0), which should usually be zeroed to avoid confusion
"""
function apply_boundaries!(C0, prob)

    ax = AXIS_DEFINITION[prob.axis]   # 1, 2, or 3
    N  = size(C0, ax)

    #dirichlet boundary handling for 2 faces of C0 perpendicular to axis
    if !isnan(prob.bound_mode[1]) selectdim(C0, ax, 1) .= prob.bound_mode[1] end
    if !isnan(prob.bound_mode[2]) selectdim(C0, ax, N) .= prob.bound_mode[2] end

end




##--- define prebuilt stop conditions for convenience---


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

#assumes intake and outake is along axis corresponding to bound_types[1]
#this should be communicated better or made more general. similar problems exist elsewhere
"""
stop_at_delta_flux(delta, D, dx, mask, axis)
stop_at_delta_flux(delta, problem::TransientProblem)

creates a function to evaluate if the incoming flux and outgoing flux have a difference at or below delta
The flux at the two faces on 'prob.axis' should approach 0 as the distribution goes to equilibrium
put a smaller delta value to go closer to equilibrium before stopping run

#Arguments
    delta: the scalar for which function will return true when the flux difference is at or below it
    problem: relevant TransientProblem
"""
function stop_at_delta_flux(delta, D, dx, mask, axis)
    return (t_hist, C_hist) -> abs(get_flux(C_hist[end], D,dx,mask, axis; ind = :end)-get_flux(C_hist[end], D,dx,mask, axis; ind=1)) <= delta
end
stop_at_delta_flux(delta, problem::TransientProblem) = stop_at_delta_flux(delta, problem.D_free, problem.dx, problem.img, problem.axis)



