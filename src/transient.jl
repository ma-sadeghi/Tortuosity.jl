# Transient diffusion solver for 3D porous materials

struct TransientProblem{T,DType}
    dx::Float64
    dt::Float64
    D::DType
    img::BitArray{3}
    grid_to_vec::Array{Int,3}
    axis::Symbol
    bc_inlet::Union{Nothing,T}
    bc_outlet::Union{Nothing,T}
    A::Union{SparseMatrixCSC{T,Int},CUDA.CUSPARSE.CuSparseMatrixCSC{T,Int32}}
end

# Concentration history is stored on CPU to avoid GPU memory exhaustion.
# The ODE integrator alone would keep all snapshots on the compute device.
struct TransientState{T,I}
    integrator::I
    t::Vector{Float64}
    C::Vector{Vector{T}}
end

"""
    TransientProblem(img, dt; axis=:z, bc_inlet=1, bc_outlet=0,
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
- `bc_inlet`: Dirichlet concentration at the inlet face along `axis`.
              Use `nothing` for an insulated (Neumann) boundary. Defaults to `1`.
- `bc_outlet`: Dirichlet concentration at the outlet face along `axis`.
               Use `nothing` for an insulated (Neumann) boundary. Defaults to `0`.
- `D`: scalar diffusion coefficient or scalar field of diffusivity
       at each pixel with shape img used inside pore voxels.
       Defaults to `1.0` for easy comparison.
- `dx`: physical spacing between adjacent voxel centers. If `nothing`,
        it is set to `1/(N_axis - 1)` so that the domain spans `[0,1]`
        along the chosen axis for easy comparison.
- `dtype`: numeric type used for the operator and solution arrays
           (e.g., `Float32` or `Float64`). Defaults to `Float32`.
- `gpu`: whether to run solver on the GPU. Defaults to `true`.
"""
function TransientProblem(
    img,
    dt;
    axis::Symbol=:z,
    bc_inlet=1,
    bc_outlet=0,
    D=1.0,
    dx=nothing,
    dtype=Float32,
    gpu=true,
)
    img = atleast_3d(img)
    img = BitArray(img .!= 0)
    @assert D isa Number || size(img) == size(D) "For scalar field D, size should match img size"
    D = D isa Number ? dtype(D) : dtype.(D)
    bc_inlet = isnothing(bc_inlet) ? nothing : dtype(bc_inlet)
    bc_outlet = isnothing(bc_outlet) ? nothing : dtype(bc_outlet)

    @assert size(img, axis_dim(axis)) > 1 "Image must have at least 2 voxels along the chosen axis"
    # Default dx so domain spans [0, 1] along axis
    isnothing(dx) && (dx = 1 / (size(img, axis_dim(axis)) - 1))

    grid_to_vec = build_grid_to_vec(img)
    A = build_transient_operator(img, D, bc_inlet, bc_outlet; axis=axis, dx=dx, gpu=gpu)

    return TransientProblem(dx, dt, D, img, grid_to_vec, axis, bc_inlet, bc_outlet, A)
end

"""
    init_state(prob::TransientProblem; C0=nothing, alg=ROCK4(),
               reltol=1e-3, abstol=1e-6)

Initialize a `TransientState` for the given `TransientProblem`, applying boundary
conditions to `C0` and setting up the ODE integrator.

# Arguments
- `prob`: a `TransientProblem` defining the problem.

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
        @assert size(C0) == size(prob.img) "C0 dims must match img"
    end

    apply_boundaries!(C0, prob)
    C0 = C0[prob.img]

    gpu = prob.A isa CUDA.CUSPARSE.CuSparseMatrixCSC
    if gpu
        C0 = cu(C0)
    end

    # A is captured via closure; passing it through ODEProblem causes Int32 truncation errors
    dC!(dC, C, p, t) = mul!(dC, prob.A, C)

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
        step!(state.integrator, prob.dt, true) # force step to land exactly at t + dt
        push!(state.t, state.integrator.t)
        push!(state.C, Array(state.integrator.u))

        verbose && @info "reached simulation time $(state.t[end])"

        if stop_condition(state.t, state.C)
            break
        end
    end
end

"""
    build_transient_operator(img, D, bc_inlet, bc_outlet; axis, dx, gpu)

Build the sparse finite-difference operator `A` such that `dC/dt = A * C` for the
pore-voxel concentration vector. Dirichlet boundary rows are zeroed so that
boundary values remain constant during integration.

# Arguments
- `img`: 3D binary array representing porous material.
- `D`: scalar diffusion coefficient, or a 3D scalar field with the same size as `img`.
- `bc_inlet`: Dirichlet value at the inlet face, or `nothing` for insulated.
- `bc_outlet`: Dirichlet value at the outlet face, or `nothing` for insulated.

# Keyword Arguments
- `axis`: `:x`, `:y`, or `:z` — the axis with non-insulated boundary faces.
- `dx`: physical spacing between adjacent voxel centers.
- `gpu`: if `true`, build a CUDA sparse matrix.
"""
function build_transient_operator(img, D, bc_inlet, bc_outlet; axis, dx, gpu)
    gpu && (img = cu(img))

    nnodes = sum(img)

    # 1. Connectivity (CPU or GPU)
    conns = create_connectivity_list(img)

    # 2. Edge diffusivity (scalar or field)
    if !(D isa Number)
        D_local = atleast_3d(D)
        gpu && (D_local = cu(D_local))
    end

    gd = D isa Number ? D : interpolate_edge_values(D_local[img], conns)

    # 3. Adjacency matrix (CPU or GPU)
    am = create_adjacency_matrix(conns; n=nnodes, weights=gd)

    # 4. Laplacian
    A = laplacian(am)

    # Negate and scale: Laplacian L has negative off-diagonals, but dC/dt = D/dx² * (-L) * C
    nonzeros(A) .= nonzeros(A) ./ (-dx^2)

    # Collect Dirichlet boundary nodes; faces default to insulated (Neumann)
    inlet_face, outlet_face = axis_faces(axis)
    bc_nodes = Int[]
    if !isnothing(bc_inlet)
        append!(bc_nodes, find_boundary_nodes(img, inlet_face))
    end
    if !isnothing(bc_outlet)
        append!(bc_nodes, find_boundary_nodes(img, outlet_face))
    end
    bc_nodes = gpu ? Int32.(bc_nodes) : bc_nodes

    # Zero rows so Dirichlet values remain constant during integration
    zero_rows!(A, bc_nodes)

    return A
end

# CPU counterpart to the GPU kernel in kernels/sparse.jl
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
whose boundary condition is not `nothing` is overwritten with that value. Face
voxels that correspond to obstacles (`D = 0`) are also set, so they should be
zeroed separately if needed.
"""
function apply_boundaries!(C0, prob)
    ax = axis_dim(prob.axis)   # 1, 2, or 3
    N = size(C0, ax)

    if !isnothing(prob.bc_inlet)
        selectdim(C0, ax, 1) .= prob.bc_inlet
    end
    if !isnothing(prob.bc_outlet)
        selectdim(C0, ax, N) .= prob.bc_outlet
    end
end

function stop_at_time(t_final)
    return (t_hist, C_hist) -> t_hist[end] >= t_final
end

"""
    stop_at_avg_concentration(C_final, img)
    stop_at_avg_concentration(C_final, prob::TransientProblem)

Create a stop condition that returns `true` when the average concentration across
pore voxels reaches `C_final`. Obstacle voxels are assumed to remain at zero.
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
    return (t_hist, C_hist) ->
        abs(get_flux(C_hist[end], prob; ind=:end) - get_flux(C_hist[end], prob; ind=1)) <=
        delta
end
