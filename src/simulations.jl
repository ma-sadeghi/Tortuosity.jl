# Steady-state diffusion simulation setup and utilities.

# Harmonic mean of node diffusivities at each edge: 2*D_a*D_b / (D_a + D_b).
# This is the standard finite-volume interface conductance for unit-spacing grids.
"""
    interpolate_edge_values(node_vals, conns)

Compute edge weights from node diffusivities using the harmonic mean:
`2 * D_a * D_b / (D_a + D_b)`. This is the standard finite-volume interface
conductance for unit-spacing grids.

# Arguments
- `node_vals`: diffusivity value for each node (1D vector, length = number of nodes).
- `conns`: `nedges x 2` connectivity matrix where each row is a `(source, target)` pair.
"""
function interpolate_edge_values(node_vals, conns)
    @assert length(node_vals) == maximum(conns)
    P1 = @view conns[:, 1]
    P2 = @view conns[:, 2]
    edge_vals = 1 ./ (1 ./ node_vals[P1] / 2 .+ 1 ./ node_vals[P2] / 2)
    return edge_vals
end

"""
    TortuositySimulation{A}

Holds the data for a steady-state diffusion problem on a binary pore image.

# Fields
- `img::A`: boolean pore mask (`true` = pore).
- `axis::Symbol`: transport direction (`:x`, `:y`, or `:z`).
- `prob::LinearProblem`: the assembled linear system ready for `solve(sim.prob, alg)`.
"""
struct TortuositySimulation{A<:AbstractArray{Bool}}
    img::A
    axis::Symbol
    prob::LinearProblem
end

function Base.show(io::IO, ts::TortuositySimulation)
    gpu = _on_gpu(ts.prob.b)
    msg = "TortuositySimulation(shape=$(size(ts.img)), axis=$(ts.axis), gpu=$(gpu))"
    return print(io, msg)
end

"""
    TortuositySimulation(img; axis, D=nothing, gpu=nothing, verbose=false)

Construct a `TortuositySimulation` for steady-state diffusion on a binary pore
image. Builds the graph Laplacian, applies Dirichlet boundary conditions
(`c = 1` at inlet, `c = 0` at outlet), and returns a ready-to-solve `LinearProblem`.

# Arguments
- `img`: boolean array where `true` = pore, `false` = solid. Promoted to 3D if needed.

# Keyword Arguments
- `axis`: transport direction (`:x`, `:y`, or `:z`).
- `D`: diffusivity. `nothing` for uniform (default), or an array matching `img` shape
  for spatially variable diffusivity.
- `gpu`: `true` to force GPU, `false` for CPU, `nothing` (default) to auto-detect
  (uses GPU when >= 100k pore voxels and a GPU backend is available).
- `verbose`: print progress messages. Default: `false`.
"""
function TortuositySimulation(img; axis, D=nothing, gpu=nothing, verbose=false)
    verbose && @info "Preprocessing image..."
    img = atleast_3d(img)
    @assert img isa AbstractArray{Bool} "Image must be a boolean array"
    D = isnothing(D) ? nothing : atleast_3d(D)

    # Deal with variable diffusivity
    if D isa AbstractArray
        @assert size(D) == size(img) "Diffusivity matrix D must match image size"
        @assert count(D .> 0) == count(img) "Diffusivity matrix D must have the same \
            number of non-zero elements as the image"
    end

    nnodes = sum(img)
    # Auto-detect GPU: use if backend is available and image is large enough
    if isnothing(gpu)
        gpu = !isnothing(_preferred_gpu_backend[]) && nnodes >= 100_000
    elseif gpu && isnothing(_preferred_gpu_backend[])
        error("`gpu=true` was requested but no GPU backend is registered. \
               Load a GPU package first (e.g. `using CUDA`, `using Metal`, or `using AMDGPU`).")
    end

    # Compute boundary nodes BEFORE GPU transfer (cheap CPU operation)
    verbose && @info "Setting up boundary conditions..."
    inlet, outlet = axis_faces(axis)
    inlet_nodes = find_boundary_nodes(img, inlet)
    outlet_nodes = find_boundary_nodes(img, outlet)

    # Move to GPU if needed. Keep `img` on CPU for the struct (postprocessing
    # helpers like tortuosity() expect a CPU mask); `img_dev` is the copy
    # handed to the kernels.
    verbose && gpu && @info "Using GPU..."
    T = gpu ? Float32 : Float64
    img_dev = gpu ? _gpu_adapt[](img) : img
    D_dev = isnothing(D) ? nothing : (gpu ? _gpu_adapt[](D) : D)
    b = gpu ? fill!(_gpu_adapt[](zeros(T, nnodes)), zero(T)) : zeros(T, nnodes)
    D0 = T(1)

    verbose && @info "Creating connectivity list and adjacency matrices..."
    conns = create_connectivity_list(img_dev)

    # Voxel size = 1 => gd = D*A/l = D (since D is at nodes -> interpolate to edges)
    # NOTE: D[img] since D might contain non-conducting values (e.g., when using a subdomain)
    gd = isnothing(D_dev) ? D0 : interpolate_edge_values(D_dev[img_dev], conns)
    am = create_adjacency_matrix(conns; n=nnodes, weights=gd)
    # For diffusion, L of the adjacency matrix is the coefficient matrix
    A = laplacian(am)

    # Apply a fixed concentration drop of 1.0 between inlet and outlet
    bc_nodes = vcat(inlet_nodes, outlet_nodes)
    bc_vals = vcat(fill(T(1), length(inlet_nodes)), fill(T(0), length(outlet_nodes)))
    # For GPU: transfer bc_vals to GPU
    if gpu
        bc_vals = _gpu_adapt[](bc_vals)
    end
    apply_dirichlet_bc_fast!(A, b; nodes=bc_nodes, vals=bc_vals)

    return TortuositySimulation(img, axis, LinearProblem(A, b))
end
