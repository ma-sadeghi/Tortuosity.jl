# function interpolate_edge_values(node_vals, conns)
#     @assert length(node_vals) == maximum(conns)
#     nedges = size(conns, 1)
#     edge_vals = similar(node_vals, nedges)
#     for i in 1:nedges
#         m, n = conns[i, :]
#         edge_vals[i] = 1 / (1 / node_vals[m] + 1 / node_vals[n])
#     end
#     return edge_vals
# end

function interpolate_edge_values(node_vals, conns)
    # Ensure both arrays are on the same device (GPU if conns is on GPU)
    @assert length(node_vals) == maximum(conns)

    # Extract node indices for all edges at once
    P1 = @view conns[:, 1]
    P2 = @view conns[:, 2]

    # Vectorized harmonic mean calculation
    edge_vals = 1 ./ (1 ./ node_vals[P1] .+ 1 ./ node_vals[P2])

    return edge_vals
end

struct TortuositySimulation
    img::AbstractArray{Bool}
    axis::Symbol
    prob::LinearProblem
end

function Base.show(io::IO, ts::TortuositySimulation)
    gpu = ts.prob.b isa CuArray
    msg = "TortuositySimulation(shape=$(size(ts.img)), axis=$(ts.axis), gpu=$(gpu))"
    return print(io, msg)
end

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
    # If gpu is not specified, use GPU if the image is large enough
    gpu = !isnothing(gpu) ? gpu : (nnodes >= 100_000) && CUDA.functional()

    # Move stuff to GPU if needed
    verbose && gpu && @info "Using GPU..."
    img = gpu ? cu(img) : img
    D0 = gpu ? 1.0f0 : 1.0
    D = isnothing(D) ? nothing : (gpu ? cu(D) : D)
    b = gpu ? CUDA.zeros(nnodes) : zeros(nnodes)

    verbose && @info "Creating connectivity list and adjacency matrices..."
    conns = create_connectivity_list(img)

    # Voxel size = 1 => gd = D⋅A/ℓ = D (since D is at nodes -> interpolate to edges)
    # NOTE: D[img] since D might contain non-conducting values (e.g., when using a subdomain)
    gd = isnothing(D) ? D0 : interpolate_edge_values(D[img], conns)
    am = create_adjacency_matrix(conns; n=nnodes, weights=gd)
    # For diffusion, ∇² of the adjacency matrix is the coefficient matrix
    A = laplacian(am)

    verbose && @info "Setting up boundary conditions..."
    axis_to_boundaries = Dict(
        :x => (:left, :right), :y => (:front, :back), :z => (:bottom, :top)
    )

    inlet, outlet = axis_to_boundaries[axis]
    inlet = find_boundary_nodes(img, inlet)
    outlet = find_boundary_nodes(img, outlet)

    # Apply a fixed concentration drop of 1.0 between inlet and outlet
    bc_nodes = vcat(inlet, outlet)
    bc_vals = vcat(fill(1.0, length(inlet)), fill(0.0, length(outlet)))
    # Pre-process for GPU if needed
    bc_nodes = gpu ? Int32.(bc_nodes) : bc_nodes
    bc_vals = gpu ? cu(bc_vals) : bc_vals
    apply_dirichlet_bc🚀!(A, b; nodes=bc_nodes, vals=bc_vals)

    return TortuositySimulation(img, axis, LinearProblem(A, b))
end
