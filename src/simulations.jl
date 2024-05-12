function interpolate_edge_values(node_vals, conns)
    @assert length(node_vals) == maximum(conns)
    nedges = size(conns, 1)
    edge_vals = similar(node_vals, nedges)
    for i in 1:nedges
        m, n = conns[i, :]
        edge_vals[i] = 1 / (1/node_vals[m] + 1/node_vals[n])
    end
    return edge_vals
end


struct TortuositySimulation
    img::AbstractArray{Bool}
    axis::Symbol
    prob::LinearProblem
end


function TortuositySimulation(img; axis, D=nothing, gpu=nothing)
    nnodes = sum(img)
    conns = create_connectivity_list🚀🚀(img)

    # Voxel size = 1 => gd = D⋅A/ℓ = D (since D is at nodes -> interpolate to edges)
    gd = D === nothing ? 1.0 : interpolate_edge_values(D, conns)

    am = create_adjacency_matrix(conns, n=nnodes, weights=gd, gpu=gpu)
    # For diffusion, ∇² of the adjacency matrix is the coefficient matrix
    A = laplacian(am)
    b = zeros(nnodes)

    axis_to_boundaries = Dict(
        :x => (:left, :right),
        :y => (:front, :back),
        :z => (:bottom, :top)
    )

    inlet, outlet = axis_to_boundaries[axis]
    inlet = find_boundary_nodes(img, inlet)
    outlet = find_boundary_nodes(img, outlet)

    # Apply a fixed concentration drop of 1.0 between inlet and outlet
    apply_dirichlet_bc🚀!(A, b, nodes=inlet, vals=1.0)
    apply_dirichlet_bc🚀!(A, b, nodes=outlet, vals=0.0)

    # Offload to GPU if requested, otherwise default to GPU if nnodes >= 100_000
    gpu = gpu === nothing ? (nnodes >= 100_000 ? true : false) : gpu
    A, b = gpu ? (cu(A), cu(b)) : (A, b)

    return TortuositySimulation(img, axis, LinearProblem(A, b))
end
