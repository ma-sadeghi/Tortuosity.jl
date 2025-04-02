function interpolate_edge_values(node_vals, conns)
    @assert length(node_vals) == maximum(conns)
    nedges = size(conns, 1)
    edge_vals = similar(node_vals, nedges)
    for i in 1:nedges
        m, n = conns[i, :]
        edge_vals[i] = 1 / (1 / node_vals[m] + 1 / node_vals[n])
    end
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

function TortuositySimulation(img; axis, D=nothing, gpu=nothing)
    nnodes = sum(img)

    # If gpu is not specified, use GPU if the image is large enough
    gpu = !isnothing(gpu) ? gpu : (nnodes >= 100_000) && CUDA.functional()
    img = gpu ? cu(img) : img

    conns = create_connectivity_list(img)

    # Voxel size = 1 => gd = D⋅A/ℓ = D (since D is at nodes -> interpolate to edges)
    gd = D === nothing ? 1.0f0 : interpolate_edge_values(D, conns)
    am = create_adjacency_matrix(conns; n=nnodes, weights=gd)
    # For diffusion, ∇² of the adjacency matrix is the coefficient matrix
    A = laplacian(am)
    b = gpu ? CUDA.zeros(nnodes) : zeros(nnodes)

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
