using LinearSolve

includet("topotools.jl")
includet("pdetools.jl")


function tortuosity_fdm(img; axis=:x)
    nnodes = sum(img)
    conns = create_connectivity_list🚀(img)
    # * Voxel size = 1 => ℓ = A = 1, set D = 1 => diffusive conductance = 1
    am = create_adjacency_matrix(conns, n=nnodes, weights=1.0)
    # * For diffusion, ∇² of the adjacency matrix is the coefficient matrix
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

    # * Apply a fixed concentration drop of 1.0 between inlet and outlet
    apply_dirichlet_bc🚀!(A, b, nodes=inlet, vals=1.0)
    apply_dirichlet_bc🚀!(A, b, nodes=outlet, vals=0.0)

    return LinearProblem(A, b)
end
