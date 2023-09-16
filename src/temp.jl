using LinearSolve
using Plots
using Random

includet("imgen.jl")
includet("pdetools.jl")
includet("topotools.jl")

Random.seed!(2)
shape = (256, 256)
porosity = 0.65
blobiness = 1
gd = 1.0

im = blobs(shape, porosity, blobiness)
im = denoise(im, 2)
display(heatmap(im, aspect_ratio=:equal, color=:viridis, clim=(0, 1)))

nnodes = sum(im)
conns = create_connectivity_list(im)
am = create_adjacency_matrix(conns, nnodes=nnodes, weights=gd)
A = laplacian(am)
b = zeros(nnodes)

left = find_boundary_nodes(im, :left)
right = find_boundary_nodes(im, :right)

apply_dirichlet_bc!(A, b, bc_nodes=left, bc_values=1.0)
apply_dirichlet_bc!(A, b, bc_nodes=right, bc_values=0.0)

prob = LinearProblem(A, b)
sol = solve(prob, KrylovJL_CG(), verbose=true, reltol=1e-6)

c = zeros(size(im)) * NaN
c[im] = Array(sol.u)

display(heatmap(c[:, :, 1], aspect_ratio=:equal, color=:viridis, clim=(0, 1)))
