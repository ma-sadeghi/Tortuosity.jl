using SparseArrays
using LinearSolve
using Random
using SpecialFunctions
using Images, ImageFiltering, ImageMorphology
using Plots


function generate_random_noise(dimx, dimy)
    rand(Bool, dimx, dimy)
end


function norm_to_uniform(im)
    lb, ub = minimum(im), maximum(im)
    im = (im .- mean(im)) / std(im)
    im = 1/2 * erfc.(-im / sqrt(2))
    im = (im .- lb) / (ub - lb)
    im = im * (ub - lb) .+ lb
end


function apply_gaussian_blur!(im, sigma)
    imfilter!(im, Kernel.gaussian(sigma))
end


function to_binary(im, threshold=0.5)
    map(x -> x < threshold ? true : false, im)
end


function disk(r)
    Bool.([sqrt((i - r - 1)^2 + (j - r - 1)^2) <= r for i in 1:2*r+1, j in 1:2*r+1])
end


function denoise_image(im, kernel_radius)
    selem = disk(kernel_radius)
    im = closing(im, selem)
    opening(im, selem)
end


function generate_blobs(shape, porosity, blobiness)
    im = generate_random_noise(shape...)
    sigma = mean(shape) / 40 / blobiness
    im = apply_gaussian_blur(im, sigma)
    im = norm_to_uniform(im)
    to_binary(im, porosity)
end


function laplacian(adjacency)
    degrees = vec(sum(adjacency, dims=2))
    degree_matrix = SparseArrays.spdiagm(0 => degrees)
    return degree_matrix - adjacency
end


function find_adjacent_indices(im)
    nx, ny, nz = size(im)
    idx = reshape(1:nx*ny*nz, nx, ny, nz)
    xconns = hcat(idx[1:nx-1, :, :][:], idx[2:nx, :, :][:])
    yconns = hcat(idx[:, 1:ny-1, :][:], idx[:, 2:ny, :][:])
    zconns = hcat(idx[:, :, 1:nz-1][:], idx[:, :, 2:nz][:])
    vcat(xconns, yconns, zconns)
end


function create_adjacency_matrix(conns; weights=1)
    nedges = size(conns, 1)
    conns = vcat(conns, conns[:, [2, 1]])
    if length(weights) == 1
        weights = fill(weights, nedges*2)
    elseif length(weights) == nedges
        weights = vcat(weights, weights)
    end
    sparse(conns[:, 1], conns[:, 2], weights, nnodes, nnodes)
end


function apply_dirichlet_bc!(A, b; bc_nodes, bc_values)
    # Add contribution from BCs to the RHS
    x_bc = zeros(nnodes)
    x_bc[bc_nodes] .= bc_values
    b = b .- A * x_bc
    # Zero out rows and columns corresponding to BCs
    I, J, V = findnz(A)
    row_indices = findall(in.(I, Ref(bc_nodes)))
    col_indices = findall(in.(J, Ref(bc_nodes)))
    V[col_indices] .= 0.0
    V[row_indices] .= 0.0
    diag_values = SparseArrays.diag(A)[bc_nodes]
    A = sparse(I, J, V, nnodes, nnodes)
    dropzeros!(A)
    # Ensure Dirichlet BCs are satisfied
    A[SparseArrays.diagind(A)[bc_nodes]] .= diag_values
    b[bc_nodes] .= bc_values .* diag_values
    return A, b
end


# Random.seed!(1)
# shape = (256, 256)
# porosity = 0.65
# blobiness = 1

# im = generate_blobs(shape, porosity, blobiness)
# im = denoise_image(im, 2)
# display(heatmap(im, aspect_ratio=:equal, color=:viridis, clim=(0, 1)))

g = 1.0
shape = (5, 4, 1)
nx, ny, nz = shape
im = ones(Bool, shape...)
nnodes = sum(im)

conns = find_adjacent_indices(im)
am = create_adjacency_matrix(conns, weights=g)
A = laplacian(am)
b = zeros(nnodes)

idx = reshape(1:nx*ny*nz, nx, ny, nz)
left = idx[1, :, :][:]
right = idx[end, :, :][:]
bc_left = 1.0
bc_right = 0.0

A, b = apply_dirichlet_bc!(A, b, bc_nodes=left, bc_values=bc_left)
A, b = apply_dirichlet_bc!(A, b, bc_nodes=right, bc_values=bc_right)

prob = LinearProblem(A, b)
sol = solve(prob, KrylovJL_CG(), verbose=true, reltol=1e-6)
