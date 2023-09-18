# %% Imports
using CUDA
using CUDA.CUSPARSE
using LinearSolve
using NaNStatistics
using Plots
using Random
using SparseArrays
using Statistics

includet("imgen.jl")
includet("numpytools.jl")
includet("pdetools.jl")
includet("plottools.jl")
includet("simulations.jl")
includet("topotools.jl")
includet("utils.jl")

# %% Initialize and set paths
Random.seed!(2)
set_plotsjl_defaults()
gpu_id = 1

# %% Generate/load the image
@info "Generating/loading the voxel image"
img = blobs(shape=(256, 256, 256), porosity=0.65, blobiness=1)
img = denoise(img, 2)
display(imshow(img, z_idx=1))
@info "Image size: $(size(img))"

# %% Build Ax = b on CPU/GPU
@info "Setting up τ simulation (assembling Ax = b)"
prob = tortuosity_fdm(img, axis=:x)
gpu = sum(img) >= 100_000 ? true : false
@info "Offloading the system to GPU: $gpu"
prob = gpu ? LinearProblem(cu(prob.A), cu(prob.b)) : prob

# %% Solve Ax = b using an iterative solver
@info "Solving the system using KrylovJL_CG"
@time sol = solve(prob, KrylovJL_CG(), verbose=true, reltol=1.0f-5)
@info "Average concentration: $(mean(sol.u))"
# %% Visualize the solution (2D slice)
isinteractive() && display(imshow(img, vals=sol.u, z_idx=100))

c = vec_to_field(sol.u, img)
tau = calc_tortuosity(c, img)
ff = calc_formation_factor(c, img)
eps = sum(img) / length(img)

@info "Tortuosity factor: $tau"
@info "Formation factor: $ff"
@info "Porosity: $eps"
