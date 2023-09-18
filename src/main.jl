# %% Imports
using CUDA
using CUDA.CUSPARSE
using LinearSolve
using NaNStatistics
using Plots
using Printf
using Random
using SparseArrays
using Statistics

includet("dnstools.jl")
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
img = blobs(shape=(64, 64, 64), porosity=0.65, blobiness=1)
img = denoise(img, 2)
isinteractive() && display(imshow(img, slice=1))
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

# %% Compute the tortuosity factor and visualize the solution
c = vec_to_field(sol.u, img)
tau = compute_tortuosity_factor(c, :x)
ff = compute_formation_factor(c, :x)
eps = sum(img) / length(img)
@info "τ: $(@sprintf("%.3f", tau)), ℱℱ: $(@sprintf("%.3f", ff)), ε = $(@sprintf("%.3f", eps))"
isinteractive() && display(imshow(c, slice=1))
