# %% Imports
using CUDA
using LinearSolve
using NPZ
using Plots
using Printf
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
gpu_id = 0

# %% Generate/load the image
@info "Generating/loading the voxel image"
# img = blobs(shape=(64, 64, 64), porosity=0.65, blobiness=1)
# img = denoise(img, 2)
fpath = "./images/sample_B9_v0_eps=H.npz"
img = npzread(fpath)["arr_0"][1:100, 1:100, 1:100]
display(heatmap(img[:, :, :], aspect_ratio=:equal, clim=(0, 1)))
# TODO: permute x and z axes until I fix the bug in dnstools.jl
img = permutedims(img, [3, 2, 1])
@info "Image size: $(size(img))"

# %% Build Ax = b on CPU/GPU
@info "Setting up τ simulation (assembling Ax = b)"
prob = tortuosity_fdm(img, axis=:x)
gpu = sum(img) >= 100_000 ? true : false
gpu && device!(gpu_id)
@info "Offloading the system to GPU: $gpu"
prob = gpu ? LinearProblem(cu(prob.A), cu(prob.b)) : prob
reltol = eltype(prob.b)(1e-5)

# %% Solve Ax = b using an iterative solver
@info "Solving the system using KrylovJL_CG"
@time sol = solve(prob, KrylovJL_CG(), verbose=true, reltol=reltol)
@info "Average concentration: $(mean(sol.u))"

# %% Compute the tortuosity factor and visualize the solution
c = vec_to_field(sol.u, img)
tau = compute_tortuosity_factor(c, :x)
ff = compute_formation_factor(c, :x)
eps = sum(img) / length(img)
@info "τ: $(@sprintf("%.3f", tau)), ℱℱ: $(@sprintf("%.3f", ff)), ε = $(@sprintf("%.3f", eps))"
display(heatmap(c[:, :, 50], aspect_ratio=:equal, clim=(0, 1)))
