# %% Imports
using Revise
using Tortuosity
using CUDA
using LinearSolve
using Plots
using Printf
using BenchmarkTools

# %% Generate/load the image
img = Imaginator.blobs(shape=(64, 64, 1), porosity=0.65, blobiness=0.5, seed=2);
display(heatmap(img[:, :, 1], aspect_ratio=:equal, clim=(0, 1)))

# %% Build Ax = b on CPU/GPU
sim = TortuositySimulation(img, axis=:x, gpu=false)

# %% Solve Ax = b using an iterative solver
@time sol = solve(sim.prob, KrylovJL_CG(), verbose=false, reltol=eltype(sim.prob.b)(1e-5));

# %% Compute the tortuosity factor and visualize the solution
c_grid = vec_to_field(sol.u, img)
τ = tortuosity(c_grid, :x)
F = formation_factor(c_grid, :x)
@info "τ: $(@sprintf("%.5f", τ)), ℱ: $(@sprintf("%.5f", F))"
display(heatmap(c_grid[:, :, 1], aspect_ratio=:equal, clim=(0, 1)))
