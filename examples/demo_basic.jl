# %% Imports
using BenchmarkTools
using CUDA
using LinearSolve
using Plots
using Printf
using Tortuosity
using Tortuosity: TortuositySimulation, formation_factor, tortuosity, vec_to_grid

# %% Generate/load the image
show_plots = false
img = Imaginator.blobs(; shape=(64, 64, 1), porosity=0.65, blobiness=0.5, seed=2);
show_plots && display(heatmap(img[:, :, 1]; aspect_ratio=:equal, clim=(0, 1)));

# %% Build Ax = b on CPU/GPU
sim = TortuositySimulation(img; axis=:x, gpu=false);

# %% Solve Ax = b using an iterative solver
@time sol = solve(sim.prob, KrylovJL_CG(); verbose=false, reltol=1e-5);

# %% Compute the tortuosity factor and visualize the solution
c_grid = vec_to_grid(sol.u, img)
τ = tortuosity(c_grid, :x)
F = formation_factor(c_grid, :x)
@info "τ: $(@sprintf("%.5f", τ)), ℱ: $(@sprintf("%.5f", F))"
show_plots && display(heatmap(c_grid[:, :, 1]; aspect_ratio=:equal, clim=(0, 1)));
