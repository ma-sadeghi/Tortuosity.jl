# %% Imports
using Plots
using Printf
using Tortuosity
using Tortuosity: Imaginator, TortuositySimulation, tortuosity, vec_to_grid

# %% Generate/load the image
show_plots = false
gpu = false
img = Imaginator.blobs(; shape=(64, 64, 1), porosity=0.5, blobiness=1, seed=2);
img = Imaginator.trim_nonpercolating_paths(img, :x);
show_plots && display(heatmap(img[:, :, 1]; aspect_ratio=:equal, clim=(0, 1)));

# %% Build Ax = b on CPU/GPU
sim = TortuositySimulation(img; axis=:x, gpu=gpu);

# %% Solve Ax = b using an iterative solver
@time sol = solve(sim.prob, KrylovJL_CG(); verbose=false, reltol=1e-5);

# %% Compute the tortuosity factor and visualize the solution
c_grid = vec_to_grid(sol.u, img)
τ = tortuosity(c_grid, :x)
F = formation_factor(c_grid, :x)
@info "τ: $(@sprintf("%.5f", τ)))"
show_plots && display(heatmap(c_grid[:, :, 1]; aspect_ratio=:equal, clim=(0, 1)));
