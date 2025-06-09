# %% ------------------------------------------------------
# Imports and config

using Plots
using Printf
using Tortuosity
using Tortuosity: Imaginator, TortuositySimulation, tortuosity, vec_to_grid

PLOT = false
USE_GPU = true

# %% ------------------------------------------------------
# Generate/load the image

shape = (64, 64, 64)
img = Imaginator.blobs(; shape=shape, porosity=0.5, blobiness=1, seed=2);
img = Imaginator.trim_nonpercolating_paths(img; axis=:x);
PLOT && display(heatmap(img[:, :, shape[3] ÷ 2]; aspect_ratio=:equal, clim=(0, 1)));

# %% ------------------------------------------------------
# Build Ax = b on CPU/GPU and solve the system

sim = TortuositySimulation(img; axis=:x, gpu=USE_GPU);
sol = solve(sim.prob, KrylovJL_CG(); verbose=false, reltol=1e-5);

# %% ------------------------------------------------------
# Compute the tortuosity factor and visualize the solution

c = vec_to_grid(sol.u, img);
τ = tortuosity(c; axis=:x);
@info "τ: $(@sprintf("%.5f", τ))"
PLOT && display(heatmap(c[:, :, 1]; aspect_ratio=:equal, clim=(0, 1)));
