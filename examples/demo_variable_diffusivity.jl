# %% ------------------------------------------------------
# Imports

using Plots
using Printf
using Tortuosity
using Tortuosity: tortuosity, vec_to_grid
using HDF5

PLOT = false
USE_GPU = true

# %% ------------------------------------------------------
# Generate/load the image

@info "Generating/loading the voxel image"
img = Imaginator.blobs(; shape=(64, 64, 1), porosity=0.65, blobiness=0.5, seed=2)

PLOT && display(heatmap(img[:, :, 1]; aspect_ratio=:equal, clim=(0, 1)))

# Example 1: Use entire image as domain
# D = zeros(size(img))
# D[img] .= 1.0       # More conductive phase
# D[.!img] .= 1e-1    # Less conductive phase
# domain = D .> 0     # Define domain as only-conducting voxels (i.e., the entire image)

# Example 2: Use a subdomain (e.g., only the true voxels)
D = zeros(size(img))
D[img] .= rand(count(img))  # Random conductivity for true voxels
D[.!img] .= 0               # Non-conducting
domain = D .> 0             # Define domain as only-conducting voxels

# %% ------------------------------------------------------
# Build Ax = b on CPU/GPU and solve the system of equations

@info "Setting up τ simulation (assembling Ax = b)"
sim = TortuositySimulation(domain; axis=:x, D=D, gpu=USE_GPU);
@info "Solving the system of equations"
sol = solve(sim.prob, KrylovJL_CG())
@info "Average concentration: $(mean(sol.u))"

# %% ------------------------------------------------------
# Compute the tortuosity factor and visualize the solution

c = vec_to_grid(sol.u, domain)
PLOT && display(heatmap(c[:, :, 1]; aspect_ratio=:equal, clim=(0, 1)))
tau = tortuosity(c; axis=:x, D=D)
@info "τ (variable diffusivity): $(@sprintf("%.5f", tau))"

# %% ------------------------------------------------------
# Compare with the ground truth (solid is non-conducting)

sim_gt = TortuositySimulation(img; axis=:x)
sol_gt = solve(sim_gt.prob, KrylovJL_CG())
c_gt = vec_to_grid(sol_gt.u, img)
tau_gt = tortuosity(c_gt; axis=:x)
PLOT && display(heatmap(c_gt[:, :, 1]; aspect_ratio=:equal, clim=(0, 1)))
@info "τ (ground truth): $(@sprintf("%.5f", tau_gt))"
