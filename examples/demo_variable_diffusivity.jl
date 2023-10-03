# %% Imports
using CUDA
using LinearSolve
using Plots
using Printf
using Tortuosity

# %% Generate/load the image
@info "Generating/loading the voxel image"
img = Imaginator.blobs(shape=(64, 64, 1), porosity=0.65, blobiness=0.5, seed=2)
display(heatmap(img[:, :, 1], aspect_ratio=:equal, clim=(0, 1)))
ε = sum(img) / length(img)
# Build diffusivity matrix
D = fill(NaN, size(img))
D[img] .= 1.0       # Fluid phase
D[.!img] .= 1e-4    # Solid phase

# %% Build Ax = b on CPU/GPU
@info "Setting up τ simulation (assembling Ax = b)"
domain = ones(Bool, size(img))
sim = TortuositySimulation(domain, axis=:x, D=D);

# %% Solve Ax = b using an iterative solver
@info "Solving the system of equations"
sol = solve(sim.prob, KrylovJL_CG())
# sol = solve(sim.prob, UMFPACKFactorization())
@info "Average concentration: $(mean(sol.u))"

# %% Compute the tortuosity factor and visualize the solution
c = vec_to_field(sol.u, domain)
c_masked = vec_to_field(sol.u[img[:]], img)
display(heatmap(c[:, :, 1], aspect_ratio=:equal, clim=(0, 1)))
display(heatmap(c_masked[:, :, 1], aspect_ratio=:equal, clim=(0, 1)))
tau = tortuosity(c, :x, eps=ε, D=D)
@info "τ (variable diffusivity): $(@sprintf("%.5f", tau))"

# %% Compare with the ground truth (solid is non-conducting)
sim_gt = TortuositySimulation(img, axis=:x)
sol_gt = solve(sim_gt.prob, KrylovJL_CG())
c_gt = vec_to_field(sol_gt.u, img)
tau_gt = tortuosity(c_gt, :x)
display(heatmap(c_gt[:, :, 1], aspect_ratio=:equal, clim=(0, 1)))
@info "τ (ground truth): $(@sprintf("%.5f", tau_gt))"
