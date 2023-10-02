# %% Imports
using Tortuosity
using CUDA
using LinearSolve
using Plots
using Printf

# %% Generate/load the image
@info "Generating/loading the voxel image"
img = imgen.blobs(shape=(64, 64, 1), porosity=0.65, blobiness=0.5, seed=2);
display(heatmap(img[:, :, 1], aspect_ratio=:equal, clim=(0, 1)))

# %% Build Ax = b on CPU/GPU
@info "Setting up τ simulation (assembling Ax = b)"
tausim = TortuositySimulation(img, axis=:x, gpu=true);
reltol = eltype(tausim.prob.b)(1e-5)

# %% Solve Ax = b using an iterative solver
@info "Solving the system using KrylovJL_CG"
@time sol = solve(tausim.prob, KrylovJL_CG(), verbose=true, reltol=reltol);
@info "Average concentration: $(mean(sol.u))"

# %% Compute the tortuosity factor and visualize the solution
c = vec_to_field(sol.u, img)
tau = compute_tortuosity_factor(c, :x)
ff = compute_formation_factor(c, :x)
eps = sum(img) / length(img)
@info "τ: $(@sprintf("%.5f", tau)), ℱℱ: $(@sprintf("%.5f", ff)), ε = $(@sprintf("%.5f", eps))"
display(heatmap(c[:, :, 1], aspect_ratio=:equal, clim=(0, 1)))
