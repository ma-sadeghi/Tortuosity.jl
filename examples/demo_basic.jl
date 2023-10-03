# %% Imports
using Tortuosity
using CUDA
using LinearSolve
using Plots
using Printf

# %% Generate/load the image
@info "Generating/loading the voxel image"
img = Imaginator.blobs(shape=(64, 64, 1), porosity=0.65, blobiness=0.5, seed=2);
display(heatmap(img[:, :, 1], aspect_ratio=:equal, clim=(0, 1)))

# %% Build Ax = b on CPU/GPU
@info "Setting up τ simulation (assembling Ax = b)"
sim = TortuositySimulation(img, axis=:x, gpu=true);

# %% Solve Ax = b using an iterative solver
@info "Solving the system using KrylovJL_CG"
@time sol = solve(sim.prob, KrylovJL_CG(), verbose=true, reltol=eltype(sim.prob.b)(1e-5));
@info "Average concentration: $(mean(sol.u))"

# %% Compute the tortuosity factor and visualize the solution
c = vec_to_field(sol.u, img)
tau = tortuosity(c, :x)
ff = formation_factor(c, :x)
ε = phase_fraction(img, 1)
@info "τ: $(@sprintf("%.5f", tau)), ℱ: $(@sprintf("%.5f", ff)), ε = $(@sprintf("%.5f", ε))"
display(heatmap(c[:, :, 1], aspect_ratio=:equal, clim=(0, 1)))
