# Tortuosity.jl

`Tortuosity.jl` is a GPU-accelerated solver to compute the tortuosity factor ($\tau$) of voxel images of porous media. You can think of `Tortuosity.jl` as the equivalent of [TauFactor](https://www.mathworks.com/matlabcentral/fileexchange/57956-taufactor) toolbox in MATLAB, or [taufactor](https://github.com/tldr-group/taufactor) in Python, but a bit faster.

## Installation

To install `Tortuosity.jl`, simply run the following command in the Julia REPL:

```julia
using Pkg
Pkg.add("Tortuosity")
```

## Basic usage

To compute the tortuosity factor of a voxel image, you can use the following workflow:

```julia
using Plots
using Tortuosity
using Tortuosity: tortuosity, vec_to_grid

USE_GPU = false

# Generate a test image
img = Imaginator.blobs(; shape=(64, 64, 1), porosity=0.65, blobiness=0.5, seed=2);
img = Imaginator.trim_nonpercolating_paths(img, axis=:x)

# Visualize the image
heatmap(img[:, :, 1]; aspect_ratio=:equal, clim=(0, 1))
savefig("img-readme.svg"); nothing # hide

# Define the simulation
sim = TortuositySimulation(img; axis=:x, gpu=USE_GPU);

# Solve the system of equations
sol = solve(sim.prob, KrylovJL_CG(); verbose=false, reltol=1e-5);

# Convert the solution vector to an Nd grid
c = vec_to_grid(sol.u, img)
# Compute the tortuosity factor
τ = tortuosity(c; axis=:x)
println("τ = $τ")

# Visualize the concentration field
heatmap(c[:, :, 1]; aspect_ratio=:equal, clim=(0, 1))
savefig("c-readme.svg"); nothing # hide
```

```@example
HTML("""<figure><img src="img-readme.svg"><figcaption>Original binary image</figcaption></figure>""") # hide
```

```@example
HTML("""<figure><img src="c-readme.svg"><figcaption>Concentration field</figcaption></figure>""") # hide
```
