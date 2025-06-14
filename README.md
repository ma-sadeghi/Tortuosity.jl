# Tortuosity.jl

[![Build Status](https://github.com/ma-sadeghi/Tortuosity.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ma-sadeghi/Tortuosity.jl/actions/workflows/CI.yml?query=branch%3Amain)

> [!WARNING]  
> We've just released `Tortuosity.jl` on the official Julia package registry. However, it is still under active development, and the API might change.

`Tortuosity.jl` is a Julia package for calculating the tortuosity factor of volumetric images. It is designed to be fast and efficient, leveraging the power of Julia's multiple dispatch to support GPU acceleration right out of the box. You can consider it as a Julia version of the well-known [TauFactor](https://github.com/tldr-group/taufactor), but more efficient and robust (read the [comparison](https://ma-sadeghi.github.com/Tortuosity.jl/taufactor) for more details).

`Tortuosity.jl` is designed to be granular, allowing users to see what's happening under the hood, and potentially modify the steps to suit their needs, e.g., using a different matrix solver, etc.

## Installation

To install the package, use the Julia package manager. Open Julia and run:

```julia
using Pkg
Pkg.add("Tortuosity")
```

## Usage

```julia
using Tortuosity
using Tortuosity: tortuosity, vec_to_grid

USE_GPU = false

# Generate a test image
img = Imaginator.blobs(; shape=(64, 64, 1), porosity=0.65, blobiness=0.5, seed=2);
img = Imaginator.trim_nonpercolating_paths(img, axis=:x)

# Define the simulation
sim = TortuositySimulation(img; axis=:x, gpu=USE_GPU);

# Solve the system of equations
sol = solve(sim.prob, KrylovJL_CG(); verbose=false, reltol=1e-5);

# Convert the solution vector to an Nd grid
c = vec_to_grid(sol.u, img)
# Compute the tortuosity factor
τ = tortuosity(c; axis=:x)
println("τ = $τ")
```
