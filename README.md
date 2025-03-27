# Tortuosity.jl

[![Build Status](https://github.com/ma-sadeghi/Tortuosity.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ma-sadeghi/Tortuosity.jl/actions/workflows/CI.yml?query=branch%3Amain)

> [!WARNING]  
> This package is still under development and not yet registered. It is not recommended for production use. We hope to release the first version in March 2025.

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
using Tortuosity: TortuositySimulation, formation_factor, tortuosity, vec_to_grid

# Generate a test image
gpu = true
img = Imaginator.blobs(; shape=(64, 64, 1), porosity=0.65, blobiness=0.5, seed=2);
img = Imaginator.trim_nonpercolating_paths(img, axis=:x)

# Define the simulation
sim = TortuositySimulation(img; axis=:x, gpu=gpu);

# Solve the system of equations
sol = solve(sim.prob, KrylovJL_CG(); verbose=false, reltol=1e-5);

# Convert the solution vector to an Nd grid
c_grid = vec_to_grid(sol.u, img)  
# Compute the tortuosity factor
τ = tortuosity(c_grid, axis=:x)
println("τ = $τ")
```
