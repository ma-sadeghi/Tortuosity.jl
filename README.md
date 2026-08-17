# Tortuosity.jl

[![Build Status](https://github.com/ma-sadeghi/Tortuosity.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ma-sadeghi/Tortuosity.jl/actions/workflows/CI.yml?query=branch%3Amain)

> [!WARNING]  
> We've just released `Tortuosity.jl` on the official Julia package registry. However, it is still under active development, and the API might change.

`Tortuosity.jl` is a Julia package for calculating the tortuosity factor of volumetric images. It is designed to be fast and efficient, leveraging the power of Julia's multiple dispatch to support GPU acceleration right out of the box. You can consider it as a Julia version of the well-known [TauFactor](https://github.com/tldr-group/taufactor).

`Tortuosity.jl` is designed to be granular, allowing users to see what's happening under the hood, and potentially modify the steps to suit their needs, e.g., using a different matrix solver, etc.

## Installation

To install the package, use the Julia package manager. Open Julia and run:

```julia
using Pkg
Pkg.add("Tortuosity")
```

Some entry points live in package extensions, and Julia does not install an extension's package for you. Add the ones you need:

```julia
Pkg.add("ImageFiltering")  # Imaginator.blobs, Imaginator.apply_gaussian_blur
Pkg.add("LsqFit")          # fit_effective_diffusivity, fit_voxel_diffusivity
Pkg.add("HDF5")            # Tortuosity.export_to_hdf5
```

GPU backends work the same way: `Pkg.add("CUDA")` (or `"Metal"`, or `"AMDGPU"`), then load it before building a simulation.

## Usage

```julia
using ImageFiltering  # optional dependency, needed by Imaginator.blobs
using Tortuosity
using Tortuosity: tortuosity, reconstruct_field

USE_GPU = false

# Generate a test image
img = Imaginator.blobs(; shape=(64, 64, 1), porosity=0.65, blobiness=0.5, seed=2);
img = Imaginator.trim_nonpercolating_paths(img, axis=:x)

# Define the simulation
sim = SteadyDiffusionProblem(img; axis=:x, gpu=USE_GPU);

# Solve the system of equations
sol = solve(sim.prob, KrylovJL_CG(); verbose=false, reltol=1e-5);

# Convert the solution vector to an Nd grid
c = reconstruct_field(sol.u, img)
# Compute the tortuosity factor
τ = tortuosity(c, img; axis=:x)
println("τ = $τ")
```

### Two operator paths

The steady operator comes in two forms, both fully supported. They produce the same pore numbering, the same right-hand side and the same `τ`, so everything downstream is identical and you can switch with one keyword.

| | assembled (default) | matrix-free |
|---|---|---|
| how to ask for it | `SteadyDiffusionProblem(img; axis=:x)` | `SteadyDiffusionProblem(img; axis=:x, matrixfree=true)` |
| what it stores | the sparse matrix: column pointers, row indices, values | one `Int32` index array over the grid |
| device memory | ~40 bytes per grid voxel | ~14 bytes per grid voxel |
| largest cube on a 24 GiB card | ~850³ | 1000³ comfortably, 1100³ at the limit |
| GPU apply at 800³ | 29.1 ms (CUSPARSE CSR) | 15.7 ms |
| GPU solve at 800³, preconditioned | 21.2 s, peak 22.4 GiB | 17.2 s, peak 9.9 GiB |
| CPU apply at 200³, 20 threads | 36.7 ms (`SparseArrays`) | 5.4 ms |

Both paths take the same number of Krylov iterations at every size and agree on τ to 4–5 significant figures.

Take the assembled path when you want the CUSPARSE-backed matrix itself, or anything that reads matrix entries. Take the matrix-free path when the image is large, when memory is the binding constraint, or when you simply want the solve to be faster.

There is also a package-owned `solve` that picks the preconditioner and the tolerance for you:

```julia
sim = SteadyDiffusionProblem(img; axis=:x, matrixfree=true)
sol = solve(sim)          # two-level preconditioner above 100k pore voxels; reltol from the element type
```

`solve(sim.prob, alg; ...)` stays exactly as it was — the unopinionated form that takes LinearSolve's defaults.

## Documentation

Full documentation, including tutorials and the API reference, is at [ma-sadeghi.github.io/Tortuosity.jl](https://ma-sadeghi.github.io/Tortuosity.jl/stable/).

## Contributing and support

Bug reports, feature requests, and questions are all welcome on the [issue tracker](https://github.com/ma-sadeghi/Tortuosity.jl/issues). See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and the pull request workflow.

## License

MIT — see [LICENSE](LICENSE).
