# Tortuosity.jl

`Tortuosity.jl` is a GPU-accelerated Julia package for computing the tortuosity factor ($\tau$) of voxel images of porous media. The tortuosity factor quantifies how much a porous microstructure slows down diffusive transport relative to free diffusion. $\tau = 1$ means no hindrance, and higher values mean slower transport.

The package supports both **steady-state** and **transient** diffusion, with uniform or spatially varying diffusivity. It runs on CPU, or on any of the supported GPU backends: NVIDIA (CUDA), Apple Silicon (Metal), and AMD (ROCm/AMDGPU).

It is similar to [TauFactor](https://www.mathworks.com/matlabcentral/fileexchange/57956-taufactor) in MATLAB and [taufactor](https://github.com/tldr-group/taufactor) in Python.

## Installation

```julia
using Pkg
Pkg.add("Tortuosity")
```

**Requirements:** Julia 1.10+. GPU acceleration is optional — install and load the corresponding package (`CUDA.jl`, `Metal.jl`, or `AMDGPU.jl`) to activate it. See [GPU backends](@ref) below.

## Optional dependencies

Parts of the API live in package extensions, so `using Tortuosity` stays light for people who do not need them. `Pkg.add("Tortuosity")` does **not** install them, because Julia resolves only a package's hard dependencies. Add the ones you need yourself, then load them before you call the entry points they back.

| Add and load this | To use |
|-------------------|--------|
| `ImageFiltering` | `Imaginator.blobs`, `Imaginator.apply_gaussian_blur` |
| `LsqFit`         | `fit_effective_diffusivity`, `fit_voxel_diffusivity` |
| `HDF5`           | `Tortuosity.export_to_hdf5` |

```julia
using Pkg
Pkg.add("ImageFiltering")   # and "LsqFit" and/or "HDF5", as needed
using ImageFiltering
```

Calling one of these without its package raises an error naming the package to load. Everything else — problem construction, solving, the observables, `Imaginator.trim_nonpercolating_paths` — works with `using Tortuosity` alone.

## Quick example

```@example
using ImageFiltering  # optional dependency, needed by Imaginator.blobs
using Tortuosity

img = Imaginator.blobs(; shape=(64, 64, 1), porosity=0.65, blobiness=0.5, seed=2)
img = Imaginator.trim_nonpercolating_paths(img, axis=:x)
sim = SteadyDiffusionProblem(img; axis=:x, gpu=false)
sol = solve(sim.prob, KrylovJL_CG(); verbose=false, reltol=1e-5)
τ = tortuosity(sol.u, sim)
println("τ = $τ")
```

## GPU backends

Tortuosity ships CPU kernels unconditionally. GPU kernels live in package extensions, one for each supported backend, and load lazily when you import the backend package:

| Backend | Package | Hardware |
|---------|---------|----------|
| CUDA    | [`CUDA.jl`](https://github.com/JuliaGPU/CUDA.jl)   | NVIDIA GPUs |
| Metal   | [`Metal.jl`](https://github.com/JuliaGPU/Metal.jl) | Apple Silicon |
| AMDGPU  | [`AMDGPU.jl`](https://github.com/JuliaGPU/AMDGPU.jl) | AMD GPUs (ROCm) |

Backend packages are optional dependencies too, so install the one matching your hardware, then load it **before** constructing a simulation:

```julia
using Pkg; Pkg.add("CUDA")   # or "Metal" / "AMDGPU"

using CUDA      # or: using Metal  / using AMDGPU
using Tortuosity

sim = SteadyDiffusionProblem(img; axis=:x)     # auto-detects the loaded backend
```

The `gpu` keyword of [`SteadyDiffusionProblem`](@ref) and [`TransientDiffusionProblem`](@ref) controls whether solver kernels run on GPU:

- **`gpu=nothing`** (default) — auto-detect. Uses GPU when a backend package is loaded *and* the image has at least 100,000 pore voxels, and otherwise runs on CPU. If you pass a large image but have not loaded a backend package, you will see a one-time `@info` message that points back to this section.
- **`gpu=true`** — force GPU. Errors immediately if no backend is loaded.
- **`gpu=false`** — force CPU, even when a backend is available.

!!! warning "Silent CPU fallback"
    If no backend package has been loaded, auto-detect runs on CPU without an error, because the intent is to never force `using CUDA` on users who do not need it. If you expect GPU performance, either pass `gpu=true`, which errors on a missing backend, or import one of `CUDA.jl`, `Metal.jl`, or `AMDGPU.jl` before you construct the simulation.

## Learn more

Follow the tutorials in order for a guided introduction:

1. **[Steady-State Tortuosity](tutorials/steady_state.md)** — the core workflow, explained step by step
2. **[Variable Diffusivity](tutorials/variable_diffusivity.md)** — assign per-voxel diffusivity
3. **[Transient Diffusion](tutorials/transient.md)** — time-dependent concentration fields
4. **[Advanced Transient](tutorials/advanced_transient.md)** — stop conditions, voxel-wise fitting, periodic boundaries

Or jump to a reference page:

- **[Imaginator](imaginator.md)** — synthetic image generation and manipulation
- **[API Reference](api.md)** — function signatures and descriptions
