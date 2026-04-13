# Tortuosity.jl

`Tortuosity.jl` is a GPU-accelerated Julia package for computing the tortuosity factor ($\tau$) of voxel images of porous media. The tortuosity factor quantifies how much a porous microstructure slows down diffusive transport relative to free diffusion — $\tau = 1$ means no hindrance, higher values mean slower transport.

The package supports both **steady-state** and **transient** diffusion, uniform or spatially varying diffusivity, and runs on CPU or any of the supported GPU backends: NVIDIA (CUDA), Apple Silicon (Metal), and AMD (ROCm/AMDGPU).

It is similar to [TauFactor](https://www.mathworks.com/matlabcentral/fileexchange/57956-taufactor) in MATLAB and [taufactor](https://github.com/tldr-group/taufactor) in Python.

## Installation

```julia
using Pkg
Pkg.add("Tortuosity")
```

**Requirements:** Julia 1.10+. GPU acceleration is optional — load the corresponding package (`CUDA.jl`, `Metal.jl`, or `AMDGPU.jl`) to activate it. See [GPU backends](@ref) below.

## Quick example

```@example
using Tortuosity

img = Imaginator.blobs(; shape=(64, 64, 1), porosity=0.65, blobiness=0.5, seed=2)
img = Imaginator.trim_nonpercolating_paths(img, axis=:x)
sim = SteadyDiffusionProblem(img; axis=:x, gpu=false)
sol = solve(sim.prob, KrylovJL_CG(); verbose=false, reltol=1e-5)
τ = tortuosity(reconstruct_field(sol.u, img), img; axis=:x)
println("τ = $τ")
```

## GPU backends

Tortuosity ships CPU kernels unconditionally. GPU kernels live in package extensions — one for each supported backend, loaded lazily when the backend package is imported:

| Backend | Package | Hardware |
|---------|---------|----------|
| CUDA    | [`CUDA.jl`](https://github.com/JuliaGPU/CUDA.jl)   | NVIDIA GPUs |
| Metal   | [`Metal.jl`](https://github.com/JuliaGPU/Metal.jl) | Apple Silicon |
| AMDGPU  | [`AMDGPU.jl`](https://github.com/JuliaGPU/AMDGPU.jl) | AMD GPUs (ROCm) |

To activate a backend, load the corresponding package **before** constructing a simulation:

```julia
using CUDA      # or: using Metal  / using AMDGPU
using Tortuosity

sim = SteadyDiffusionProblem(img; axis=:x)     # auto-detects the loaded backend
```

The `gpu` keyword of [`SteadyDiffusionProblem`](@ref) and [`TransientDiffusionProblem`](@ref) controls whether solver kernels run on GPU:

- **`gpu=nothing`** (default) — auto-detect. Uses GPU when a backend package is loaded *and* the image has at least 100,000 pore voxels; otherwise runs on CPU. If you pass a large image but have not loaded a backend package, you'll see a one-time `@info` message pointing back to this section.
- **`gpu=true`** — force GPU. Errors immediately if no backend is loaded.
- **`gpu=false`** — force CPU, even when a backend is available.

!!! warning "Silent CPU fallback"
    The auto-detect mode will run on CPU without erroring if no backend package has been loaded — the intent is to never force `using CUDA` on users who don't need it. If you're expecting GPU performance, either pass `gpu=true` (which errors on a missing backend) or make sure one of `CUDA.jl`, `Metal.jl`, or `AMDGPU.jl` is imported before constructing the simulation.

## Learn more

Follow the tutorials in order for a guided introduction:

1. **[Steady-State Tortuosity](tutorials/steady_state.md)** — the core workflow, explained step by step
2. **[Variable Diffusivity](tutorials/variable_diffusivity.md)** — assign per-voxel diffusivity
3. **[Transient Diffusion](tutorials/transient.md)** — time-dependent concentration fields
4. **[Advanced Transient](tutorials/advanced_transient.md)** — stop conditions, voxel-wise fitting, periodic boundaries

Or jump to a reference page:

- **[Imaginator](imaginator.md)** — synthetic image generation and manipulation
- **[API Reference](api.md)** — function signatures and descriptions
