# Tortuosity.jl

`Tortuosity.jl` is a GPU-accelerated Julia package for computing the tortuosity factor ($\tau$) of voxel images of porous media. The tortuosity factor quantifies how much a porous microstructure slows down diffusive transport relative to free diffusion — $\tau = 1$ means no hindrance, higher values mean slower transport.

The package supports both **steady-state** and **transient** diffusion, uniform or spatially varying diffusivity, and runs on CPU or CUDA GPU.

It is similar to [TauFactor](https://www.mathworks.com/matlabcentral/fileexchange/57956-taufactor) in MATLAB and [taufactor](https://github.com/tldr-group/taufactor) in Python.

## Installation

```julia
using Pkg
Pkg.add("Tortuosity")
```

**Requirements:** Julia 1.10+. For GPU acceleration, a CUDA-capable GPU is needed.

## Quick example

```@example
using Tortuosity

img = Imaginator.blobs(; shape=(64, 64, 1), porosity=0.65, blobiness=0.5, seed=2)
img = Imaginator.trim_nonpercolating_paths(img, axis=:x)
sim = TortuositySimulation(img; axis=:x, gpu=false)
sol = solve(sim.prob, KrylovJL_CG(); verbose=false, reltol=1e-5)
τ = tortuosity(vec_to_grid(sol.u, img), img; axis=:x)
println("τ = $τ")
```

## Learn more

Follow the tutorials in order for a guided introduction:

1. **[Steady-State Tortuosity](tutorials/steady_state.md)** — the core workflow, explained step by step
2. **[Variable Diffusivity](tutorials/variable_diffusivity.md)** — assign per-voxel diffusivity
3. **[Transient Diffusion](tutorials/transient.md)** — time-dependent concentration fields
4. **[Advanced Transient](tutorials/advanced_transient.md)** — stop conditions, voxel-wise fitting, periodic boundaries

Or jump to a reference page:

- **[Imaginator](imaginator.md)** — synthetic image generation and manipulation
- **[API Reference](api.md)** — function signatures and descriptions
