# Variable Diffusivity

In the [previous tutorial](steady_state.md), we assumed uniform diffusivity ($D = 1$) everywhere. Here we assign a spatially varying diffusivity field — useful for heterogeneous media like fractured rock or composite materials.

## How it works

Instead of a scalar `D`, you pass a 3D array matching the image shape. Each voxel gets its own diffusivity value. The solver uses harmonic-mean interpolation at voxel interfaces, and the effective diffusivity (and hence tortuosity) is computed from the resulting flux.

## Example: porous subdomain with random diffusivity

We generate a binary pore image and assign a random diffusivity to each pore voxel:

```@example vardiff
using ImageFiltering  # optional dependency, needed by Imaginator.blobs
using Plots
using Tortuosity

img = Imaginator.blobs(; shape=(64, 64, 1), porosity=0.65, blobiness=0.5, seed=2)

# Random diffusivity in [0, 1] for pore voxels, zero for solid
D = rand(Float64, size(img))
D[.!img] .= 0
domain = D .> 0  # conducting voxels

sim = SteadyDiffusionProblem(domain; axis=:x, D=D, gpu=false)
sol = solve(sim.prob, KrylovJL_CG(); verbose=false, reltol=1e-5)

c = reconstruct_field(sol.u, domain)
τ = tortuosity(c, domain; axis=:x, D=D)
println("τ = $τ")

heatmap(c[:, :, 1]; aspect_ratio=:equal, clim=(0, 1))
savefig("c-vardiff.svg"); nothing # hide
```

```@example vardiff
HTML("""<figure><img src=$(joinpath(Main.buildpath,"c-vardiff.svg"))><figcaption>Concentration field with spatially varying diffusivity</figcaption></figure>""") # hide
```

The key difference from the uniform case: we pass `D=D` to both `SteadyDiffusionProblem` and `tortuosity()`.

## Example: two-phase composite

Consider a composite electrode: an active phase with a slower-diffusing binder dispersed through it. Both phases conduct, so neither is solid, but the binder resists diffusion far more. Model this by treating the entire image as the domain and giving the binder voxels the lower diffusivity.

```@example vardiff2
using ImageFiltering  # optional dependency, needed by Imaginator.blobs
using Tortuosity

img = Imaginator.blobs(; shape=(64, 64, 1), porosity=0.65, blobiness=0.5, seed=2)

D = zeros(size(img))
D[img] .= 1.0    # Active phase (fast diffusion)
D[.!img] .= 0.2  # Binder (slower diffusion)
domain = D .> 0   # Everything is conducting

sim = SteadyDiffusionProblem(domain; axis=:x, D=D, gpu=false)
sol = solve(sim.prob, KrylovJL_CG(); verbose=false, reltol=1e-5)

c = reconstruct_field(sol.u, domain)
τ = tortuosity(c, domain; axis=:x, D=D)
println("τ = $τ")
```

!!! note
    Tortuosity is traditionally a geometric property: it measures how much the solid walls and internal microstructure force diffusing species to take longer, more winding paths compared to a straight line. When every voxel conducts and nothing blocks the path, there is no geometric hindrance, so the computed $\tau$ reflects diffusivity contrast between phases, not structural tortuosity in the traditional sense. The concentration field is the primary quantity of interest in this case.

Here, we pass `domain` (not `img`) to the constructor because all voxels are conducting — `domain` is derived from where `D > 0`.

## Next steps

- [Transient Diffusion](transient.md) — time-dependent concentration fields
