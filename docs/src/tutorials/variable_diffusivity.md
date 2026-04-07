# Variable Diffusivity

In the [previous tutorial](steady_state.md), we assumed uniform diffusivity ($D = 1$) everywhere. Here we assign a spatially varying diffusivity field — useful for heterogeneous media like fractured rock or composite materials.

## How it works

Instead of a scalar `D`, you pass a 3D array matching the image shape. Each voxel gets its own diffusivity value. The solver uses harmonic-mean interpolation at voxel interfaces, and the effective diffusivity (and hence tortuosity) is computed from the resulting flux.

## Example: porous subdomain with random diffusivity

We generate a binary pore image and assign a random diffusivity to each pore voxel:

```@example vardiff
using Plots
using Tortuosity
using Tortuosity: tortuosity, vec_to_grid

img = Imaginator.blobs(; shape=(64, 64, 1), porosity=0.65, blobiness=0.5, seed=2)

# Random diffusivity in [0, 1] for pore voxels, zero for solid
D = rand(Float64, size(img))
D[.!img] .= 0
domain = D .> 0  # conducting voxels

sim = TortuositySimulation(domain; axis=:x, D=D, gpu=false)
sol = solve(sim.prob, KrylovJL_CG(); verbose=false, reltol=1e-5)

c = vec_to_grid(sol.u, domain)
τ = tortuosity(c, domain; axis=:x, D=D)
println("τ = $τ")

heatmap(c[:, :, 1]; aspect_ratio=:equal, clim=(0, 1))
savefig("c-vardiff.svg"); nothing # hide
```

```@example vardiff
HTML("""<figure><img src=$(joinpath(Main.buildpath,"c-vardiff.svg"))><figcaption>Concentration field with spatially varying diffusivity</figcaption></figure>""") # hide
```

The key difference from the uniform case: we pass `D=D` to both `TortuositySimulation` and `tortuosity()`.

## Using the entire image as domain

You can also treat the entire image as the domain, assigning different diffusivities to "pore" and "solid" phases. This is useful for non-porous heterogeneous media:

```@example vardiff2
using Tortuosity
using Tortuosity: tortuosity, vec_to_grid

img = Imaginator.blobs(; shape=(64, 64, 1), porosity=0.65, blobiness=0.5, seed=2)

D = zeros(size(img))
D[img] .= 1.0    # More conductive phase
D[.!img] .= 0.2  # Less conductive phase
domain = D .> 0   # Everything is conducting

sim = TortuositySimulation(domain; axis=:x, D=D, gpu=false)
sol = solve(sim.prob, KrylovJL_CG(); verbose=false, reltol=1e-5)

c = vec_to_grid(sol.u, domain)
τ = tortuosity(c, domain; axis=:x, D=D)
println("τ = $τ")
```

!!! note
    Tortuosity is ill-defined when the entire image is the domain, since there is no distinct void phase. The concentration field is the primary quantity of interest in this case.

Here, we pass `domain` (not `img`) to the constructor because all voxels are conducting — `domain` is derived from where `D > 0`.

## Next steps

- [Transient Diffusion](transient.md) — time-dependent concentration fields
