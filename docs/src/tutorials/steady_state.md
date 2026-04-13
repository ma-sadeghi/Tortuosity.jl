# Steady-State Tortuosity

This tutorial walks through the core `Tortuosity.jl` workflow: loading a porous image, solving the steady-state diffusion equation, and extracting transport properties.

## Background

When you solve the steady-state diffusion equation ($\nabla^2 c = 0$) on a porous geometry with fixed concentrations at the inlet and outlet, the resulting concentration field encodes how the pore structure resists transport. The **tortuosity factor** $\tau$ measures this resistance:

$$\tau = \frac{\varepsilon \, D_0}{D_\text{eff}}$$

where $\varepsilon$ is the porosity, $D_0$ is the free-phase diffusivity, and $D_\text{eff}$ is the effective diffusivity computed from the flux through the medium.

## Step 1: Generate or load an image

The input is a 3D `BitArray` where `true` = pore and `false` = solid. You can load your own image or generate a synthetic one with the `Imaginator` submodule (see [Imaginator](../imaginator.md) for details).

```@example steady
using Plots
using Tortuosity

# Generate a 2D test image (3rd dimension is a singleton for visualization)
img = Imaginator.blobs(; shape=(64, 64, 1), porosity=0.65, blobiness=0.5, seed=2)
```

Before solving, we must remove pore clusters that don't connect the inlet face to the outlet face. Isolated clusters create a singular linear system.

```@example steady
img = Imaginator.trim_nonpercolating_paths(img, axis=:x)
φ = Imaginator.phase_fraction(img, true)
println("Porosity after trimming: $φ")

heatmap(img[:, :, 1]; aspect_ratio=:equal, clim=(0, 1), title="Pore space (white = pore)")
savefig("img-ss.svg"); nothing # hide
```

```@example steady
HTML("""<figure><img src=$(joinpath(Main.buildpath,"img-ss.svg"))><figcaption>Binary pore image after removing non-percolating paths</figcaption></figure>""") # hide
```

## Step 2: Set up the simulation

`SteadyDiffusionProblem` discretizes the Laplacian on the pore space and builds a linear system $Ax = b$ with Dirichlet boundary conditions ($c = 1$ at the inlet, $c = 0$ at the outlet).

```@example steady
sim = SteadyDiffusionProblem(img; axis=:x, gpu=false)
```

Key parameters:
- **`axis`** — direction of the concentration gradient (`:x`, `:y`, or `:z`). The inlet is the first face, the outlet is the last face along this axis.
- **`gpu`** — `true` to force GPU, `false` for CPU, `nothing` (default) to auto-detect. Auto-detect uses GPU when one of `CUDA.jl`, `Metal.jl`, or `AMDGPU.jl` has been loaded and the image has at least 100,000 pore voxels. See [GPU backends](@ref) for how to activate each backend.
- **`D`** — diffusivity (scalar or array). Defaults to `1.0`. See [Variable Diffusivity](variable_diffusivity.md) for non-uniform cases.

## Step 3: Solve

We solve the linear system using conjugate gradient (CG). Any solver from [LinearSolve.jl](https://docs.sciml.ai/LinearSolve/) can be used.

```@example steady
sol = solve(sim.prob, KrylovJL_CG(); verbose=false, reltol=1e-5)
nothing # hide
```

`reltol` controls the convergence tolerance. Tighter tolerances give more accurate results at the cost of more iterations.

## Step 4: Extract results

The solution `sol.u` is a 1D vector containing concentrations at pore voxels only (solid voxels are excluded to save memory). Use `reconstruct_field` to reconstruct the full 3D concentration field, with `NaN` for solid voxels.

```@example steady
c = reconstruct_field(sol.u, img)

heatmap(c[:, :, 1]; aspect_ratio=:equal, clim=(0, 1), title="Concentration field")
savefig("c-ss.svg"); nothing # hide
```

```@example steady
HTML("""<figure><img src=$(joinpath(Main.buildpath,"c-ss.svg"))><figcaption>Steady-state concentration field (inlet at left, outlet at right)</figcaption></figure>""") # hide
```

Now compute the transport properties:

```@example steady
Deff = effective_diffusivity(c, img; axis=:x)
τ = tortuosity(c, img; axis=:x)
F = formation_factor(c, img; axis=:x)
println("Effective diffusivity: $Deff")
println("Tortuosity factor:    $τ")
println("Formation factor:     $F")
```

These quantities are related: $\tau = \varepsilon / D_\text{eff}$ and $F = 1 / D_\text{eff}$.

!!! tip "3D images"
    This tutorial uses a 2D slice for easy visualization, but the workflow is identical for 3D images — just change `shape` to, e.g., `(64, 64, 64)`.

## Next steps

- [Variable Diffusivity](variable_diffusivity.md) — what if diffusivity isn't uniform?
- [Transient Diffusion](transient.md) — time-dependent concentration fields
