# API Reference

This page is generated from the in-source docstrings. Every function or type
below is defined in `Tortuosity.jl` and is available after `using Tortuosity`
unless explicitly noted.

For how to enable a GPU backend, see [GPU backends](@ref).

## Types

```@docs
SteadyDiffusionProblem
TransientDiffusionProblem
TransientState
```

## Steady-state solvers and analysis

The linear system assembled by [`SteadyDiffusionProblem`](@ref) is a standard
[LinearSolve.jl](https://docs.sciml.ai/LinearSolve/) `LinearProblem`, so the
`solve` function and any compatible algorithm from that package apply. For
diffusion Laplacians the Krylov conjugate-gradient method
(`KrylovJL_CG()` — re-exported from LinearSolve.jl) is the recommended default:

```julia
sol = solve(sim.prob, KrylovJL_CG(); reltol=1e-6)
```

Once a steady-state concentration field has been solved for, the following
helpers derive the usual transport descriptors.

```@docs
tortuosity
effective_diffusivity
formation_factor
reconstruct_field
```

## Transient solver

```@docs
init_state
solve!
```

### Stop conditions

`solve!` accepts any `f(t_hist, C_hist) -> Bool` as a stop condition. The
package ships four built-in constructors:

```@docs
stop_at_time
stop_at_flux_balance
stop_at_avg_concentration
stop_at_periodic
```

## Measurements on transient fields

```@docs
flux
slice_concentration
mass_uptake
```

## Fitting

```@docs
fit_effective_diffusivity
fit_voxel_diffusivity
```

## Analytical reference solutions

Closed-form solutions to 1-D slab diffusion with constant diffusivity (Crank,
*The Mathematics of Diffusion*, 2nd ed.). Used both by [`fit_effective_diffusivity`](@ref)
for parameter fitting and as a ground truth for verifying numerical results.

```@docs
slab_concentration
slab_mass_uptake
slab_flux
slab_cumulative_flux
```

## Imaginator submodule

`Imaginator` is a small submodule for generating and preprocessing synthetic
3D voxel images of porous media. See [Imaginator](imaginator.md) for an
illustrated walk-through; the reference-level descriptions live here.

### Image generation

```@docs
Tortuosity.Imaginator.blobs
```

### Image analysis

```@docs
Tortuosity.Imaginator.phase_fraction
Tortuosity.Imaginator.trim_nonpercolating_paths
Tortuosity.Imaginator.faces
```

### Image processing utilities

```@docs
Tortuosity.Imaginator.denoise
Tortuosity.Imaginator.disk
Tortuosity.Imaginator.ball
Tortuosity.Imaginator.apply_gaussian_blur
Tortuosity.Imaginator.to_binary
Tortuosity.Imaginator.norm_to_uniform
```
