# API Reference

This page is generated from the in-source docstrings. Every function or type
below is defined in `Tortuosity.jl` and is available after `using Tortuosity`
unless explicitly noted.

For how to enable a GPU backend, see [GPU backends](@ref).

## Types

```@docs
SteadyDiffusionProblem
TransientDiffusionProblem
Tortuosity.TransientSolution
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

On large images the number of conjugate-gradient iterations grows with the edge
length, and a preconditioner is what stops that growth. Build one with
[`two_level_preconditioner`](@ref) and hand it to the solver as `Pl`.

```@docs
two_level_preconditioner
Tortuosity.TwoLevelPreconditioner
```

Rather than make those choices yourself, you can hand the whole problem to the
package and let it decide the preconditioner and the tolerance:

```julia
sol = solve(sim)                    # two-level coarse space above 100k pore voxels
sol = solve(sim; precond=:none)     # or opt out
```

`solve(sim.prob, alg; ...)` is unaffected by this and remains the unopinionated
form that takes LinearSolve's defaults.

### Matrix-free operator

`SteadyDiffusionProblem(img; axis, matrixfree=true)` builds the operator as a
7-point stencil applied straight from the pore mask instead of an assembled
sparse matrix. It stores one `Int32` index array over the grid — about 14 bytes
per grid voxel against the assembled path's 40 — which is what makes images past
roughly 850³ solvable on a 24 GiB card, and its apply is about twice as fast on
GPU and six times as fast threaded on CPU.

The two paths are peers. They produce the same pore numbering, the same
right-hand side, the same number of Krylov iterations and the same `τ`, so
everything downstream is unchanged and the keyword is the only difference. Keep
the assembled path when you want the matrix itself — it is the CUSPARSE-backed
one, and the only one whose entries can be read.

```@docs
Tortuosity.MaskedLaplacian
Tortuosity.build_steady_operator
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

The transient solver follows the SciML convention: build a
[`TransientDiffusionProblem`](@ref) and pass it to `solve(prob, alg; ...)`. The
returned [`TransientSolution`](@ref Tortuosity.TransientSolution) holds
CPU-resident snapshots at the requested `saveat` intervals. The `solve` method
takes the same kwargs OrdinaryDiffEq does (`reltol`, `abstol`, `tspan`,
`callback`, …) plus a required `saveat`. See the
[Transient Diffusion](tutorials/transient.md) tutorial for a worked example.

### Stop conditions

Stop-condition callbacks terminate the solve when a diffusion-specific
convergence criterion is met. They compose with `CallbackSet` and any other
SciML-compatible callback.

```@docs
StopAtSteadyState
StopAtFluxBalance
StopAtSaturation
StopAtPeriodicState
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
