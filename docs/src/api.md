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

The linear system assembled by [`SteadyDiffusionProblem`](@ref) is a standard [LinearSolve.jl](https://docs.sciml.ai/LinearSolve/) `LinearProblem`, so `solve(sim.prob, alg)` accepts any compatible algorithm. `KrylovJL_CG()` remains the direct, unopinionated conjugate-gradient route:

```julia
sol = solve(sim.prob, KrylovJL_CG(); reltol=1e-6)
```

On large images the number of conjugate-gradient iterations grows with the edge
length, and a preconditioner is what stops that growth. Build one with
[`two_level_preconditioner`](@ref) and hand it to the solver as `Pl`.

```@docs
two_level_preconditioner
Tortuosity.TwoLevelPreconditioner
Tortuosity.CoarseLevel
Tortuosity.Aggregation
```

Rather than make those choices yourself, you can hand the whole problem to the package and let it decide the algorithm, preconditioner, and tolerance:

```julia
sol = solve(sim)                    # automatic two-level coarse space and tolerance
sol = solve(sim; precond=:none)     # or opt out
```

On Float64 host systems with at least 10,000 unknowns, `solve(sim)` selects [`Tortuosity.HostCG`](@ref), a deterministic threaded conjugate-gradient implementation that fuses bandwidth-bound vector operations and uses a row-parallel gather for image-built assembled matrices. Smaller hosts and GPU systems retain `KrylovJL_CG`. Passing an algorithm explicitly always honors it, and `solve(sim.prob, alg; ...)` remains the unopinionated LinearSolve form.

```@docs
Tortuosity.HostCG
Tortuosity.HostCGWorkspace
```

The automatic coarse space starts at 3,000 pore nodes on CUDA, 8,000 on CPU, and 100,000 on Metal or AMDGPU. Domains with at most 32 voxels along the transport axis retain the 100,000-node threshold because their unpreconditioned solve is already short. These thresholds reflect where setup plus refinement becomes cheaper than the unpreconditioned iteration count on each measured path.

### Matrix-free operator

`SteadyDiffusionProblem(img; axis, matrixfree=true)` builds the operator as a
7-point stencil applied straight from the pore mask instead of an assembled
sparse matrix. It stores one `Int32` index array over the grid — four bytes per
grid voxel, against roughly 59 bytes per *pore* voxel for the assembled
matrix — and its apply is about twice as fast on GPU and two to three times as fast
threaded on CPU. Measured end to end, peak device memory is 32.0 bytes per pore
node plus 4.00 bytes per grid voxel, with at most 8 bytes per open inlet-face
voxel for direct flux readout. The O(N²) inlet term is negligible beside the
O(N³) solve storage. The matrix-free path is 1.7× to 3.2× leaner than the
assembled path depending on porosity. That margin is what lets a 1000³ image at
any porosity fit on a 48 GiB card where the assembled path runs out above ε ≈ 0.4.

That is a memory limit rather than a refusal. Both paths run at any size their
storage fits: the assembled one widens its indices to 64-bit once an image
carries more than 306,783,378 pore voxels, and past that point it simply needs a
card or a host with the room. Nothing switches between the two on your behalf.

The two paths are peers. They produce the same pore numbering, the same
right-hand side, the same number of Krylov iterations and the same `τ`, so
everything downstream is unchanged and the keyword is the only difference. Keep
the assembled path when you want the matrix itself — it is the CUSPARSE-backed
one, and the only one whose entries can be read.

```@docs
Tortuosity.MaskedLaplacian
Tortuosity.build_steady_operator
```

Once you have solved for a steady-state concentration field, these helpers
derive the usual transport descriptors.

Pass the pore-vector solution and its problem directly when you only need a transport scalar:

```julia
τ = tortuosity(sol.u, sim)
D_eff = effective_diffusivity(sol.u, sim)
F = formation_factor(sol.u, sim)
```

This path reduces the inlet flux from the linear system without allocating a full image or copying the complete solution from a GPU. Use `reconstruct_field` only when you also need the concentration field.

```@docs
tortuosity
effective_diffusivity
formation_factor
reconstruct_field
```

## Cavern detection

Porosity counts a blind pocket exactly as it counts a through-going channel, so part of a pore space can be fully connected and still carry no transport at all. Cavern detection separates the two: it solves the steady problem, thresholds the per-voxel flux, and returns a mask of the stagnant volume — the "caverns" — along with the fraction of pore volume they account for. Reach for it when you want to know where a tortuosity number comes from, to report the transporting share of a pore space rather than its porosity, or to strip dead ends before a further analysis.

Both names are unexported; call them as `Tortuosity.find_caverns` and `Tortuosity.flux_out`.

```@docs
Tortuosity.find_caverns
Tortuosity.flux_out
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
illustrated walk-through. The reference-level descriptions live here.

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
