# API Reference

This page lists the main types and functions in `Tortuosity.jl`. Functions marked with **(exported)** are available after `using Tortuosity`. All others require a qualified import, e.g., `using Tortuosity: tortuosity`.

## Types

### `TortuositySimulation` (exported)

```julia
TortuositySimulation(img; axis, D=nothing, gpu=nothing, verbose=false)
```

Sets up a steady-state diffusion problem on a binary pore image. Builds the Laplacian operator and applies Dirichlet boundary conditions ($c=1$ at inlet, $c=0$ at outlet).

- **`img`** — `BitArray` where `true` = pore, `false` = solid.
- **`axis`** — transport direction (`:x`, `:y`, or `:z`).
- **`D`** — diffusivity. Scalar (uniform) or array matching `img` shape. Default: `1.0`.
- **`gpu`** — `true` for CUDA, `false` for CPU, `nothing` for auto-detect (GPU when >100k pore voxels).
- **`verbose`** — print solver progress.

The resulting `sim.prob` is a `LinearProblem` that can be solved with any [LinearSolve.jl](https://docs.sciml.ai/LinearSolve/) solver.

### `TransientProblem` (exported)

```julia
TransientProblem(img, dt; axis=:z, bc_inlet=1, bc_outlet=0,
                 D=1.0, dx=nothing, dtype=Float32, gpu=nothing)
```

Sets up a transient diffusion problem.

- **`img`** — `BitArray` pore mask.
- **`dt`** — snapshot interval (saved concentration interval, not the internal ODE timestep).
- **`axis`** — transport direction.
- **`bc_inlet`**, **`bc_outlet`** — boundary conditions: `Number` (Dirichlet), `nothing` (insulated), or `Function` (time-dependent Dirichlet).
- **`D`** — diffusivity (scalar or array). Default: `1.0`.
- **`dx`** — voxel spacing. Default: computed as `1 / (N-1)` where `N` is the number of voxels along `axis`.
- **`dtype`** — floating-point precision. Default: `Float32`.
- **`gpu`** — same as `TortuositySimulation`.

### `TransientState` (exported)

```julia
TransientState(integrator, t, C)
```

Holds the state of a transient simulation. Created by `init_state`. Fields:

- **`integrator`** — ODE integrator instance (from OrdinaryDiffEq.jl).
- **`t::Vector{Float64}`** — time history.
- **`C::Vector{Vector{T}}`** — concentration history. Each entry is a 1D pore-only vector.

## Steady-state functions

### `solve` (exported)

```julia
solve(sim.prob, alg; kwargs...)
```

Solves the steady-state linear system. Accepts any [LinearSolve.jl](https://docs.sciml.ai/LinearSolve/) algorithm. `KrylovJL_CG()` (exported) is the recommended default.

### `tortuosity`

```julia
tortuosity(c, img; axis, slice=1, eps=nothing, D=1.0, dx=1.0, L=nothing, Δc=nothing)
```

Computes $\tau = \varepsilon / D_\text{eff}$ from a concentration field.

### `effective_diffusivity`

```julia
effective_diffusivity(c, img; axis, slice=1, D=1.0, dx=1.0, L=nothing, Δc=nothing)
```

Computes $D_\text{eff}$ by measuring flux through a cross-sectional `slice`.

### `formation_factor`

```julia
formation_factor(c, img; axis, slice=1, D=1.0, dx=1.0, L=nothing, Δc=nothing)
```

Computes $F = 1 / D_\text{eff}$.

### `vec_to_grid`

```julia
vec_to_grid(u, img)
```

Expands a pore-only solution vector `u` into a full-sized array matching `img`. Solid voxels are filled with `NaN`.

## Transient functions

### `init_state` (exported)

```julia
init_state(prob::TransientProblem; C0=nothing, alg=ROCK4(), reltol=1e-3, abstol=1e-6)
```

Initializes a `TransientState` from a `TransientProblem`.

- **`C0`** — initial concentration field (3D array matching `img`). Default: zeros with boundary conditions applied.
- **`alg`** — ODE solver algorithm. Default: `ROCK4()` (explicit stabilized method). Only explicit methods are supported.
- **`reltol`**, **`abstol`** — ODE solver tolerances.

### `solve!` (exported)

```julia
solve!(state::TransientState, prob::TransientProblem, stop_condition; max_iter=500, verbose=false)
```

Advances the transient simulation until `stop_condition(t_hist, C_hist)` returns `true` or `max_iter` snapshots are reached.

### `stop_at_time` (exported)

```julia
stop_at_time(t_final)
```

Returns a stop condition that triggers at time `t_final`.

### `stop_at_delta_flux`

```julia
stop_at_delta_flux(delta, prob::TransientProblem)
```

Stop condition: triggers when the absolute difference between inlet and outlet flux falls below `delta`.

### `stop_at_avg_concentration`

```julia
stop_at_avg_concentration(C_final, prob::TransientProblem)
stop_at_avg_concentration(C_final, img::BitArray)
```

Stop condition: triggers when mean pore concentration reaches `C_final`.

### `stop_at_periodic`

```julia
stop_at_periodic(freq, prob::TransientProblem;
                 reltol=1e-2, Nphase=4, frac_period=0.3, depth=1.0)
```

Stop condition for periodic steady-state detection under oscillating boundary conditions.

## Measurement functions

### `compute_flux`

```julia
compute_flux(C, D, dx, img, axis; ind=:end, grid_to_vec=nothing)
compute_flux(Cs::AbstractVector, D, dx, img, axis; kwargs...)
```

Computes the diffusive flux between two adjacent voxel planes at index `ind` along `axis`. The vector overload maps over a list of concentration snapshots.

### `get_slice_conc`

```julia
get_slice_conc(C, img, axis, ind; grid_to_vec=nothing, pore_only=false)
get_slice_conc(Cs::AbstractVector, img, axis, ind; kwargs...)
```

Returns the mean concentration along a 2D cross-section at index `ind`.

### `compute_mass_intake`

```julia
compute_mass_intake(C_hist, img)
```

Computes total pore-averaged concentration at each timestep.

## Fitting functions

### `fit_effective_diffusivity`

```julia
fit_effective_diffusivity(t, C, prob::TransientProblem, method::Symbol;
                          depth=0.5, t_fit=(0, t[end]), terms=100)
fit_effective_diffusivity(sim::TransientState, prob::TransientProblem, method; kwargs...)
```

Fits the 1D analytical slab solution to transient data to extract $D_\text{eff}$ and $\tau$.

- **`method`** — `:conc` (concentration), `:mass` (mass uptake), or `:flux` (diffusive flux).
- **`depth`** — normalized position along the axis where data is sampled.
- **`t_fit`** — time window for fitting.

### `fit_voxel_diffusivity`

```julia
fit_voxel_diffusivity(sim::TransientState, prob::TransientProblem;
                      depth=0.5, n_samples=200, t_fit=(0, sim.t[end]),
                      terms=100, fit_depth=false)
```

Fits per-voxel tortuosity at a cross-sectional plane. Returns `(tau_vals, SE_tau, voxel_inds)`.

## Analytical solutions

All functions accept scalar or vector `t`. Common keyword arguments: `C1=1`, `C2=0` (boundary concentrations), `C0=0` (initial), `L=1` (slab length), `terms=100` (Fourier series terms).

```julia
slab_concentration(D, x, t; C1=1, C2=0, C0=0, L=1, terms=100)
slab_mass_uptake(D, t; C1=1, C2=0, C0=0, L=1, terms=100)
slab_flux(D, x, t; C1=1, C2=0, C0=0, L=1, terms=100)
slab_cumulative_flux(D, t; C1=1, C2=0, C0=0, L=1, terms=100)
```

## Imaginator submodule

### Image generation

```julia
Imaginator.blobs(; shape, porosity, blobiness, seed=nothing)
```

Generates a random binary porous image. See [Imaginator](imaginator.md) for parameter details.

### Image analysis

```julia
Imaginator.phase_fraction(img)            # Dict of all phase fractions
Imaginator.phase_fraction(img, label)     # fraction for a single label
Imaginator.trim_nonpercolating_paths(img; axis)  # remove isolated clusters
Imaginator.faces(shape; inlet, outlet)    # boundary face masks
```

### Image processing

```julia
Imaginator.denoise(img, kernel_radius)    # morphological denoising
Imaginator.disk(r)                        # 2D structuring element
Imaginator.ball(r)                        # 3D structuring element
Imaginator.apply_gaussian_blur(img, sigma)
Imaginator.to_binary(img, threshold=0.5)
Imaginator.norm_to_uniform(img; scale)
```
