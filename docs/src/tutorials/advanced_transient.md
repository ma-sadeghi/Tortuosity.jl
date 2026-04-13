# Advanced Transient Techniques

This page covers advanced features of the transient solver: the full set of stop conditions, per-voxel tortuosity fitting, time-dependent boundaries, and the analytical reference solutions.

## All stop conditions

The transient `solve!` function accepts any function with signature `f(t_hist, C_hist) -> Bool`. The package provides four built-in constructors:

| Constructor | Stops when... | Typical use case |
|-------------|--------------|------------------|
| `stop_at_time(t)` | Simulation time reaches `t` | Fixed-duration runs |
| `stop_at_flux_balance(delta, prob)` | Inlet–outlet flux difference ≤ `delta` | Steady-state detection |
| `stop_at_avg_concentration(c, prob)` | Mean pore concentration reaches `c` | Insulated-outlet saturation |
| `stop_at_periodic(freq, prob; reltol=1e-2, ...)` | Periodic steady state detected | Oscillating boundary conditions |

`stop_at_periodic` has additional keyword arguments:

- **`reltol`** — relative tolerance for periodicity (default `1e-2`).
- **`Nphase`** — number of phase points to compare across the trailing window (default `4`).
- **`frac_period`** — fraction of the period to test (default `0.3`).
- **`depth`** — normalized depth in $(0, 1]$ at which concentration is evaluated (default `1.0`).

Custom stop conditions are just functions:

```julia
# Stop after 100 diffusion time units
my_stop = (t_hist, C_hist) -> t_hist[end] > 100.0

# Stop when max concentration exceeds 0.95
my_stop = (t_hist, C_hist) -> maximum(C_hist[end]) > 0.95
```

## Tortuosity distribution

Steady-state tortuosity is a single number for the whole image. But transport through a porous medium follows many different paths — some short and straight, others long and winding. `fit_voxel_diffusivity` quantifies this by fitting the concentration history at individual voxels to the analytical homogeneous solution, yielding a per-voxel tortuosity estimate.

```@example advtrans
using Plots
using Tortuosity

# 3D image — we need spatial variation along the transport axis
img = Imaginator.blobs(; shape=(64, 64, 32), porosity=0.4, blobiness=0.5, seed=3)
img = Imaginator.trim_nonpercolating_paths(img; axis=:z)

# C=1 at inlet, insulated at outlet — concentration fills the pore space over time
prob = TransientDiffusionProblem(img, 0.1; bc_inlet=1, bc_outlet=nothing, axis=:z, gpu=false)
sim = init_state(prob)

solve!(sim, prob, stop_at_avg_concentration(0.98, prob))

# Fit tortuosity at 400 randomly sampled voxels at the outlet (depth=1.0)
tau_vals, SE_tau, voxel_inds = fit_voxel_diffusivity(sim, prob; depth=1.0, n_samples=400)

histogram(tau_vals,
    xlabel = "Tortuosity", ylabel = "Count",
    title = "Per-Voxel Tortuosity Distribution at the Outlet",
    legend = false
)
savefig("tau_hist.svg"); nothing # hide
```

```@example advtrans
HTML("""<figure><img src=$(joinpath(Main.buildpath,"tau_hist.svg"))><figcaption>Distribution of fitted tortuosity across outlet voxels — the spread reflects path-length variability through the pore network</figcaption></figure>""") # hide
```

Key parameters of `fit_voxel_diffusivity`:
- **`depth`** — normalized position along the transport axis in $(0, 1]$ where voxels are sampled.
- **`n_samples`** — number of voxels to sample.

## Time-dependent boundaries

`TransientDiffusionProblem` supports boundary conditions that are functions of time. Two common use cases:

**Smoother startup** — ramp the boundary to reduce numerical error from the large initial concentration jump.

**Periodic probing** — a sine wave inlet at periodic steady state produces an outlet wave with decayed amplitude and shifted phase. The decay and shift depend on the pore structure, revealing dead-end channels and bottlenecks that steady-state tortuosity cannot capture.

```@example timedep
using Plots
using Tortuosity

# 1D homogeneous image for clarity
N = 64
img = trues(1, 1, N)

freq = 0.5
T = 1 / freq
dt = T / 30  # ~30 snapshots per period

# Sine wave inlet (0 to 1), insulated outlet
prob = TransientDiffusionProblem(img, dt;
    bc_inlet = t -> (sin(2π * freq * t) + 1) / 2,
    bc_outlet = nothing, axis=:z, gpu=false
)
# Start at the time-averaged BC value for faster convergence to periodic steady state
sim = init_state(prob; C0 = 0.5 .* ones(size(img)))

# Phase 1: run to periodic steady state
solve!(sim, prob, stop_at_periodic(freq, prob; reltol=1e-3))

# Phase 2: capture one clean period
solve!(sim, prob, stop_at_time(sim.t[end] + T))

# Animate the last period
start_ind = searchsortedfirst(sim.t, sim.t[end] - T)
anim = @animate for k in start_ind:length(sim.t)
    C_grid = reconstruct_field(sim.C[k], img)
    plot(range(0, 1, N), C_grid[:, 1, :][1, :],
        title = "Sine Wave Inlet — Periodic Steady State",
        ylim = (0, 1), legend = false,
        ylabel = "Concentration", xlabel = "Depth",
        linewidth = 2
    )
end

runtime = 2 # hide
using Logging # hide
with_logger(NullLogger()) do # hide
    gif(anim, "sin_inlet.gif", fps=length(sim.t) / runtime) # hide
end # hide
nothing # hide
```

```@example timedep
HTML("""<figure><img src=$(joinpath(Main.buildpath,"sin_inlet.gif"))><figcaption>Concentration profile oscillating at periodic steady state</figcaption></figure>""") # hide
```

## Analytical reference solutions

The package includes analytical solutions for 1D slab diffusion, useful for validating simulations and fitting transport parameters. All functions accept scalar or vector time inputs.

| Function | Returns |
|----------|---------|
| `slab_concentration(D, x, t)` | Concentration at position `x` and time `t` |
| `slab_mass_uptake(D, t)` | Normalized mass uptake over time |
| `slab_flux(D, x, t)` | Diffusive flux at position `x` |
| `slab_cumulative_flux(D, t)` | Cumulative flux through the outlet |

Common keyword arguments: `C1` and `C2` (boundary concentrations, default `1` and `0`), `C0` (initial concentration, default `0`), `L` (slab length, default `1`), `terms` (number of Fourier series terms, default `100`).

See the [API Reference](../api.md) for full signatures.
