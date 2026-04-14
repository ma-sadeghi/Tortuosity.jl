# Advanced Transient Techniques

This page covers advanced features of the transient solver: per-voxel tortuosity fitting, time-dependent boundaries for frequency-response measurements, and the analytical reference solutions.

## Per-voxel tortuosity distribution

Steady-state tortuosity is a single number for the whole image. But transport through a porous medium follows many different paths — some short and straight, others long and winding. `fit_voxel_diffusivity` quantifies this by fitting the concentration history at individual voxels to the analytical homogeneous solution, yielding a per-voxel tortuosity estimate.

```@example advtrans
using Plots
using Tortuosity

# 3D image — we need spatial variation along the transport axis
img = Imaginator.blobs(; shape=(64, 64, 32), porosity=0.4, blobiness=0.5, seed=3)
img = Imaginator.trim_nonpercolating_paths(img; axis=:z)

# c=1 at inlet, insulated at outlet — concentration fills the pore space over time
prob = TransientDiffusionProblem(img; bc_inlet=1, bc_outlet=nothing, axis=:z, gpu=false)

sol = solve(prob, ROCK4();
    saveat   = 0.1,
    callback = StopAtSaturation(0.98),
    tspan    = (0.0, 50.0))

# Fit tortuosity at 400 randomly sampled voxels at the outlet (depth=1.0)
tau_vals, SE_tau, voxel_inds = fit_voxel_diffusivity(sol, prob; depth=1.0, n_samples=400)

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

# Sine wave inlet (0 to 1), insulated outlet
prob = TransientDiffusionProblem(img;
    bc_inlet = t -> (sin(2π * freq * t) + 1) / 2,
    bc_outlet = nothing, axis=:z, gpu=false)

# Phase 1: run to periodic steady state, starting at the time-averaged BC value.
sol_warmup = solve(prob, ROCK4();
    saveat   = T/30,
    callback = StopAtPeriodicState(freq, prob; reltol=1e-3),
    tspan    = (0.0, 50.0),
    u0       = 0.5 .* ones(size(img)))

# Phase 2: capture one clean period continuing from the warmup end state.
t_warmup_end = sol_warmup.t[end]
sol = solve(prob, ROCK4();
    saveat = T/30,
    tspan  = (t_warmup_end, t_warmup_end + T),
    u0     = reconstruct_field(sol_warmup.u[end], img))

# Animate the last period
anim = @animate for k in eachindex(sol.t)
    c_grid = reconstruct_field(sol.u[k], img)
    plot(range(0, 1, N), c_grid[:, 1, :][1, :],
        title = "Sine Wave Inlet — Periodic Steady State",
        ylim = (0, 1), legend = false,
        ylabel = "Concentration", xlabel = "Depth",
        linewidth = 2
    )
end

runtime = 2 # hide
using Logging # hide
with_logger(NullLogger()) do # hide
    gif(anim, "sin_inlet.gif", fps=length(sol.t) / runtime) # hide
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

Common keyword arguments: `c1` and `c2` (boundary concentrations, default `1` and `0`), `c0` (initial concentration, default `0`), `L` (slab length, default `1`), `terms` (number of Fourier series terms, default `100`).

See the [API Reference](../api.md) for full signatures.
