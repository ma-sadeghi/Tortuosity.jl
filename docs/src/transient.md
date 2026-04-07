# Transient Solver

`Tortuosity.jl` includes a transient diffusion solver for computing concentration distributions over time in porous materials. It supports Dirichlet and insulated (zero-flux Neumann) boundary conditions, and provides functions to extract observables such as the flux between voxel planes perpendicular to the concentration gradient axis.

Porous features like dead-end channels and bottlenecks can produce transient behavior that deviates from homogeneous predictions and cannot be captured by steady-state tortuosity alone. The transient solver can quantify these effects.

## Boundary condition types

`TransientProblem` accepts the following boundary condition types for `bc_inlet` and `bc_outlet`:

| Type | Meaning | Example |
|------|---------|---------|
| `Number` | Dirichlet (fixed concentration) | `bc_inlet=1.0` |
| `nothing` | Insulated (zero-flux Neumann) | `bc_outlet=nothing` |
| `Function` | Time-dependent Dirichlet | `bc_inlet=t -> sin(2π*t)` |

## Comparing to homogeneous solutions

The example below runs a transient simulation on a porous image and compares the outlet flux to the analytical solution for a homogeneous slab with diffusivity $D_\text{eff} = 1/\tau_\text{ss}$.

```@example
using Tortuosity
using Tortuosity: stop_at_delta_flux, vec_to_grid, tortuosity, compute_flux, slab_flux

USE_GPU = false

axis = :x

# Generate a test image
img = Imaginator.blobs(; shape=(64, 64, 1), porosity=0.65, blobiness=0.5, seed=2)
φ = Imaginator.phase_fraction(img, true)

# Steady-state solution for reference tortuosity
sim_ss = TortuositySimulation(img; axis=axis, gpu=USE_GPU);
sol_ss = solve(sim_ss.prob, KrylovJL_CG(); verbose=false, reltol=1e-5);

C_ss = vec_to_grid(sol_ss.u, img)
τ_ss = tortuosity(C_ss, img; axis=:x)

# Transient solution
# dt is the snapshot interval (not the internal ODE timestep)
dt = 0.05

prob = TransientProblem(img, dt; bc_inlet=1, bc_outlet=0, axis=axis, gpu=USE_GPU)
sim = init_state(prob);

# Stop when inlet and outlet fluxes converge (near steady state)
stop_condition = stop_at_delta_flux(0.005, prob)
solve!(sim, prob, stop_condition)

# Outlet flux at each snapshot
flux_out = map(C -> compute_flux(C, prob.D, prob.dx, prob.img, prob.axis; ind=:end, grid_to_vec=prob.grid_to_vec), sim.C)

# Analytical outlet flux for a homogeneous slab with D_eff = 1/τ_ss
t_ana = range(0, 1.5*sim.t[end], 200)[2:end]
J_ana = φ .* slab_flux(1/τ_ss, 1, t_ana)  # x=1 is the outlet face of a unit-length slab

# Visualize
using Plots

heatmap(img[:, :, 1]; aspect_ratio=:equal, clim=(0, 1))
savefig("img-plot.svg"); nothing # hide

plot(sim.t, flux_out,
    title = "Outlet Flux over Time", xlabel = "time", ylabel = "flux",
    seriestype = :scatter, label = "transient data for image",
    legend = :bottomright
)

plot!(t_ana, J_ana,
    label = "homogeneous solution with D = 1/τ_ss \n τ_ss = $(round(τ_ss, digits = 2))"
)

savefig("outlet_flux.svg"); nothing #hide

```

```@example
HTML("""<figure><img src=$(joinpath(Main.buildpath,"img-plot.svg"))><figcaption>Original binary image</figcaption></figure>""") # hide
```

```@example
HTML("""<figure><img src=$(joinpath(Main.buildpath,"outlet_flux.svg"))><figcaption>Comparison of porous material transient data with analytical homogeneous solution</figcaption></figure>""") # hide
```

!!! note
    The difference between the porous and homogeneous solutions may be exaggerated for low-resolution images or images with a low domain-to-pore size ratio.

## Stop conditions

The transient `solve!()` function steps the `TransientState` forward in time, appending a concentration vector to `state.C` every `dt` time units and evaluating a stop condition. Stop conditions have the signature `f(t_hist, C_hist) -> Bool`. When the return is `true`, `solve!()` terminates.

Built-in stop condition constructors:

- **`stop_at_time(t)`** — stops once the solver reaches time `t`.
- **`stop_at_delta_flux(delta, prob)`** — stops when the absolute difference between inlet and outlet flux falls below `delta`. Useful for detecting steady state under time-independent boundary conditions.
- **`stop_at_avg_concentration(c, prob)`** — stops when the average pore concentration reaches `c`. For example, with `bc_inlet=1` and `bc_outlet=nothing`, stop at `c_avg=0.99`.
- **`stop_at_periodic(freq, prob; reltol=1e-2, Nphase=4, frac_period=0.3, depth=1.0)`** — stops when `Nphase` phase points are within `reltol` of the previous period. Intended for problems with time-periodic boundary conditions.

Custom stop conditions can be any function with the right signature:

```julia
my_stop = (t_hist, C_hist) -> t_hist[end] > 5.0
```

## Tortuosity distribution

One indicator of how a porous material deviates from homogeneous behavior is the distribution of tortuosity across individual pore paths. `fit_voxel_diffusivity()` samples voxel concentration histories at a cross-sectional plane and fits each to the analytical homogeneous solution, yielding a per-voxel tortuosity estimate via least-squares regression.

```@example
using Tortuosity
using Tortuosity: stop_at_avg_concentration, fit_voxel_diffusivity

USE_GPU = false

axis = :z

# Generate a 3D test image and trim isolated pores (porosity 0.4 has many dead ends)
img = Imaginator.blobs(; shape=(64, 64, 32), porosity=0.4, blobiness=0.5, seed=3)
img = Imaginator.trim_nonpercolating_paths(img; axis=axis)
φ = Imaginator.phase_fraction(img, true)

# Transient solution: C=1 at inlet, insulated at outlet
dt = 0.1

prob = TransientProblem(img, dt; bc_inlet=1, bc_outlet=nothing, axis=axis, gpu=USE_GPU)
sim = init_state(prob);

stop_condition = stop_at_avg_concentration(0.98, prob)
solve!(sim, prob, stop_condition)

# Fit tortuosity at a sample of voxels at the outlet (depth=1.0)
tau_vals, SE_tau, voxel_inds_1d = fit_voxel_diffusivity(sim, prob; depth=1.0, n_samples=400)

# Plot a histogram of the tortuosity values
using Plots

histogram(tau_vals,
    xlabel = "Tortuosity",
    ylabel = "Bin Count",
    title = "Tortuosity Distribution",
    legend = false
)

savefig("tortuosity_histogram.svg"); nothing #hide

```

```@example
HTML("""<figure><img src=$(joinpath(Main.buildpath,"tortuosity_histogram.svg"))><figcaption>Histogram of fitted tortuosity at the outlet (depth = 1.0)</figcaption></figure>""") # hide
```

## Time-dependent boundary

`TransientProblem` supports boundary conditions that are functions of time. This is useful for:

- **Smoother startup** — ramp the boundary to reduce numerical error from large initial concentration gradients.
- **Periodic probing** — a sine wave inlet at periodic steady state produces an outlet wave with decayed amplitude and phase offset, which can reveal dead-end channels and other structural features.

```@example
using Tortuosity
using Tortuosity: stop_at_periodic, stop_at_time, vec_to_grid

USE_GPU = false
axis = :z

# 1D homogeneous image
N = 64
img = trues(1,1,N)

# Parameters
freq = 0.5
T = 1/freq
dt = T/30  # ~30 snapshots per period

# Sine wave inlet (varying from 0 to 1) with insulated outlet
prob = TransientProblem(img, dt; bc_inlet=t -> (sin(2π*freq*t)+1)/2, bc_outlet=nothing, axis=axis, gpu=USE_GPU)
# Initial condition at the time-averaged value of the BC (faster convergence to periodic steady state)
sim = init_state(prob; C0=0.5 .* ones(size(img)))

# Run to periodic steady state
stop_condition = stop_at_periodic(freq, prob; reltol=1e-3)
solve!(sim, prob, stop_condition)

# Run for one more period to capture the steady-state waveform
stop_condition = stop_at_time(sim.t[end] + T)
solve!(sim, prob, stop_condition)

# Animate the concentration distribution over the last period
using Plots

start_ind = searchsortedfirst(sim.t, sim.t[end] - T)
anim = @animate for k in start_ind:length(sim.t)
    C_grid = vec_to_grid(sim.C[k], img)
    plot(range(0, 1, N), C_grid[:, 1, :][1, :],
        title = "Concentration With Sine Wave Inlet",
        ylim = (0, 1), legend = false,
        ylabel = "Concentration",
        xlabel = "Depth",
        linewidth = 2
    )
end

runtime = 2 #seconds # hide
using Logging # hide
with_logger(NullLogger()) do # hide
    gif(anim, "sin_inlet.gif", fps=length(sim.t)/runtime) # hide
end # hide
nothing # hide

```

```@example
HTML("""<figure><img src=$(joinpath(Main.buildpath,"sin_inlet.gif"))><figcaption>Concentration distribution at periodic steady state with a sine wave inlet</figcaption></figure>""") # hide
```
