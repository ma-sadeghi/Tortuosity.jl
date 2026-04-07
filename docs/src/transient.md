# Transient Solver

`Tortuosity.jl` includes transient functionality to compute the concentration distribution over time in a porous material with some Dirichlet or insulated boundaries and initial conditions. It also includes functions to get observables from the solution, such as the flux between two voxel planes perpendicular to the axis of the concentration gradient.

There may be effects from dead end channels, bottlenecks, and other porous features that vary from the behavior of a homogenous material, and cannot be predicted from the tortuosity calculated from steady state. This feature can be used to quantify and investigate those variations.


## Comparing To Homogenous Solutions

Note that the difference between the porous and homogenous solution may be exaggerated for a low resolution image or an image with low domain size to pore size ratio.

```@example
using Tortuosity
using Tortuosity: stop_at_delta_flux, vec_to_grid, tortuosity, compute_flux, slab_flux

USE_GPU = false

axis = :x

# Generate a test image
img = Imaginator.blobs(; shape=(64, 64, 1), porosity=0.65, blobiness=0.5, seed=2)
φ = Imaginator.phase_fraction(img, true)

# First get the steady state solution
sim_ss = TortuositySimulation(img; axis=axis, gpu=USE_GPU);
sol_ss = solve(sim_ss.prob, KrylovJL_CG(); verbose=false, reltol=1e-5);

# Convert the solution vector to an Nd grid
C_ss = vec_to_grid(sol_ss.u, img)
# Compute the tortuosity factor from steady state
τ_ss = tortuosity(C_ss; axis=:x)

# Now the transient solution:
dt = 0.05

prob = TransientProblem(img, dt; bc_inlet=1, bc_outlet=0, axis=axis, gpu=USE_GPU)
sim = init_state(prob);

# Define a stop condition for near steady state
stop_condition = stop_at_delta_flux(0.005, prob)
solve!(sim, prob, stop_condition)

# Get the outlet flux over time from the transient data
flux_out = map(C -> compute_flux(C, prob.D, prob.dx, prob.img, prob.axis; ind=:end, grid_to_vec=prob.grid_to_vec), sim.C)

# Get the analytical outlet flux for a homogenous material
t_ana = range(0, 1.5*sim.t[end], 200)[2:end]
J_ana = φ .* slab_flux(1/τ_ss, 1, t_ana)

# Visualize the image
using Plots

heatmap(img[:, :, 1]; aspect_ratio=:equal, clim=(0, 1))
savefig("img-plot.svg"); nothing # hide

# Plot the porous and analytical-homogenous solution
plot(sim.t, flux_out,
    title = "Outlet Flux over Time", xlabel = "time", ylabel = "flux",
    seriestype = :scatter, label = "transient data for image",
    legend = :bottomright
)

plot!(t_ana, J_ana,
    label = "homogenous solution with D = 1/τ_ss \n τ_ss = $(round(τ_ss, digits = 2))"
)

savefig("outlet_flux.svg"); nothing #hide

```

```@example
HTML("""<figure><img src=$(joinpath(Main.buildpath,"img-plot.svg"))><figcaption>Original binary image</figcaption></figure>""") # hide
```

```@example
HTML("""<figure><img src=$(joinpath(Main.buildpath,"outlet_flux.svg"))><figcaption>Comparison of porous material transient data with analytical solution for homogenous case</figcaption></figure>""") # hide
```

## Stop Conditions

The transient `solve!()` function steps the TransientState forward in time, appending a concentration vector to state.C every dt time units, and evaluating a stop condition. Stop conditions are in the form `f(t_hist, C_hist) -> Bool`. When the return value is true, the `solve!()` function terminates.

Built in stop condition constructors include:
- `stop_at_time(t)`:
        stops running once the solver reaches time t
- `stop_at_delta_flux(delta, prob::TransientProblem)`:
        stops running once the inlet flux and outlet flux are within tolerance delta. Good for stopping close to when the concentration gradient reaches steady state, as the flux will be constant everywhere for time-independent boundary conditions.
- `stop_at_avg_concentration(c, prob::TransientProblem)`:
        stops running once the average pore concentration reaches concentration c. Good for boundary conditions where the steady state average concentration is predictable. For example, stop a (1, insulated) boundary solution at c_avg=0.99.
- `stop_at_periodic(freq, prob; reltol=1e-3, Nphase=4, frac_period=0.3, depth=1.0)`:
        stops when Nphase points are within reltol of points one period of time behind them (relative to the amplitude of the previous period). Intended for problems with a time-periodic boundary condition.

Custom stop condition example:
    `my_stop = (t_hist, C_hist) -> t_hist[end] > 5.0`


## Tortuosity Distribution

One indicator of how different a porous material may act compared to a homogenous material is the deviation in tortuosity of the various paths that a substance can take through it. `fit_voxel_diffusivity()` takes the concentration data from a regularly-spaced sample of voxels at a certain plane along the concentration gradient axis of an image, and performs a least-squares fit with the homogenous concentration over time solution for tortuosity at each voxel.

```@example
using Tortuosity
using Tortuosity: stop_at_avg_concentration, fit_voxel_diffusivity

USE_GPU = false

axis = :z

# Generate a test image
img = Imaginator.blobs(; shape=(64, 64, 32), porosity=0.4, blobiness=0.5, seed=3)
img = Imaginator.trim_nonpercolating_paths(img; axis=axis)
φ = Imaginator.phase_fraction(img, true)

# Get the transient solution:
dt = 0.1

# C=1 at the inlet, insulated (nothing) at the outlet
prob = TransientProblem(img, dt; bc_inlet=1, bc_outlet=nothing, axis=axis, gpu=USE_GPU)
sim = init_state(prob);

stop_condition = stop_at_avg_concentration(0.98, prob)
solve!(sim, prob, stop_condition)

# Get tortuosity at a sample of voxels at the outlet
depth = 1.0
tau_vals, SE_tau, voxel_inds_1d = fit_voxel_diffusivity(sim, prob; depth=depth, n_samples=400)

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
HTML("""<figure><img src=$(joinpath(Main.buildpath,"tortuosity_histogram.svg"))><figcaption>Histogram of fitted tortuosity at the outlet</figcaption></figure>""") # hide
```

## Time-Dependent Boundary

`TransientProblem` supports boundary conditions that are a function of time. This feature is useful for:
- Smoother startup behavior: minimize numerical error from large concentration gradients at early time by ramping the boundary condition.
- Probing the material with periodic signals: a sine wave inlet at periodic steady state produces an outlet wave with decayed amplitude and offset phase, which can reveal dead-end channels.

```@example
using Tortuosity
using Tortuosity: stop_at_periodic, stop_at_time, vec_to_grid

USE_GPU = false
axis = :z

# 1D homogenous image
N = 64
img = trues(1,1,N)

# Parameters
freq = 0.5
T = 1/freq
dt = T/30

# Boundary with sin wave varying from 0 to 1 and insulated outlet
prob = TransientProblem(img, dt; bc_inlet=t -> (sin(2π*freq*t)+1)/2, bc_outlet=nothing, axis=axis, gpu=USE_GPU)
sim = init_state(prob; C0=0.5 .* ones(size(img)))

# Run to periodic steady state
stop_condition = stop_at_periodic(freq, prob; reltol=1e-3)
solve!(sim, prob, stop_condition)

# Run for another period
stop_condition = stop_at_time(sim.t[end] + T)
solve!(sim, prob, stop_condition)

# Animate the concentration distribution for a periodic steady state period
using Plots

start_ind = searchsortedfirst(sim.t, sim.t[end] - T)
anim = @animate for k in start_ind:length(sim.t)
    C_grid = vec_to_grid(sim.C[k], img)
    plot(range(0, 1, N), C_grid[:, 1, :][1, :],
        title = "Concentration Distribution With Sine Wave Inlet",
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
HTML("""<figure><img src=$(joinpath(Main.buildpath,"sin_inlet.gif"))><figcaption>Concentration distribution after reaching periodic steady state with a sine wave time dependent boundary</figcaption></figure>""") # hide
```
