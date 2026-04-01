# Transient Solver

`Tortuosity.jl` includes transient functionality to compute the concentration distribution over time in a porous material with some dirichlet or insulated boundaries and initial conditions. It also includes functions to get observables from the solution, like the flux between two voxel planes perpendicular to the axis of the concentration gradient. There may be effects from dead end channels, bottlenecks, and other porous features that vary from the behavior of a homogenous material, and cannot be predicted from the tortuosity calculated from steady state. This feature can be used to quantify and investigate those variations.


## Comparing To Homogenous Solution

Note that the difference between the porous and homogenous solution may be exaggerated for a low resolution image or an image with low domain size to pore size ratio.

```@example
using Tortuosity
using Tortuosity: stop_at_delta_flux, vec_to_grid, tortuosity

USE_GPU = false

axis = :x

# Generate a test image
img = Imaginator.blobs(; shape=(64, 64, 1), porosity=0.65, blobiness=0.5, seed=2)
φ = Tortuosity.porosity(img)

# First get the steady state solution
# Define the simulation
sim_ss = TortuositySimulation(img; axis=axis, gpu=USE_GPU);

# Solve the system of equations
sol_ss = solve(sim_ss.prob, KrylovJL_CG(); verbose=false, reltol=1e-5);

# Convert the solution vector to an Nd grid
C_ss = vec_to_grid(sol_ss.u, img)
# Compute the tortuosity factor from steady state
τ_ss = tortuosity(C_ss; axis=:x)

# Now the transient solution:
# Define the regularity with which data is saved in simulation time
dt = 0.05

# Define the simulation with the same dirichlet boundaries as the steady state solution (1,0)
# and default initial concentration of 0 everywhere
prob = TransientProblem(img, dt; bound_mode = (1, 0), axis = axis, gpu = USE_GPU)
sim = init_state(prob);

# Define a stop condition for near steady state using shorthand
stop_condition = stop_at_delta_flux(0.005, prob)

# Run the solver until stop condition is reached
solve!(sim, prob, stop_condition)

# Get the outlet flux over time from the transient data, at the outlet by default
flux_out = Tortuosity.get_flux(sim.C, prob)

# Get the solution of outlet flux over time for a homogenous material
# with a diffusivity based on the steady state solution for this image
t_ana = range(0, 1.5*sim.t[end], 200)[2:end] #trim 0 to avoid infinite series artifact
J_ana = φ .* Tortuosity.analytic_flux(1/τ_ss, 1, t_ana) #scale by porosity

# Visualize the image
using Plots

heatmap(img[:, :, 1]; aspect_ratio=:equal, clim=(0, 1))
savefig("img-plot.svg"); nothing # hide

#plot the porous and analytical-homogenous solution
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

The transient solve!() function steps the TransientState forward in time, appending a concentration vector to state.C every dt time units, and evaluating a stop condition. Stop conditions are in the form 'f(t_hist, C_hist) -> Bool', when the return value is true, the solve!() function terminates.

Built in stop conditions constructors include:
    stop_at_time(t): 
        stops running once the solver reaches time t
    stop_at_delta_flux(delta, prob::TransientProblem): 
        stops running once the inlet flux and outlet flux are within tolerance delta, good for stopping close to when the concentration gradient reaches steady state, as the flux will be constant everywhere for time-independent boundary conditions
    stop_at_avg_concentration(c, prob::TransientProblem): 
        stops running once the average pore       concentration reaches concentration c, good for boundary conditions where the steady state average concentration is predictable. For example stop a (1, insulated) boundary solution at c_avg=0.99
    stop_at_periodic(freq, prob; reltol=1e-3, Nphase=4, frac_period=0.3, depth = 1.0): 
        stops when Nphase points are within reltol of points one period of time behind them (relative to the amplitude of the previous period). Intended for problems with a time-periodic boundary condition

Custom stop condition example:
    my_stop = (t_hist, C_hist) -> t_hist[end] > 5.0


## Tortuosity Distribution

One indicator of how different a porous material may act compared to a homogenous material is the deviation in tortuosity of the various paths that a substance can take through it. One of the Tortuosity.jl transient analysis tools is voxel_tortuosity(), a function that takes the concentration data from a regularly spaced sample of voxels at a certain plane along the concentration gradient axis of an image, and preforms a least-squares fit with the homogenous concentration over time solution for tortuosity at each voxel.

```@example
using Tortuosity
using Tortuosity: stop_at_avg_concentration, voxel_tortuosity

USE_GPU = false

axis = :z

# Generate a test image
img = Imaginator.blobs(; shape=(64, 64, 32), porosity=0.4, blobiness=0.5, seed=3)
img = Imaginator.trim_nonpercolating_paths(img; axis=axis)
φ = Tortuosity.porosity(img)

# Get the transient solution:
# Define the regularity with which data is saved in simulation time
dt = 0.1

# Define the simulation boundaries with C=1 at the inlet, insulated (NaN) at the outlet
# default initial concentration of 0 everywhere, will go to 1 everywhere
prob = TransientProblem(img, dt; bound_mode = (1, NaN), axis = axis, gpu = USE_GPU)
sim = init_state(prob);

# Define a stop condition for near steady state using shorthand
stop_condition = stop_at_avg_concentration(0.98, prob)

# Run the solver until stop condition is reached
solve!(sim, prob, stop_condition)

# Get tortuosity at a sample of voxels at the outlet
# SE_D, standard deviation of each fit, can be used to assess if the fits are good
# the list of voxel_inds fitted to is available if the spatial locations are of interest
depth = 1.0
tau_vals, SE_tau, voxel_inds_1d = voxel_tortuosity(sim, prob; depth =depth, n_samples = 400)


# Visualize the image
using Plots

histogram( tau_vals,
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

TortuosityProblem supports boundary conditions that are a function of time, although this functionality does not have extensive analysis tools.
This feature is potentially useful for:
- Smoother startup behavior
    If the user wants to minimize numerical error caused by the large concentration gradient associated with zero initial conditions and a non-zero Dirichlet boundary, a boundary condition that starts at the zero and decays towards a final value can be used instead, along with the analytical solution for a homogenous material.

- Probing the material with periodic signals:
    By setting the inlet boundary to a sin wave with a certain frequency, at periodic steady state the outlet concentration over time becomes a wave of the same frequency but decayed amplitude and offset phase. The amplitude and phase offset for a certain frequency can contain information about dead-end side channels in the porous material.

```@example
using Tortuosity
using Tortuosity: stop_at_periodic, stop_at_time

USE_GPU = false
axis = :z

# 1D homogenous image
N = 64
img = trues(1,1,N)

# Parameters
freq = 0.5
T = 1/freq
dt = T/30 #30 datapoints per period

# Boundary with sin wave varying from 0 to 1 and insulated outlet
bounds = (t -> (sin(2π*freq*t)+1)/2, NaN)

#define problem, set initial concentration to average of value of boundary function
prob = TransientProblem(img, dt; bound_mode = bounds, axis = axis, gpu = USE_GPU)
sim = init_state(prob; C0 = 0.5 .*ones(size(img)))

# Run to periodic steady state
stop_condition = stop_at_periodic(freq, prob; reltol = 1e-3)
solve!(sim, prob, stop_condition)

#run the same TransientState for another period
stop_condition = stop_at_time(sim.t[end] + T)
solve!(sim, prob, stop_condition)

# Animate the concentration distribution for a periodic steady state period
using Plots

runtime = 2 #seconds
start_ind = searchsortedfirst(sim.t, sim.t[end] - T)
anim = @animate for k in start_ind:length(sim.t)

    plot(range(0,1,N),Tortuosity.slice_conc_dist(sim.C[k], prob, pore_only = true),
        title = "Concentration Distribution With Sin Wave Inlet",
        ylim = (0,1), legend = false,
        ylabel = "Average Pore Concentration",
        xlabel = "Depth",
        linewidth = 2
    )
end


gif(anim, "sin_inlet.gif", fps=length(sim.t)/runtime); nothing #hide

```

```@example
HTML("""<figure><img src=$(joinpath(Main.buildpath,"sin_inlet.gif"))><figcaption>Concentration distribution after reaching periodic steady state with a sin wave time dependent boundary</figcaption></figure>""") # hide
```