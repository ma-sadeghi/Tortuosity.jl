# Transient Solver

You can use `Tortuosity.jl` to compute the concentration distribution over time in a porous material with some dirichlet or insulated boundaries and initial conditions. The effects of dead end channels, bottlenecks, and other porous features may cause differences that cannot be predicted from the tortuosity calculated from steady state. This feature can be used to quantify and investigate those differences.


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

# Get the outlet flux over time from the transient data
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
HTML("""<figure><img src=$(joinpath(Main.buildpath,"outlet_flux.svg"))><figcaption>Comparison of porous transient data with analytical solution for homogenous case</figcaption></figure>""") # hide
```

