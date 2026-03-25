# %% ------------------------------------------------------
# Imports

using Plots
using Printf
using Tortuosity
using Tortuosity: stop_at_delta_flux, effective_diffusivity, analytic_mass

PLOT = true
USE_GPU = false

# %% ------------------------------------------------------
# Generate/load the image

shape = (64, 64, 64)
img = Imaginator.blobs(; shape=shape, porosity=0.65, blobiness=1, seed=2);
img = Imaginator.trim_nonpercolating_paths(img; axis=:x);
PLOT && display(heatmap(img[:, :, shape[3] ÷ 2]; aspect_ratio=:equal, clim=(0, 1)));

# %% ------------------------------------------------------
# Build A*C = dC on CPU/GPU and run the transient solver until it reaches stop_condition
# Save a data point every dt time units in simulation time
dt = 0.1

problem = TransientProblem(img, dt; axis = :x, gpu = USE_GPU,
    bound_mode=(1,0) # referring to Dirichlet bound at inlet and outlet
)
sim = init_state(problem);

#stop run when outlet flux is very close to inlet flux
stop_condition = stop_at_delta_flux(0.005, problem)
solve!(sim, problem, stop_condition)

# %% ------------------------------------------------------
# Compute the tortuosity factor and visualize the solution

τ, D_eff, xdata, ydata, fit, f = effective_diffusivity(sim, problem, :flux, depth = 1)

@info "τ: $(@sprintf("%.5f", τ))"

if PLOT
    t_analytic = range(sim.t[1], sim.t[end], 100)
    J_analytic = f(t_analytic, fit.param)

    p = plot(t_analytic, J_analytic,
        title = "Outlet Flux Curve",
        xlabel = "time",
        ylabel = "outlet flux",
        label = "fitted curve, τ = $(@sprintf("%.5f", τ))",
        legend = :bottomright
    )
    plot!(p[1], xdata, ydata,
        seriestype = :scatter,
        label = "simulation data, D_free = 1.0"
    )
end