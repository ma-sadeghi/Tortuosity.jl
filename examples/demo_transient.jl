# %% ------------------------------------------------------
# Imports

using Plots
using Printf
using Tortuosity
using Tortuosity: Imaginator, stop_at_delta_flux, effective_diffusivity, analytic_mass

PLOT = false
USE_GPU = false

# %% ------------------------------------------------------
# Generate/load the image

shape = (64, 64, 64)
img = Imaginator.blobs(; shape=shape, porosity=0.5, blobiness=1, seed=2);
img = Imaginator.trim_nonpercolating_paths(img; axis=:x);
PLOT && display(heatmap(img[:, :, shape[3] ÷ 2]; aspect_ratio=:equal, clim=(0, 1)));

# %% ------------------------------------------------------
# Build A*C = dC on CPU/GPU and run the transient solver until it reaches stop_condition
# Save a data point every dt time units in simulation time
dt = 0.05

problem = TransientProblem(img, dt; axis = :x, gpu = USE_GPU,
    bound_mode=(1,0)
)
sim = init_state(problem);

#stop run when outlet flux is very close to inlet flux
stop_condition = stop_at_delta_flux(0.005, problem)
solve!(sim, problem, stop_condition)

# %% ------------------------------------------------------
# Compute the tortuosity factor and visualize the solution
# very much a work in progress part

D_eff_pore, φ, xdata, ydata, fit, f = effective_diffusivity(sim, problem, :mass)

τ = 1/D_eff_pore #not the same as D_eff continuum
@info "τ: $(@sprintf("%.5f", τ))"

if PLOT
    t_analytic = range(sim.t[1], sim.t[end], 100)
    m_analytic = f(t_analytic, fit.param)

    p = plot(t_analytic, m_analytic,
        title = "Mass Intake Curve",
        xlabel = "time",
        ylabel = "mass intake",
        label = "fitted curve, τ = $(@sprintf("%.5f", τ))",
        legend = :bottomright
    )
    plot!(p[1], xdata, ydata,
        seriestype = :scatter,
        label = "simulation data, D_free = 1.0"
    )
end