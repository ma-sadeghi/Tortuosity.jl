# %%
# Imports

using Plots
using Printf
using Tortuosity
using Tortuosity: Imaginator

PLOT = true
USE_GPU = false

# %%
# Generate/load the image

shape = (64, 64, 64)
img = Imaginator.blobs(; shape=shape, porosity=0.65, blobiness=1, seed=2);
img = Imaginator.trim_nonpercolating_paths(img; axis=:x);
PLOT && display(heatmap(img[:, :, shape[3] ÷ 2]; aspect_ratio=:equal, clim=(0, 1)));

# %%
# Build the problem and run the transient solver until inlet/outlet flux
# balance falls below the tolerance — the porous-media-native "near steady
# state" check.

problem = TransientDiffusionProblem(img; axis=:x, gpu=USE_GPU, bc_inlet=1, bc_outlet=0)

@time sol = solve(problem, ROCK4();
    saveat   = 0.1,
    callback = StopAtFluxBalance(problem; abstol=0.01),
    tspan    = (0.0, 200.0),
    reltol   = 1e-3,
)

@info "sol.retcode = $(sol.retcode), t[end] = $(sol.t[end]), snapshots = $(length(sol.t))"

# %%
# Compute the tortuosity factor and visualize the solution

τ, D_eff, xdata, ydata, fit, f = fit_effective_diffusivity(sol, problem, :flux; depth=1.0)

@info "τ: $(@sprintf("%.5f", τ))"

if PLOT
    t_analytic = range(sol.t[1], sol.t[end], 100)
    J_analytic = f(t_analytic, fit.param)

    p = plot(t_analytic, J_analytic;
        title="Outlet Flux Curve",
        xlabel="time",
        ylabel="outlet flux",
        label="fitted curve, τ = $(@sprintf("%.5f", τ))",
        legend=:bottomright,
    )
    plot!(p[1], xdata, ydata; seriestype=:scatter, label="simulation data, D_free = 1.0")
end
