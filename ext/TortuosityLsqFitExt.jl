# Optional dependency: routes Tortuosity's curve-fitting hooks to LsqFit.jl.
module TortuosityLsqFitExt

using LsqFit
using Tortuosity
using PrecompileTools: @setup_workload, @compile_workload, workload_enabled

Tortuosity._curve_fit(model, xdata, ydata, p0) = LsqFit.curve_fit(model, xdata, ydata, p0)

Tortuosity._stderror(fit) = LsqFit.stderror(fit)

# The effective-diffusivity fit used to sit in the CPU workload in
# src/Tortuosity.jl; it moved here with LsqFit so users who do load LsqFit still
# get a compiled first call. `workload_enabled(Tortuosity)` rather than the
# extension's own preference: `set_preferences!` refuses an extension UUID, so
# only the parent package's `precompile_workload` preference can switch this
# off. On by default.
@setup_workload begin
    if workload_enabled(Tortuosity)
        img = ones(Bool, 12, 12, 12)
        @compile_workload begin
            prob = Tortuosity.TransientDiffusionProblem(
                img; axis=:z, bc_inlet=1, bc_outlet=0
            )
            tsol = Tortuosity.solve(prob, Tortuosity.ROCK4(); saveat=0.1, tspan=(0.0, 0.2))
            Tortuosity.fit_effective_diffusivity(tsol, prob, :mass)
        end
    end
end

end
