# Thin extension: registers Metal backend for GPU auto-detection.
module TortuosityMetalExt

using Metal
using Tortuosity
using KernelAbstractions
using PrecompileTools: @setup_workload, @compile_workload

function __init__()
    if Metal.functional()
        Tortuosity._preferred_gpu_backend[] = MetalBackend()
        Tortuosity._gpu_adapt[] = Metal.mtl
    end
end

Tortuosity._on_gpu(::MtlArray) = true

# Mirror the CPU precompile workload in src/Tortuosity.jl for the Metal GPU path.
# Only runs when a Metal device is actually present at extension-precompile time;
# on machines without a GPU it's a no-op and users pay full TTFX on first solve.
@setup_workload begin
    if Metal.functional()
        img = ones(Bool, 12, 12, 12)
        @compile_workload begin
            Tortuosity._preferred_gpu_backend[] = MetalBackend()
            Tortuosity._gpu_adapt[] = Metal.mtl
            try
                sim = Tortuosity.SteadyDiffusionProblem(img; axis=:x, gpu=true)
                Tortuosity.solve(sim.prob, Tortuosity.KrylovJL_CG())

                prob = Tortuosity.TransientDiffusionProblem(img; axis=:z, bc_inlet=1, bc_outlet=0, gpu=true)
                Tortuosity.solve(prob, Tortuosity.ROCK4(); saveat=0.1, tspan=(0.0, 0.2))
            finally
                Tortuosity._preferred_gpu_backend[] = nothing
                Tortuosity._gpu_adapt[] = identity
            end
        end
    end
end

end
