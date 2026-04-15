# Thin extension: registers AMDGPU backend for GPU auto-detection.
module TortuosityAMDGPUExt

using AMDGPU
using Tortuosity
using KernelAbstractions
using PrecompileTools: @setup_workload, @compile_workload

function __init__()
    if AMDGPU.functional()
        Tortuosity._preferred_gpu_backend[] = ROCBackend()
        Tortuosity._gpu_adapt[] = AMDGPU.roc
    end
end

Tortuosity._on_gpu(::ROCArray) = true

# Mirror the CPU precompile workload in src/Tortuosity.jl for the ROCm GPU path.
# Only runs when an AMDGPU device is actually present at extension-precompile
# time; on machines without a GPU it's a no-op and users pay full TTFX on
# first solve.
@setup_workload begin
    if AMDGPU.functional()
        img = ones(Bool, 12, 12, 12)
        @compile_workload begin
            Tortuosity._preferred_gpu_backend[] = ROCBackend()
            Tortuosity._gpu_adapt[] = AMDGPU.roc
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
