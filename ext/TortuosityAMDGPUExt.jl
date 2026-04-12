# Thin extension: registers AMDGPU backend for GPU auto-detection.
module TortuosityAMDGPUExt

using AMDGPU
using Tortuosity
using KernelAbstractions

function __init__()
    if AMDGPU.functional()
        Tortuosity._preferred_gpu_backend[] = ROCBackend()
        Tortuosity._gpu_adapt[] = AMDGPU.roc
    end
end

Tortuosity._on_gpu(::ROCArray) = true

end
