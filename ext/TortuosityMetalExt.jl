# Thin extension: registers Metal backend for GPU auto-detection.
module TortuosityMetalExt

using Metal
using Tortuosity
using KernelAbstractions

function __init__()
    if Metal.functional()
        Tortuosity._preferred_gpu_backend[] = MetalBackend()
        Tortuosity._gpu_adapt[] = Metal.mtl
    end
end

Tortuosity._on_gpu(::MtlArray) = true

end
