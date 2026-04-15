# Thin extension: registers CUDA backend for GPU auto-detection +
# CUSPARSE fast-paths for operations that benefit from vendor libraries.
module TortuosityCUDAExt

using CUDA
using CUDA.CUSPARSE
using Tortuosity
using Tortuosity: PortableSparseCSC
using KernelAbstractions
using LinearAlgebra
using SparseArrays
using PrecompileTools: @setup_workload, @compile_workload

function __init__()
    if CUDA.functional()
        Tortuosity._preferred_gpu_backend[] = CUDABackend()
        Tortuosity._gpu_adapt[] = CUDA.cu
    end
end

Tortuosity._on_gpu(::CuArray) = true
Tortuosity._on_gpu(::CUDA.CUSPARSE.CuSparseMatrixCSC) = true

# --- Fast path: wrap PortableSparseCSC as CuSparseMatrixCSC for CUSPARSE SpMV ---
# CUSPARSE expects Int32 indices. Wrapping is cheap (just stores pointers), but
# within a Krylov solve `mul!` is called hundreds of times so even cheap
# allocations accumulate. We cache the wrapper in `A._cache` and invalidate via
# a pointer check — if any of `A`'s underlying vectors has been reassigned
# (e.g. by `dropzeros!`) the cached wrapper's fields point to stale buffers
# and we rebuild.

@inline function _as_cusparse(
    A::PortableSparseCSC{Tv,Int32,V,Vi}
) where {Tv,V<:CuVector,Vi<:CuVector{Int32}}
    cached = A._cache[]
    if cached isa CUDA.CUSPARSE.CuSparseMatrixCSC{Tv,Int32} &&
       pointer(cached.nzVal) == pointer(A.nzval) &&
       pointer(cached.rowVal) == pointer(A.rowval) &&
       pointer(cached.colPtr) == pointer(A.colptr)
        return cached
    end
    wrapped = CUDA.CUSPARSE.CuSparseMatrixCSC{Tv,Int32}(
        A.colptr, A.rowval, A.nzval, (A.m, A.n)
    )
    A._cache[] = wrapped
    return wrapped
end

# Fallback when index type is not Int32 — convert. This path allocates so should
# be avoided in the hot loop by constructing PortableSparseCSC with Int32 indices.
# Not cached: this path is only hit on misconfiguration.
function _as_cusparse(
    A::PortableSparseCSC{Tv,Ti,V,Vi}
) where {Tv,Ti,V<:CuVector,Vi<:CuVector}
    colptr32 = convert(CuVector{Int32}, A.colptr)
    rowval32 = convert(CuVector{Int32}, A.rowval)
    return CUDA.CUSPARSE.CuSparseMatrixCSC{Tv,Int32}(
        colptr32, rowval32, A.nzval, (A.m, A.n)
    )
end

# CUSPARSE-accelerated mul! for PortableSparseCSC backed by CuVector storage
function LinearAlgebra.mul!(
    y::CuVector, A::PortableSparseCSC{Tv,Ti,V,Vi}, x::CuVector
) where {Tv,Ti,V<:CuVector,Vi<:CuVector}
    return mul!(y, _as_cusparse(A), x)
end

# 5-argument mul!(y, A, x, alpha, beta) — used by some Krylov solvers
function LinearAlgebra.mul!(
    y::CuVector, A::PortableSparseCSC{Tv,Ti,V,Vi}, x::CuVector,
    alpha::Number, beta::Number,
) where {Tv,Ti,V<:CuVector,Vi<:CuVector}
    return mul!(y, _as_cusparse(A), x, alpha, beta)
end

# Mirror the CPU precompile workload in src/Tortuosity.jl for the CUDA GPU path.
# Only runs when a CUDA device is actually present at extension-precompile time;
# on machines without a GPU it's a no-op and users pay full TTFX on first solve.
# Note: `__init__` hasn't run yet during precompile, so we register the backend
# refs manually inside the workload and restore them after.
@setup_workload begin
    if CUDA.functional()
        img = ones(Bool, 12, 12, 12)
        @compile_workload begin
            Tortuosity._preferred_gpu_backend[] = CUDABackend()
            Tortuosity._gpu_adapt[] = CUDA.cu
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
