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

Tortuosity._free!(x::CuArray) = CUDA.unsafe_free!(x)

# --- Fast path: wrap PortableSparseCSC as CuSparseMatrixCSC for CUSPARSE SpMV ---
# CUSPARSE expects Int32 indices. Wrapping is cheap (just stores pointers), but
# within a Krylov solve `mul!` is called hundreds of times so even cheap
# allocations accumulate, so we cache the wrapper in `A._cache`.
#
# Freshness is the mutators' responsibility: every routine that reassigns
# `A.colptr`, `A.rowval`, or `A.nzval` calls `_invalidate_cache!` first, so a
# wrapper found here always describes the current buffers. Comparing pointers
# instead would be unsound — a buffer released back to the CUDA pool can be
# handed straight back out at the same address, and the wrapper would then pass
# the check while describing an array of a different length.

@inline function _as_cusparse(
    A::PortableSparseCSC{Tv,Int32,V,Vi}
) where {Tv,V<:CuVector,Vi<:CuVector{Int32}}
    cached = A._cache[]
    cached isa CUDA.CUSPARSE.CuSparseMatrixCSC{Tv,Int32} && return cached
    wrapped = CUDA.CUSPARSE.CuSparseMatrixCSC{Tv,Int32}(
        A.colptr, A.rowval, A.nzval, (A.m, A.n)
    )
    A._cache[] = wrapped
    return wrapped
end

# Fallback when the index type is not Int32: CUSPARSE needs Int32, so `colptr`
# and `rowval` are converted. Cached like the fast path — uncached this
# reconverted both index arrays on *every* `mul!`, several GB per Krylov
# iteration on a large matrix, which reads as an unexplained slowdown rather
# than as an error. Construct with Int32 indices to skip the conversion.
#
# Unlike the fast path this caches copies of the index arrays, so it relies on
# `rowval`/`colptr` never being edited in place — the mutators here either only
# touch `nzval` or reassign, and reassignment invalidates the cache.
function _as_cusparse(
    A::PortableSparseCSC{Tv,Ti,V,Vi}
) where {Tv,Ti,V<:CuVector,Vi<:CuVector}
    cached = A._cache[]
    cached isa CUDA.CUSPARSE.CuSparseMatrixCSC{Tv,Int32} && return cached
    wrapped = CUDA.CUSPARSE.CuSparseMatrixCSC{Tv,Int32}(
        convert(CuVector{Int32}, A.colptr), convert(CuVector{Int32}, A.rowval),
        A.nzval, (A.m, A.n),
    )
    A._cache[] = wrapped
    return wrapped
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
