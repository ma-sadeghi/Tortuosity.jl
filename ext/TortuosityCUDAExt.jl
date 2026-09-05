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
Tortuosity._on_gpu(::CUDA.CUSPARSE.CuSparseMatrix) = true
@static if pkgversion(CUDA) >= v"5.4"
    Tortuosity._async_return_safe(::CuArray) = true
end
Tortuosity._steady_workgroup(a::CuArray) =
    size(a, 3) == 1 ? (64, 4, 1) : (32, 2, 2)
Tortuosity._precond_min_nodes(::CuArray) = 3_000
function Tortuosity._refinement_correction_reltol(::CUDABackend, n, nvoxels)
    small = n < Tortuosity.LOOSE_CORRECTION_MIN_NODES
    small && return Tortuosity.REFINEMENT_CORRECTION_RELTOL
    ε = n / nvoxels
    0.5 <= ε < 0.7 && return Tortuosity.MID_GPU_CORRECTION_RELTOL
    ε >= 0.9 && return Tortuosity.HIGH_GPU_CORRECTION_RELTOL
    return Tortuosity.REFINEMENT_CORRECTION_RELTOL
end

Tortuosity._free!(x::CuArray) = CUDA.unsafe_free!(x)

# --- Fast path: wrap PortableSparseCSC for CUSPARSE SpMV ---
# CUSPARSE indexes with either 32-bit or 64-bit integers — `cusparseCreateCsr`
# takes the index type as an argument, and CUDA.jl maps `Int64` to
# `CUSPARSE_INDEX_64I` — so both of the types `build_steady_system` produces are
# wrapped as they stand, with no conversion. Wrapping is cheap (just stores
# pointers), but within a Krylov solve `mul!` is called hundreds of times so even
# cheap allocations accumulate, so we cache the wrapper in `A._cache`.
#
# A matrix that its builder declared `symmetric` is wrapped as CSR rather than
# CSC. The three arrays are the same bytes read the other way round, and CSC
# read as CSR is `transpose(A)`, which for a symmetric matrix is `A` — so the
# product is unchanged. What changes is the kernel cuSPARSE picks: CSR SpMV
# gathers each output entry in one thread, while CSC SpMV has to scatter with
# atomics. Measured on an RTX PRO 5000 at 800^3 (1.75e9 nonzeros, Float32):
# 32.9 ms per CSC `mul!` against 28.0 ms per CSR one, and the CSR timings
# repeat to within 2 % where the CSC ones spread over 15 %.
#
# Freshness is the mutators' responsibility: every routine that touches `A`
# calls `_invalidate_cache!` first, so a wrapper found here always describes the
# current buffers and the symmetry claim behind its format still holds.
# Comparing pointers instead would be unsound — a buffer released back to the
# CUDA pool can be handed straight back out at the same address, and the wrapper
# would then pass the check while describing an array of a different length.

@inline function _as_cusparse(
    A::PortableSparseCSC{Tv,Ti,V,Vi}
) where {Tv,Ti<:Union{Int32,Int64},V<:CuVector,Vi<:CuVector{Ti}}
    cached = A._cache[]
    if A.symmetric
        cached isa CUDA.CUSPARSE.CuSparseMatrixCSR{Tv,Ti} && return cached
        wrapped = CUDA.CUSPARSE.CuSparseMatrixCSR{Tv,Ti}(
            A.colptr, A.rowval, A.nzval, (A.m, A.n)
        )
    else
        cached isa CUDA.CUSPARSE.CuSparseMatrixCSC{Tv,Ti} && return cached
        wrapped = CUDA.CUSPARSE.CuSparseMatrixCSC{Tv,Ti}(
            A.colptr, A.rowval, A.nzval, (A.m, A.n)
        )
    end
    A._cache[] = wrapped
    return wrapped
end

# Fallback for an index type CUSPARSE does not take at all — anything but Int32
# or Int64 — whose `colptr` and `rowval` are converted to Int32. Cached like the
# fast path: uncached this reconverted both index arrays on *every* `mul!`,
# several GB per Krylov iteration on a large matrix, which reads as an
# unexplained slowdown rather than as an error.
#
# The range check is what keeps the conversion honest. Narrowing an index that
# does not fit raises `InexactError` inside a device broadcast kernel, where
# there is nothing to catch it and the message names neither the matrix nor the
# reason; refuse on the host instead, before a kernel is launched.
#
# Unlike the fast path this caches copies of the index arrays, so it relies on
# `rowval`/`colptr` never being edited in place. `_invalidate_cache!` covers it:
# every mutator calls it, whether it reassigns the arrays or edits them.
function _as_cusparse(
    A::PortableSparseCSC{Tv,Ti,V,Vi}
) where {Tv,Ti,V<:CuVector,Vi<:CuVector}
    (max(A.m, A.n) <= typemax(Int32) && nnz(A) + 1 <= typemax(Int32)) || throw(ArgumentError(
        "a $(A.m)×$(A.n) `PortableSparseCSC{$(Tv),$(Ti)}` with $(nnz(A)) stored entries \
         needs more index range than CUSPARSE's 32-bit indexing offers, and $(Ti) is \
         not one of the types it indexes with directly (`Int32`, `Int64`); rebuild it \
         with `Int64` indices"
    ))
    cached = A._cache[]
    colptr32() = convert(CuVector{Int32}, A.colptr)
    rowval32() = convert(CuVector{Int32}, A.rowval)
    if A.symmetric
        cached isa CUDA.CUSPARSE.CuSparseMatrixCSR{Tv,Int32} && return cached
        wrapped = CUDA.CUSPARSE.CuSparseMatrixCSR{Tv,Int32}(
            colptr32(), rowval32(), A.nzval, (A.m, A.n)
        )
    else
        cached isa CUDA.CUSPARSE.CuSparseMatrixCSC{Tv,Int32} && return cached
        wrapped = CUDA.CUSPARSE.CuSparseMatrixCSC{Tv,Int32}(
            colptr32(), rowval32(), A.nzval, (A.m, A.n)
        )
    end
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
#
# The workload macros resolve the `precompile_workload` preference against the
# module they expand in, which here is the extension — and `set_preferences!`
# refuses an extension's UUID, so that preference is unreachable. Consulting
# `Tortuosity` explicitly is what lets
# `set_preferences!(Tortuosity, "precompile_workload" => false)` switch this
# workload off along with the CPU one. Enabled by default, as for users.
@setup_workload begin
    if Tortuosity._workload_enabled() && CUDA.functional()
        img = ones(Bool, 12, 12, 12)
        @compile_workload begin
            Tortuosity._preferred_gpu_backend[] = CUDABackend()
            Tortuosity._gpu_adapt[] = CUDA.cu
            try
                sim = Tortuosity.SteadyDiffusionProblem(img; axis=:x, gpu=true)
                sol = Tortuosity.solve(sim.prob, Tortuosity.KrylovJL_CG())
                Tortuosity.tortuosity(sol.u, sim)

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
