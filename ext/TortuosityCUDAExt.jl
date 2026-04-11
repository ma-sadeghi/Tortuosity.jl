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

function __init__()
    if CUDA.functional()
        Tortuosity._preferred_gpu_backend[] = CUDABackend()
        Tortuosity._gpu_adapt[] = CUDA.cu
    end
end

Tortuosity._on_gpu(::CuArray) = true
Tortuosity._on_gpu(::CUDA.CUSPARSE.CuSparseMatrixCSC) = true

# --- Fast path: wrap PortableSparseCSC as CuSparseMatrixCSC for CUSPARSE SpMV ---
# CUSPARSE expects Int32 indices. Constructing the wrapper is cheap (just
# stores pointers); the cost is dominated by SpMV itself.

@inline function _as_cusparse(
    A::PortableSparseCSC{Tv,Int32,V,Vi}
) where {Tv,V<:CuVector,Vi<:CuVector{Int32}}
    return CUDA.CUSPARSE.CuSparseMatrixCSC{Tv,Int32}(
        A.colptr, A.rowval, A.nzval, (A.m, A.n)
    )
end

# Fallback when index type is not Int32 — convert. This path allocates so should
# be avoided in the hot loop by constructing PortableSparseCSC with Int32 indices.
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

# --- get_diag fast path: raw @cuda kernel avoids KA wrapper overhead ---
function _get_diag_cuda_kernel!(diag_vals, nzVal, rowVal, colPtr, N_diag)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if k <= N_diag && k > 0
        @inbounds idx_start = colPtr[k]
        @inbounds idx_end = colPtr[k + 1] - 1
        result = zero(eltype(diag_vals))
        @inbounds for idx in idx_start:idx_end
            if rowVal[idx] == k
                result = nzVal[idx]
                break
            end
        end
        @inbounds diag_vals[k] = result
    end
    return nothing
end

function Tortuosity.get_diag(A::PortableSparseCSC{Tv,Ti,V,Vi}) where {Tv,Ti,V<:CuVector,Vi<:CuVector}
    N_diag = min(A.m, A.n)
    N_diag == 0 && return similar(A.nzval, Tv, 0)
    diag_vals = similar(A.nzval, Tv, N_diag)
    threads = min(N_diag, 256)
    blocks = cld(N_diag, threads)
    @cuda threads=threads blocks=blocks _get_diag_cuda_kernel!(
        diag_vals, A.nzval, A.rowval, A.colptr, N_diag
    )
    return diag_vals
end

end
