# Portable sparse CSC matrix type that works with any array backend (CPU, CUDA, Metal, AMD).
using KernelAbstractions
using Atomix

"""
    PortableSparseCSC{T,Ti,V,Vi} <: AbstractMatrix{T}

Backend-agnostic sparse matrix in Compressed Sparse Column (CSC) format.
Works with any array backend — `Vector` (CPU), `CuVector` (CUDA),
`MtlVector` (Metal), `ROCVector` (AMD) — through duck typing.

Implements `mul!(y, A, x)` via a KA SpMV kernel, enabling use with
Krylov.jl and LinearSolve.jl solvers.
"""
mutable struct PortableSparseCSC{
    T,Ti<:Integer,V<:AbstractVector{T},Vi<:AbstractVector{Ti}
} <: AbstractMatrix{T}
    m::Int
    n::Int
    colptr::Vi
    rowval::Vi
    nzval::V
end

Base.size(A::PortableSparseCSC) = (A.m, A.n)
SparseArrays.nnz(A::PortableSparseCSC) = length(A.nzval)
SparseArrays.nonzeros(A::PortableSparseCSC) = A.nzval
SparseArrays.rowvals(A::PortableSparseCSC) = A.rowval
SparseArrays.getcolptr(A::PortableSparseCSC) = A.colptr

function Base.getindex(::PortableSparseCSC, ::Integer, ::Integer)
    error("Scalar indexing not supported for PortableSparseCSC; use mul! for SpMV")
end

# --- SpMV kernel ---

@kernel function _spmv_kernel!(
    y, @Const(colptr), @Const(rowval), @Const(nzval), @Const(x), n
)
    j = @index(Global)
    if j <= n
        @inbounds for idx in colptr[j]:(colptr[j + 1] - 1)
            r = rowval[idx]
            v = nzval[idx] * x[j]
            Atomix.@atomic y[r] += v
        end
    end
end

function LinearAlgebra.mul!(
    y::AbstractVector, A::PortableSparseCSC, x::AbstractVector
)
    fill!(y, zero(eltype(y)))
    n = A.n
    if n > 0 && nnz(A) > 0
        backend = get_backend(A.nzval)
        _spmv_kernel!(backend)(y, A.colptr, A.rowval, A.nzval, x, n; ndrange=n)
        KernelAbstractions.synchronize(backend)
    end
    return y
end

function Base.:*(A::PortableSparseCSC, x::AbstractVector)
    T = promote_type(eltype(A), eltype(x))
    y = fill!(similar(A.nzval, T, A.m), zero(T))
    return mul!(y, A, x)
end

# --- Laplacian: L = D - A in a single pass ---
# D is the degree matrix (diagonal of row sums), A is the adjacency matrix.
# Assumes A has no self-loops (no diagonal entries), which holds for graph Laplacians.

@kernel function _laplacian_colptr_kernel!(L_colptr, @Const(A_colptr), n)
    j = @index(Global)
    if j <= n + 1
        # Each column gains one entry (the diagonal), so offset by (j - 1)
        @inbounds L_colptr[j] = A_colptr[j] + (j - 1)
    end
end

@kernel function _laplacian_entries_kernel!(
    L_rowval, L_nzval, @Const(L_colptr),
    @Const(A_rowval), @Const(A_nzval), @Const(A_colptr),
    @Const(degrees), n,
)
    j = @index(Global)
    if j <= n
        @inbounds A_start = A_colptr[j]
        @inbounds A_end = A_colptr[j + 1] - 1
        @inbounds L_pos = L_colptr[j]

        offset = 0
        diag_inserted = false

        for idx in A_start:A_end
            @inbounds row = A_rowval[idx]

            if !diag_inserted && row >= j
                if row == j
                    # Self-loop (rare): L[j,j] = degree[j] - A[j,j]
                    @inbounds L_rowval[L_pos + offset] = j
                    @inbounds L_nzval[L_pos + offset] = degrees[j] - A_nzval[idx]
                    offset += 1
                    diag_inserted = true
                    continue
                else
                    # Insert diagonal before this entry
                    @inbounds L_rowval[L_pos + offset] = j
                    @inbounds L_nzval[L_pos + offset] = degrees[j]
                    offset += 1
                    diag_inserted = true
                end
            end

            # Off-diagonal entry: negate
            @inbounds L_rowval[L_pos + offset] = row
            @inbounds L_nzval[L_pos + offset] = -A_nzval[idx]
            offset += 1
        end

        # Diagonal not yet inserted (all row indices < j)
        if !diag_inserted
            @inbounds L_rowval[L_pos + offset] = j
            @inbounds L_nzval[L_pos + offset] = degrees[j]
        end
    end
end

"""
    laplacian(am::PortableSparseCSC)

Compute the graph Laplacian `L = D - A` where `D = diag(A * 1)`.
Builds `L` in a single pass using KA kernels — no intermediate matrices.
"""
function laplacian(am::PortableSparseCSC{T}) where {T}
    m, n = size(am)
    @assert m == n "Adjacency matrix must be square"

    # Row sums (degrees) via SpMV: degrees = A * ones(n)
    ones_v = fill!(similar(am.nzval, n), one(T))
    degrees = fill!(similar(am.nzval, m), zero(T))
    mul!(degrees, am, ones_v)

    # L has nnz(A) + n entries (one new diagonal per column)
    nnz_L = nnz(am) + n
    L_colptr = similar(am.colptr, n + 1)
    L_rowval = similar(am.rowval, nnz_L)
    L_nzval = similar(am.nzval, nnz_L)

    backend = get_backend(am.nzval)

    _laplacian_colptr_kernel!(backend)(L_colptr, am.colptr, n; ndrange=n + 1)
    KernelAbstractions.synchronize(backend)

    _laplacian_entries_kernel!(backend)(
        L_rowval, L_nzval, L_colptr,
        am.rowval, am.nzval, am.colptr,
        degrees, n; ndrange=n,
    )
    KernelAbstractions.synchronize(backend)

    return PortableSparseCSC(m, n, L_colptr, L_rowval, L_nzval)
end
