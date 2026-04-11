# KernelAbstractions-based sparse matrix kernels for PortableSparseCSC operations.
import SparseArrays: dropzeros!

using KernelAbstractions
using Atomix

# --- Diagonal operations ---

"""
    set_diag_kernel!(nzVal, rowVal, colPtr, vals, N_diag)

KA kernel: set existing non-zero diagonal elements in a CSC sparse matrix.
Each thread handles one column/diagonal index `k` and searches for `rowVal[idx] == k`.
"""
@kernel function set_diag_kernel!(nzVal, @Const(rowVal), @Const(colPtr), @Const(vals), N_diag)
    k = @index(Global)
    if k <= N_diag && k > 0
        @inbounds idx_start = colPtr[k]
        @inbounds idx_end = colPtr[k + 1] - 1
        for idx in idx_start:idx_end
            @inbounds if rowVal[idx] == k
                @inbounds nzVal[idx] = vals[k]
                break
            end
        end
    end
end

"""
    set_diag!(A::PortableSparseCSC, vals)

Set the values of *existing* non-zero diagonal elements of `A` in place.
"""
function set_diag!(A::PortableSparseCSC{Tv}, vals::AbstractVector) where {Tv}
    N_diag = min(A.m, A.n)
    length(vals) != N_diag && throw(
        DimensionMismatch("Length of vals ($(length(vals))) must match min(size(A)...) ($N_diag)")
    )
    # Ensure vals is on the same backend as A and has element type Tv
    cu_vals = if vals isa typeof(A.nzval) && eltype(vals) === Tv
        vals
    else
        v = similar(A.nzval, Tv, N_diag)
        copyto!(v, Tv.(vals))
        v
    end

    backend = get_backend(A.nzval)
    wg = min(N_diag, 256)
    set_diag_kernel!(backend, wg)(A.nzval, A.rowval, A.colptr, cu_vals, N_diag; ndrange=N_diag)
    return nothing
end

"""
    get_diag_kernel!(diag_vals, nzVal, rowVal, colPtr, N_diag)

KA kernel: extract diagonal elements from a CSC sparse matrix. Each thread
unconditionally writes its result slot (so the caller does not need fill!),
using a local accumulator to avoid an unnecessary zero write to global memory.
"""
@kernel function get_diag_kernel!(diag_vals, @Const(nzVal), @Const(rowVal), @Const(colPtr), N_diag)
    k = @index(Global)
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
end

"""
    get_diag(A::PortableSparseCSC) -> AbstractVector

Extract the diagonal elements of `A`. Structurally absent diagonal entries are zero.
"""
function get_diag(A::PortableSparseCSC{Tv}) where {Tv}
    N_diag = min(A.m, A.n)
    N_diag == 0 && return similar(A.nzval, Tv, 0)
    # Kernel writes unconditionally, so we don't need fill!
    diag_vals = similar(A.nzval, Tv, N_diag)
    backend = get_backend(A.nzval)
    wg = min(N_diag, 256)
    get_diag_kernel!(backend, wg)(diag_vals, A.nzval, A.rowval, A.colptr, N_diag; ndrange=N_diag)
    # No explicit sync — downstream consumers (or CUDA.@sync wrappers) handle ordering
    return diag_vals
end

# --- Row/column zeroing ---

"""
    zero_rows_kernel!(nzVal, rowVal, is_target_row, nnz_val)

KA kernel: zero out nonzero values whose row index is flagged in `is_target_row`.
"""
@kernel function zero_rows_kernel!(nzVal, @Const(rowVal), @Const(is_target_row), nnz_val)
    k = @index(Global)
    if k <= nnz_val && k > 0
        @inbounds r = rowVal[k]
        if r > 0 && r <= length(is_target_row)
            @inbounds if is_target_row[r]
                @inbounds nzVal[k] = zero(eltype(nzVal))
            end
        end
    end
end

"""
    zero_cols_kernel!(nzVal, colPtr, target_cols, num_target_cols)

KA kernel: zero out all nonzero values in the specified target columns.
"""
@kernel function zero_cols_kernel!(nzVal, @Const(colPtr), @Const(target_cols), num_target_cols)
    idx_in_target = @index(Global)
    if idx_in_target <= num_target_cols && idx_in_target > 0
        @inbounds target_c = target_cols[idx_in_target]
        @inbounds col_start = colPtr[target_c]
        @inbounds col_end = colPtr[target_c + 1] - 1
        for k in col_start:col_end
            if k > 0 && k <= length(nzVal)
                @inbounds nzVal[k] = zero(eltype(nzVal))
            end
        end
    end
end

"""
    zero_rows_cols!(A::PortableSparseCSC, idxs)

Zero out all entries `A[i, j]` where `i in idxs` or `j in idxs`, in place.
"""
function zero_rows_cols!(A::PortableSparseCSC, idxs::AbstractVector{<:Integer})
    num_rows, num_cols = size(A)
    nnz_val = nnz(A)
    (nnz_val == 0 || isempty(idxs)) && return nothing

    Ti = eltype(A.rowval)
    idxs_Ti = Ti.(idxs)
    valid_rows = filter(i -> 1 <= i <= num_rows, idxs_Ti)
    valid_cols = filter(i -> 1 <= i <= num_cols, idxs_Ti)

    backend = get_backend(A.nzval)

    # Zero rows
    unique_rows = unique(valid_rows)
    if !isempty(unique_rows)
        is_target_row = fill!(similar(A.nzval, Bool, num_rows), false)
        gpu_rows = similar(A.rowval, length(unique_rows))
        copyto!(gpu_rows, unique_rows)
        is_target_row[gpu_rows] .= true

        zero_rows_kernel!(backend)(A.nzval, A.rowval, is_target_row, nnz_val; ndrange=nnz_val)
        KernelAbstractions.synchronize(backend)
    end

    # Zero columns
    unique_cols = unique(valid_cols)
    if !isempty(unique_cols)
        gpu_cols = similar(A.rowval, length(unique_cols))
        copyto!(gpu_cols, unique_cols)
        num_tc = length(unique_cols)
        zero_cols_kernel!(backend)(A.nzval, A.colptr, gpu_cols, num_tc; ndrange=num_tc)
        KernelAbstractions.synchronize(backend)
    end

    return nothing
end

"""
    zero_rows!(A::PortableSparseCSC, rows)

Zero out all entries in the specified `rows` of `A` in place, then drop
structural zeros.
"""
function zero_rows!(A::PortableSparseCSC, rows::AbstractVector{<:Integer})
    num_rows, _ = size(A)
    nnz_val = nnz(A)
    (nnz_val == 0 || isempty(rows)) && return nothing

    Ti = eltype(A.rowval)
    rows_Ti = Ti.(rows)
    valid_rows = unique(filter(i -> 1 <= i <= num_rows, rows_Ti))
    isempty(valid_rows) && return nothing

    backend = get_backend(A.nzval)
    is_target_row = fill!(similar(A.nzval, Bool, num_rows), false)
    gpu_rows = similar(A.rowval, length(valid_rows))
    copyto!(gpu_rows, valid_rows)
    is_target_row[gpu_rows] .= true

    zero_rows_kernel!(backend)(A.nzval, A.rowval, is_target_row, nnz_val; ndrange=nnz_val)
    KernelAbstractions.synchronize(backend)

    dropzeros!(A)
    return nothing
end

# --- Sparse compaction (dropzeros) ---

"""
    compact_and_count_kernel!(new_nzVal, new_rowVal, new_col_counts,
                              nzVal_old, rowVal_old, colPtr_old,
                              flags, scan_output, nnz_old)

KA kernel: compact nonzero values (flagged for retention) into new arrays
and atomically count entries per column for CSC `colPtr` reconstruction.
"""
@kernel function compact_and_count_kernel!(
    new_nzVal, new_rowVal, new_col_counts,
    @Const(nzVal_old), @Const(rowVal_old), @Const(colPtr_old),
    @Const(flags), @Const(scan_output), nnz_old,
)
    k = @index(Global)
    if k <= nnz_old && k > 0
        @inbounds if flags[k]
            @inbounds new_idx = scan_output[k] + 1
            @inbounds val = nzVal_old[k]
            @inbounds row = rowVal_old[k]
            if new_idx > 0 && new_idx <= length(new_nzVal)
                @inbounds new_nzVal[new_idx] = val
                @inbounds new_rowVal[new_idx] = row
            end
            @inbounds c = searchsortedlast(colPtr_old, k)
            if c > 0 && c <= length(new_col_counts)
                Atomix.@atomic new_col_counts[c] += 1
            end
        end
    end
end

_drop_tol(::Type{T}) where {T<:AbstractFloat} = eps(real(T))
_drop_tol(::Type{Complex{T}}) where {T<:AbstractFloat} = eps(real(T))
_drop_tol(::Type{T}) where {T} = zero(T)

"""
    dropzeros!(A::PortableSparseCSC; tol=_drop_tol(Tv))

Remove explicit zeros (values with `abs(v) <= tol`) from `A` by rebuilding
the CSC structure. Modifies `A` in place.
"""
function dropzeros!(A::PortableSparseCSC{Tv}; tol=_drop_tol(Tv)) where {Tv}
    nnz_old = nnz(A)
    _, num_cols = size(A)
    nnz_old == 0 && return nothing

    Ti = eltype(A.rowval)
    backend = get_backend(A.nzval)

    # Phase 1: flag elements to keep
    flags = abs.(A.nzval) .> tol
    flags_Ti = similar(A.rowval, length(flags))
    flags_Ti .= Ti.(flags)
    scan_inclusive = accumulate(+, flags_Ti)

    nnz_new = nnz_old > 0 ? Int(maximum(scan_inclusive)) : 0

    # Nothing to drop
    if nnz_new == nnz_old
        return nothing
    end

    # All zeros
    if nnz_new == 0
        A.nzval = similar(A.nzval, Tv, 0)
        A.rowval = similar(A.rowval, Ti, 0)
        A.colptr = fill!(similar(A.colptr, num_cols + 1), one(Ti))
        return nothing
    end

    # Exclusive scan for kernel indexing
    scan_output = fill!(similar(A.rowval, nnz_old), zero(Ti))
    scan_output_view = @view scan_output[2:end]
    scan_inclusive_view = @view scan_inclusive[1:(end - 1)]
    copyto!(scan_output_view, scan_inclusive_view)

    # Phase 2: allocate outputs
    new_nzVal = similar(A.nzval, Tv, nnz_new)
    new_rowVal = similar(A.rowval, Ti, nnz_new)
    new_col_counts = fill!(similar(A.rowval, num_cols), zero(Ti))

    # Phase 3: compact and count
    compact_and_count_kernel!(backend)(
        new_nzVal, new_rowVal, new_col_counts,
        A.nzval, A.rowval, A.colptr,
        flags, scan_output, nnz_old; ndrange=nnz_old,
    )
    KernelAbstractions.synchronize(backend)

    # Phase 4: build new colptr
    new_colPtr = similar(A.colptr, num_cols + 1)
    inclusive_scan_counts = accumulate(+, new_col_counts)
    fill_val = one(Ti)
    new_colPtr_view1 = @view new_colPtr[1:1]
    fill!(new_colPtr_view1, fill_val)
    if num_cols > 0
        new_colPtr_view2 = @view new_colPtr[2:end]
        new_colPtr_view2 .= inclusive_scan_counts .+ one(Ti)
    end

    # Phase 5: update A in place
    A.nzval = new_nzVal
    A.rowval = new_rowVal
    A.colptr = new_colPtr
    KernelAbstractions.synchronize(backend)

    return nothing
end
