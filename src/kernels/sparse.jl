import SparseArrays: dropzeros!  # Needed for extending for CUDA.CUSPARSE.CuSparseMatrixCSC

"""
    set_diag_kernel!(nzVal, rowVal, colPtr, vals, N_diag)

CUDA kernel to set the values of existing non-zero diagonal elements
in a CSC sparse matrix representation.
"""
function set_diag_kernel!(
    nzVal::CuDeviceVector{Tv},
    rowVal::CuDeviceVector{Ti},
    colPtr::CuDeviceVector{Ti},
    vals::CuDeviceVector{Tv},
    N_diag::Ti,
) where {Tv,Ti}
    k = Ti((blockIdx().x - 1) * blockDim().x + threadIdx().x) # This thread handles column/diagonal k (1-based)

    if k <= N_diag && k > 0 # Check if k is within the valid diagonal range
        # Get the range of indices in rowVal/nzVal for column k
        # Note: colPtr uses 1-based indexing in Julia/CUDA.jl
        @inbounds idx_start = colPtr[k]
        @inbounds idx_end = colPtr[k + 1] - 1 # End index is inclusive

        # Search within column k's entries for row k (the diagonal element)
        for idx in idx_start:idx_end
            @inbounds if rowVal[idx] == k
                @inbounds nzVal[idx] = vals[k] # Update the value if diagonal element exists
                break # Found the diagonal element for this column, move to next thread
            end
        end
    end
    return nothing
end

"""
    set_diag!(A::CUDA.CUSPARSE.CuSparseMatrixCSC{Tv, Ti}, vals::AbstractVector) where {Tv, Ti}

Efficiently sets the values of *existing* non-zero diagonal elements of a CuSparseMatrixCSC `A`.

The function modifies the `nzVal` array of `A` in place. Diagonal elements `A[k,k]`
that were originally zero will *not* be inserted, and their corresponding value in `vals`
will be ignored.

Arguments:
- A: The CuSparseMatrixCSC matrix on the GPU to modify.
- vals: A vector (CPU or GPU) containing the desired values for the diagonal.
        It must have length `min(size(A)...)`. If it's a CPU vector, it will be
        transferred to the GPU. Its element type will be converted to match `A`.

Returns:
- `nothing` (The matrix `A` is modified in place).
"""
function set_diag!(
    A::CUDA.CUSPARSE.CuSparseMatrixCSC{Tv,Ti}, vals::AbstractVector
) where {Tv,Ti}
    num_rows, num_cols = size(A)
    N_diag = min(num_rows, num_cols)

    if length(vals) != N_diag
        throw(
            DimensionMismatch(
                "Length of `vals` ($(length(vals))) must match min(size(A)...) ($N_diag)"
            ),
        )
    end

    # Ensure vals is a CuVector of the correct type
    # Convert element type first to avoid potential intermediate allocation of wrong type
    vals_typed = convert.(Tv, vals)
    cu_vals = CuArray(vals_typed) # Transfer to GPU if it wasn't already

    # Get pointers to the matrix data on GPU
    nzVal = A.nzVal
    rowVal = A.rowVal
    colPtr = A.colPtr

    # Determine kernel launch configuration
    # Launch one thread per diagonal element we need to check
    threads = min(N_diag, 256) # Common thread block size, capped by N_diag
    blocks = cld(N_diag, threads) # Ceiling division to ensure enough blocks

    # Launch the kernel
    CUDA.@sync @cuda threads = threads blocks = blocks set_diag_kernel!(
        nzVal, rowVal, colPtr, cu_vals, Ti(N_diag)
    )

    return nothing # Modified A in place
end

"""
    get_diag_kernel!(diag_vals, nzVal, rowVal, colPtr, N_diag)

CUDA kernel to extract diagonal elements from a CSC sparse matrix representation.
"""
function get_diag_kernel!(
    diag_vals::CuDeviceVector{Tv},
    nzVal::CuDeviceVector{Tv},
    rowVal::CuDeviceVector{Ti},
    colPtr::CuDeviceVector{Ti},
    N_diag::Ti,
) where {Tv,Ti}
    k = Ti((blockIdx().x - 1) * blockDim().x + threadIdx().x) # This thread handles column/diagonal k (1-based)

    if k <= N_diag && k > 0 # Check if k is within the valid diagonal range
        # Get the range of indices in rowVal/nzVal for column k
        @inbounds idx_start = colPtr[k]
        @inbounds idx_end = colPtr[k + 1] - 1 # End index is inclusive

        # Search within column k's entries for row k (the diagonal element)
        for idx in idx_start:idx_end
            @inbounds if rowVal[idx] == k
                @inbounds diag_vals[k] = nzVal[idx] # Store the found diagonal value
                break # Found the diagonal element for this column, move to next thread
            end
        end
        # If the loop completes without finding rowVal[idx] == k,
        # diag_vals[k] remains 0 (due to pre-initialization).
    end
    return nothing
end

"""
    get_diag(A::CUDA.CUSPARSE.CuSparseMatrixCSC{Tv, Ti}) where {Tv, Ti}

Efficiently extracts the diagonal elements of a CuSparseMatrixCSC `A` on the GPU.

Returns a `CuVector{Tv}` containing the diagonal values. If a diagonal element `A[k,k]`
is not explicitly stored in the sparse matrix (i.e., it's zero), the corresponding
value in the output vector will be zero.

Arguments:
- A: The CuSparseMatrixCSC matrix on the GPU.

Returns:
- CuVector{Tv}: A vector containing the diagonal elements of `A`.
"""
function get_diag(A::CUDA.CUSPARSE.CuSparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    num_rows, num_cols = size(A)
    N_diag = min(num_rows, num_cols)

    if N_diag == 0
        return CUDA.zeros(Tv, 0) # Return empty CuVector if matrix is empty or has 0 rows/cols
    end

    # Get pointers to the matrix data on GPU
    nzVal = A.nzVal
    rowVal = A.rowVal
    colPtr = A.colPtr

    # Allocate output vector on GPU, initialized to zeros
    diag_vals = CUDA.zeros(Tv, N_diag)

    # Determine kernel launch configuration
    # Launch one thread per diagonal element we need to extract
    threads = min(N_diag, 256) # Common thread block size, capped by N_diag
    blocks = cld(N_diag, threads) # Ceiling division to ensure enough blocks

    # Launch the kernel
    CUDA.@sync @cuda threads = threads blocks = blocks get_diag_kernel!(
        diag_vals, nzVal, rowVal, colPtr, Ti(N_diag)
    )

    return diag_vals
end

# Kernel 1: Zeroes elements based on row index lookup table
function zero_rows_kernel!(
    nzVal::CuDeviceVector{Tv},
    rowVal::CuDeviceVector{Ti},
    is_target_row::CuDeviceVector{Bool},
    nnz::Ti,
) where {Tv,Ti}
    k = Ti((blockIdx().x - 1) * blockDim().x + threadIdx().x) # Global index for k-th non-zero element

    if k <= nnz && k > 0
        @inbounds r = rowVal[k]
        # Check row index bounds before accessing lookup table
        if r > 0 && r <= length(is_target_row)
            @inbounds if is_target_row[r]
                @inbounds nzVal[k] = zero(Tv)
            end
        end
    end
    return nothing
end

# Kernel 2: Zeroes elements based on target column indices
function zero_cols_kernel!(
    nzVal::CuDeviceVector{Tv},
    colPtr::CuDeviceVector{Ti},
    target_cols::CuDeviceVector{Ti}, # Vector of unique column indices to zero
    num_target_cols::Ti,
) where {Tv,Ti}
    idx_in_target = Ti((blockIdx().x - 1) * blockDim().x + threadIdx().x) # Index into target_cols vector

    if idx_in_target <= num_target_cols && idx_in_target > 0
        @inbounds target_c = target_cols[idx_in_target] # The actual column index to zero

        # Get the range of indices in nzVal for this target column
        # Assuming target_c is valid (1 <= target_c <= num_cols)
        @inbounds col_start = colPtr[target_c]
        @inbounds col_end = colPtr[target_c + 1] - 1 # Inclusive end index

        # Loop through all elements in this column and zero them
        for k in col_start:col_end
            # Check k bounds just in case colPtr gives invalid range (should not happen in valid CSC)
            if k > 0 && k <= length(nzVal)
                @inbounds nzVal[k] = zero(Tv)
            end
        end
    end
    return nothing
end

"""
    zero_rows_cols!(A::CUDA.CUSPARSE.CuSparseMatrixCSC{Tv, Ti}, idxs::AbstractVector{<:Integer}) where {Tv, Ti}

Efficiently zeroes out all non-zero elements `A[i, j]` of a CuSparseMatrixCSC `A`
where the row index `i` is in `idxs` OR the column index `j` is in `idxs`.

The function modifies the `nzVal` array of `A` in place. Assumes the kernel
functions `zero_rows_kernel!` and `zero_cols_kernel!` are defined elsewhere.

Arguments:
- A: The CuSparseMatrixCSC matrix on the GPU to modify.
- idxs: A vector of integer indices. Rows `i` where `i ∈ idxs` and columns `j` where `j ∈ idxs` will be zeroed out.
        Indices out of matrix bounds are ignored.

Returns:
- `nothing` (The matrix `A` is modified in place).
"""
function zero_rows_cols!(
    A::CUDA.CUSPARSE.CuSparseMatrixCSC{Tv,Ti}, idxs::AbstractVector{<:Integer}
) where {Tv,Ti}
    num_rows, num_cols = size(A)
    nnz = length(A.nzVal)
    if nnz == 0 || isempty(idxs)
        return nothing # Nothing to do
    end

    # --- Preprocessing ---
    # Filter indices based on matrix dimensions and ensure unique indices on GPU
    # Convert idxs to the matrix's index type Ti early on
    idxs_Ti = convert.(Ti, idxs)

    valid_rows_idx = filter(i -> 1 <= i <= num_rows, idxs_Ti)
    valid_cols_idx = filter(i -> 1 <= i <= num_cols, idxs_Ti)

    # Use unique directly on filtered vectors before converting to CuArray
    cu_target_rows = CuArray(unique(valid_rows_idx))
    cu_target_cols = CuArray(unique(valid_cols_idx))

    num_target_rows = length(cu_target_rows)
    num_target_cols = length(cu_target_cols)

    # Get matrix data
    nzVal = A.nzVal
    rowVal = A.rowVal
    colPtr = A.colPtr

    # --- Pass 1: Zero Rows ---
    if num_target_rows > 0
        # Create boolean lookup table for rows
        is_target_row = CUDA.zeros(Bool, num_rows)
        # Check is needed before indexing if num_target_rows could be 0 after unique/filtering
        if num_target_rows > 0
            is_target_row[cu_target_rows] .= true # Efficiently mark target rows
        end

        threads_k1 = min(nnz, 256)
        blocks_k1 = cld(nnz, threads_k1)
        CUDA.@sync @cuda threads = threads_k1 blocks = blocks_k1 zero_rows_kernel!(
            nzVal, rowVal, is_target_row, Ti(nnz)
        )
        # Allow GC to collect lookup table
        is_target_row = nothing
        # CUDA.unsafe_free!(is_target_row) # Optional: explicit free
    end

    # --- Pass 2: Zero Columns ---
    if num_target_cols > 0
        threads_k2 = min(num_target_cols, 256)
        blocks_k2 = cld(num_target_cols, threads_k2)
        CUDA.@sync @cuda threads = threads_k2 blocks = blocks_k2 zero_cols_kernel!(
            nzVal, colPtr, cu_target_cols, Ti(num_target_cols)
        )
    end

    # Allow GC to collect temporary index arrays
    cu_target_rows = nothing
    cu_target_cols = nothing

    return nothing
end

# Combined kernel for compaction and calculating column counts for CSC rebuild
function compact_and_count_kernel!(
    new_nzVal::CuDeviceVector{Tv},
    new_rowVal::CuDeviceVector{Ti},
    new_col_counts::CuDeviceVector{Ti}, # Atomic counts per column
    nzVal_old::CuDeviceVector{Tv},
    rowVal_old::CuDeviceVector{Ti},
    colPtr_old::CuDeviceVector{Ti},
    flags::CuDeviceVector{Bool},        # flags[k] is true if nzVal_old[k] should be kept
    scan_output::CuDeviceVector{Ti},   # Exclusive scan of flags (Ti type)
    nnz_old::Ti,
) where {Tv,Ti}
    k = Ti((blockIdx().x - 1) * blockDim().x + threadIdx().x) # Global index for old element k (1-based)

    if k <= nnz_old && k > 0
        @inbounds if flags[k] # Check if this element should be kept
            # Calculate new index using exclusive scan result
            @inbounds new_idx = scan_output[k] + Ti(1) # 1-based index for new arrays

            # Copy data to compacted arrays
            @inbounds val = nzVal_old[k]
            @inbounds row = rowVal_old[k]
            # Ensure indices are within bounds before writing (robustness check)
            if new_idx > 0 && new_idx <= length(new_nzVal)
                @inbounds new_nzVal[new_idx] = val
                @inbounds new_rowVal[new_idx] = row
            end

            # Find the original column 'c' for element 'k'
            # searchsortedlast(colPtr, k) finds the largest index `c` such that colPtr[c] <= k.
            # This corresponds to the 1-based column index.
            # Assumes CUDA.searchsortedlast works efficiently on CuDeviceVector inside kernel.
            @inbounds c = searchsortedlast(colPtr_old, k)

            # Atomically increment the count for this column `c`
            # Ensure column index c is valid before atomic operation
            if c > 0 && c <= length(new_col_counts)
                CUDA.@atomic new_col_counts[c] += Ti(1)
                # else
                # Handle error or unexpected column index 'c' if necessary,
                # though valid k should yield valid c for well-formed CSC.
            end
        end
    end
    return nothing
end

"""
    dropzeros!(A::CUDA.CUSPARSE.CuSparseMatrixCSC{Tv, Ti}; tol=eps(real(Tv))) where {Tv, Ti}

Removes explicit zeros (or values close to zero within tolerance `tol`)
from a `CuSparseMatrixCSC` `A` by rebuilding its structure. Modifies `A` in place.

Arguments:
- A: The `CuSparseMatrixCSC` on the GPU to modify.
- tol: Tolerance for considering a value as zero. Defaults to `eps` of the real part of the value type. Values `v` where `abs(v) <= tol` are dropped.

Returns:
- `nothing` (The matrix `A` is modified in place).

Note: This operation is computationally intensive as it requires rebuilding
the sparse matrix structure (nzVal, rowVal, colPtr). Relies on CUDA.jl features like
scan (`accumulate`), atomics, and potentially `searchsortedlast` within kernels.
Ensure your CUDA.jl version supports these device-side functions efficiently.
"""
function dropzeros!(
    A::CUDA.CUSPARSE.CuSparseMatrixCSC{Tv,Ti}; tol=eps(real(Tv))
) where {Tv,Ti}
    nnz_old = A.nnz
    num_rows, num_cols = size(A)

    if nnz_old == 0
        return nothing # Nothing to drop
    end

    nzVal_old = A.nzVal
    rowVal_old = A.rowVal
    colPtr_old = A.colPtr

    # --- Phase 1: Compaction Prep & New NNZ ---
    # Create flags: true if element's absolute value > tol
    flags = abs.(nzVal_old) .> tol # Boolean CuVector

    # Calculate inclusive prefix sum of flags, converting flags to the index type Ti first
    # This gives the running count of elements to keep.
    scan_inclusive = CUDA.accumulate(+, convert(CuArray{Ti}, flags))

    # Get the total count of non-zeros to keep (the last element of the inclusive scan)
    nnz_new =
        (nnz_old > 0 && length(scan_inclusive) > 0) ? Ti(maximum(scan_inclusive)) : Ti(0)

    # Check if anything needs to be dropped
    if nnz_new == nnz_old
        # Clean up intermediate arrays even if no change
        flags = nothing
        scan_inclusive = nothing
        CUDA.synchronize()
        return nothing # No zeros (within tolerance) found
    end

    # Handle case where all elements become zero
    if nnz_new == 0
        A.nzVal = CUDA.zeros(Tv, 0)
        A.rowVal = CUDA.zeros(Ti, 0)
        A.colPtr = CUDA.ones(Ti, num_cols + 1) # Point all columns to index 1
        A.nnz = 0
        # Clean up intermediate arrays
        flags = nothing
        scan_inclusive = nothing
        CUDA.synchronize()
        return nothing
    end

    # Calculate exclusive scan needed for kernel indexing (scan_output[k] = count before k)
    scan_output = CUDA.zeros(Ti, nnz_old)      # Allocate space for exclusive scan result
    scan_output[2:end] .= scan_inclusive[1:(end - 1)]
    # scan_output[1] remains 0

    # --- Phase 2: Allocate Outputs ---
    new_nzVal = CUDA.CuArray{Tv}(undef, nnz_new)
    new_rowVal = CUDA.CuArray{Ti}(undef, nnz_new)
    new_col_counts = CUDA.zeros(Ti, num_cols) # For atomic increments

    # --- Phase 3: Combined Kernel (Compact nzVal/rowVal & Count Columns) ---
    threads_k1 = 256 # Typical block size
    blocks_k1 = cld(nnz_old, threads_k1)
    CUDA.@sync @cuda threads = threads_k1 blocks = blocks_k1 compact_and_count_kernel!(
        new_nzVal,
        new_rowVal,
        new_col_counts,
        nzVal_old,
        rowVal_old,
        colPtr_old,
        flags,
        scan_output,
        Ti(nnz_old),
    )

    # Free intermediate arrays no longer needed
    flags = nothing
    scan_output = nothing
    scan_inclusive = nothing # Already used to calculate scan_output

    # --- Phase 4: Calculate New colPtr from Counts ---
    new_colPtr = CUDA.CuArray{Ti}(undef, num_cols + 1)
    # Compute inclusive scan of counts to get the end+1 position for each column pointer
    inclusive_scan_counts = CUDA.accumulate(+, new_col_counts)
    new_colPtr[1:1] .= 1 # First pointer is always 1
    # Check if num_cols > 0 before indexing new_colPtr[2:end] and inclusive_scan_counts
    if num_cols > 0
        new_colPtr[2:end] .= inclusive_scan_counts .+ 1 # Add 1 for 1-based indexing start
    end

    # Free intermediate count arrays
    new_col_counts = nothing
    inclusive_scan_counts = nothing

    # --- Phase 5: Update Matrix A in place ---
    # Replace the fields of the mutable struct A
    A.nzVal = new_nzVal
    A.rowVal = new_rowVal
    A.colPtr = new_colPtr
    A.nnz = nnz_new # Update nnz count

    CUDA.synchronize() # Ensure all GPU work is done before function returns
    return nothing
end
