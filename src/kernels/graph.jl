# ================================================================
# Helper Kernel: Fill the idx array on the GPU
# ================================================================
function fill_idx_kernel!(idx_gpu, linear_indices, num_true)
    # Calculate the global linear index for this thread
    thread_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if thread_idx <= num_true
        # Get the linear index in the original image 'im'
        original_linear_idx = linear_indices[thread_idx]
        # Write the sequential index (1, 2, 3...) into idx_gpu
        # at the location corresponding to the true value in 'im'
        @inbounds idx_gpu[original_linear_idx] = thread_idx
    end
    return nothing
end

# ================================================================
# Kernel 1: Count Connections
# ================================================================
function count_connections_kernel!(d_conn_count, im_gpu, idx_gpu, nx, ny, nz)
    # Calculate the global linear index for the voxel this thread processes
    linear_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    # Map linear index back to 3D Cartesian index (adjust for thread counts > N)
    if linear_idx <= length(im_gpu)
        cid = CartesianIndices(im_gpu)[linear_idx]
        i, j, k = cid.I

        local_conn_count = 0
        # Only proceed if the current voxel is true
        if @inbounds im_gpu[i, j, k]
            # Check neighbors (ensure bounds checking)
            # Check k-1 (behind)
            if k > 1 && @inbounds im_gpu[i, j, k - 1]
                local_conn_count += 1
            end
            # Check j-1 (left)
            if j > 1 && @inbounds im_gpu[i, j - 1, k]
                local_conn_count += 1
            end
            # Check i-1 (up)
            if i > 1 && @inbounds im_gpu[i - 1, j, k]
                local_conn_count += 1
            end
            # Check i+1 (down) - No need to check for GPU kernel,
            # the corresponding neighbour will add the connection
            # Same applies for j+1 and k+1
            # --> Original logic adds each connection twice (once for each voxel)
            # --> We can optimize by only checking negative directions (k-1, j-1, i-1)
            # --> Let's stick to the original logic for direct porting first,
            #     then consider optimization. We will count each edge twice here.
            # Check i+1 (down)
            if i < nx && @inbounds im_gpu[i + 1, j, k]
                local_conn_count += 1
            end
            # Check j+1 (right)
            if j < ny && @inbounds im_gpu[i, j + 1, k]
                local_conn_count += 1
            end
            # Check k+1 (front)
            if k < nz && @inbounds im_gpu[i, j, k + 1]
                local_conn_count += 1
            end
        end

        # Atomically add the local count to the global counter if > 0
        if local_conn_count > 0
            CUDA.@atomic d_conn_count[1] += local_conn_count
            # Alternate for specific atomic ops:
            # CUDA.atomic_add!(pointer(d_conn_count), local_conn_count)
        end
    end
    return nothing
end

# ================================================================
# Kernel 2: Write Connections
# ================================================================
function write_connections_kernel!(conns_gpu, d_row_counter, im_gpu, idx_gpu, nx, ny, nz)
    # Calculate the global linear index for the voxel this thread processes
    linear_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    # Map linear index back to 3D Cartesian index (adjust for thread counts > N)
    if linear_idx <= length(im_gpu)
        cid = CartesianIndices(im_gpu)[linear_idx]
        i, j, k = cid.I

        # Only proceed if the current voxel is true
        if @inbounds im_gpu[i, j, k]
            current_val_idx = @inbounds idx_gpu[i, j, k] # Get index of current voxel

            # --- Check Neighbors ---
            # Check k-1 (behind)
            if k > 1 && @inbounds im_gpu[i, j, k - 1]
                neighbor_val_idx = @inbounds idx_gpu[i, j, k - 1]
                # Atomically get next row index and increment the counter
                row = CUDA.atomic_add!(pointer(d_row_counter), 1) + 1 # atomic_add returns old value
                @inbounds conns_gpu[row, 1] = neighbor_val_idx
                @inbounds conns_gpu[row, 2] = current_val_idx
            end
            # Check j-1 (left)
            if j > 1 && @inbounds im_gpu[i, j - 1, k]
                neighbor_val_idx = @inbounds idx_gpu[i, j - 1, k]
                row = CUDA.atomic_add!(pointer(d_row_counter), 1) + 1
                @inbounds conns_gpu[row, 1] = neighbor_val_idx
                @inbounds conns_gpu[row, 2] = current_val_idx
            end
            # Check i-1 (up)
            if i > 1 && @inbounds im_gpu[i - 1, j, k]
                neighbor_val_idx = @inbounds idx_gpu[i - 1, j, k]
                row = CUDA.atomic_add!(pointer(d_row_counter), 1) + 1
                @inbounds conns_gpu[row, 1] = neighbor_val_idx
                @inbounds conns_gpu[row, 2] = current_val_idx
            end
            # Check i+1 (down)
            if i < nx && @inbounds im_gpu[i + 1, j, k]
                neighbor_val_idx = @inbounds idx_gpu[i + 1, j, k]
                row = CUDA.atomic_add!(pointer(d_row_counter), 1) + 1
                @inbounds conns_gpu[row, 1] = neighbor_val_idx
                @inbounds conns_gpu[row, 2] = current_val_idx
            end
            # Check j+1 (right)
            if j < ny && @inbounds im_gpu[i, j + 1, k]
                neighbor_val_idx = @inbounds idx_gpu[i, j + 1, k]
                row = CUDA.atomic_add!(pointer(d_row_counter), 1) + 1
                @inbounds conns_gpu[row, 1] = neighbor_val_idx
                @inbounds conns_gpu[row, 2] = current_val_idx
            end
            # Check k+1 (front)
            if k < nz && @inbounds im_gpu[i, j, k + 1]
                neighbor_val_idx = @inbounds idx_gpu[i, j, k + 1]
                row = CUDA.atomic_add!(pointer(d_row_counter), 1) + 1
                @inbounds conns_gpu[row, 1] = neighbor_val_idx
                @inbounds conns_gpu[row, 2] = current_val_idx
            end
        end
    end
    return nothing
end

# ================================================================
# Kernel 1 (New): Create Histogram of Connections by First Index
# ================================================================
# This kernel counts how many connections will have 'k' as their first element
function histogram_connections_kernel!(
    d_histogram, d_total_conn_count, im_gpu, idx_gpu, nx, ny, nz
)
    linear_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if linear_idx <= length(im_gpu)
        cid = CartesianIndices(im_gpu)[linear_idx]
        i, j, k = cid.I
        local_conn_count = 0
        if @inbounds im_gpu[i, j, k]
            # Check neighbors (k-1, j-1, i-1, i+1, j+1, k+1)
            # Note: We need the *value* from idx_gpu for the neighbor to index the histogram
            # Check k-1
            if k > 1 && @inbounds im_gpu[i, j, k - 1]
                neighbor_val_idx = @inbounds idx_gpu[i, j, k - 1]
                if neighbor_val_idx > 0 # Ensure neighbor is part of the indexed set
                    CUDA.@atomic d_histogram[neighbor_val_idx] += 1
                    local_conn_count += 1
                end
            end
            # Check j-1
            if j > 1 && @inbounds im_gpu[i, j - 1, k]
                neighbor_val_idx = @inbounds idx_gpu[i, j - 1, k]
                if neighbor_val_idx > 0
                    CUDA.@atomic d_histogram[neighbor_val_idx] += 1
                    local_conn_count += 1
                end
            end
            # Check i-1
            if i > 1 && @inbounds im_gpu[i - 1, j, k]
                neighbor_val_idx = @inbounds idx_gpu[i - 1, j, k]
                if neighbor_val_idx > 0
                    CUDA.@atomic d_histogram[neighbor_val_idx] += 1
                    local_conn_count += 1
                end
            end
            # Check i+1
            if i < nx && @inbounds im_gpu[i + 1, j, k]
                neighbor_val_idx = @inbounds idx_gpu[i + 1, j, k]
                if neighbor_val_idx > 0
                    CUDA.@atomic d_histogram[neighbor_val_idx] += 1
                    local_conn_count += 1
                end
            end
            # Check j+1
            if j < ny && @inbounds im_gpu[i, j + 1, k]
                neighbor_val_idx = @inbounds idx_gpu[i, j + 1, k]
                if neighbor_val_idx > 0
                    CUDA.@atomic d_histogram[neighbor_val_idx] += 1
                    local_conn_count += 1
                end
            end
            # Check k+1
            if k < nz && @inbounds im_gpu[i, j, k + 1]
                neighbor_val_idx = @inbounds idx_gpu[i, j, k + 1]
                if neighbor_val_idx > 0
                    CUDA.@atomic d_histogram[neighbor_val_idx] += 1
                    local_conn_count += 1
                end
            end
        end
        # Keep track of total connections simultaneously (optional, could sum histogram)
        if local_conn_count > 0
            CUDA.@atomic d_total_conn_count[1] += local_conn_count
        end
    end
    return nothing
end

# ================================================================
# GPU Exclusive Scan Helper (using cumsum!)
# ================================================================
# Simple kernel for shifting - potentially faster than CPU roundtrip for large N
function shift_kernel!(dest, src, n)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx == 1 && n >= 1
        @inbounds dest[idx] = 0
    elseif 1 < idx <= n
        @inbounds dest[idx] = src[idx - 1]
    end
    return nothing
end

# Computes exclusive scan: out[i] = sum(inp[1]...inp[i-1])
function exclusive_scan!(out::CuVector{T}, inp::CuVector{T}) where {T}
    n = length(inp)
    if n == 0
        return out
    end
    if length(out) != n
        throw(DimensionMismatch("Output vector must have same length as input"))
    end

    # Use inclusive scan provided by CUDA.jl
    temp_inclusive = similar(out) # Need temporary storage
    CUDA.cumsum!(temp_inclusive, inp)
    CUDA.synchronize() # Ensure cumsum is done

    # Launch kernel to perform the shift for exclusive scan
    # out[1] = 0; out[2:n] = temp_inclusive[1:n-1]
    threads = 256
    blocks = cld(n, threads)
    CUDA.@sync @cuda threads = threads blocks = blocks shift_kernel!(out, temp_inclusive, n)

    # Optional: Free temporary array if needed sooner rather than later
    # CUDA.unsafe_free!(temp_inclusive)
    return out
end

# ================================================================
# Kernel 2 (New): Write Connections using Offsets
# ================================================================
# Writes connections to their final sorted slot using atomically incremented offsets
function write_connections_offset_kernel!(
    conns_gpu, d_bucket_write_counters, im_gpu, idx_gpu, nx, ny, nz
)
    linear_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if linear_idx <= length(im_gpu)
        cid = CartesianIndices(im_gpu)[linear_idx]
        i, j, k = cid.I
        if @inbounds im_gpu[i, j, k]
            current_val_idx = @inbounds idx_gpu[i, j, k]
            if current_val_idx == 0
                return nothing
            end # Should not happen if idx filled correctly

            # Check neighbors (k-1, j-1, i-1, i+1, j+1, k+1)
            # Check k-1
            if k > 1 && @inbounds im_gpu[i, j, k - 1]
                neighbor_val_idx = @inbounds idx_gpu[i, j, k - 1]
                if neighbor_val_idx > 0 # If neighbor is valid
                    # Atomically get the next write position for this neighbor_val_idx bucket
                    write_row = CUDA.atomic_add!(
                        pointer(d_bucket_write_counters, neighbor_val_idx), 1
                    )
                    # Write the connection (add 1 for 1-based Julia indexing)
                    @inbounds conns_gpu[write_row + 1, 1] = neighbor_val_idx
                    @inbounds conns_gpu[write_row + 1, 2] = current_val_idx
                end
            end
            # Check j-1
            if j > 1 && @inbounds im_gpu[i, j - 1, k]
                neighbor_val_idx = @inbounds idx_gpu[i, j - 1, k]
                if neighbor_val_idx > 0
                    write_row = CUDA.atomic_add!(
                        pointer(d_bucket_write_counters, neighbor_val_idx), 1
                    )
                    @inbounds conns_gpu[write_row + 1, 1] = neighbor_val_idx
                    @inbounds conns_gpu[write_row + 1, 2] = current_val_idx
                end
            end
            # Check i-1
            if i > 1 && @inbounds im_gpu[i - 1, j, k]
                neighbor_val_idx = @inbounds idx_gpu[i - 1, j, k]
                if neighbor_val_idx > 0
                    write_row = CUDA.atomic_add!(
                        pointer(d_bucket_write_counters, neighbor_val_idx), 1
                    )
                    @inbounds conns_gpu[write_row + 1, 1] = neighbor_val_idx
                    @inbounds conns_gpu[write_row + 1, 2] = current_val_idx
                end
            end
            # Check i+1
            if i < nx && @inbounds im_gpu[i + 1, j, k]
                neighbor_val_idx = @inbounds idx_gpu[i + 1, j, k]
                if neighbor_val_idx > 0
                    write_row = CUDA.atomic_add!(
                        pointer(d_bucket_write_counters, neighbor_val_idx), 1
                    )
                    @inbounds conns_gpu[write_row + 1, 1] = neighbor_val_idx
                    @inbounds conns_gpu[write_row + 1, 2] = current_val_idx
                end
            end
            # Check j+1
            if j < ny && @inbounds im_gpu[i, j + 1, k]
                neighbor_val_idx = @inbounds idx_gpu[i, j + 1, k]
                if neighbor_val_idx > 0
                    write_row = CUDA.atomic_add!(
                        pointer(d_bucket_write_counters, neighbor_val_idx), 1
                    )
                    @inbounds conns_gpu[write_row + 1, 1] = neighbor_val_idx
                    @inbounds conns_gpu[write_row + 1, 2] = current_val_idx
                end
            end
            # Check k+1
            if k < nz && @inbounds im_gpu[i, j, k + 1]
                neighbor_val_idx = @inbounds idx_gpu[i, j, k + 1]
                if neighbor_val_idx > 0
                    write_row = CUDA.atomic_add!(
                        pointer(d_bucket_write_counters, neighbor_val_idx), 1
                    )
                    @inbounds conns_gpu[write_row + 1, 1] = neighbor_val_idx
                    @inbounds conns_gpu[write_row + 1, 2] = current_val_idx
                end
            end
        end
    end
    return nothing
end
