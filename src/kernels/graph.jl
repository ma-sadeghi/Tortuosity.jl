# KernelAbstractions-based graph kernels for building connectivity lists on any GPU backend.

using KernelAbstractions
using Atomix

"""
    fill_idx_kernel!(idx_gpu, linear_indices, num_true)

KA kernel: write sequential indices (1, 2, ...) into `idx_gpu` at positions
given by `linear_indices`. Maps each `true` voxel to its pore-voxel number.
"""
@kernel function fill_idx_kernel!(idx_gpu, @Const(linear_indices), num_true)
    thread_idx = @index(Global)
    if thread_idx <= num_true
        @inbounds original_linear_idx = linear_indices[thread_idx]
        @inbounds idx_gpu[original_linear_idx] = thread_idx
    end
end

"""
    histogram_connections_kernel!(d_histogram, d_total_conn_count, im_gpu, idx_gpu, nx, ny, nz)

KA kernel: build a histogram counting connections per source node. Each thread
processes one voxel and atomically increments `d_histogram[neighbor_idx]` for
each face-connected neighbor. Also accumulates `d_total_conn_count`.
"""
@kernel function histogram_connections_kernel!(
    d_histogram, d_total_conn_count, @Const(im_gpu), @Const(idx_gpu), nx, ny, nz,
)
    linear_idx = @index(Global)
    if linear_idx <= length(im_gpu)
        cid = CartesianIndices(im_gpu)[linear_idx]
        i, j, k = cid.I
        local_conn_count = 0
        if @inbounds im_gpu[i, j, k]
            # Check k-1
            if k > 1 && @inbounds im_gpu[i, j, k - 1]
                neighbor_val_idx = @inbounds idx_gpu[i, j, k - 1]
                if neighbor_val_idx > 0
                    Atomix.@atomic d_histogram[neighbor_val_idx] += 1
                    local_conn_count += 1
                end
            end
            # Check j-1
            if j > 1 && @inbounds im_gpu[i, j - 1, k]
                neighbor_val_idx = @inbounds idx_gpu[i, j - 1, k]
                if neighbor_val_idx > 0
                    Atomix.@atomic d_histogram[neighbor_val_idx] += 1
                    local_conn_count += 1
                end
            end
            # Check i-1
            if i > 1 && @inbounds im_gpu[i - 1, j, k]
                neighbor_val_idx = @inbounds idx_gpu[i - 1, j, k]
                if neighbor_val_idx > 0
                    Atomix.@atomic d_histogram[neighbor_val_idx] += 1
                    local_conn_count += 1
                end
            end
            # Check i+1
            if i < nx && @inbounds im_gpu[i + 1, j, k]
                neighbor_val_idx = @inbounds idx_gpu[i + 1, j, k]
                if neighbor_val_idx > 0
                    Atomix.@atomic d_histogram[neighbor_val_idx] += 1
                    local_conn_count += 1
                end
            end
            # Check j+1
            if j < ny && @inbounds im_gpu[i, j + 1, k]
                neighbor_val_idx = @inbounds idx_gpu[i, j + 1, k]
                if neighbor_val_idx > 0
                    Atomix.@atomic d_histogram[neighbor_val_idx] += 1
                    local_conn_count += 1
                end
            end
            # Check k+1
            if k < nz && @inbounds im_gpu[i, j, k + 1]
                neighbor_val_idx = @inbounds idx_gpu[i, j, k + 1]
                if neighbor_val_idx > 0
                    Atomix.@atomic d_histogram[neighbor_val_idx] += 1
                    local_conn_count += 1
                end
            end
        end
        if local_conn_count > 0
            Atomix.@atomic d_total_conn_count[1] += local_conn_count
        end
    end
end

"""
    shift_kernel!(dest, src, n)

KA kernel: right-shift `src` into `dest` with `dest[1] = 0`, converting an
inclusive prefix sum into an exclusive prefix sum.
"""
@kernel function shift_kernel!(dest, @Const(src), n)
    idx = @index(Global)
    if idx == 1 && n >= 1
        @inbounds dest[idx] = 0
    elseif 1 < idx <= n
        @inbounds dest[idx] = src[idx - 1]
    end
end

"""
    exclusive_scan!(out, inp)

Compute the exclusive prefix sum: `out[i] = sum(inp[1:i-1])`.
Uses `cumsum!` for the inclusive scan, then shifts via [`shift_kernel!`](@ref).
Works on any backend (CPU via Base, GPU via GPUArrays).
"""
function exclusive_scan!(out::AbstractVector{T}, inp::AbstractVector{T}) where {T}
    n = length(inp)
    n == 0 && return out
    length(out) != n && throw(DimensionMismatch("Output must match input length"))

    temp_inclusive = similar(out)
    cumsum!(temp_inclusive, inp)

    backend = get_backend(out)
    shift_kernel!(backend)(out, temp_inclusive, n; ndrange=n)
    KernelAbstractions.synchronize(backend)

    return out
end

"""
    write_connections_offset_kernel!(conns_gpu, d_bucket_write_counters, im_gpu, idx_gpu, nx, ny, nz)

KA kernel: write connections into pre-computed sorted positions using per-bucket
atomic counters (from [`exclusive_scan!`](@ref)). Produces a connectivity list
sorted by the first (source) index.
"""
@kernel function write_connections_offset_kernel!(
    conns_gpu, d_bucket_write_counters, @Const(im_gpu), @Const(idx_gpu), nx, ny, nz,
)
    linear_idx = @index(Global)
    if linear_idx <= length(im_gpu)
        cid = CartesianIndices(im_gpu)[linear_idx]
        i, j, k = cid.I
        if @inbounds im_gpu[i, j, k]
            current_val_idx = @inbounds idx_gpu[i, j, k]

            if current_val_idx > 0
                # Check k-1
                if k > 1 && @inbounds im_gpu[i, j, k - 1]
                    neighbor_val_idx = @inbounds idx_gpu[i, j, k - 1]
                    if neighbor_val_idx > 0
                        ref = Atomix.IndexableRef(d_bucket_write_counters, (Int(neighbor_val_idx),))
                        write_row = Atomix.modify!(ref, +, 1).first
                        @inbounds conns_gpu[write_row + 1, 1] = neighbor_val_idx
                        @inbounds conns_gpu[write_row + 1, 2] = current_val_idx
                    end
                end
                # Check j-1
                if j > 1 && @inbounds im_gpu[i, j - 1, k]
                    neighbor_val_idx = @inbounds idx_gpu[i, j - 1, k]
                    if neighbor_val_idx > 0
                        ref = Atomix.IndexableRef(d_bucket_write_counters, (Int(neighbor_val_idx),))
                        write_row = Atomix.modify!(ref, +, 1).first
                        @inbounds conns_gpu[write_row + 1, 1] = neighbor_val_idx
                        @inbounds conns_gpu[write_row + 1, 2] = current_val_idx
                    end
                end
                # Check i-1
                if i > 1 && @inbounds im_gpu[i - 1, j, k]
                    neighbor_val_idx = @inbounds idx_gpu[i - 1, j, k]
                    if neighbor_val_idx > 0
                        ref = Atomix.IndexableRef(d_bucket_write_counters, (Int(neighbor_val_idx),))
                        write_row = Atomix.modify!(ref, +, 1).first
                        @inbounds conns_gpu[write_row + 1, 1] = neighbor_val_idx
                        @inbounds conns_gpu[write_row + 1, 2] = current_val_idx
                    end
                end
                # Check i+1
                if i < nx && @inbounds im_gpu[i + 1, j, k]
                    neighbor_val_idx = @inbounds idx_gpu[i + 1, j, k]
                    if neighbor_val_idx > 0
                        ref = Atomix.IndexableRef(d_bucket_write_counters, (Int(neighbor_val_idx),))
                        write_row = Atomix.modify!(ref, +, 1).first
                        @inbounds conns_gpu[write_row + 1, 1] = neighbor_val_idx
                        @inbounds conns_gpu[write_row + 1, 2] = current_val_idx
                    end
                end
                # Check j+1
                if j < ny && @inbounds im_gpu[i, j + 1, k]
                    neighbor_val_idx = @inbounds idx_gpu[i, j + 1, k]
                    if neighbor_val_idx > 0
                        ref = Atomix.IndexableRef(d_bucket_write_counters, (Int(neighbor_val_idx),))
                        write_row = Atomix.modify!(ref, +, 1).first
                        @inbounds conns_gpu[write_row + 1, 1] = neighbor_val_idx
                        @inbounds conns_gpu[write_row + 1, 2] = current_val_idx
                    end
                end
                # Check k+1
                if k < nz && @inbounds im_gpu[i, j, k + 1]
                    neighbor_val_idx = @inbounds idx_gpu[i, j, k + 1]
                    if neighbor_val_idx > 0
                        ref = Atomix.IndexableRef(d_bucket_write_counters, (Int(neighbor_val_idx),))
                        write_row = Atomix.modify!(ref, +, 1).first
                        @inbounds conns_gpu[write_row + 1, 1] = neighbor_val_idx
                        @inbounds conns_gpu[write_row + 1, 2] = current_val_idx
                    end
                end
            end
        end
    end
end

# --- COO to CSC conversion kernels ---

"""
    _histogram_cols_kernel!(col_counts, J, nedges)

KA kernel: count entries per column for COO-to-CSC conversion.
"""
@kernel function _histogram_cols_kernel!(col_counts, @Const(J), nedges)
    k = @index(Global)
    if k <= nedges
        @inbounds j = J[k]
        Atomix.@atomic col_counts[j] += 1
    end
end

"""
    _build_colptr_kernel!(colptr, scan, n)

KA kernel: build colptr from inclusive prefix sum of column counts.
"""
@kernel function _build_colptr_kernel!(colptr, @Const(scan), n)
    j = @index(Global)
    if j == 1
        @inbounds colptr[1] = 1
    end
    if j <= n
        @inbounds colptr[j + 1] = scan[j] + 1
    end
end

"""
    _scatter_coo_to_csc_kernel!(rowval, nzval, write_offsets, I, J, V, nedges)

KA kernel: scatter COO entries into CSC arrays using atomic write counters.
"""
@kernel function _scatter_coo_to_csc_kernel!(
    rowval, nzval, write_offsets, @Const(I), @Const(J), @Const(V), nedges,
)
    k = @index(Global)
    if k <= nedges
        @inbounds j = Int(J[k])
        ref = Atomix.IndexableRef(write_offsets, (j,))
        pos = Atomix.modify!(ref, +, 1).first
        @inbounds rowval[pos] = I[k]
        @inbounds nzval[pos] = V[k]
    end
end
