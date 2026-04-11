# Graph topology tools: connectivity lists, adjacency matrices, boundary detection.

"""
    spdiagm(v)

Construct a sparse diagonal matrix from vector `v`. Dispatches to
`SparseArrays.spdiagm` on CPU.
"""
spdiagm(v::AbstractVector) = SparseArrays.spdiagm(0 => v)

"""
    laplacian(am)

Compute the graph Laplacian `L = D - A` where `D` is the degree matrix and
`A` is the adjacency matrix `am`.
"""
function laplacian(am)
    degrees = vec(sum(am; dims=2))
    degree_matrix = spdiagm(degrees)
    return degree_matrix - am
end
# NOTE: laplacian(am::PortableSparseCSC) is defined in sparse_type.jl

"""
    create_connectivity_list(img::AbstractArray{Bool,3}; inds=nothing)

Build an `nedges x 2` connectivity matrix listing all face-connected neighbor
pairs in the 3D boolean image `img`. Each row `[i, j]` indicates that pore
voxels `i` and `j` share a face. Connections are listed in both directions.

Uses CPU loops for standard arrays, KA kernels for GPU arrays.

# Keyword Arguments
- `inds`: pre-computed index array mapping grid positions to sequential pore
  indices. Default: computed internally.
"""
function create_connectivity_list(img::AbstractArray{Bool,3}; inds=nothing)
    if _on_gpu(img)
        return _create_connectivity_list_ka(img; inds=inds)
    end
    return _create_connectivity_list_cpu(img; inds=inds)
end

# Handle 2D input by promoting to 3D
function create_connectivity_list(img::AbstractArray{Bool}; inds=nothing)
    img3d = ndims(img) == 2 ? reshape(img, size(img)..., 1) : img
    return create_connectivity_list(img3d; inds=inds)
end

function _create_connectivity_list_cpu(img::AbstractArray{Bool,3}; inds=nothing)
    nx, ny, nz = size(img)

    if isnothing(inds)
        idx = similar(img, Int)
        idx[img] .= 1:sum(img)
    else
        idx = inds
    end

    total_conns = count(img) * 6
    conns = Matrix{Int}(undef, total_conns, 2)
    row = 0

    for cid in CartesianIndices(img)
        i, j, k = cid.I
        if img[i, j, k]
            if k > 1 && img[i, j, k - 1]
                row += 1
                conns[row, :] .= idx[i, j, k - 1], idx[i, j, k]
            end
            if j > 1 && img[i, j - 1, k]
                row += 1
                conns[row, :] .= idx[i, j - 1, k], idx[i, j, k]
            end
            if i > 1 && img[i - 1, j, k]
                row += 1
                conns[row, :] .= idx[i - 1, j, k], idx[i, j, k]
            end
            if i < nx && img[i + 1, j, k]
                row += 1
                conns[row, :] .= idx[i + 1, j, k], idx[i, j, k]
            end
            if j < ny && img[i, j + 1, k]
                row += 1
                conns[row, :] .= idx[i, j + 1, k], idx[i, j, k]
            end
            if k < nz && img[i, j, k + 1]
                row += 1
                conns[row, :] .= idx[i, j, k + 1], idx[i, j, k]
            end
        end
    end

    return conns[1:row, :]
end

"""
GPU implementation using a two-pass histogram strategy with KA kernels.
"""
function _create_connectivity_list_ka(img; inds=nothing)
    img = ndims(img) == 2 ? reshape(img, size(img)..., 1) : img
    nx, ny, nz = size(img)
    N = length(img)
    backend = get_backend(img)

    idx_gpu = nothing
    num_true = 0

    if isnothing(inds)
        linear_indices_gpu = findall(img)
        num_true = length(linear_indices_gpu)
        num_true == 0 && return Matrix{Int}(undef, 0, 2)
        idx_gpu = fill!(similar(img, Int32), Int32(0))
        wg = min(num_true, 256)
        fill_idx_kernel!(backend, wg)(idx_gpu, linear_indices_gpu, num_true; ndrange=num_true)
        # No sync — next kernel on same stream waits automatically
    else
        if size(inds) != size(img)
            error("Provided `inds` array must have the same dimensions as `img`")
        end
        idx_gpu = _on_gpu(inds) ? inds : _gpu_adapt[](inds)
        num_true = Int(maximum(idx_gpu))
        num_true == 0 && return Matrix{Int}(undef, 0, 2)
    end

    # Pass 1: histogram. Use Int32 buckets — half the memory traffic.
    d_histogram = fill!(similar(idx_gpu, Int32, num_true), Int32(0))
    d_total_conn_count = fill!(similar(idx_gpu, Int32, 1), Int32(0))
    histogram_connections_kernel!(backend, 256)(
        d_histogram, d_total_conn_count, img, idx_gpu, nx, ny, nz; ndrange=N,
    )
    # We need total_conns on the host before allocating conns_gpu, so this
    # implicit sync via Array() is unavoidable.
    total_conns = Int(Array(d_total_conn_count)[1])
    total_conns == 0 && return Matrix{Int}(undef, 0, 2)

    # Exclusive scan for write offsets
    d_bucket_write_counters = similar(idx_gpu, Int32, num_true)
    exclusive_scan!(d_bucket_write_counters, d_histogram)

    # Pass 2: write connections
    conns_gpu = similar(idx_gpu, Int32, total_conns, 2)
    write_connections_offset_kernel!(backend, 256)(
        conns_gpu, d_bucket_write_counters, img, idx_gpu, nx, ny, nz; ndrange=N,
    )
    KernelAbstractions.synchronize(backend)

    return conns_gpu
end

"""
    create_adjacency_matrix(conns; n, weights=1)

Build a sparse adjacency matrix in CSC format from an `nedges x 2` connectivity
list. Returns `SparseMatrixCSC` for CPU arrays, `PortableSparseCSC` for GPU arrays.

# Keyword Arguments
- `n`: number of nodes (determines matrix size `n x n`).
- `weights`: scalar weight for all edges, or a vector of per-edge weights.
"""
function create_adjacency_matrix(conns::Array{Int,2}; n, weights=1)
    nedges = size(conns, 1)
    w = length(weights) == 1 ? fill(weights, nedges) : weights

    # Build CSC directly from pre-sorted connectivity (sorted by column, then row).
    # This avoids the 3-4x memory overhead and sorting cost of sparse().
    rowVal = conns[:, 1]
    colPtr = zeros(Int, n + 1)
    for k in 1:nedges
        colPtr[conns[k, 2] + 1] += 1
    end
    cumsum!(colPtr, colPtr)
    colPtr .+= 1
    return SparseMatrixCSC(n, n, colPtr, rowVal, w)
end

function create_adjacency_matrix(
    conns::AbstractMatrix{<:Integer}; n::Integer, weights=1.0f0,
)
    backend = get_backend(conns)
    nedges = size(conns, 1)

    Tv = weights isa Number ? typeof(weights) : eltype(weights)
    # Infer index type from the input — Int32 keeps GPU memory traffic minimal
    # and is what CUSPARSE expects.
    Ti = eltype(conns)

    if nedges == 0
        colptr = fill!(similar(conns, Ti, n + 1), one(Ti))
        rowval = similar(conns, Ti, 0)
        nzval = similar(conns, Tv, 0)
        return PortableSparseCSC(n, n, colptr, rowval, nzval)
    end

    # Create value vector
    V = if weights isa Number
        fill!(similar(conns, Tv, nedges), weights)
    else
        weights
    end

    I = conns[:, 1]  # row indices
    J = conns[:, 2]  # column indices

    # Step 1: histogram of column indices
    col_counts = fill!(similar(conns, Ti, n), zero(Ti))
    _histogram_cols_kernel!(backend)(col_counts, J, nedges; ndrange=nedges)
    KernelAbstractions.synchronize(backend)

    # Step 2: build colptr from cumulative sum
    colptr = fill!(similar(conns, Ti, n + 1), zero(Ti))
    temp_scan = accumulate(+, col_counts)
    _build_colptr_kernel!(backend)(colptr, temp_scan, n; ndrange=max(n, 1))
    KernelAbstractions.synchronize(backend)

    # Step 3: scatter COO entries into CSC position using atomic counters
    # Initialize write offsets from colptr[1:n]
    write_offsets = similar(conns, Ti, n)
    copyto!(write_offsets, colptr[1:n])

    rowval = similar(conns, Ti, nedges)
    nzval = similar(conns, Tv, nedges)

    _scatter_coo_to_csc_kernel!(backend)(
        rowval, nzval, write_offsets, I, J, V, nedges; ndrange=nedges,
    )
    KernelAbstractions.synchronize(backend)

    return PortableSparseCSC(n, n, colptr, rowval, nzval)
end

"""
    find_boundary_nodes(img, face)

Return the pore-voxel indices on a given `face` of the 3D image `img`.
`face` is one of `:left`, `:right`, `:front`, `:back`, `:bottom`, `:top`.
Operates on CPU; call before transferring `img` to GPU.
"""
function find_boundary_nodes(img, face)
    # Transfer to CPU if on GPU (boundary detection is cheap, avoid GPU indexing issues)
    img_cpu = _on_gpu(img) ? Array(img) : img
    nnodes = sum(img_cpu)
    indices = fill(-1, size(img_cpu))
    indices[img_cpu] .= 1:nnodes

    face_dict = Dict(
        :left => indices[1, :, :],
        :right => indices[end, :, :],
        :bottom => indices[:, :, 1],
        :top => indices[:, :, end],
        :front => indices[:, 1, :],
        :back => indices[:, end, :],
    )

    nodes = face_dict[face][:]
    return nodes[nodes .> 0]
end
