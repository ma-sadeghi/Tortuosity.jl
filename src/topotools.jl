# Graph topology tools: connectivity lists, adjacency matrices, boundary detection.

"""
    spdiagm(v)

Construct a sparse diagonal matrix from vector `v`. Dispatches to
`SparseArrays.spdiagm` on CPU.
"""
spdiagm(v::AbstractVector) = SparseArrays.spdiagm(0 => v)

# Docstring for `laplacian` lives on the stub in sparse_type.jl; the
# PortableSparseCSC-specific method is also in sparse_type.jl.
function laplacian(am)
    degrees = vec(sum(am; dims=2))
    degree_matrix = spdiagm(degrees)
    return degree_matrix - am
end

"""
    build_connectivity_list(img::AbstractArray{Bool,3}; inds=nothing)

Build an `nedges x 2` connectivity matrix listing all face-connected neighbor
pairs in the 3D boolean image `img`. Each row `[i, j]` indicates that pore
voxels `i` and `j` share a face. Connections are listed in both directions.

Uses CPU loops for standard arrays, KA kernels for GPU arrays.

# Keyword Arguments
- `inds`: pre-computed index array mapping grid positions to sequential pore
  indices. Default: computed internally.
"""
function build_connectivity_list(img::AbstractArray{Bool,3}; inds=nothing)
    if _on_gpu(img)
        return _build_connectivity_list_ka(img; inds=inds)
    end
    return _build_connectivity_list_cpu(img; inds=inds)
end

# Handle 2D input by promoting to 3D
function build_connectivity_list(img::AbstractArray{Bool}; inds=nothing)
    img3d = ndims(img) == 2 ? reshape(img, size(img)..., 1) : img
    return build_connectivity_list(img3d; inds=inds)
end

function _build_connectivity_list_cpu(img::AbstractArray{Bool,3}; inds=nothing)
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
function _build_connectivity_list_ka(img; inds=nothing)
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
    histogram_connections_kernel!(backend, 256)(
        d_histogram, img, idx_gpu, nx, ny, nz; ndrange=N,
    )

    # Exclusive scan for write offsets. The kernel deliberately does not
    # maintain its own total-connection counter — combining a per-bucket atomic
    # with a second shared-counter atomic in the same kernel exposes a
    # Metal/Atomix bug where the shared counter silently loses updates under
    # contention (#80). Instead, derive total_conns from the scan we need
    # anyway: the matching inclusive-scan-last-element equals
    # `exclusive[end] + histogram[end]`. Two single-element host reads beat a
    # full GPU reduction at moderate problem sizes.
    d_bucket_write_counters = similar(idx_gpu, Int32, num_true)
    exclusive_scan!(d_bucket_write_counters, d_histogram)
    total_conns = Int(Array(@view d_bucket_write_counters[end:end])[1]) +
                  Int(Array(@view d_histogram[end:end])[1])
    total_conns == 0 && return Matrix{Int}(undef, 0, 2)

    # Pass 2: write connections
    conns_gpu = similar(idx_gpu, Int32, total_conns, 2)
    write_connections_offset_kernel!(backend, 256)(
        conns_gpu, d_bucket_write_counters, img, idx_gpu, nx, ny, nz; ndrange=N,
    )
    KernelAbstractions.synchronize(backend)

    return conns_gpu
end

"""
    build_adjacency_matrix(conns; n, weights=1)

Build a sparse adjacency matrix in CSC format from an `nedges x 2` connectivity
list. Returns `SparseMatrixCSC` for CPU arrays, `PortableSparseCSC` for GPU arrays.

# Keyword Arguments
- `n`: number of nodes (determines matrix size `n x n`).
- `weights`: scalar weight for all edges, or a vector of per-edge weights.
"""
function build_adjacency_matrix(conns::Array{Int,2}; n, weights=1)
    nedges = size(conns, 1)
    w = length(weights) == 1 ? fill(weights, nedges) : weights

    # Build CSC directly from pre-sorted connectivity (sorted by column, then row).
    # This avoids the 3-4x memory overhead and sorting cost of sparse().
    rowval = conns[:, 1]
    colptr = zeros(Int, n + 1)
    for k in 1:nedges
        colptr[conns[k, 2] + 1] += 1
    end
    cumsum!(colptr, colptr)
    colptr .+= 1
    return SparseMatrixCSC(n, n, colptr, rowval, w)
end

function build_adjacency_matrix(
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
    nx, ny, nz = size(img_cpu)

    dim, fidx = if face === :left
        (1, 1)
    elseif face === :right
        (1, nx)
    elseif face === :front
        (2, 1)
    elseif face === :back
        (2, ny)
    elseif face === :bottom
        (3, 1)
    elseif face === :top
        (3, nz)
    else
        error("unknown face :$face")
    end

    # Single column-major walk: track the running pore-voxel ordinal and record it
    # whenever the current cell sits on the target face. The previous implementation
    # allocated a full `Int` array the size of `img` plus a `Dict` of 6 slice copies,
    # which dominated transient setup at large sizes (see design.md).
    nodes = Int[]
    ordinal = 0
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        if img_cpu[i, j, k]
            ordinal += 1
            on_face = (dim == 1 && i == fidx) ||
                      (dim == 2 && j == fidx) ||
                      (dim == 3 && k == fidx)
            on_face && push!(nodes, ordinal)
        end
    end
    return nodes
end
