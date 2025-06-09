function spdiagm(v::CuArray)
    nnz = length(v)
    colPtr = cu(collect(1:(nnz + 1)))
    rowVal = cu(collect(1:nnz))
    nzVal = v
    dims = (nnz, nnz)
    return CUSPARSE.CuSparseMatrixCSC(colPtr, rowVal, nzVal, dims)
end

function laplacian(am)
    degrees = vec(sum(am; dims=2))
    degree_matrix = SparseArrays.spdiagm(degrees)
    return degree_matrix - am
end

function create_connectivity_list(img::AbstractArray{Bool,3}; inds=nothing)
    img = ndims(img) == 2 ? reshape(img, size(img)..., 1) : img
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

    # Resize the connections matrix
    return conns[1:row, :]
end

# ================================================================
# Main GPU Function (Histogram Strategy)
# ================================================================
function create_connectivity_list(img::CuArray{Bool}; inds=nothing)
    # --- Preprocessing & Data Transfer ---
    img = ndims(img) == 2 ? reshape(img, size(img)..., 1) : img
    nx, ny, nz = size(img)
    N = length(img)
    img = cu(img)

    idx_gpu = nothing
    num_true = 0

    # --- Determine num_true and create idx_gpu ---
    if isnothing(inds)
        linear_indices_gpu = findall(img) # Potentially expensive?
        num_true = length(linear_indices_gpu)
        if num_true == 0
            return Matrix{Int}(undef, 0, 2)
        end
        idx_gpu = CUDA.zeros(Int, size(img)...)
        threads_fill = min(num_true, 256)
        blocks_fill = cld(num_true, threads_fill)
        CUDA.@sync @cuda threads = threads_fill blocks = blocks_fill fill_idx_kernel!(
            idx_gpu, linear_indices_gpu, num_true
        )
        # Consider freeing linear_indices_gpu if memory is tight
    else
        if size(inds) != size(img)
            error("Provided `inds` array must have the same dimensions as `im`")
        end
        idx_gpu = CuArray(inds)
        # Need num_true to size histogram. Finding max index is needed.
        # This could be slow if inds is large and not dense.
        @warn "Warning: Calculating max index from provided 'inds' on GPU..."
        num_true = Int(maximum(idx_gpu)) # Assumes indices are 1 to num_true
        @info "Inferred num_true: ", num_true
        if num_true == 0
            return Matrix{Int}(undef, 0, 2)
        end
    end

    # --- Kernel Launch Setup ---
    threads = 256
    blocks = cld(N, threads)

    # --- Pass 1: Calculate Histogram ---
    d_histogram = CUDA.zeros(Int, num_true) # One counter per possible index value
    d_total_conn_count = CUDA.zeros(Int, 1) # To get total count easily
    CUDA.@sync @cuda threads = threads blocks = blocks histogram_connections_kernel!(
        d_histogram, d_total_conn_count, img, idx_gpu, nx, ny, nz
    )

    total_conns = Int(Array(d_total_conn_count)[1])
    # total_conns = Int(sum(d_histogram)) # Alternative way to get count
    if total_conns == 0
        return Matrix{Int}(undef, 0, 2)
    end

    # --- Calculate Exclusive Scan (Write Offsets) ---
    d_bucket_write_counters = CUDA.zeros(Int, num_true) # Initialize counters for atomic adds
    exclusive_scan!(d_bucket_write_counters, d_histogram) # Computes offsets in-place

    # --- Allocate Final Array ---
    conns_gpu = CuArray{Int32}(undef, total_conns, 2)

    # --- Pass 2: Write Connections using Offsets ---
    # d_bucket_write_counters now holds the starting offset for each bucket
    CUDA.@sync @cuda threads = threads blocks = blocks write_connections_offset_kernel!(
        conns_gpu, d_bucket_write_counters, img, idx_gpu, nx, ny, nz
    )

    return conns_gpu
end

function create_adjacency_matrix(conns::Array{Int,2}; n, weights=1)
    # Ensure conns describes bidirectional connections
    nedges = size(conns, 1)
    if length(weights) == 1
        weights = fill(weights, nedges)
    end

    # Build colPtr, rowVal, and nzVal
    rowVal = conns[:, 1]
    colPtr = Vector{Int}(undef, n + 1)
    j = 1  # colPtr index
    for (i, col) in enumerate(@view conns[:, 2])
        if i == 1
            colPtr[j] = 1
        end
        if col != j
            colPtr[j + 1] = i
            j += 1
        end
    end
    colPtr[end] = nedges + 1

    return am = SparseMatrixCSC(n, n, colPtr, rowVal, weights)
    # am = sparse(conns[:, 1], conns[:, 2], weights, n, n)
end

function create_adjacency_matrix🐢(conns::CuArray{Int64,2}; n, weights=1)
    dims = (n, n)
    nedges = size(conns, 1)
    # NOTE: Promoting to Cint makes creating COO non-allocating and faster, but slower for CSC
    # I, J, V = CuVector{Cint}(conns[:, 1]), CuVector{Cint}(conns[:, 2]), CUDA.ones(Float32, nedges)
    I, J, V = conns[:, 1], conns[:, 2], CUDA.ones(Float32, nedges)
    am = CUSPARSE.CuSparseMatrixCOO(I, J, V, dims, nedges)
    am = CUDA.CUSPARSE.CuSparseMatrixCSC(am)
    return am
end

"""
    create_adjacency_matrix(conns::CuArray{Ti, 2}; n::Integer,
                            weights = 1.0f0) where {Ti<:Integer}

Creates a sparse adjacency matrix `A` in CSC format on the GPU from a list of connections.

Given a list of connections where `conns[k, 1]` is the source node `i` and `conns[k, 2]`
is the target node `j` for the k-th connection, the function constructs a sparse
matrix `A` of size `n x n` such that `A[i, j] = weight`.
The value type (`Tv`) of the matrix is inferred directly from the `weights` argument.

If multiple connections exist between the same `(i, j)` pair, their weights are summed.

Arguments:
- `conns::CuArray{Ti, 2}`: A `nedges x 2` GPU array containing connection pairs.
                          `conns[:, 1]` are row indices (sources), `conns[:, 2]` are
                          column indices (targets). Assumed to be 1-based indices.
                          Ti is the integer type (e.g., Int32, Int64).
- `n::Integer`: The number of nodes in the graph, determining the matrix size (`n x n`).

Keyword Arguments:
- `weights`: The weights for the connections. Determines the value type (`Tv`) of the
            resulting matrix. Can be:
            - A scalar value (Number): All connections will have this weight. Defaults to
              `1.0f0` (resulting in `Float32` matrix). Passing `1` results in `Int`,
              `1.0` results in `Float64`, etc.
            - A `CuVector{<:Real}`: A GPU vector of length `nedges` specifying the weight
              for each connection in `conns`. The element type determines the matrix `Tv`.

Returns:
- `CUDA.CUSPARSE.CuSparseMatrixCSC{Tv, Ti}`: The resulting adjacency matrix on the GPU,
                                             where `Tv` is inferred from `weights`.
"""
function create_adjacency_matrix(
    conns::CuArray{Ti,2}; n::Integer, weights=1.0f0
) where {Ti<:Integer}

    # Check input shape consistency (allow empty connections)
    if size(conns, 2) != 2 && size(conns, 1) != 0
        throw(ArgumentError("`conns` must have exactly 2 columns. Found $(size(conns))."))
    end
    if n <= 0
        throw(ArgumentError("Number of nodes `n` must be positive."))
    end

    nedges = size(conns, 1)

    # Determine Value Type (ActualTv) and prepare Value Vector (V) based *only* on weights
    local V::CuVector
    local ActualTv::Type

    if weights isa CuVector
        # Check length only if there are connections to match
        if nedges > 0 && length(weights) != nedges
            msg = "Length of `weights` ($(length(weights))) must match `conns` ($size(conns))."
            throw(DimensionMismatch(msg))
        end
        ActualTv = eltype(weights) # Infer type from vector
        V = weights # Use the vector directly

    elseif weights isa Number
        ActualTv = typeof(weights) # Infer type directly from scalar
    # We will create V later, only if nedges > 0
    else
        throw(ArgumentError("`weights` must be a Number (scalar) or a CuVector."))
    end

    # Handle empty connections case gracefully
    if nedges == 0
        I = CUDA.zeros(Ti, 0)
        J = CUDA.zeros(Ti, 0)
        V_empty = CUDA.zeros(ActualTv, 0) # Use the inferred ActualTv
        coo = CuSparseMatrixCOO{ActualTv,Ti}(I, J, V_empty, (n, n))
        csc = CuSparseMatrixCSC{ActualTv,Ti}(coo)
        return csc
    end

    # Create V for scalar case (now that nedges > 0 is confirmed)
    if weights isa Number
        # Fill with the scalar 'weights', CUDA.fill infers type correctly
        V = CUDA.fill(weights, nedges)
    end
    # V is already assigned if weights was a CuVector

    # Extract Row (I) and Column (J) indices from columns
    I = conns[:, 1]
    J = conns[:, 2]

    # Step 1: Create COO matrix - this handles summing duplicates
    # Explicitly specify ActualTv and Ti
    coo = CuSparseMatrixCOO{ActualTv,Ti}(I, J, V, (n, n), nedges)

    # Step 2: Convert COO to CSC - uses efficient CUSPARSE routines
    csc = CuSparseMatrixCSC(coo)

    return csc
end

function find_boundary_nodes(img, face)
    nnodes = sum(img)
    indices = fill(-1, size(img))
    indices[img] .= 1:nnodes

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
