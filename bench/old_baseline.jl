# Standalone recreation of the OLD CUDA-specific code used as baseline.
# Only the GPU paths — no CPU stuff. Loaded after `using CUDA` is in scope.
module OldBaseline

using CUDA
using CUDA.CUSPARSE
using SparseArrays
using LinearAlgebra
import Tortuosity
import OrdinaryDiffEq: ODEProblem, init, ROCK4

# ========================================================================
# GRAPH KERNELS (old, CUDA.@cuda, CUDA.@atomic)
# ========================================================================

function fill_idx_kernel!(idx_gpu, linear_indices, num_true)
    thread_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if thread_idx <= num_true
        @inbounds original_linear_idx = linear_indices[thread_idx]
        @inbounds idx_gpu[original_linear_idx] = thread_idx
    end
    return nothing
end

function histogram_connections_kernel!(d_histogram, d_total_conn_count, im_gpu, idx_gpu, nx, ny, nz)
    linear_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if linear_idx <= length(im_gpu)
        cid = CartesianIndices(im_gpu)[linear_idx]
        i, j, k = cid.I
        local_conn_count = 0
        if @inbounds im_gpu[i, j, k]
            if k > 1 && @inbounds im_gpu[i, j, k - 1]
                neighbor_val_idx = @inbounds idx_gpu[i, j, k - 1]
                if neighbor_val_idx > 0
                    CUDA.@atomic d_histogram[neighbor_val_idx] += 1
                    local_conn_count += 1
                end
            end
            if j > 1 && @inbounds im_gpu[i, j - 1, k]
                neighbor_val_idx = @inbounds idx_gpu[i, j - 1, k]
                if neighbor_val_idx > 0
                    CUDA.@atomic d_histogram[neighbor_val_idx] += 1
                    local_conn_count += 1
                end
            end
            if i > 1 && @inbounds im_gpu[i - 1, j, k]
                neighbor_val_idx = @inbounds idx_gpu[i - 1, j, k]
                if neighbor_val_idx > 0
                    CUDA.@atomic d_histogram[neighbor_val_idx] += 1
                    local_conn_count += 1
                end
            end
            if i < nx && @inbounds im_gpu[i + 1, j, k]
                neighbor_val_idx = @inbounds idx_gpu[i + 1, j, k]
                if neighbor_val_idx > 0
                    CUDA.@atomic d_histogram[neighbor_val_idx] += 1
                    local_conn_count += 1
                end
            end
            if j < ny && @inbounds im_gpu[i, j + 1, k]
                neighbor_val_idx = @inbounds idx_gpu[i, j + 1, k]
                if neighbor_val_idx > 0
                    CUDA.@atomic d_histogram[neighbor_val_idx] += 1
                    local_conn_count += 1
                end
            end
            if k < nz && @inbounds im_gpu[i, j, k + 1]
                neighbor_val_idx = @inbounds idx_gpu[i, j, k + 1]
                if neighbor_val_idx > 0
                    CUDA.@atomic d_histogram[neighbor_val_idx] += 1
                    local_conn_count += 1
                end
            end
        end
        if local_conn_count > 0
            CUDA.@atomic d_total_conn_count[1] += local_conn_count
        end
    end
    return nothing
end

function shift_kernel!(dest, src, n)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx == 1 && n >= 1
        @inbounds dest[idx] = 0
    elseif 1 < idx <= n
        @inbounds dest[idx] = src[idx - 1]
    end
    return nothing
end

function exclusive_scan!(out::CuVector{T}, inp::CuVector{T}) where {T}
    n = length(inp)
    n == 0 && return out
    length(out) != n && throw(DimensionMismatch("Output must match input"))
    temp_inclusive = similar(out)
    CUDA.cumsum!(temp_inclusive, inp)
    CUDA.synchronize()
    threads = 256
    blocks = cld(n, threads)
    CUDA.@sync @cuda threads=threads blocks=blocks shift_kernel!(out, temp_inclusive, n)
    return out
end

function write_connections_offset_kernel!(conns_gpu, d_bucket_write_counters, im_gpu, idx_gpu, nx, ny, nz)
    linear_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if linear_idx <= length(im_gpu)
        cid = CartesianIndices(im_gpu)[linear_idx]
        i, j, k = cid.I
        if @inbounds im_gpu[i, j, k]
            current_val_idx = @inbounds idx_gpu[i, j, k]
            if current_val_idx == 0
                return nothing
            end
            if k > 1 && @inbounds im_gpu[i, j, k - 1]
                neighbor_val_idx = @inbounds idx_gpu[i, j, k - 1]
                if neighbor_val_idx > 0
                    write_row = CUDA.atomic_add!(pointer(d_bucket_write_counters, neighbor_val_idx), 1)
                    @inbounds conns_gpu[write_row + 1, 1] = neighbor_val_idx
                    @inbounds conns_gpu[write_row + 1, 2] = current_val_idx
                end
            end
            if j > 1 && @inbounds im_gpu[i, j - 1, k]
                neighbor_val_idx = @inbounds idx_gpu[i, j - 1, k]
                if neighbor_val_idx > 0
                    write_row = CUDA.atomic_add!(pointer(d_bucket_write_counters, neighbor_val_idx), 1)
                    @inbounds conns_gpu[write_row + 1, 1] = neighbor_val_idx
                    @inbounds conns_gpu[write_row + 1, 2] = current_val_idx
                end
            end
            if i > 1 && @inbounds im_gpu[i - 1, j, k]
                neighbor_val_idx = @inbounds idx_gpu[i - 1, j, k]
                if neighbor_val_idx > 0
                    write_row = CUDA.atomic_add!(pointer(d_bucket_write_counters, neighbor_val_idx), 1)
                    @inbounds conns_gpu[write_row + 1, 1] = neighbor_val_idx
                    @inbounds conns_gpu[write_row + 1, 2] = current_val_idx
                end
            end
            if i < nx && @inbounds im_gpu[i + 1, j, k]
                neighbor_val_idx = @inbounds idx_gpu[i + 1, j, k]
                if neighbor_val_idx > 0
                    write_row = CUDA.atomic_add!(pointer(d_bucket_write_counters, neighbor_val_idx), 1)
                    @inbounds conns_gpu[write_row + 1, 1] = neighbor_val_idx
                    @inbounds conns_gpu[write_row + 1, 2] = current_val_idx
                end
            end
            if j < ny && @inbounds im_gpu[i, j + 1, k]
                neighbor_val_idx = @inbounds idx_gpu[i, j + 1, k]
                if neighbor_val_idx > 0
                    write_row = CUDA.atomic_add!(pointer(d_bucket_write_counters, neighbor_val_idx), 1)
                    @inbounds conns_gpu[write_row + 1, 1] = neighbor_val_idx
                    @inbounds conns_gpu[write_row + 1, 2] = current_val_idx
                end
            end
            if k < nz && @inbounds im_gpu[i, j, k + 1]
                neighbor_val_idx = @inbounds idx_gpu[i, j, k + 1]
                if neighbor_val_idx > 0
                    write_row = CUDA.atomic_add!(pointer(d_bucket_write_counters, neighbor_val_idx), 1)
                    @inbounds conns_gpu[write_row + 1, 1] = neighbor_val_idx
                    @inbounds conns_gpu[write_row + 1, 2] = current_val_idx
                end
            end
        end
    end
    return nothing
end

function create_connectivity_list_old(img::CuArray{Bool})
    img = ndims(img) == 2 ? reshape(img, size(img)..., 1) : img
    nx, ny, nz = size(img)
    N = length(img)
    linear_indices_gpu = findall(img)
    num_true = length(linear_indices_gpu)
    num_true == 0 && return Matrix{Int}(undef, 0, 2)
    idx_gpu = CUDA.zeros(Int, size(img)...)
    threads_fill = min(num_true, 256)
    blocks_fill = cld(num_true, threads_fill)
    CUDA.@sync @cuda threads=threads_fill blocks=blocks_fill fill_idx_kernel!(idx_gpu, linear_indices_gpu, num_true)

    threads = 256
    blocks = cld(N, threads)

    d_histogram = CUDA.zeros(Int, num_true)
    d_total_conn_count = CUDA.zeros(Int, 1)
    CUDA.@sync @cuda threads=threads blocks=blocks histogram_connections_kernel!(
        d_histogram, d_total_conn_count, img, idx_gpu, nx, ny, nz)

    total_conns = Int(Array(d_total_conn_count)[1])
    total_conns == 0 && return Matrix{Int}(undef, 0, 2)

    d_bucket_write_counters = CUDA.zeros(Int, num_true)
    exclusive_scan!(d_bucket_write_counters, d_histogram)

    conns_gpu = CuArray{Int32}(undef, total_conns, 2)
    CUDA.@sync @cuda threads=threads blocks=blocks write_connections_offset_kernel!(
        conns_gpu, d_bucket_write_counters, img, idx_gpu, nx, ny, nz)

    return conns_gpu
end

# ========================================================================
# ADJACENCY MATRIX (old, via CUSPARSE COO→CSC)
# ========================================================================

function create_adjacency_matrix_old(conns::CuArray{Ti,2}; n::Integer, weights=1.0f0) where {Ti<:Integer}
    nedges = size(conns, 1)
    Tv = weights isa Number ? typeof(weights) : eltype(weights)
    V = weights isa Number ? CUDA.fill(Tv(weights), nedges) : weights
    I = conns[:, 1]
    J = conns[:, 2]
    coo = CuSparseMatrixCOO{Tv,Ti}(I, J, V, (n, n), nedges)
    csc = CuSparseMatrixCSC(coo)
    return csc
end

# ========================================================================
# LAPLACIAN (old, via CUSPARSE sum + spdiagm + subtract)
# ========================================================================

function spdiagm_old(v::CuArray)
    nnz = length(v)
    colPtr = CuArray(collect(Cint(1):Cint(nnz + 1)))
    rowVal = CuArray(collect(Cint(1):Cint(nnz)))
    nzVal = v
    dims = (nnz, nnz)
    return CUSPARSE.CuSparseMatrixCSC(colPtr, rowVal, nzVal, dims)
end

function laplacian_old(am)
    degrees = vec(sum(am; dims=2))
    degree_matrix = spdiagm_old(degrees)
    return degree_matrix - am
end

# ========================================================================
# SPARSE OPERATIONS (old CUDA kernels)
# ========================================================================

function set_diag_kernel_old!(nzVal::CuDeviceVector{Tv}, rowVal::CuDeviceVector{Ti},
                               colPtr::CuDeviceVector{Ti}, vals::CuDeviceVector{Tv},
                               N_diag::Ti) where {Tv,Ti}
    k = Ti((blockIdx().x - 1) * blockDim().x + threadIdx().x)
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
    return nothing
end

function set_diag_old!(A::CUDA.CUSPARSE.CuSparseMatrixCSC{Tv,Ti}, vals::AbstractVector) where {Tv,Ti}
    num_rows, num_cols = size(A)
    N_diag = min(num_rows, num_cols)
    vals_typed = convert.(Tv, vals)
    cu_vals = CuArray(vals_typed)
    threads = min(N_diag, 256)
    blocks = cld(N_diag, threads)
    CUDA.@sync @cuda threads=threads blocks=blocks set_diag_kernel_old!(
        A.nzVal, A.rowVal, A.colPtr, cu_vals, Ti(N_diag))
    return nothing
end

function get_diag_kernel_old!(diag_vals::CuDeviceVector{Tv}, nzVal::CuDeviceVector{Tv},
                               rowVal::CuDeviceVector{Ti}, colPtr::CuDeviceVector{Ti},
                               N_diag::Ti) where {Tv,Ti}
    k = Ti((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    if k <= N_diag && k > 0
        @inbounds idx_start = colPtr[k]
        @inbounds idx_end = colPtr[k + 1] - 1
        for idx in idx_start:idx_end
            @inbounds if rowVal[idx] == k
                @inbounds diag_vals[k] = nzVal[idx]
                break
            end
        end
    end
    return nothing
end

function get_diag_old(A::CUDA.CUSPARSE.CuSparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    num_rows, num_cols = size(A)
    N_diag = min(num_rows, num_cols)
    diag_vals = CUDA.zeros(Tv, N_diag)
    threads = min(N_diag, 256)
    blocks = cld(N_diag, threads)
    CUDA.@sync @cuda threads=threads blocks=blocks get_diag_kernel_old!(
        diag_vals, A.nzVal, A.rowVal, A.colPtr, Ti(N_diag))
    return diag_vals
end

function zero_rows_kernel_old!(nzVal::CuDeviceVector{Tv}, rowVal::CuDeviceVector{Ti},
                                is_target_row::CuDeviceVector{Bool}, nnz_val::Ti) where {Tv,Ti}
    k = Ti((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    if k <= nnz_val && k > 0
        @inbounds r = rowVal[k]
        if r > 0 && r <= length(is_target_row)
            @inbounds if is_target_row[r]
                @inbounds nzVal[k] = zero(Tv)
            end
        end
    end
    return nothing
end

function zero_cols_kernel_old!(nzVal::CuDeviceVector{Tv}, colPtr::CuDeviceVector{Ti},
                                target_cols::CuDeviceVector{Ti}, num_target_cols::Ti) where {Tv,Ti}
    idx_in_target = Ti((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    if idx_in_target <= num_target_cols && idx_in_target > 0
        @inbounds target_c = target_cols[idx_in_target]
        @inbounds col_start = colPtr[target_c]
        @inbounds col_end = colPtr[target_c + 1] - 1
        for k in col_start:col_end
            if k > 0 && k <= length(nzVal)
                @inbounds nzVal[k] = zero(Tv)
            end
        end
    end
    return nothing
end

function zero_rows_cols_old!(A::CUDA.CUSPARSE.CuSparseMatrixCSC{Tv,Ti}, idxs::AbstractVector{<:Integer}) where {Tv,Ti}
    num_rows, num_cols = size(A)
    nnz = length(A.nzVal)
    (nnz == 0 || isempty(idxs)) && return nothing
    idxs_Ti = convert.(Ti, idxs)
    valid_rows_idx = filter(i -> 1 <= i <= num_rows, idxs_Ti)
    valid_cols_idx = filter(i -> 1 <= i <= num_cols, idxs_Ti)
    cu_target_rows = CuArray(unique(valid_rows_idx))
    cu_target_cols = CuArray(unique(valid_cols_idx))
    num_target_rows = length(cu_target_rows)
    num_target_cols = length(cu_target_cols)

    if num_target_rows > 0
        is_target_row = CUDA.zeros(Bool, num_rows)
        is_target_row[cu_target_rows] .= true
        threads_k1 = min(nnz, 256)
        blocks_k1 = cld(nnz, threads_k1)
        CUDA.@sync @cuda threads=threads_k1 blocks=blocks_k1 zero_rows_kernel_old!(
            A.nzVal, A.rowVal, is_target_row, Ti(nnz))
    end
    if num_target_cols > 0
        threads_k2 = min(num_target_cols, 256)
        blocks_k2 = cld(num_target_cols, threads_k2)
        CUDA.@sync @cuda threads=threads_k2 blocks=blocks_k2 zero_cols_kernel_old!(
            A.nzVal, A.colPtr, cu_target_cols, Ti(num_target_cols))
    end
    return nothing
end

function compact_and_count_kernel_old!(
    new_nzVal::CuDeviceVector{Tv}, new_rowVal::CuDeviceVector{Ti},
    new_col_counts::CuDeviceVector{Ti}, nzVal_old::CuDeviceVector{Tv},
    rowVal_old::CuDeviceVector{Ti}, colPtr_old::CuDeviceVector{Ti},
    flags::CuDeviceVector{Bool}, scan_output::CuDeviceVector{Ti}, nnz_old::Ti) where {Tv,Ti}
    k = Ti((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    if k <= nnz_old && k > 0
        @inbounds if flags[k]
            @inbounds new_idx = scan_output[k] + Ti(1)
            @inbounds val = nzVal_old[k]
            @inbounds row = rowVal_old[k]
            if new_idx > 0 && new_idx <= length(new_nzVal)
                @inbounds new_nzVal[new_idx] = val
                @inbounds new_rowVal[new_idx] = row
            end
            @inbounds c = searchsortedlast(colPtr_old, k)
            if c > 0 && c <= length(new_col_counts)
                CUDA.@atomic new_col_counts[c] += Ti(1)
            end
        end
    end
    return nothing
end

function dropzeros_old!(A::CUDA.CUSPARSE.CuSparseMatrixCSC{Tv,Ti}; tol=eps(real(Tv))) where {Tv,Ti}
    nnz_old = A.nnz
    num_rows, num_cols = size(A)
    nnz_old == 0 && return nothing
    nzVal_old = A.nzVal
    rowVal_old = A.rowVal
    colPtr_old = A.colPtr
    flags = abs.(nzVal_old) .> tol
    scan_inclusive = CUDA.accumulate(+, convert(CuArray{Ti}, flags))
    nnz_new = (nnz_old > 0 && length(scan_inclusive) > 0) ? Ti(maximum(scan_inclusive)) : Ti(0)
    if nnz_new == nnz_old
        return nothing
    end
    if nnz_new == 0
        A.nzVal = CUDA.zeros(Tv, 0)
        A.rowVal = CUDA.zeros(Ti, 0)
        A.colPtr = CUDA.ones(Ti, num_cols + 1)
        A.nnz = 0
        return nothing
    end
    scan_output = CUDA.zeros(Ti, nnz_old)
    scan_output[2:end] .= scan_inclusive[1:(end - 1)]
    new_nzVal = CUDA.CuArray{Tv}(undef, nnz_new)
    new_rowVal = CUDA.CuArray{Ti}(undef, nnz_new)
    new_col_counts = CUDA.zeros(Ti, num_cols)
    threads_k1 = 256
    blocks_k1 = cld(nnz_old, threads_k1)
    CUDA.@sync @cuda threads=threads_k1 blocks=blocks_k1 compact_and_count_kernel_old!(
        new_nzVal, new_rowVal, new_col_counts, nzVal_old, rowVal_old, colPtr_old,
        flags, scan_output, Ti(nnz_old))
    new_colPtr = CUDA.CuArray{Ti}(undef, num_cols + 1)
    inclusive_scan_counts = CUDA.accumulate(+, new_col_counts)
    new_colPtr[1:1] .= 1
    if num_cols > 0
        new_colPtr[2:end] .= inclusive_scan_counts .+ 1
    end
    A.nzVal = new_nzVal
    A.rowVal = new_rowVal
    A.colPtr = new_colPtr
    A.nnz = nnz_new
    CUDA.synchronize()
    return nothing
end

# zero_rows_old! — zeros out specified rows then drops zeros
function zero_rows_old!(
    A::CUDA.CUSPARSE.CuSparseMatrixCSC{Tv,Ti}, rows::AbstractVector{<:Integer}
) where {Tv,Ti}
    num_rows, _ = size(A)
    nnz = length(A.nzVal)
    (nnz == 0 || isempty(rows)) && return nothing
    rows_Ti = convert.(Ti, rows)
    valid_rows = filter(i -> 1 <= i <= num_rows, rows_Ti)
    cu_rows = CuArray(unique(valid_rows))
    num_target_rows = length(cu_rows)
    num_target_rows == 0 && return nothing
    is_target_row = CUDA.zeros(Bool, num_rows)
    is_target_row[cu_rows] .= true
    threads = min(nnz, 256)
    blocks = cld(nnz, threads)
    CUDA.@sync @cuda threads=threads blocks=blocks zero_rows_kernel_old!(
        A.nzVal, A.rowVal, is_target_row, Ti(nnz)
    )
    dropzeros_old!(A)
    return nothing
end

# ========================================================================
# Full workflow (old)
# ========================================================================
function multihotvec_old(indices, n; vals=1.0, gpu=false)
    if vals isa AbstractArray
        @assert length(indices) == length(vals)
        vals = gpu ? CuArray(vals) : vals
    end
    vec = gpu ? CUDA.zeros(eltype(vals), n) : zeros(eltype(vals), n)
    vec[indices] .= vals
    return vec
end

function find_boundary_nodes_old(img, face)
    img_cpu = img isa CuArray ? Array(img) : img
    nnodes = sum(img_cpu)
    indices = fill(-1, size(img_cpu))
    indices[img_cpu] .= 1:nnodes
    fd = Dict(
        :left => indices[1, :, :], :right => indices[end, :, :],
        :bottom => indices[:, :, 1], :top => indices[:, :, end],
        :front => indices[:, 1, :], :back => indices[:, end, :],
    )
    nodes = fd[face][:]
    return nodes[nodes .> 0]
end

function apply_dirichlet_bc_old!(A::CUDA.CUSPARSE.CuSparseMatrixCSC, b; nodes, vals)
    diag_vals = get_diag_old(A)
    x_bc = multihotvec_old(nodes, length(b); vals=CuArray(vals), gpu=true)
    b .-= A * x_bc
    zero_rows_cols_old!(A, nodes)
    set_diag_old!(A, diag_vals)
    b[nodes] .= vals .* CUDA.@allowscalar diag_vals[nodes]
    dropzeros_old!(A)
end

# ========================================================================
# Transient pipeline (old)
# ========================================================================
# These functions mirror the OLD transient code from src/transient.jl
# (pre-refactor), using OldBaseline primitives to keep the old GPU path
# fully isolated from the new code.

# Transient reference implementations (`build_transient_operator_old`,
# `transient_problem_old`, `init_state_old`) were removed when issue #54
# dropped `TransientState` and the closure-based stop-condition API. The
# steady-state `tortuosity_simulation_old` path below still exercises the
# full old CUDA kernel chain (connectivity list + adjacency + laplacian +
# apply_dirichlet_bc), which is what the GPU parity tests care about.

function tortuosity_simulation_old(img_gpu::CuArray)
    nnodes = sum(img_gpu)
    img = img_gpu

    inlet = find_boundary_nodes_old(img, :left)
    outlet = find_boundary_nodes_old(img, :right)

    b = CUDA.zeros(Float32, nnodes)
    conns = create_connectivity_list_old(img)
    am = create_adjacency_matrix_old(conns; n=nnodes, weights=CUDA.fill(1.0f0, size(conns,1)))
    A = laplacian_old(am)

    bc_nodes = Int32.(vcat(inlet, outlet))
    bc_vals = CuArray(vcat(fill(1.0f0, length(inlet)), fill(0.0f0, length(outlet))))
    apply_dirichlet_bc_old!(A, b; nodes=bc_nodes, vals=bc_vals)

    return A, b
end

end  # module OldBaseline
