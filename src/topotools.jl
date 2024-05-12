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

function create_connectivity_list_L(im)
    im = ndims(im) == 2 ? reshape(im, size(im)..., 1) : im
    nnodes = sum(im)
    nx, ny, nz = size(im)
    idx = fill(-1, size(im))
    idx[im] .= 1:nnodes
    xconns = hcat(idx[1:(nx - 1), :, :][:], idx[2:nx, :, :][:])
    yconns = hcat(idx[:, 1:(ny - 1), :][:], idx[:, 2:ny, :][:])
    zconns = hcat(idx[:, :, 1:(nz - 1)][:], idx[:, :, 2:nz][:])
    conns = vcat(xconns, yconns, zconns)
    nodes = findall(row -> all(row .!= -1), eachrow(conns))
    return conns = conns[nodes, :]
end

function create_connectivity_list🚀_L(im, sort=true, triu=false)
    im = ndims(im) == 2 ? reshape(im, size(im)..., 1) : im
    nx, ny, nz = size(im)

    idx = fill(-1, size(im))
    idx[im] .= 1:sum(im)

    # Generate possible connections
    xconns = hcat(idx[1:(nx - 1), :, :][:], idx[2:nx, :, :][:])
    yconns = hcat(idx[:, 1:(ny - 1), :][:], idx[:, 2:ny, :][:])
    zconns = hcat(idx[:, :, 1:(nz - 1)][:], idx[:, :, 2:nz][:])
    conns = vcat(xconns, yconns, zconns)

    # Filter connections
    mask = .!any(conns .== -1; dims=2)
    conns = conns[mask[:, 1], :]

    # Return full or upper triangular connections
    conns = triu ? conns : vcat(conns, conns[:, [2, 1]])

    # Sort the connections (by the second column)
    conns = sort ? sortslices(conns; dims=1, by=x -> (x[2], x[1])) : conns

    return conns
end

function create_connectivity_list🚀(im; inds=nothing, sort=true, triu=false)
    im = ndims(im) == 2 ? reshape(im, size(im)..., 1) : im
    nx, ny, nz = size(im)

    if inds === nothing
        idx = similar(im, Int)
        idx[im] .= 1:sum(im)
    else
        idx = inds
    end

    total_conns = count(im) * 3
    conns = Matrix{Int}(undef, total_conns, 2)
    row = 0

    # x-connections
    for k in 1:nz, j in 1:ny, i in 1:(nx - 1)
        if im[i, j, k] && im[i + 1, j, k]
            row += 1
            conns[row, :] .= idx[i, j, k], idx[i + 1, j, k]
        end
    end

    # y-connections
    for k in 1:nz, j in 1:(ny - 1), i in 1:nx
        if im[i, j, k] && im[i, j + 1, k]
            row += 1
            conns[row, :] .= idx[i, j, k], idx[i, j + 1, k]
        end
    end

    # z-connections
    for k in 1:(nz - 1), j in 1:ny, i in 1:nx
        if im[i, j, k] && im[i, j, k + 1]
            row += 1
            conns[row, :] .= idx[i, j, k], idx[i, j, k + 1]
        end
    end

    # Resize the connections matrix
    conns = conns[1:row, :]

    # Return full or upper triangular connections
    conns = triu ? conns : vcat(conns, conns[:, [2, 1]])

    # Sort the connections (by the second column)
    return conns = sort ? sortslices(conns; dims=1, by=x -> (x[2], x[1])) : conns
end

function create_connectivity_list🚀🚀(im; inds=nothing)
    im = ndims(im) == 2 ? reshape(im, size(im)..., 1) : im
    nx, ny, nz = size(im)

    if inds === nothing
        idx = similar(im, Int)
        idx[im] .= 1:sum(im)
    else
        idx = inds
    end

    total_conns = count(im) * 6
    conns = Matrix{Int}(undef, total_conns, 2)
    row = 0

    for cid in CartesianIndices(im)
        i, j, k = cid.I
        if im[i, j, k]
            if k > 1 && im[i, j, k - 1]
                row += 1
                conns[row, :] .= idx[i, j, k - 1], idx[i, j, k]
            end
            if j > 1 && im[i, j - 1, k]
                row += 1
                conns[row, :] .= idx[i, j - 1, k], idx[i, j, k]
            end
            if i > 1 && im[i - 1, j, k]
                row += 1
                conns[row, :] .= idx[i - 1, j, k], idx[i, j, k]
            end
            if i < nx && im[i + 1, j, k]
                row += 1
                conns[row, :] .= idx[i + 1, j, k], idx[i, j, k]
            end
            if j < ny && im[i, j + 1, k]
                row += 1
                conns[row, :] .= idx[i, j + 1, k], idx[i, j, k]
            end
            if k < nz && im[i, j, k + 1]
                row += 1
                conns[row, :] .= idx[i, j, k + 1], idx[i, j, k]
            end
        end
    end

    # Resize the connections matrix
    return conns[1:row, :]
end

function create_adjacency_matrix(conns; n, weights=1, gpu=false)
    # TODO: Memory-inefficient implementation, remove the following code
    # nedges = size(conns, 1)
    # if length(weights) == 1
    #     weights = fill(weights, nedges)
    # elseif length(weights) == nedges
    #     weights = vcat(weights, weights)
    # end
    # am = sparse(conns[:, 1], conns[:, 2], weights, n, n)
    # return cu(am)

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

    am = SparseMatrixCSC(n, n, colPtr, rowVal, weights)

    # am = sparse(conns[:, 1], conns[:, 2], weights, n, n)

    return gpu ? cu(am) : am

    # TODO: cu(am) does the job, remove the following code
    # colPtr = CuVector{Cint}(colPtr)
    # rowVal = CuVector{Cint}(rowVal)
    # nzVal = CuVector{Cfloat}(weights)
    # dims = (n, n)
    # CUSPARSE.CuSparseMatrixCSC(colPtr, rowVal, nzVal, dims)
end

function find_boundary_nodes(im, face)
    nnodes = sum(im)
    indices = fill(-1, size(im))
    indices[im] .= 1:nnodes

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
