function spdiagm(v::CuArray)
    nnz = length(v)
    colPtr = cu(collect(1:nnz+1))
    rowVal = cu(collect(1:nnz))
    nzVal = v
    dims = (nnz, nnz)
    return CUSPARSE.CuSparseMatrixCSC(colPtr, rowVal, nzVal, dims)
end


function laplacian(am)
    degrees = vec(sum(am, dims=2))
    degree_matrix = SparseArrays.spdiagm(degrees)
    return degree_matrix - am
end


function create_connectivity_list_L(im)
    im = ndims(im) == 2 ? reshape(im, size(im)..., 1) : im
    nnodes = sum(im)
    nx, ny, nz = size(im)
    idx = fill(-1, size(im))
    idx[im] .= 1:nnodes
    xconns = hcat(idx[1:nx-1, :, :][:], idx[2:nx, :, :][:])
    yconns = hcat(idx[:, 1:ny-1, :][:], idx[:, 2:ny, :][:])
    zconns = hcat(idx[:, :, 1:nz-1][:], idx[:, :, 2:nz][:])
    conns = vcat(xconns, yconns, zconns)
    nodes = findall(row -> all(row .!= -1), eachrow(conns))
    conns = conns[nodes, :]
end


function create_connectivity_list🚀_L(im)
    im = ndims(im) == 2 ? reshape(im, size(im)..., 1) : im
    nx, ny, nz = size(im)

    idx = fill(-1, size(im))
    idx[im] .= 1:sum(im)

    # Generate possible connections
    xconns = hcat(idx[1:nx-1, :, :][:], idx[2:nx, :, :][:])
    yconns = hcat(idx[:, 1:ny-1, :][:], idx[:, 2:ny, :][:])
    zconns = hcat(idx[:, :, 1:nz-1][:], idx[:, :, 2:nz][:])
    conns = vcat(xconns, yconns, zconns)

    # Filter connections
    mask = .!any(conns .== -1, dims=2)
    return conns[mask[:,1], :]
end


# NOTE: This function has two implementations. The first one is using a copy
# of the image to store the indices of the nodes. The second one is using a
# dictionary to store the indices of the nodes. The former is faster, but
# memory inefficient, especially for low porosity images. The latter is slower,
# but memory efficient.
function create_connectivity_list🚀(im)
    im = ndims(im) == 2 ? reshape(im, size(im)..., 1) : im
    nx, ny, nz = size(im)

    idx = fill(-1, size(im))
    idx[im] .= 1:sum(im)

    # Uncomment the following lines to use the dictionary implementation
    # s2i = LinearIndices(im)
    # n2c = reverse_lookup(im)

    total_conns = count(im) * 3
    conns = Matrix{Int}(undef, total_conns, 2)
    row = 0

    # x-connections
    for k in 1:nz, j in 1:ny, i in 1:nx-1
        if im[i, j, k] && im[i+1, j, k]
            row += 1
            conns[row, :] .= idx[i, j, k], idx[i+1, j, k]
            # conns[row, :] .= n2c[s2i[i, j, k]], n2c[s2i[i+1, j, k]]
        end
    end

    # y-connections
    for k in 1:nz, j in 1:ny-1, i in 1:nx
        if im[i, j, k] && im[i, j+1, k]
            row += 1
            conns[row, :] .= idx[i, j, k], idx[i, j+1, k]
            # conns[row, :] .= n2c[s2i[i, j, k]], n2c[s2i[i, j+1, k]]
        end
    end

    # z-connections
    for k in 1:nz-1, j in 1:ny, i in 1:nx
        if im[i, j, k] && im[i, j, k+1]
            row += 1
            conns[row, :] .= idx[i, j, k], idx[i, j, k+1]
            # conns[row, :] .= n2c[s2i[i, j, k]], n2c[s2i[i, j, k+1]]
        end
    end

    # Resize the connections matrix
    return @view conns[1:row, :]
end


function create_adjacency_matrix(conns; n, weights=1)
    nedges = size(conns, 1)
    conns = vcat(conns, conns[:, [2, 1]])
    if length(weights) == 1
        weights = fill(weights, nedges*2)
    elseif length(weights) == nedges
        weights = vcat(weights, weights)
    end
    sparse(conns[:, 1], conns[:, 2], weights, n, n)
end


function find_boundary_nodes(im, face)
    nnodes = sum(im)
    indices = fill(-1, size(im))
    indices[im] .= 1:nnodes

    face_dict = Dict(
        :left   => indices[1, :, :],
        :right  => indices[end, :, :],
        :bottom => indices[:, :, 1],
        :top    => indices[:, :, end],
        :front  => indices[:, 1, :],
        :back   => indices[:, end, :]
    )

    nodes = face_dict[face][:]
    return nodes[nodes .> 0]
end
