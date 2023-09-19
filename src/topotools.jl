using SparseArrays


function laplacian(adjacency)
    degrees = vec(sum(adjacency, dims=2))
    degree_matrix = SparseArrays.spdiagm(0 => degrees)
    return degree_matrix - adjacency
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


function create_connectivity_list🚀(im)
    im = ndims(im) == 2 ? reshape(im, size(im)..., 1) : im
    nx, ny, nz = size(im)

    idx = fill(-1, size(im))
    idx[im] .= 1:sum(im)

    total_conns = (nx-1)*ny*nz + nx*(ny-1)*nz + nx*ny*(nz-1)
    conns = Matrix{Int}(undef, total_conns, 2)

    # Create connections
    ptr = 1

    # x-connections
    for k in 1:nz
        for j in 1:ny
            for i in 1:nx-1
                if im[i, j, k] == 1 && im[i+1, j, k] == 1
                    conns[ptr, :] .= idx[i, j, k], idx[i+1, j, k]
                    ptr += 1
                end
            end
        end
    end

    # y-connections
    for k in 1:nz
        for j in 1:ny-1
            for i in 1:nx
                if im[i, j, k] == 1 && im[i, j+1, k] == 1
                    conns[ptr, :] .= idx[i, j, k], idx[i, j+1, k]
                    ptr += 1
                end
            end
        end
    end

    # z-connections
    for k in 1:nz-1
        for j in 1:ny
            for i in 1:nx
                if im[i, j, k] == 1 && im[i, j, k+1] == 1
                    conns[ptr, :] .= idx[i, j, k], idx[i, j, k+1]
                    ptr += 1
                end
            end
        end
    end

    # Resize the connections matrix
    return conns[1:ptr-1, :]
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
