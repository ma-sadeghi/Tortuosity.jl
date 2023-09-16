using SparseArrays


function laplacian(adjacency)
    degrees = vec(sum(adjacency, dims=2))
    degree_matrix = SparseArrays.spdiagm(0 => degrees)
    return degree_matrix - adjacency
end


function create_connectivity_list(im)
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


function create_adjacency_matrix(conns; nnodes, weights=1)
    nedges = size(conns, 1)
    conns = vcat(conns, conns[:, [2, 1]])
    if length(weights) == 1
        weights = fill(weights, nedges*2)
    elseif length(weights) == nedges
        weights = vcat(weights, weights)
    end
    sparse(conns[:, 1], conns[:, 2], weights, nnodes, nnodes)
end


function find_boundary_nodes(im, face)
    nnodes = sum(im)
    indices = fill(-1, size(im))
    indices[im] .= 1:nnodes

    face_dict = Dict(
        :left   => indices[1, :, :],
        :right  => indices[end, :, :],
        :bottom => indices[:, 1, :],
        :top    => indices[:, end, :],
        :front  => indices[:, :, 1],
        :back   => indices[:, :, end]
    )

    nodes = face_dict[face][:]
    return nodes[nodes .> 0]
end
