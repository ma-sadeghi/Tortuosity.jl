# Axis helpers: map axis symbols to dimension indices, face names, and orthogonal dims

function axis_dim(ax::Symbol)
    d = Dict(:x => 1, :y => 2, :z => 3)
    haskey(d, ax) || error("axis must be :x, :y, or :z")
    return d[ax]
end

function axis_faces(ax::Symbol)
    d = Dict(:x => (:left, :right), :y => (:front, :back), :z => (:bottom, :top))
    haskey(d, ax) || error("axis must be :x, :y, or :z")
    return d[ax]
end

function orthogonal_dims(ax::Symbol)
    d = Dict(:x => (2, 3), :y => (1, 3), :z => (1, 2))
    haskey(d, ax) || error("axis must be :x, :y, or :z")
    return d[ax]
end
orthogonal_dims(d::Int) = orthogonal_dims((:x, :y, :z)[d])

# Extract a 2D slice without reconstructing the full 3D concentration field

function slice_vec_indices(img::BitArray, grid_to_vec::Array{Int}, axis::Symbol, idx::Int)
    ax = axis_dim(axis)
    ind_slice = selectdim(grid_to_vec, ax, idx)
    img_slice = selectdim(img, ax, idx)

    return vec(ind_slice[img_slice])
end

function vec_to_slice(u, img::BitArray, grid_to_vec::Array{Int}, axis::Symbol, idx::Int)
    @assert length(u) == count(img) "Length of u must match the number of true voxels in img"

    ax = axis_dim(axis)
    ind_slice = selectdim(grid_to_vec, ax, idx)
    img_slice = selectdim(img, ax, idx)

    c = fill(NaN, size(img_slice))
    c[img_slice] .= Array(u)[vec(ind_slice[img_slice])]
    return c
end
