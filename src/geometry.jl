# Axis helpers: map axis symbols to dimension indices, face names, and orthogonal dims

function axis_dim(ax::Symbol)
    return if ax == :x
        1
    elseif ax == :y
        2
    elseif ax == :z
        3
    else
        error("axis must be :x, :y, or :z")
    end
end

function axis_faces(ax::Symbol)
    return if ax == :x
        (:left, :right)
    elseif ax == :y
        (:front, :back)
    elseif ax == :z
        (:bottom, :top)
    else
        error("axis must be :x, :y, or :z")
    end
end

orthogonal_dims(ax::Symbol) = orthogonal_dims(axis_dim(ax))
function orthogonal_dims(d::Int)
    return if d == 1
        (2, 3)
    elseif d == 2
        (1, 3)
    elseif d == 3
        (1, 2)
    else
        error("dim must be 1, 2, or 3")
    end
end

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
