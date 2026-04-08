# Axis helpers: map axis symbols to dimension indices, face names, and orthogonal dims

"""
    axis_dim(ax::Symbol) -> Int

Convert an axis symbol (`:x`, `:y`, `:z`) to its dimension index (1, 2, 3).
"""
function axis_dim(ax::Symbol)
    ax === :x && return 1
    ax === :y && return 2
    ax === :z && return 3
    error("axis must be :x, :y, or :z, got :$ax")
end

"""
    axis_faces(ax::Symbol) -> Tuple{Symbol, Symbol}

Return the `(inlet, outlet)` face names for a given axis:
`:x` → `(:left, :right)`, `:y` → `(:front, :back)`, `:z` → `(:bottom, :top)`.
"""
function axis_faces(ax::Symbol)
    ax === :x && return (:left, :right)
    ax === :y && return (:front, :back)
    ax === :z && return (:bottom, :top)
    error("axis must be :x, :y, or :z, got :$ax")
end

"""
    orthogonal_dims(ax::Symbol) -> Tuple{Int, Int}
    orthogonal_dims(d::Int) -> Tuple{Int, Int}

Return the two dimension indices orthogonal to `ax`:
`:x` → `(2, 3)`, `:y` → `(1, 3)`, `:z` → `(1, 2)`.
"""
function orthogonal_dims(ax::Symbol)
    ax === :x && return (2, 3)
    ax === :y && return (1, 3)
    ax === :z && return (1, 2)
    error("axis must be :x, :y, or :z, got :$ax")
end
orthogonal_dims(d::Int) = orthogonal_dims((:x, :y, :z)[d])

# Extract a 2D slice without reconstructing the full 3D concentration field

"""
    slice_vec_indices(img, grid_to_vec, axis, idx)

Return the vector indices corresponding to the pore voxels in a 2D slice of a
3D porous medium. Extracts a single slice along `axis` at position `idx`, then
returns the 1D vector indices (into the pore-only vectorization) for the pore
voxels in that slice.
"""
function slice_vec_indices(img::BitArray, grid_to_vec::Array{Int}, axis::Symbol, idx::Int)
    ax = axis_dim(axis)
    ind_slice = selectdim(grid_to_vec, ax, idx)
    img_slice = selectdim(img, ax, idx)

    return vec(ind_slice[img_slice])
end

"""
    vec_to_slice(u, img, grid_to_vec, axis, idx)

Reconstruct a 2D slice of the concentration field from a pore-only 1D vector.
Pore voxels receive their values from `u`, solid voxels are filled with `NaN`.
"""
function vec_to_slice(u, img::BitArray, grid_to_vec::Array{Int}, axis::Symbol, idx::Int)
    @assert length(u) == count(img) "Length of u must match the number of true voxels in img"

    ax = axis_dim(axis)
    ind_slice = selectdim(grid_to_vec, ax, idx)
    img_slice = selectdim(img, ax, idx)

    c = fill(NaN, size(img_slice))
    c[img_slice] .= Array(u)[vec(ind_slice[img_slice])]
    return c
end
