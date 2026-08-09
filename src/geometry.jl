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
    slice_indices(pore_index::Array{Int,3}, axis::Symbol, idx::Int)

Return the pore-vector indices of the pore voxels on a 2D slice along `axis` at
position `idx`. `pore_index` is the lookup table built by `build_pore_index`
— solid entries are `0`, pore entries are the 1-based pore-vector position.
"""
function slice_indices(pore_index::Array{Int,3}, axis::Symbol, idx::Int)
    ax = axis_dim(axis)
    ind_slice = selectdim(pore_index, ax, idx)
    return filter(!iszero, vec(ind_slice))
end

"""
    reconstruct_slice(u, pore_index::Array{Int,3}, axis::Symbol, idx::Int)

Reconstruct a 2D slice of the concentration field from the pore-only vector `u`
at position `idx` along `axis`. Pore voxels receive their values from `u`,
solid voxels are filled with `NaN`. Mirrors [`reconstruct_field`](@ref) but for
a single axial slice.
"""
function reconstruct_slice(u, pore_index::Array{Int,3}, axis::Symbol, idx::Int)
    ax = axis_dim(axis)
    ind_slice = selectdim(pore_index, ax, idx)
    c = fill(NaN, size(ind_slice))
    mask = ind_slice .> 0
    inds = ind_slice[mask]
    # Gather on whichever device `u` lives on and bring back only the slice.
    # `Array(u)[inds]` would copy the entire solution to the host to read one
    # face of it — at 800³ that is 1 GB moved to serve 2.6 MB, on every call,
    # and `flux` calls this twice each time a stop condition fires.
    c[mask] .= if _on_gpu(u)
        d_inds = similar(u, Int32, length(inds))
        copyto!(d_inds, Int32.(inds))
        vals = Array(u[d_inds])
        _free!(d_inds)
        vals
    else
        u[inds]
    end
    return c
end
