## definition of axis symbols and related shorthand

const AXIS_DEFINITION = Dict(
    :x => 1,
    :y => 2,
    :z => 3
)

#used for kwargs in functions to act on dims other than primary dim (like summing over perp. dims)
const AXIS_COMPLEMENT = Dict(
    1 => (2,3),
    2 => (1,3),
    3 => (1,2),
    :x => (2,3),
    :y => (1,3),
    :z => (1,2)
)

## for going from 3D voxel image coords to 1D pore-voxel vector index
function build_grid_to_vec(img::BitArray)

    grid_to_vec = zeros(Int, size(img))
    grid_to_vec[img] = 1:count(img)
    return grid_to_vec
end

## avoid building entire 3D concentration distribution from vector when only a slice is needed

"""
    slice_vec_indices(img, grid_to_vec, axis, idx)
    slice_vec_indices(prob::TransientProblem, idx)

Return the vector indices corresponding to the pore voxels in a 2D slice of a
3D porous medium.

This function extracts a single slice of the 3D domain along the specified
`axis` at position `idx`, then returns the 1D vector indices (into the pore-only
vectorization of the domain) for the voxels that belong to the pore space in
that slice.

# Arguments
- `img::BitArray`: 3D boolean mask of the pore space (`true` = pore).
- `grid_to_vec::Array{Int}`: Mapping from 3D grid coordinates to 1D pore-vector indices.
- `axis::Symbol`: Axis along which the slice is taken (`:x`, `:y`, or `:z`).
- `idx::Int`: Slice index along the chosen axis.

# Returns
- `Vector{Int}`: The 1D indices of pore voxels belonging to the selected slice.

# Notes
- The `TransientProblem` method extracts the required fields automatically.
"""
function slice_vec_indices(img::BitArray, grid_to_vec::Array{Int}, axis::Symbol, idx::Int) 
    
    ax = AXIS_DEFINITION[axis]
    ind_slice = selectdim(grid_to_vec, ax, idx)
    img_slice = selectdim(img, ax, idx)

    return vec(ind_slice[img_slice])
end
slice_vec_indices(prob::TransientProblem, idx::Int) = 
    slice_vec_indices(prob.img, prob.grid_to_vec, prob.axis, idx)

#get a plane/slice of the 3D concentration distribution from the pore only 1D vector
#similar to vec_to_grid, but it requires grid_to_vec which makes it unique to transient struct
"""
    vec_to_slice(u, img, grid_to_vec, axis, idx)
    vec_to_slice(u, prob::TransientProblem, idx)

Reconstruct a 2D slice of the 3D concentration field from a pore-only 1D vector.

This function takes the 1D concentration vector `u` (defined only on pore
voxels), and maps its values back onto a 2D slice of the full 3D grid. Pore
voxels in the slice receive their corresponding values from `u`, while solid
voxels are filled with `NaN`.

# Arguments
- `u`: 1D concentration vector defined on pore voxels.
- `img::BitArray`: 3D boolean mask of the pore space (`true` = pore).
- `grid_to_vec::Array{Int}`: Mapping from 3D grid coordinates to 1D pore-vector indices.
- `axis::Symbol`: Axis along which the slice is taken (`:x`, `:y`, or `:z`).
- `idx::Int`: Slice index along the chosen axis.

# Returns
- `Array{Float64,2}`: A 2D array representing the slice, with pore values filled
  from `u` and solid voxels set to `NaN`.

# Notes
- The `TransientProblem` method extracts the required fields automatically.
"""
function vec_to_slice(u, img::BitArray, grid_to_vec::Array{Int}, axis::Symbol, idx::Int)
    @assert length(u) == count(img) "Length of u must match the number of true voxels in img"
    
    ax = AXIS_DEFINITION[axis]
    ind_slice = selectdim(grid_to_vec, ax, idx)
    img_slice = selectdim(img, ax, idx)

    c = fill(NaN, size(img_slice))
    c[img_slice] .= Array(u)[vec(ind_slice[img_slice])]
    return c
end
vec_to_slice(u, prob::TransientProblem, idx::Int) = 
    vec_to_slice(u, prob.img, prob.grid_to_vec, prob.axis, idx)