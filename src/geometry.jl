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

#get the indexes of the pore only 1D vector corresponding to a slice in 3D space
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