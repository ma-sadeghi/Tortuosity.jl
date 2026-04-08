"""
    set_plotsjl_defaults()

Apply default Plots.jl settings: 600×400 size, 1.25× thickness and font scaling.
"""
function set_plotsjl_defaults()
    default()
    default(; size=(600, 400), thickness_scaling=1.25)
    return scalefontsizes(1.25)
end

"""
    imshow(im; slice=1)

Display a 2D heatmap of `im` (or a z-slice of a 3D array) using Plots.jl
with equal aspect ratio and viridis colormap clamped to `[0, 1]`.
"""
function imshow(im; slice=1)
    im_slice = ndims(im) == 3 ? im[:, :, slice] : im
    return heatmap(im_slice; aspect_ratio=:equal, color=:viridis, clim=(0, 1))
end

# function imshow(im; vals=nothing, slice=1)
#     fillval = isa(vals, Nothing) ? 0 : NaN
#     u = fill(fillval, size(im))
#     u[im] .= isa(vals, Nothing) ? 1 : Array(vals)
#     u_slice = ndims(im) == 3 ? u[:, :, slice] : u
#     heatmap(u_slice, aspect_ratio=:equal, color=:viridis, clim=(0, 1))
# end
