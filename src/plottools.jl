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
    imshow(img; slice=1)

Display a 2D heatmap of `img` (or a z-slice of a 3D array) using Plots.jl
with equal aspect ratio and viridis colormap clamped to `[0, 1]`.
"""
function imshow(img; slice=1)
    img_slice = ndims(img) == 3 ? img[:, :, slice] : img
    return heatmap(img_slice; aspect_ratio=:equal, color=:viridis, clim=(0, 1))
end
