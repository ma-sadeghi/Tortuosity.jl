using Plots


function imshow(im; z_idx=1)
    im_slice = ndims(im) == 3 ? im[:, :, z_idx] : im
    heatmap(im_slice, aspect_ratio=:equal, color=:viridis, clim=(0, 1))
end
