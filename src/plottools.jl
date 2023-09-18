using Plots


function set_plotsjl_defaults()
    default()
    default(size=(600, 400), thickness_scaling=1.25)
    scalefontsizes(1.25)
end


# function imshow(im; z_idx)
#     im_slice = ndims(im) == 3 ? im[:, :, z_idx] : im
#     heatmap(im_slice, aspect_ratio=:equal, color=:viridis, clim=(0, 1))
# end


function imshow(im; vals=nothing, z_idx=1)
    fillval = isa(vals, Nothing) ? 0 : NaN
    u = fill(fillval, size(im))
    u[im] .= isa(vals, Nothing) ? 1 : Array(vals)
    u_slice = ndims(im) == 3 ? u[:, :, z_idx] : u
    heatmap(u_slice, aspect_ratio=:equal, color=:viridis, clim=(0, 1))
end


function vec_to_field(u, im)
    c = fill(NaN, size(im))
    c[im] = Array(u)
    return c
end
