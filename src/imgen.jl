module Imaginator

using ImageFiltering
using ImageMorphology
using Images
using PythonCall
using Random
using SpecialFunctions
using Statistics

function norm_to_uniform(img; scale=(minimum(img), maximum(img)))
    lb, ub = scale
    img = (img .- mean(img)) / std(img)
    img = 1 / 2 * erfc.(-img / sqrt(2))
    img = (img .- lb) / (ub - lb)
    return img = img * (ub - lb) .+ lb
end

function apply_gaussian_blur(img, sigma)
    # sigma = tuple(fill(sigma, ndims(im))...)
    # kernel = Kernel.gaussian(sigma)
    # imfilter(im, kernel, "symmetric")
    ndi = pyimport("scipy.ndimage")
    im_f = ndi.gaussian_filter(img, sigma)
    return pyconvert(Array, im_f)
end

function to_binary(img, threshold=0.5)
    return map(x -> x < threshold ? true : false, img)
end

function disk(r)
    return Bool.([
        sqrt((i - r - 1)^2 + (j - r - 1)^2) <= r for i in 1:(2 * r + 1), j in 1:(2 * r + 1)
    ])
end

function ball(r)
    return Bool.([
        sqrt((i - r - 1)^2 + (j - r - 1)^2 + (k - r - 1)^2) <= r for i in 1:(2 * r + 1),
        j in 1:(2 * r + 1), k in 1:(2 * r + 1)
    ])
end

function denoise(img, kernel_radius)
    selem = ndims(img) == 3 ? ball(kernel_radius) : disk(kernel_radius)
    img = closing(img, selem)
    return opening(img, selem)
end

function blobs(; shape, porosity, blobiness, seed=nothing)
    Random.seed!(seed)
    im = rand(shape...)
    sigma = mean(shape) / 40 / blobiness
    im = apply_gaussian_blur(im, sigma)
    im = norm_to_uniform(im; scale=(0, 1))
    return to_binary(im, porosity)
end

function trim_nonpercolating_paths(img, axis)
    ps = pyimport("porespy")
    shape = img isa Py ? img.shape : size(img)
    axis_idx = Dict(:x => 1, :y => 2, :z => 3)[axis]
    inlet = ps.generators.faces(shape; inlet=axis_idx - 1)  # Python 0-based indexing
    outlet = ps.generators.faces(shape; outlet=axis_idx - 1)  # Python 0-based indexing
    img = ps.filters.trim_nonpercolating_paths(img; inlets=inlet, outlets=outlet)
    return pyconvert(Array{Bool}, img)
end

end  # module Imaginator
