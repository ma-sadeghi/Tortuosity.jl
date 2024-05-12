module Imaginator

using Images
using ImageFiltering
using ImageMorphology
using Random
using SpecialFunctions
using Statistics


function norm_to_uniform(im; scale=(minimum(im), maximum(im)))
    lb, ub = scale
    im = (im .- mean(im)) / std(im)
    im = 1/2 * erfc.(-im / sqrt(2))
    im = (im .- lb) / (ub - lb)
    im = im * (ub - lb) .+ lb
end


function apply_gaussian_blur(im, sigma)
    r = size(im) .* 2 .- 1
    sigma = tuple(fill(sigma, ndims(im))...)
    kernel = Kernel.gaussian(sigma, r)
    imfilter(im, kernel, "symmetric")
end


function to_binary(im, threshold=0.5)
    map(x -> x < threshold ? true : false, im)
end


function disk(r)
    Bool.([sqrt((i - r - 1)^2 + (j - r - 1)^2) <= r for i in 1:2*r+1, j in 1:2*r+1])
end


function ball(r)
    Bool.([sqrt((i - r - 1)^2 + (j - r - 1)^2 + (k - r - 1)^2) <= r for i in 1:2*r+1, j in 1:2*r+1, k in 1:2*r+1])
end


function denoise(im, kernel_radius)
    selem = ndims(im) == 3 ? ball(kernel_radius) : disk(kernel_radius)
    im = closing(im, selem)
    opening(im, selem)
end


function blobs(;shape, porosity, blobiness, seed=nothing)
    Random.seed!(seed)
    im = rand(shape...)
    sigma = mean(shape) / 40 / blobiness
    im = apply_gaussian_blur(im, sigma)
function trim_nonpercolating_paths(img, axis)
    ps = pyimport("porespy")
    axis_idx = Dict(:x => 1, :y => 2, :z => 3)[axis]
    inlet = ps.generators.faces(size(img); inlet=axis_idx - 1)  # Python 0-based indexing
    outlet = ps.generators.faces(size(img); outlet=axis_idx - 1)  # Python 0-based indexing
    img = ps.filters.trim_nonpercolating_paths(img; inlets=inlet, outlets=outlet)
    return pyconvert(Array{Bool,3}, img)
end

end  # module Imaginator
