module Imaginator

using Images
using ImageFiltering
using ImageMorphology
using Random
using SpecialFunctions
using Statistics


function norm_to_uniform(im)
    lb, ub = minimum(im), maximum(im)
    im = (im .- mean(im)) / std(im)
    im = 1/2 * erfc.(-im / sqrt(2))
    im = (im .- lb) / (ub - lb)
    im = im * (ub - lb) .+ lb
end


function apply_gaussian_blur(im, sigma)
    kernel = Kernel.gaussian(fill(sigma, ndims(im)))
    imfilter(im, kernel)
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
    im = rand(Bool, shape...)
    sigma = mean(shape) / 40 / blobiness
    im = apply_gaussian_blur(im, sigma)
    im = norm_to_uniform(im)
    to_binary(im, porosity)
end

end  # module Imaginator
