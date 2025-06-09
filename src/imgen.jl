module Imaginator

using ImageFiltering
using ImageMorphology
using Images
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
    sigma = tuple(fill(sigma, ndims(img))...)
    kernel = Kernel.gaussian(sigma)
    return imfilter(img, kernel, "symmetric")
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

function faces(shape; inlet=nothing, outlet=nothing)
    if isnothing(inlet) && isnothing(outlet)
        error("Must provide at least one `inlet` or `outlet`")
    end
    img = zeros(Bool, shape)
    !isnothing(inlet) ? selectdim(img, inlet, 1) .= true : nothing
    !isnothing(outlet) ? selectdim(img, outlet, size(img)[outlet]) .= true : nothing
    return img
end

function trim_nonpercolating_paths(img; axis)
    shape = size(img)
    dim = Dict(:x => 1, :y => 2, :z => 3)[axis]
    inlet = faces(shape; inlet=dim)
    outlet = faces(shape; outlet=dim)
    labels = label_components(img)
    labels_percolating = intersect(labels[inlet], labels[outlet])
    setdiff!(labels_percolating, 0)  # Remove background label
    img_percolating = in.(labels, Ref(Set(labels_percolating)))  # Ref to avoid broadcasting
    return img_percolating
end

function phase_fraction(img, labels)
    return sum(sum(img .== label) for label in labels) / length(img)
end

function phase_fraction(img)
    labels = unique(img)
    return Dict(label => phase_fraction(img, label) for label in labels)
end

end  # module Imaginator
