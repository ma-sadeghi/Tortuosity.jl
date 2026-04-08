module Imaginator

using ImageFiltering
using ImageMorphology
using Images
using Random
using SpecialFunctions
using Statistics

"""
    norm_to_uniform(img; scale=(minimum(img), maximum(img)))

Transform pixel values to a uniform distribution via the Gaussian CDF, then
rescale to `[lb, ub]`. This is the standard PoreSpy normalization: standardize
→ erfc CDF → normalize to `[0, 1]` → rescale to `scale`.

# Keyword Arguments
- `scale`: `(lb, ub)` tuple for the output range. Default: input min/max.
"""
function norm_to_uniform(img; scale=(minimum(img), maximum(img)))
    lb, ub = scale
    img = (img .- mean(img)) / std(img)
    img = 1 / 2 * erfc.(-img / sqrt(2))
    # Normalize to [0, 1] using actual post-erfc bounds, then rescale to [lb, ub]
    img = (img .- minimum(img)) / (maximum(img) - minimum(img))
    return img * (ub - lb) .+ lb
end

"""
    apply_gaussian_blur(img, sigma)

Apply an isotropic Gaussian blur with standard deviation `sigma` in each dimension
using symmetric boundary padding.
"""
function apply_gaussian_blur(img, sigma)
    sigma = tuple(fill(sigma, ndims(img))...)
    kernel = Kernel.gaussian(sigma)
    return imfilter(img, kernel, "symmetric")
end

"""
    to_binary(img, threshold=0.5)

Threshold `img` into a `BitArray`: voxels with value `< threshold` become `true`.
"""
function to_binary(img, threshold=0.5)
    return map(x -> x < threshold ? true : false, img)
end

"""
    disk(r)

Create a 2D circular structuring element of radius `r` as a `BitMatrix`.
"""
function disk(r)
    return Bool.([
        sqrt((i - r - 1)^2 + (j - r - 1)^2) <= r for i in 1:(2 * r + 1), j in 1:(2 * r + 1)
    ])
end

"""
    ball(r)

Create a 3D spherical structuring element of radius `r` as a `BitArray{3}`.
"""
function ball(r)
    return Bool.([
        sqrt((i - r - 1)^2 + (j - r - 1)^2 + (k - r - 1)^2) <= r for i in 1:(2 * r + 1),
        j in 1:(2 * r + 1), k in 1:(2 * r + 1)
    ])
end

"""
    denoise(img, kernel_radius)

Apply morphological closing then opening to remove small noise features.
Uses a `disk` (2D) or `ball` (3D) structuring element of the given radius.
"""
function denoise(img, kernel_radius)
    selem = ndims(img) == 3 ? ball(kernel_radius) : disk(kernel_radius)
    img = closing(img, selem)
    return opening(img, selem)
end

"""
    blobs(; shape, porosity, blobiness, seed=nothing)

Generate a random binary porous image using Gaussian-blurred white noise.
Higher `blobiness` produces finer features; lower values produce coarser blobs.
The algorithm: random noise → Gaussian blur (σ = mean(shape) / 40 / blobiness)
→ uniform normalization → threshold at `porosity`.

# Keyword Arguments
- `shape`: tuple of image dimensions, e.g. `(64, 64)` or `(64, 64, 64)`.
- `porosity`: target pore fraction in `[0, 1]`.
- `blobiness`: controls feature size (higher = finer features).
- `seed`: random seed for reproducibility. Default: `nothing`.
"""
function blobs(; shape, porosity, blobiness, seed=nothing)
    Random.seed!(seed)
    im = rand(shape...)
    sigma = mean(shape) / 40 / blobiness
    im = apply_gaussian_blur(im, sigma)
    im = norm_to_uniform(im; scale=(0, 1))
    return to_binary(im, porosity)
end

"""
    faces(shape; inlet=nothing, outlet=nothing)

Create a boolean mask of the given `shape` with `true` on the specified boundary
faces. `inlet` and `outlet` are dimension indices (1, 2, or 3): `inlet` marks
the first slice, `outlet` marks the last slice along that dimension.
At least one of `inlet` or `outlet` must be provided.
"""
function faces(shape; inlet=nothing, outlet=nothing)
    if isnothing(inlet) && isnothing(outlet)
        error("Must provide at least one `inlet` or `outlet`")
    end
    img = zeros(Bool, shape)
    !isnothing(inlet) ? selectdim(img, inlet, 1) .= true : nothing
    !isnothing(outlet) ? selectdim(img, outlet, size(img)[outlet]) .= true : nothing
    return img
end

"""
    trim_nonpercolating_paths(img; axis)

Remove pore clusters that do not percolate from inlet to outlet along `axis`.
Returns a new boolean image containing only the connected pore space that
spans the full domain along the specified axis.

# Keyword Arguments
- `axis`: percolation direction (`:x`, `:y`, or `:z`).
"""
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

"""
    phase_fraction(img, label)
    phase_fraction(img, labels::AbstractArray)
    phase_fraction(img)

Compute the volume fraction of a phase in `img`.

- Single `label`: fraction of voxels equal to `label`.
- Array of `labels`: sum of individual phase fractions.
- No label: returns a `Dict` mapping each unique value to its fraction.
"""
function phase_fraction(img, label)
    return count(img .== label) / length(img)
end

function phase_fraction(img, labels::AbstractArray)
    return sum(phase_fraction(img, label) for label in labels)
end

function phase_fraction(img)
    labels = unique(img)
    return Dict(label => phase_fraction(img, label) for label in labels)
end

end  # module Imaginator
