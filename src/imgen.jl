module Imaginator

using ImageMorphology
using Random
using SpecialFunctions
using Statistics

# Hooks for the optional dependencies this submodule needs; see src/weakdeps.jl.
using ..Tortuosity: _gaussian_blur

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
    mu, sd = mean(img), std(img)
    # Two passes over one buffer. Only the reductions force a pass boundary: the
    # CDF map needs `mean`/`std` of the input, and the rescale needs the extrema
    # of the CDF output. Everything between them is elementwise, so writing it as
    # a chain of separate expressions would materialise ten full-size temporaries
    # — 41 GB at 800³ — for arithmetic that fits in a register.
    out = @. 1 / 2 * erfc(-((img - mu) / sd) / sqrt(2))
    # Normalize to [0, 1] using actual post-erfc bounds, then rescale to [lb, ub]
    lo, hi = extrema(out)
    @. out = (out - lo) / (hi - lo) * (ub - lb) + lb
    return out
end

"""
    apply_gaussian_blur(img, sigma)

Apply an isotropic Gaussian blur with standard deviation `sigma` in each dimension
using symmetric boundary padding.

Requires `ImageFiltering.jl`, an optional dependency that does not install with
Tortuosity: run `Pkg.add("ImageFiltering")`, then `using ImageFiltering`, before
calling this.
"""
function apply_gaussian_blur(img, sigma)
    return _gaussian_blur(img, sigma)
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

Requires `ImageFiltering.jl`, an optional dependency that does not install with
Tortuosity: run `Pkg.add("ImageFiltering")`, then `using ImageFiltering`, before
calling this.
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
    # `label_components` numbers components densely from 1, with 0 for
    # background, so a label is already an index: membership is a table lookup
    # rather than a hash probe. The probe ran once per voxel — 512M times at
    # 800³, single-threaded — which was the real cost of this function.
    keep = zeros(Bool, maximum(labels) + 1)
    keep[labels_percolating .+ 1] .= true
    img_percolating = (label -> @inbounds keep[label + 1]).(labels)
    return img_percolating
end

"""
Count pore voxels in components that connect both transport faces.

This is the allocation-lean scalar counterpart of
[`trim_nonpercolating_paths`](@ref). It labels into 32-bit storage whenever the
image fits and counts matching labels directly, rather than allocating two
full face masks and a second image only for its population count.
"""
function _count_percolating(img; axis)
    Ti = length(img) <= typemax(Int32) ? Int32 : Int
    return _count_percolating(img, Val(axis), Ti)
end

_boundary_slice(a, ::Val{:x}, i) = @view a[i, :, :]
_boundary_slice(a, ::Val{:y}, i) = @view a[:, i, :]
_boundary_slice(a, ::Val{:z}, i) = @view a[:, :, i]
_axis_length(a, ::Val{:x}) = size(a, 1)
_axis_length(a, ::Val{:y}) = size(a, 2)
_axis_length(a, ::Val{:z}) = size(a, 3)

function _count_percolating(img, axis, ::Type{Ti}) where {Ti<:Integer}
    labels = similar(img, Ti)
    label_components!(labels, img)

    inlet = _boundary_slice(labels, axis, 1)
    outlet = _boundary_slice(labels, axis, _axis_length(labels, axis))
    max_boundary = max(maximum(inlet), maximum(outlet))
    max_boundary == 0 && return 0

    touches_inlet = falses(max_boundary)
    percolating = falses(max_boundary)
    @inbounds for label in inlet
        label > 0 && (touches_inlet[label] = true)
    end
    @inbounds for label in outlet
        label > 0 && touches_inlet[label] && (percolating[label] = true)
    end

    n = 0
    @inbounds for label in labels
        n += label > 0 && label <= max_boundary && percolating[label]
    end
    return n
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
