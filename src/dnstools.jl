using NaNStatistics


# NOTE: Why do we only compute the rate along the specified axis? What about
# the lateral rates (like what Transport.rate does in OpenPNM)? We can do
# that, but it won't make a difference because the lateral rates get cancelled
# out. Picture two adjacent pores i, j: The rate from i to j is the same as
# the rate from j to i, but with opposite sign. So when we sum the rates, the
# lateral rates cancel out.

function effective_diffusivity(scalar_field, axis; D=nothing)
    Δc = 1.0
    axis_idx = Dict(:x => 1, :y => 2, :z => 3)[axis]
    L = size(scalar_field)[axis_idx]
    A = prod(size(scalar_field)) / L

    # Extract slices based on the specified axis
    function slice_at_dim(dim, index)
        return ntuple(i -> i == dim ? index : :, 3)
    end

    # Extract the conductance of first and second slices if D is provided
    # NOTE: voxel size = 1 => gd = D⋅A/ℓ = D
    D1 = D === nothing ? 1.0 : D[slice_at_dim(axis_idx, 1)...]
    D2 = D === nothing ? 1.0 : D[slice_at_dim(axis_idx, 2)...]
    D12 = 1 ./ (0.5 ./ D1 + 0.5 ./ D2)

    first_slice = scalar_field[slice_at_dim(axis_idx, 1)...]
    second_slice = scalar_field[slice_at_dim(axis_idx, 2)...]
    rate = nansum(D12 .* (first_slice - second_slice))
    Deff = rate * (L-1) / A / Δc
    return Deff
end


function tortuosity(c, axis; D=nothing, eps=nothing)
    # !: Assumes that c is NaN-filled outside the pore space
    ε = eps === nothing ? sum(isfinite.(c)) / prod(size(c)) : eps
    Deff = effective_diffusivity(c, axis, D=D)
    return ε / Deff
end


function formation_factor(c, axis; D=nothing)
    Deff = effective_diffusivity(c, axis, D=D)
    return 1 / Deff
end


function phase_fraction(img, labels)
    return sum(sum(img .== label) for label in labels) / length(img)
end


function phase_fraction(img)
    labels = unique(img)
    return Dict(label => phase_fraction(img, label) for label in labels)
end
