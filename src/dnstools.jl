using NaNStatistics


function compute_effective_diffusivity(scalar_field, axis)
    Δc = 1.0
    axis_idx = Dict(:x => 1, :y => 2, :z => 3)[axis]
    L = size(scalar_field)[axis_idx]
    A = prod(size(scalar_field)) / L

    # Extract slices based on the specified axis
    function slice_at_dim(dim, index)
        return ntuple(i -> i == dim ? index : :, 3)
    end

    first_slice = scalar_field[slice_at_dim(axis_idx, 1)...]
    second_slice = scalar_field[slice_at_dim(axis_idx, 2)...]
    rate = nansum(first_slice - second_slice)
    Deff = rate * (L-1) / A / Δc
    return Deff
end


function compute_formation_factor(c, axis)
    Deff = compute_effective_diffusivity(c, axis)
    return 1 / Deff
end


function compute_tortuosity_factor(c, axis)
    # !: Assumes that c is NaN-filled outside the pore space
    ε = sum(isfinite.(c)) / prod(size(c))
    Deff = compute_effective_diffusivity(c, axis)
    return ε / Deff
end
