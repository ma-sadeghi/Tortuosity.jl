# NOTE: Why do we only compute the rate along the specified axis? What about
#  the lateral rates (like what Transport.rate does in OpenPNM)? We can do
#  that, but it won't make a difference because the lateral rates get cancelled
#  out. Picture two adjacent pores i, j: The rate from i to j is the same as
#  the rate from j to i, but with opposite sign. So when we sum the rates, the
#  lateral rates cancel out.

function effective_diffusivity(
    scalar_field; axis, slice=1, D=nothing, L=nothing, Δc=nothing
)
    axis_idx = Dict(:x => 1, :y => 2, :z => 3)[axis]
    L = isnothing(L) ? size(scalar_field)[axis_idx] : L
    # NOTE: If L is provided, don't use it to compute A since it might not be the actual L
    A = prod(size(scalar_field)) / size(scalar_field, axis_idx)
    c₁ = nanmean(selectdim(scalar_field, axis_idx, 1))
    c₂ = nanmean(selectdim(scalar_field, axis_idx, size(scalar_field)[axis_idx]))
    Δc = isnothing(Δc) ? c₁ - c₂ : Δc

    # Extract the conductance of first and second slices if D is provided
    # NOTE: voxel size = 1 => gd = D⋅A/ℓ = D
    D1 = isnothing(D) ? 1.0 : selectdim(D, axis_idx, slice)
    D2 = isnothing(D) ? 1.0 : selectdim(D, axis_idx, slice + 1)
    D12 = 1 ./ (0.5 ./ D1 + 0.5 ./ D2)

    first_slice = selectdim(scalar_field, axis_idx, slice)
    second_slice = selectdim(scalar_field, axis_idx, slice + 1)
    rate = nansum(D12 .* (first_slice - second_slice))
    Deff = rate * (L - 1) / A / Δc
    return Deff
end

function tortuosity(c; axis, slice=1, eps=nothing, D=nothing, L=nothing, Δc=nothing)
    # NOTE: Assumes that c is NaN-filled outside the pore space
    ε = isnothing(eps) ? sum(isfinite.(c)) / prod(size(c)) : eps
    Deff = effective_diffusivity(c; axis=axis, slice=slice, D=D, L=L, Δc=Δc)
    return ε / Deff
end

function formation_factor(c; axis, slice=1, D=nothing, L=nothing, Δc=nothing)
    Deff = effective_diffusivity(c; axis=axis, slice=slice, D=D, L=L, Δc=Δc)
    return 1 / Deff
end
