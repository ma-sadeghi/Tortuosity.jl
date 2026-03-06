# Steady-state effective diffusivity, tortuosity, and formation factor

# NOTE: Why do we only compute the rate along the specified axis? What about
#  the lateral rates (like what Transport.rate does in OpenPNM)? We can do
#  that, but it won't make a difference because the lateral rates get cancelled
#  out. Picture two adjacent pores i, j: The rate from i to j is the same as
#  the rate from j to i, but with opposite sign. So when we sum the rates, the
#  lateral rates cancel out.

function effective_diffusivity(c, img; axis, slice=1, D=1.0, dx=1.0, L=nothing, Δc=nothing)
    ax = axis_dim(axis)
    N = size(img, ax)
    L = isnothing(L) ? (N - 1) * dx : L
    Δc = isnothing(Δc) ? nanmean(selectdim(c, ax, 1)) - nanmean(selectdim(c, ax, N)) : Δc
    J = get_flux(c, D, dx, img, axis; ind=slice)
    return J * L / Δc
end

function tortuosity(c, img; axis, slice=1, eps=nothing, D=1.0, dx=1.0, L=nothing, Δc=nothing)
    ε = isnothing(eps) ? Imaginator.phase_fraction(img, true) : eps
    Deff = effective_diffusivity(c, img; axis, slice, D, dx, L, Δc)
    return ε / Deff
end

function formation_factor(c, img; axis, slice=1, D=1.0, dx=1.0, L=nothing, Δc=nothing)
    Deff = effective_diffusivity(c, img; axis, slice, D, dx, L, Δc)
    return 1 / Deff
end
