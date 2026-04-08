# Steady-state effective diffusivity, tortuosity, and formation factor

# NOTE: Why do we only compute the rate along the specified axis? What about
#  the lateral rates (like what Transport.rate does in OpenPNM)? We can do
#  that, but it won't make a difference because the lateral rates get cancelled
#  out. Picture two adjacent pores i, j: The rate from i to j is the same as
#  the rate from j to i, but with opposite sign. So when we sum the rates, the
#  lateral rates cancel out.

"""
    effective_diffusivity(c, img; axis, ind=1, D=1.0, dx=1.0, L=nothing, Δc=nothing)

Compute the effective diffusivity `D_eff` from a steady-state concentration field
by measuring flux through the cross-section at voxel index `ind`.

`D_eff = J * L / Δc`, where `J` is the diffusive flux at `ind`, `L` is the domain
length, and `Δc` is the concentration drop between inlet and outlet faces.

# Arguments
- `c`: concentration field (full grid, same shape as `img`).
- `img`: 3D boolean pore mask (`true` = pore).

# Keyword Arguments
- `axis`: transport direction (`:x`, `:y`, or `:z`).
- `ind`: voxel index at which flux is measured. Default: `1`.
- `D`: intrinsic diffusivity (scalar). Default: `1.0`.
- `dx`: voxel spacing. Default: `1.0`.
- `L`: domain length. Default: `(N - 1) * dx` where `N` is the number of voxels along `axis`.
- `Δc`: imposed concentration drop. Default: mean inlet minus mean outlet concentration.
"""
function effective_diffusivity(c, img; axis, ind=1, D=1.0, dx=1.0, L=nothing, Δc=nothing)
    ax = axis_dim(axis)
    N = size(img, ax)
    L = isnothing(L) ? (N - 1) * dx : L
    Δc = isnothing(Δc) ? nanmean(selectdim(c, ax, 1)) - nanmean(selectdim(c, ax, N)) : Δc
    J = compute_flux(c, D, dx, img, axis; ind=ind)
    return J * L / Δc
end

"""
    tortuosity(c, img; axis, ind=1, ε=nothing, D=1.0, dx=1.0, L=nothing, Δc=nothing)

Compute the tortuosity factor `τ = ε / D_eff` from a steady-state concentration
field. When `ε` is omitted, porosity is computed automatically from `img`.

# Arguments
- `c`: concentration field (full grid, same shape as `img`).
- `img`: 3D boolean pore mask (`true` = pore).

# Keyword Arguments
- `axis`: transport direction (`:x`, `:y`, or `:z`).
- `ind`: voxel index at which flux is measured. Default: `1`.
- `ε`: porosity. Default: computed as `phase_fraction(img, true)`.
- `D`: intrinsic diffusivity (scalar). Default: `1.0`.
- `dx`: voxel spacing. Default: `1.0`.
- `L`: domain length. Default: `(N - 1) * dx`.
- `Δc`: imposed concentration drop. Default: computed from `c`.
"""
function tortuosity(c, img; axis, ind=1, ε=nothing, D=1.0, dx=1.0, L=nothing, Δc=nothing)
    ε = isnothing(ε) ? Imaginator.phase_fraction(img, true) : ε
    Deff = effective_diffusivity(c, img; axis, ind, D, dx, L, Δc)
    return ε / Deff
end

"""
    formation_factor(c, img; axis, ind=1, D=1.0, dx=1.0, L=nothing, Δc=nothing)

Compute the formation factor `F = 1 / D_eff` from a steady-state concentration field.

# Arguments
- `c`: concentration field (full grid, same shape as `img`).
- `img`: 3D boolean pore mask (`true` = pore).

# Keyword Arguments
- `axis`: transport direction (`:x`, `:y`, or `:z`).
- `ind`: voxel index at which flux is measured. Default: `1`.
- `D`: intrinsic diffusivity (scalar). Default: `1.0`.
- `dx`: voxel spacing. Default: `1.0`.
- `L`: domain length. Default: `(N - 1) * dx`.
- `Δc`: imposed concentration drop. Default: computed from `c`.
"""
function formation_factor(c, img; axis, ind=1, D=1.0, dx=1.0, L=nothing, Δc=nothing)
    Deff = effective_diffusivity(c, img; axis, ind, D, dx, L, Δc)
    return 1 / Deff
end
