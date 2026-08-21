# Steady-state effective diffusivity, tortuosity, and formation factor

# NOTE: Why do we only compute the rate along the specified axis? What about
#  the lateral rates (like what Transport.rate does in OpenPNM)? We can do
#  that, but it won't make a difference because the lateral rates get cancelled
#  out. Picture two adjacent pores i, j: The rate from i to j is the same as
#  the rate from j to i, but with opposite sign. So when we sum the rates, the
#  lateral rates cancel out.

"""
    effective_diffusivity(c, img; axis, ind=1, D=1.0, voxel_size=1.0, L=nothing, dc=nothing)

Compute the effective diffusivity `D_eff` from a steady-state concentration field
by measuring flux through the cross-section at voxel index `ind`.

`D_eff = J * L / dc`, where `J` is the diffusive flux at `ind`, `L` is the domain
length, and `dc` is the concentration drop between inlet and outlet faces.

# Arguments
- `c`: concentration field (full grid, same shape as `img`).
- `img`: 3D boolean pore mask (`true` = pore).

# Keyword Arguments
- `axis`: transport direction (`:x`, `:y`, or `:z`).
- `ind`: voxel index at which flux is measured. Default: `1`.
- `D`: intrinsic diffusivity (scalar). Default: `1.0`.
- `voxel_size`: physical voxel spacing. Default: `1.0`.
- `L`: domain length. Default: `(N - 1) * voxel_size` where `N` is the number of voxels along `axis`.
- `dc`: imposed concentration drop. Default: mean inlet minus mean outlet concentration.
"""
function effective_diffusivity(c, img; axis, ind=1, D=1.0, voxel_size=1.0, L=nothing, dc=nothing)
    ax = axis_dim(axis)
    N = size(img, ax)
    L = isnothing(L) ? (N - 1) * voxel_size : L
    dc = isnothing(dc) ? nanmean(selectdim(c, ax, 1)) - nanmean(selectdim(c, ax, N)) : dc
    J = flux(c, D, voxel_size, img, axis; ind=ind)
    return J * L / dc
end

"""
    tortuosity(c, img; axis, ind=1, ε=nothing, D=1.0, D0=nothing, voxel_size=1.0, L=nothing, dc=nothing)

Compute the tortuosity factor `τ = D0 * ε / D_eff` from a steady-state
concentration field. When `ε` is omitted, porosity is computed automatically
from `img`.

`D_eff` scales linearly with `D`, so the reference diffusivity `D0` must divide
back out for `τ` to be a property of the geometry alone. Without it, scaling a
diffusivity field by a constant would change `τ` by that constant and could
drive it below 1, which is physically impossible.

# Arguments
- `c`: concentration field (full grid, same shape as `img`).
- `img`: 3D boolean pore mask (`true` = pore).

# Keyword Arguments
- `axis`: transport direction (`:x`, `:y`, or `:z`).
- `ind`: voxel index at which flux is measured. Default: `1`.
- `ε`: porosity. Default: computed as `phase_fraction(img, true)`.
- `D`: intrinsic diffusivity, a scalar or a per-voxel field. Default: `1.0`.
- `D0`: reference diffusivity that `τ` is measured against. Default: `D` when
  `D` is a scalar, and the largest pore-phase value of `D` when it is a field.
- `voxel_size`: physical voxel spacing. Default: `1.0`.
- `L`: domain length. Default: `(N - 1) * voxel_size`.
- `dc`: imposed concentration drop. Default: computed from `c`.
"""
function tortuosity(
    c, img; axis, ind=1, ε=nothing, D=1.0, D0=nothing, voxel_size=1.0, L=nothing, dc=nothing
)
    ε = isnothing(ε) ? Imaginator.phase_fraction(img, true) : ε
    D0 = isnothing(D0) ? _reference_diffusivity(D, img) : D0
    Deff = effective_diffusivity(c, img; axis, ind, D, voxel_size, L, dc)
    return D0 * ε / Deff
end

# The reference against which `τ` and the formation factor are measured. A
# scalar `D` is itself the intrinsic diffusivity; for a per-voxel field the
# conventional reference is the fastest conducting phase, taken over the pore
# space so that whatever value fills the solid voxels cannot set the scale.
_reference_diffusivity(D::Number, img) = D
_reference_diffusivity(D, img) = maximum(D[img])

"""
    formation_factor(c, img; axis, ind=1, D=1.0, D0=nothing, voxel_size=1.0, L=nothing, dc=nothing)

Compute the formation factor `F = D0 / D_eff` from a steady-state concentration
field. As with [`tortuosity`](@ref), the reference diffusivity divides back out
so that `F` describes the geometry rather than the units `D` was given in.

# Arguments
- `c`: concentration field (full grid, same shape as `img`).
- `img`: 3D boolean pore mask (`true` = pore).

# Keyword Arguments
- `axis`: transport direction (`:x`, `:y`, or `:z`).
- `ind`: voxel index at which flux is measured. Default: `1`.
- `D`: intrinsic diffusivity, a scalar or a per-voxel field. Default: `1.0`.
- `D0`: reference diffusivity that `F` is measured against. Default: `D` when
  `D` is a scalar, and the largest pore-phase value of `D` when it is a field.
- `voxel_size`: physical voxel spacing. Default: `1.0`.
- `L`: domain length. Default: `(N - 1) * voxel_size`.
- `dc`: imposed concentration drop. Default: computed from `c`.
"""
function formation_factor(
    c, img; axis, ind=1, D=1.0, D0=nothing, voxel_size=1.0, L=nothing, dc=nothing
)
    D0 = isnothing(D0) ? _reference_diffusivity(D, img) : D0
    Deff = effective_diffusivity(c, img; axis, ind, D, voxel_size, L, dc)
    return D0 / Deff
end
