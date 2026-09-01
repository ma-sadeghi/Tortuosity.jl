# Steady-state effective diffusivity, tortuosity, and formation factor

# NOTE: Why do we only compute the rate along the specified axis? What about
#  the lateral rates (like what Transport.rate does in OpenPNM)? We can do
#  that, but it won't make a difference because the lateral rates get cancelled
#  out. Picture two adjacent pores i, j: The rate from i to j is the same as
#  the rate from j to i, but with opposite sign. So when we sum the rates, the
#  lateral rates cancel out.

function _inlet_edge_flux(u, flux::InletFlux)
    !iszero(flux.direct) && return zero(flux.direct)
    isnothing(flux.targets) && return zero(flux.direct)
    isempty(flux.targets) && return zero(eltype(flux.weights))

    targets, weights = flux.targets, flux.weights
    nodes_backend = _device_backend(targets)
    u_backend = _device_backend(u)
    same_backend = if isnothing(nodes_backend) || isnothing(u_backend)
        isnothing(nodes_backend) && isnothing(u_backend)
    else
        typeof(nodes_backend) === typeof(u_backend)
    end
    if same_backend
        return mapreduce(
            (i, w) -> w * (one(eltype(u)) - u[i]), +, targets, weights,
        )
    end

    adapted_nodes = isnothing(u_backend) ?
        Vector{eltype(targets)}(undef, length(targets)) :
        similar(u, eltype(targets), length(targets))
    adapted_weights = isnothing(u_backend) ?
        Vector{eltype(u)}(undef, length(weights)) :
        similar(u, eltype(u), length(weights))
    try
        copyto!(adapted_nodes, targets)
        copyto!(adapted_weights, weights)
        return mapreduce(
            (i, w) -> w * (one(eltype(u)) - u[i]), +, adapted_nodes, adapted_weights,
        )
    finally
        _free!(adapted_nodes)
        _free!(adapted_weights)
    end
end

function _checkpoint_tortuosity(u, sim::SteadyDiffusionProblem)
    flux = sim.flux
    isnothing(flux) && throw(ArgumentError(
        "this SteadyDiffusionProblem has no boundary-flux metadata"
    ))
    isnothing(flux.sources) && throw(ArgumentError(
        "construct the SteadyDiffusionProblem with `checkpoint_readout=true` to inspect \
         unconverged iterates"
    ))
    u_backend = _device_backend(u)
    flux_backend = _device_backend(flux.targets)
    same_backend = if isnothing(u_backend) || isnothing(flux_backend)
        isnothing(u_backend) && isnothing(flux_backend)
    else
        typeof(u_backend) === typeof(flux_backend)
    end
    same_backend || throw(ArgumentError(
        "checkpoint readout requires the solution and boundary metadata on the same backend"
    ))

    edge_flux = mapreduce(
        (source, target, weight) -> weight * (u[source] - u[target]),
        +, flux.sources, flux.targets, flux.weights,
    )
    inlet_mean = mapreduce(i -> u[i], +, flux.inlet) / length(flux.inlet)
    outlet_mean = mapreduce(i -> u[i], +, flux.outlet) / length(flux.outlet)
    dc = inlet_mean - outlet_mean

    ax = axis_dim(sim.axis)
    N = size(sim.img, ax)
    face_area = length(sim.img) ÷ N
    D_eff = edge_flux / face_area * (N - 1) / dc
    ε = length(sim.prob.b) / length(sim.img)
    return sim.D0 * ε / D_eff
end

"""
    effective_diffusivity(u, sim::SteadyDiffusionProblem; voxel_size=1.0, L=nothing, dc=1.0)

Compute effective diffusivity directly from the pore-ordered solution vector
`u` and its steady problem, without reconstructing the full concentration
field.

The problem retains a compact map of edges from the unit-concentration inlet to
the adjacent free nodes. Reducing `w * (1 - u[i])` over those edges runs where
`u` lives and transfers only the resulting scalar from a GPU, instead of
copying the pore vector to the host and allocating an image-sized concentration
field.

`dc` defaults to the imposed inlet-to-outlet concentration difference, which is
one for every `SteadyDiffusionProblem`.
"""
function effective_diffusivity(
    u::AbstractVector, sim::SteadyDiffusionProblem;
    voxel_size=1.0, L=nothing, dc=1.0,
)
    n = length(sim.prob.b)
    length(u) == n || throw(DimensionMismatch(
        "solution has length $(length(u)) but the steady problem has $(n) unknowns"
    ))
    isnothing(sim.flux) && throw(ArgumentError(
        "this SteadyDiffusionProblem was built from an existing LinearProblem and has no \
         inlet-flux metadata; construct it from an image to use direct transport observables"
    ))

    ax = axis_dim(sim.axis)
    N = size(sim.img, ax)
    L = isnothing(L) ? (N - 1) * voxel_size : L
    face_area = length(sim.img) ÷ N
    edge_flux = _inlet_edge_flux(u, sim.flux)
    total_flux = edge_flux + sim.flux.direct
    J = total_flux / voxel_size / face_area
    return J * L / dc
end

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
- `D`: intrinsic diffusivity, a scalar or a per-voxel field. Default: `1.0`.
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
    tortuosity(u, sim::SteadyDiffusionProblem; ε=nothing, D0=nothing,
               voxel_size=1.0, L=nothing, dc=1.0)

Compute tortuosity directly from a pore-ordered steady solution. This is the
fast path when the full concentration field is not otherwise needed; use
[`reconstruct_field`](@ref) only for field visualization or analysis.

The problem retains the scalar reference diffusivity selected from the `D` used
to construct it, so device-resident diffusivity fields do not need to be copied
or indexed from the host. Pass `D0` only to override that reference.
"""
function tortuosity(
    u::AbstractVector, sim::SteadyDiffusionProblem;
    ε=nothing, D0=nothing, voxel_size=1.0, L=nothing, dc=1.0,
)
    ε = isnothing(ε) ? length(sim.prob.b) / length(sim.img) : ε
    isnothing(sim.D0) && isnothing(D0) && throw(ArgumentError(
        "this SteadyDiffusionProblem has no reference-diffusivity metadata; pass `D0`, \
         or construct it from an image"
    ))
    D0 = isnothing(D0) ? sim.D0 : D0
    D_eff = effective_diffusivity(u, sim; voxel_size, L, dc)
    return D0 * ε / D_eff
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

# Disconnected pore space

When no pore path joins the inlet face to the outlet face, `D_eff` is zero and
`τ = D0·ε/D_eff` diverges. It does not come back as `Inf`: the solve leaves a
`D_eff` of rounding size rather than exactly zero, so `τ` is enormous — of order
`1e15` — with a sign that is whichever way that rounding fell, and the solver
still reports `Success`. Test `abs(τ)` rather than `τ >= 1` if you need to
detect this. [`SteadyDiffusionProblem`](@ref) warns about stranded pore volume
as it is constructed, which is the earlier and more informative signal; see its
`warn_nonpercolating` keyword.

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
#
# A fused masked reduction rather than `maximum(D[img])`: logical indexing
# materialises one value per pore voxel — 4.8 GB at 1000³ and ε = 0.6 — to
# return one scalar. `mapreduce` also stays on the arrays' backend, which lets
# construction retain `D0` without copying a device diffusivity to the host.
_reference_diffusivity(D::Number, img) = D
function _reference_diffusivity(D, img)
    axes(D) == axes(img) || throw(DimensionMismatch(
        "diffusivity has axes $(axes(D)) but the pore mask has axes $(axes(img))"
    ))
    return mapreduce(
        (d, pore) -> ifelse(pore, d, typemin(typeof(d))), max, D, img,
    )
end

"""
    formation_factor(u, sim::SteadyDiffusionProblem; D0=nothing,
                     voxel_size=1.0, L=nothing, dc=1.0)

Compute the formation factor directly from a pore-ordered steady solution,
without reconstructing the full concentration field. The reference diffusivity
comes from the `D` used to build `sim`; pass `D0` only to override it.
"""
function formation_factor(
    u::AbstractVector, sim::SteadyDiffusionProblem;
    D0=nothing, voxel_size=1.0, L=nothing, dc=1.0,
)
    isnothing(sim.D0) && isnothing(D0) && throw(ArgumentError(
        "this SteadyDiffusionProblem has no reference-diffusivity metadata; pass `D0`, \
         or construct it from an image"
    ))
    D0 = isnothing(D0) ? sim.D0 : D0
    D_eff = effective_diffusivity(u, sim; voxel_size, L, dc)
    return D0 / D_eff
end

"""
    formation_factor(c, img; axis, ind=1, D=1.0, D0=nothing, voxel_size=1.0, L=nothing, dc=nothing)

Compute the formation factor `F = D0 / D_eff` from a steady-state concentration
field. As with [`tortuosity`](@ref), the reference diffusivity divides back out
so that `F` describes the geometry rather than the units `D` was given in, and a
pore space that does not span the domain sends `F` to a rounding-signed value of
order `1e16` rather than to `Inf`.

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
