# Observables extracted from concentration fields

"""
    slice_concentration(c, img, axis, ind; pore_index=nothing, pore_only=false)
    slice_concentration(c_hist::AbstractVector{<:Array}, img, axis, ind; kwargs...)
    slice_concentration(c, prob::TransientDiffusionProblem, ind; pore_only=false)

Average concentration of a 2D slice perpendicular to `axis` at voxel index
`ind`. Obstacle voxels are treated as NaN.

`c` may be either a full 3D concentration grid (same shape as `img`) or a
1D pore-only vector; in the latter case `pore_index` must be provided so
the vector can be mapped back to grid coordinates.

# Method overloads

- `(c::AbstractArray, img, axis, ind; ...)` — one snapshot, explicit geometry.
- `(c_hist::AbstractVector{<:Array}, img, axis, ind; ...)` — maps the scalar
  computation over a vector of snapshots (e.g. `sol.u`) and returns a
  `Vector{Float64}` of per-snapshot slice averages.
- `(c, prob::TransientDiffusionProblem, ind; pore_only=false)` — convenience wrapper
  that unpacks `prob.img`, `prob.axis`, and `prob.pore_index`.

# Keyword Arguments

- `pore_index`: required if `c` is a pore-only vector; ignored otherwise.
- `pore_only`: when `true`, the average is taken over pore voxels only;
  otherwise the full slice area is used (solid voxels count as zero).
"""
function slice_concentration(c, img, axis, ind; pore_index=nothing, pore_only::Bool=false)
    full_grid = size(c) == size(img)
    @assert (full_grid || !isnothing(pore_index)) "if c is a vector of pore voxels, slice_concentration() requires pore_index array"

    ax = axis_dim(axis)
    if full_grid
        c_slice = selectdim(c, ax, ind)
    else
        c_slice = reconstruct_slice(c, pore_index, axis, ind)
    end
    slice_nodes = pore_only ? count(selectdim(img, ax, ind)) : length(c_slice)
    return nansum(c_slice) / slice_nodes
end
function slice_concentration(c_hist::AbstractVector{<:Array}, img, axis, ind; pore_index=nothing, pore_only::Bool=false)
    return map(c -> slice_concentration(c, img, axis, ind; pore_index=pore_index, pore_only=pore_only), c_hist)
end

"""
    flux(c, D, voxel_size, img, axis; ind=:end, pore_index=nothing)
    flux(c_hist::AbstractVector{<:Array}, D, voxel_size, img, axis; kwargs...)
    flux(c, prob::TransientDiffusionProblem; ind=:end)

Diffusive flux per unit face area between voxel slices `ind` and `ind + 1`
along `axis`. Computed as `mean(D_eff * dc) / voxel_size`, where `dc` is the
concentration difference across the pair of slices and `D_eff` is the
face-centered diffusivity (harmonic mean of `D` at the two adjacent voxels).

# Method overloads

- `(c::AbstractArray, D, voxel_size, img, axis; ...)` — one snapshot. `c` may be a
  full 3D grid or a pore-only 1D vector; in the latter case `pore_index`
  is required.
- `(c_hist::AbstractVector{<:Array}, D, voxel_size, img, axis; ...)` — maps the flux
  computation over a vector of snapshots, returning a `Vector{Float64}`.
- `(c, prob::TransientDiffusionProblem; ind=:end)` — convenience wrapper that unpacks
  `prob.D`, `prob.voxel_size`, `prob.img`, `prob.axis`, and `prob.pore_index`.

# Keyword Arguments

- `ind`: index at which flux is measured. `:end` (default) maps to
  `size(img, axis) - 1`, i.e. the flux across the outlet face.
- `pore_index`: required if `c` is a pore-only vector; ignored otherwise.
"""
function flux(c, D, voxel_size, img, axis; ind=:end, pore_index=nothing)
    full_grid = size(c) == size(img)
    @assert (full_grid || !isnothing(pore_index)) "if c is a vector of pore voxels, flux() requires pore_index array"

    ax = axis_dim(axis)
    if ind === :end
        ind = size(img, ax) - 1
    end
    @assert 1 <= ind < size(img, ax) "ind must satisfy 1 <= ind < size(img, axis)"

    if full_grid
        c1 = selectdim(c, ax, ind)
        c2 = selectdim(c, ax, ind + 1)
    else
        c1 = reconstruct_slice(c, pore_index, axis, ind)
        c2 = reconstruct_slice(c, pore_index, axis, ind + 1)
    end

    m1 = selectdim(img, ax, ind)
    m2 = selectdim(img, ax, ind + 1)
    dc = (c1 .* m2) .- (c2 .* m1)

    if D isa Number
        D_eff = D
    else
        D1 = selectdim(D, ax, ind)
        D2 = selectdim(D, ax, ind + 1)
        denom = @. D1 + D2
        D_eff = @. ifelse(denom == 0, zero(denom), 2 * D1 * D2 / denom)
    end

    return nansum(D_eff .* dc) / voxel_size / length(c1)
end
function flux(c_hist::AbstractVector{<:Array}, D, voxel_size, img, axis; ind=:end, pore_index=nothing)
    return map(c -> flux(c, D, voxel_size, img, axis; ind=ind, pore_index=pore_index), c_hist)
end

"""
    mass_uptake(c_hist, img; c0_total=nothing)
    mass_uptake(c_hist, prob::TransientDiffusionProblem; c0_total=0)

Change in **volume-averaged** concentration from the pre-simulation initial
state at each timestep. For each snapshot `c` in `c_hist`, returns
`(Σ c - c0_total) / length(img)` — the difference between total
concentration summed over all voxels (solid contributions are `NaN`-safe
via `nansum` and therefore treated as zero) and a scalar reference
`c0_total`, divided by the **total** number of voxels (not the pore count).

This is the discrete counterpart of [`slab_mass_uptake`](@ref), which is
defined as `M_t / M_∞` for a homogeneous slab and therefore porosity-
weighted. The `fit_effective_diffusivity` routine multiplies the analytical
`slab_mass_uptake(D, t)` by `φ · (c1 + c2) / 2` to match the volume-averaged
convention used here.

# The `c0_total` reference

`c_hist[1]` cannot be used as the initial reference for fits against
`slab_mass_uptake`: the transient solver applies Dirichlet boundary values
inside `_initial_state` **before** the first save, so `c_hist[1]` already
contains the clamped inlet/outlet face contribution and is not the true
pre-clamp initial state. Subtracting `c_hist[1]` yields an `O(1 / N)` bias
that survives to `t → ∞` and skews any `D_eff` fit against
`slab_mass_uptake` by a constant offset.

The analytical `slab_mass_uptake` assumes `c(x, 0) = c0` (typically `0`)
for all `x`, *before* the Dirichlet boundary load turns on. The correct
reference is therefore the volume-integral of the pre-clamp initial field:

- `img` overload, default `c0_total=nothing` → preserves legacy behavior
  (`c0_total = nansum(c_hist[1])`) for any callers that rely on it.
- `prob` overload, default `c0_total=0` → matches the default `solve` path
  (`u0 === nothing ⇒ c0 = zeros`). If you ran `solve(prob, alg; u0=u0)`
  with a non-zero pre-clamp `u0`, pass `c0_total = nansum(u0)` explicitly.

# Arguments
- `c_hist`: vector of concentration snapshots. Each entry may be a full 3D
  grid or, via the `TransientDiffusionProblem` overload, a pore-only
  vector that is mapped back via `prob.img`.
- `img`: boolean pore mask matching the full-grid shape.

# Keyword Arguments
- `c0_total`: scalar reference. See discussion above. `img` overload
  defaults to `nansum(c_hist[1])`; `prob` overload defaults to `0`.

# Returns
`Vector{Float64}` of volume-averaged mass uptake, one entry per snapshot.
"""
function mass_uptake(c_hist, img::AbstractArray; c0_total=nothing)
    ref = isnothing(c0_total) ? nansum(c_hist[1]) : c0_total
    return [(nansum(c) - ref) / length(img) for c in c_hist]
end

# --- Convenience wrappers that unpack TransientDiffusionProblem ---

slice_concentration(c, prob::TransientDiffusionProblem, ind; pore_only::Bool=false) =
    slice_concentration(c, prob.img, prob.axis, ind; pore_index=prob.pore_index, pore_only=pore_only)

flux(c, prob::TransientDiffusionProblem; ind=:end) =
    flux(c, prob.D, prob.voxel_size, prob.img, prob.axis; ind=ind, pore_index=prob.pore_index)

mass_uptake(c_hist, prob::TransientDiffusionProblem; c0_total::Real=0) =
    mass_uptake(c_hist, prob.img; c0_total=c0_total)
