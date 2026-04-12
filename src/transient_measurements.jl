# Observables extracted from concentration fields

"""
    get_slice_conc(C, img, axis, ind; grid_to_vec=nothing, pore_only=false)
    get_slice_conc(Cs::AbstractVector{<:Array}, img, axis, ind; kwargs...)
    get_slice_conc(C, prob::TransientProblem, ind; pore_only=false)

Average concentration of a 2D slice perpendicular to `axis` at voxel index
`ind`. Obstacle voxels are treated as NaN.

`C` may be either a full 3D concentration grid (same shape as `img`) or a
1D pore-only vector; in the latter case `grid_to_vec` must be provided so
the vector can be mapped back to grid coordinates.

# Method overloads

- `(C::AbstractArray, img, axis, ind; ...)` — one snapshot, explicit geometry.
- `(Cs::AbstractVector{<:Array}, img, axis, ind; ...)` — maps the scalar
  computation over a vector of snapshots (e.g. `state.C`) and returns a
  `Vector{Float64}` of per-snapshot slice averages.
- `(C, prob::TransientProblem, ind; pore_only=false)` — convenience wrapper
  that unpacks `prob.img`, `prob.axis`, and `prob.grid_to_vec`.

# Keyword Arguments

- `grid_to_vec`: required if `C` is a pore-only vector; ignored otherwise.
- `pore_only`: when `true`, the average is taken over pore voxels only;
  otherwise the full slice area is used (solid voxels count as zero).
"""
function get_slice_conc(C, img, axis, ind; grid_to_vec=nothing, pore_only::Bool=false)
    full_grid = size(C) == size(img)
    @assert (full_grid || !isnothing(grid_to_vec)) "if C is a vector of pore voxels, get_slice_conc() requires grid_to_vec array"

    ax = axis_dim(axis)
    if full_grid
        C_slice = selectdim(C, ax, ind)
    else
        C_slice = vec_to_slice(C, img, grid_to_vec, axis, ind)
    end
    slice_nodes = pore_only ? count(selectdim(img, ax, ind)) : length(C_slice)
    return nansum(C_slice) / slice_nodes
end
function get_slice_conc(Cs::AbstractVector{<:Array}, img, axis, ind; grid_to_vec=nothing, pore_only::Bool=false)
    return map(C -> get_slice_conc(C, img, axis, ind; grid_to_vec=grid_to_vec, pore_only=pore_only), Cs)
end

"""
    compute_flux(C, D, dx, img, axis; ind=:end, grid_to_vec=nothing)
    compute_flux(Cs::AbstractVector{<:Array}, D, dx, img, axis; kwargs...)
    compute_flux(C, prob::TransientProblem; ind=:end)

Diffusive flux per unit face area between voxel slices `ind` and `ind + 1`
along `axis`. Computed as `mean(D_eff * ΔC) / dx`, where `ΔC` is the
concentration difference across the pair of slices and `D_eff` is the
face-centered diffusivity (harmonic mean of `D` at the two adjacent voxels).

# Method overloads

- `(C::AbstractArray, D, dx, img, axis; ...)` — one snapshot. `C` may be a
  full 3D grid or a pore-only 1D vector; in the latter case `grid_to_vec`
  is required.
- `(Cs::AbstractVector{<:Array}, D, dx, img, axis; ...)` — maps the flux
  computation over a vector of snapshots, returning a `Vector{Float64}`.
- `(C, prob::TransientProblem; ind=:end)` — convenience wrapper that unpacks
  `prob.D`, `prob.dx`, `prob.img`, `prob.axis`, and `prob.grid_to_vec`.

# Keyword Arguments

- `ind`: index at which flux is measured. `:end` (default) maps to
  `size(img, axis) - 1`, i.e. the flux across the outlet face.
- `grid_to_vec`: required if `C` is a pore-only vector; ignored otherwise.
"""
function compute_flux(C, D, dx, img, axis; ind=:end, grid_to_vec=nothing)
    full_grid = size(C) == size(img)
    @assert (full_grid || !isnothing(grid_to_vec)) "if C is a vector of pore voxels, compute_flux() requires grid_to_vec array"

    ax = axis_dim(axis)
    if ind === :end
        ind = size(img, ax) - 1
    end
    @assert 1 <= ind < size(img, ax) "ind must satisfy 1 <= ind < size(img, axis)"

    if full_grid
        C1 = selectdim(C, ax, ind)
        C2 = selectdim(C, ax, ind + 1)
    else
        C1 = vec_to_slice(C, img, grid_to_vec, axis, ind)
        C2 = vec_to_slice(C, img, grid_to_vec, axis, ind + 1)
    end

    m1 = selectdim(img, ax, ind)
    m2 = selectdim(img, ax, ind + 1)
    ΔC = (C1 .* m2) .- (C2 .* m1)

    if D isa Number
        D_eff = D
    else
        D1 = selectdim(D, ax, ind)
        D2 = selectdim(D, ax, ind + 1)
        denom = @. D1 + D2
        D_eff = @. ifelse(denom == 0, zero(denom), 2 * D1 * D2 / denom)
    end

    return nansum(D_eff .* ΔC) / dx / length(C1)
end
function compute_flux(Cs::AbstractVector{<:Array}, D, dx, img, axis; ind=:end, grid_to_vec=nothing)
    return map(C -> compute_flux(C, D, dx, img, axis; ind=ind, grid_to_vec=grid_to_vec), Cs)
end

"""
    compute_mass_uptake(C_hist, img)
    compute_mass_uptake(C_hist, prob::TransientProblem)

Change in **volume-averaged** concentration from the initial state at each
timestep. For each snapshot `C` in `C_hist`, returns
`(Σ C - Σ C_hist[1]) / length(img)` — i.e. the difference between total
concentration summed over all voxels (solid contributions are `NaN`-safe
via `nansum` and therefore treated as zero) divided by the **total** number
of voxels, not the pore count.

This is the discrete counterpart of [`slab_mass_uptake`](@ref), which is
defined as `M_t / M_∞` for a homogeneous slab and therefore
porosity-weighted. The `fit_effective_diffusivity` routine multiplies the
analytical `slab_mass_uptake(D, t)` by `φ` to match the volume-averaged
convention used here.

# Arguments
- `C_hist`: a vector of concentration snapshots. Each entry may be a full
  3D grid or, via the `TransientProblem` overload, a pore-only vector that
  is mapped back via `prob.img`.
- `img`: boolean pore mask matching the full-grid shape.

# Returns
`Vector{Float64}` of volume-averaged mass uptake, one entry per snapshot.
Entry 1 is always zero.
"""
function compute_mass_uptake(C_hist, img::AbstractArray)
    C0_total = nansum(C_hist[1])
    return [(nansum(C) - C0_total) / length(img) for C in C_hist]
end

# --- Convenience wrappers that unpack TransientProblem ---

get_slice_conc(C, prob::TransientProblem, ind; pore_only::Bool=false) =
    get_slice_conc(C, prob.img, prob.axis, ind; grid_to_vec=prob.grid_to_vec, pore_only=pore_only)

compute_flux(C, prob::TransientProblem; ind=:end) =
    compute_flux(C, prob.D, prob.dx, prob.img, prob.axis; ind=ind, grid_to_vec=prob.grid_to_vec)

compute_mass_uptake(C_hist, prob::TransientProblem) =
    compute_mass_uptake(C_hist, prob.img)
