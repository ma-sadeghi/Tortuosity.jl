# Observables extracted from concentration fields

"""
    get_slice_conc(C, img, axis, ind; grid_to_vec=nothing, pore_only=false)

Average concentration of a 2D slice perpendicular to `axis` at index `ind`.
Obstacle voxels are treated as NaN. When `pore_only=true`, the average is taken
over pore voxels only; otherwise over the full slice area.
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

Flux per unit area between slices at `ind` and `ind + 1` along `axis`.
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
        D_eff = @. (2 * D1 * D2) / (D1 + D2 + eps())
    end

    return nansum(D_eff .* ΔC) / dx / length(C1)
end
function compute_flux(Cs::AbstractVector{<:Array}, D, dx, img, axis; ind=:end, grid_to_vec=nothing)
    return map(C -> compute_flux(C, D, dx, img, axis; ind=ind, grid_to_vec=grid_to_vec), Cs)
end

"""
    compute_mass_intake(C_hist, img)

Total mass intake per unit volume relative to the initial concentration field.
"""
function compute_mass_intake(C_hist, img::AbstractArray)
    C0_total = nansum(C_hist[1])
    return [(nansum(C) - C0_total) / length(img) for C in C_hist]
end
