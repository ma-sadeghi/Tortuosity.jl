# Observables extracted from concentration fields and problem definitions

porosity(img) = Imaginator.phase_fraction(img, true)
porosity(problem::TransientProblem) = porosity(problem.img)

function slice_conc_dist(C, img, axis)
    C = isa(C, AbstractVector) ? vec_to_grid(C, img) : C
    collapse = orthogonal_dims(axis)
    slice_nodes = length(selectdim(img, axis_dim(axis), 1))
    return dropdims(nansum(C; dims=collapse) ./ slice_nodes; dims=collapse)
end
slice_conc_dist(C, prob::TransientProblem) = slice_conc_dist(C, prob.img, prob.axis)

"""
    get_slice_conc(C, img, axis, ind; grid_to_vec=nothing)

Average concentration of a 2D slice perpendicular to `axis` at index `ind`,
including obstacle voxels (treated as NaN).
"""
function get_slice_conc(C, img, axis, ind; grid_to_vec=nothing)
    full_grid = size(C) == size(img)
    @assert (full_grid || !isnothing(grid_to_vec)) "if C is a vector of pore voxels, get_slice_conc() requires grid_to_vec array"

    ax = axis_dim(axis)

    C_slice =
        full_grid ? selectdim(C, ax, ind) : vec_to_slice(C, img, grid_to_vec, axis, ind)

    return nansum(C_slice) / length(C_slice)
end
function get_slice_conc(Cs::AbstractVector{<:Array}, img, axis, ind; grid_to_vec=nothing)
    return map(C -> get_slice_conc(C, img, axis, ind; grid_to_vec=grid_to_vec), Cs)
end
function get_slice_conc(C, prob::TransientProblem, ind)
    return get_slice_conc(C, prob.img, prob.axis, ind; grid_to_vec=prob.grid_to_vec)
end

"""
    flux_dist(C, D, dx, img, axis; inds=nothing)

Compute the diffusive flux between consecutive 2D slices along `axis`.
Returns a vector of fluxes for each slice pair in `inds` (defaults to all).
"""
function flux_dist(C, D, dx, img, axis; inds=nothing)
    C = isa(C, AbstractVector) ? vec_to_grid(C, img) : C

    ax = axis_dim(axis)      # 1, 2, or 3
    comp = orthogonal_dims(axis)
    dims = size(C)

    # all slice pairs (1,2), (2,3), ..., (N-1,N)
    isnothing(inds) && (inds = 1:(dims[ax] - 1))
    @assert all(1 .<= inds .<= (dims[ax] - 1)) "inds must be within 1:(size(C, axis)-1)"

    # accumulate flux contributions for each slice pair
    fluxes = similar(inds, Float64)

    for (k, i) in enumerate(inds)
        C1 = selectdim(C, ax, i)
        C2 = selectdim(C, ax, i + 1)

        m1 = selectdim(img, ax, i)
        m2 = selectdim(img, ax, i + 1)

        # Zero contribution from voxels adjacent to solid
        ΔC = C1 .* (m2 .!= 0) .- C2 .* (m1 .!= 0)

        if D isa Number
            D_eff = D
        else
            D1 = selectdim(D, ax, i)
            D2 = selectdim(D, ax, i + 1)
            D_eff = @. (2 * D1 * D2) / (D1 + D2 + eps()) # harmonic mean
        end

        plane_nodes = (dims[comp[1]]) * (dims[comp[2]])
        fluxes[k] = nansum(D_eff .* ΔC) / dx / plane_nodes
    end

    return fluxes
end
function flux_dist(C, prob::TransientProblem; inds=nothing)
    return flux_dist(C, prob.D, prob.dx, prob.img, prob.axis; inds=inds)
end

"""
    get_flux(C, D, dx, img, axis; ind=:end, grid_to_vec=nothing)

Flux per unit area between slices at `ind` and `ind + 1` along `axis`.
"""
function get_flux(C, D, dx, img, axis; ind=:end, grid_to_vec=nothing)
    full_grid = size(C) == size(img)
    @assert (full_grid || !isnothing(grid_to_vec)) "if C is a vector of pore voxels, get_flux() requires grid_to_vec array"

    ax = axis_dim(axis)
    ind === :end && (ind = size(img, ax) - 1)
    @assert 1 <= ind < size(img, ax) "ind must satisfy 1 <= ind < size(img, axis)"

    # slices of C
    C1 = full_grid ? selectdim(C, ax, ind) : vec_to_slice(C, img, grid_to_vec, axis, ind)
    C2 = if full_grid
        selectdim(C, ax, ind + 1)
    else
        vec_to_slice(C, img, grid_to_vec, axis, ind + 1)
    end

    # mask slices
    m1 = selectdim(img, ax, ind)
    m2 = selectdim(img, ax, ind + 1)

    ΔC = (C1 .* m2) .- (C2 .* m1)

    # diffusivity as scalar or scalar field
    if D isa Number
        D_eff = D
    else
        D1 = selectdim(D, ax, ind)
        D2 = selectdim(D, ax, ind + 1)
        D_eff = @. (2 * D1 * D2) / (D1 + D2 + eps())
    end

    voxel_count = length(C1)
    return nansum(D_eff .* ΔC) / dx / voxel_count
end
function get_flux(
    Cs::AbstractVector{<:Array}, D, dx, img, axis; ind=:end, grid_to_vec=nothing
)
    return map(C -> get_flux(C, D, dx, img, axis; ind=ind, grid_to_vec=grid_to_vec), Cs)
end
function get_flux(C, prob::TransientProblem; ind=:end)
    return get_flux(
        C, prob.D, prob.dx, prob.img, prob.axis; ind=ind, grid_to_vec=prob.grid_to_vec
    )
end

"""
    mass_intake(C_hist, img)

Total mass intake per unit volume relative to the initial concentration field.
"""
function mass_intake(C_hist, img::AbstractArray)
    voxels = length(img)
    mass_intake = A -> (sum(A) - sum(C_hist[1])) / voxels
    return map(mass_intake, C_hist)
end
mass_intake(C_hist, prob::TransientProblem) = mass_intake(C_hist, prob.img)
mass_intake(sim::TransientState, prob::TransientProblem) = mass_intake(sim.C, prob.img)
