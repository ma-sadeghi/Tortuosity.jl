##--- functions for extracting datapoints of interest from concentration distributions or problem---

function porosity(img)
    sum(img)/length(img) #assumes pores are represented by true/1, not very standard but that is the definition here
end
porosity(problem::TransientProblem) = porosity(problem.img)

"""
slice_conc_dist(C, img, axis; pore_only = false)

returns 1D concentration distribution collapsed onto 'axis'

kwargs
    pore_only::bool - if it is false (default), include solid voxels in concentration calculation,
                    so the result is overall concentration in that slice of material
"""
function slice_conc_dist(C, img, axis; pore_only::Bool = false)

    #accept pore-voxel vector form as well as full 3D distribution
    C = isa(C, AbstractVector) ? vec_to_grid(C, img) : C

    collapse = AXIS_COMPLEMENT[axis] #dims to sum over

    slice_nodes = pore_only ? count(img, dims = collapse) : prod(size(img)[collect(collapse)])

    return dropdims(nansum(C, dims = collapse)./slice_nodes, dims = collapse)
end
slice_conc_dist(C, prob::TransientProblem; pore_only::Bool = false) = slice_conc_dist(C, prob.img, prob.axis; pore_only = pore_only)


# argument for concentration in terms of entire volume, or just in terms of pores?
"""
returns average concentration slice perpendicular to axis at index ind (average for entire slice including obstacles)

"""
function get_slice_conc(C, img, axis, ind; grid_to_vec = nothing, pore_only::Bool = false)
    #accept pore-voxel vector form as well as full 3D distribution
    full_grid = size(C) == size(img)
    @assert (full_grid || !isnothing(grid_to_vec)) "if C is a vector of pore voxels, get_slice_conc() requires grid_to_vec array"

    ax = AXIS_DEFINITION[axis]

    C_slice = full_grid ? selectdim(C, ax, ind) : vec_to_slice(C, img, grid_to_vec, axis, ind)

    slice_nodes = pore_only ? count(selectdim(img, ax, ind)) : length(C_slice)

    return nansum(C_slice)/slice_nodes
end
#multiple Concentration fields wrapper
function get_slice_conc(Cs::AbstractVector{<:Array}, img, axis, ind; grid_to_vec=nothing, pore_only::Bool = false)
    map(C -> get_slice_conc(C, img, axis, ind; grid_to_vec=grid_to_vec, pore_only = pore_only), Cs)
end
#problem struct convenience wrapper
get_slice_conc(C, prob::TransientProblem, ind; pore_only::Bool = false) =
         get_slice_conc(C, prob.img, prob.axis, ind; grid_to_vec= prob.grid_to_vec, pore_only = pore_only)



"""
flux_dist(C, prob)
input the C distribution for a timestep

input C, and either dx or the associated TransientProblem
returns a vector of the flux between each 2d slice of voxels along direction of axis or problem.axis
    or just between ind and ind+1 for entries to inds (note ind=N is out of range)
"""
function flux_dist(C, D, dx, img, axis; inds = nothing)

    #accept pore-voxel vector form as well as full 3D distribution
    C = isa(C, AbstractVector) ? vec_to_grid(C, img) : C

    ax = AXIS_DEFINITION[axis]      # 1, 2, or 3
    comp = AXIS_COMPLEMENT[axis]
    dims  = size(C)

    # all slice pairs (1,2), (2,3), ..., (N-1,N)
    isnothing(inds) && (inds = 1:(dims[ax]-1)) #default to entire distribution
    @assert all(1 .<= inds .<= (dims[ax]-1)) "inds must be within 1:(size(C, axis)-1)"  

    # accumulate flux contributions for each slice pair
    fluxes = similar(inds, Float64)

    for (k, i) in enumerate(inds)
        C1 = selectdim(C, ax, i)
        C2 = selectdim(C, ax, i+1)

        m1 = selectdim(img, ax, i)
        m2 = selectdim(img, ax, i+1)

        #zero voxels adjacent to solid for C[solid] = 0 case, redundant for C[solid] = NaN case
        ΔC = C1 .* (m2 .!= 0) .- C2 .* (m1 .!= 0)

        #
        if D isa Number
            D_eff = D
        else #handle scalar field diffusivity
            D1 = selectdim(D, ax, i)
            D2 = selectdim(D, ax, i + 1)
            D_eff = @. (2 * D1 * D2) / (D1 + D2 + eps()) #harmonic mean
        end
        
        plane_nodes = (dims[comp[1]]) * (dims[comp[2]])
        # sum over the perpendicular axes
        fluxes[k] = nansum(D_eff.*ΔC) /dx /plane_nodes
    end

    return fluxes
end
flux_dist(C, prob::TransientProblem; inds = nothing)=  flux_dist(C, prob.D, prob.dx, prob.img, prob.axis; inds = inds)


"""
returns flux/area between slice of C at ind and ind+1
"""
function get_flux(C, D, dx, img, axis; ind=:end, grid_to_vec=nothing)
    full_grid = size(C) == size(img)
    @assert (full_grid || !isnothing(grid_to_vec)) "if C is a vector of pore voxels, get_flux() requires grid_to_vec array"

    ax  = AXIS_DEFINITION[axis]
    ind === :end && (ind = size(img, ax) - 1)
    @assert 1 <= ind < size(img, ax) "ind must satisfy 1 <= ind < size(img, axis)"  


    # slices of C
    C1 = full_grid ? selectdim(C, ax, ind)     : vec_to_slice(C, img, grid_to_vec, axis, ind)
    C2 = full_grid ? selectdim(C, ax, ind + 1) : vec_to_slice(C, img, grid_to_vec, axis, ind + 1)

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
    return nansum(D_eff.*ΔC) / dx / voxel_count
end
# Accept a vector of C arrays
function get_flux(Cs::AbstractVector{<:Array}, D, dx, img, axis; ind=:end, grid_to_vec=nothing)
    map(C -> get_flux(C, D, dx, img, axis; ind=ind, grid_to_vec=grid_to_vec), Cs)
end
#pass in the problem struct for convenience
get_flux(C, prob::TransientProblem; ind=:end)=  get_flux(C, prob.D, prob.dx, prob.img, prob.axis; ind=ind, grid_to_vec = prob.grid_to_vec)

"""
mass_intake(state)

return the time curve for total mass intake per unit volume since initial-conditions

#Arguments
    state: a DiffusionState holding the solved datapoints

"""
function mass_intake(C_hist, img::AbstractArray)

    voxels = length(img)
    mass_intake = A -> (nansum(A)-nansum(C_hist[1]))/voxels
    return map(mass_intake, C_hist)
end
mass_intake(C_hist, prob::TransientProblem) = mass_intake(C_hist, prob.img)
mass_intake(sim::TransientState, prob::TransientProblem) = mass_intake(sim.C, prob.img) 