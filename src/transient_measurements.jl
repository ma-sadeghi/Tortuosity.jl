##--- functions for extracting datapoints of interest from concentration distributions or problem---

function porosity(img)
    sum(img)/length(img) #assumes pores are represented by 1, not very standard but that is the definition here
end
porosity(problem::TransientProblem) = porosity(problem.img)


function slice_conc_dist(C, img, axis)

    #accept pore-voxel vector form as well as full 3D distribution
    C = isa(C, AbstractVector) ? vec_to_grid(C, img) : C

    collapse = AXIS_COMPLEMENT[axis] #dims to sum over
    dims_comp = size(img, AXIS_COMPLEMENT[axis])
    slice_nodes = dims_comp[1]*dims_comp[2]

    return dropdims(nansum(C, dims = collapse)./slice_nodes) #sum(img, dims=collapse), dims = collapse)
end
slice_conc_dist(C, prob::TransientProblem) = slice_conc_dist(C, prob.img, prob.axis)


"""
returns average concentration of pores in slices perpendicular to axis at index ind

"""
function get_slice_conc(C, img, axis, ind)
    #accept pore-voxel vector form as well as full 3D distribution
    C = isa(C, AbstractVector) ? vec_to_grid(C, img) : C

    ax = AXIS_DEFINITION[axis]

    C_slice = selectdim(C, ax, ind)
    img_slice = selectdim(img, ax, ind)

    return nansum(C_slice)/length(img_slice)
end
get_slice_conc(C, prob::TransientProblem, ind) = get_slice_conc(C, prob.img, prob.axis, ind)



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

    # accumulate flux contributions for each slice pair
    fluxes = similar(inds, Float64)

    for (k, i) in enumerate(inds)
        C1 = selectdim(C, ax, i)
        C2 = selectdim(C, ax, i+1)

        m1 = selectdim(img, ax, i)
        m2 = selectdim(img, ax, i+1)

        #zero voxels adjacent to solid for C[solid] = 0 case, redundant for C[solid] = NaN case
        ΔC = C1 .* (m2 .!= 0) .- C2 .* (m1 .!= 0)
        
        plane_nodes = (dims[comp[1]]) * (dims[comp[2]])
        # sum over the perpendicular axes
        fluxes[k] = D* nansum(ΔC) /dx /plane_nodes
    end

    return fluxes
end
flux_dist(C, prob::TransientProblem; inds = nothing)=  flux_dist(C, prob.D_pore, prob.dx, prob.img, prob.axis; inds = inds)


"""
returns flux/area between slice of C at ind and ind+1
"""
function get_flux(C::Array, D, dx, img, axis; ind=:end)

    #accept pore-voxel vector form as well as full 3D distribution
    C = isa(C, AbstractVector) ? vec_to_grid(C, img) : C
    
    ax = AXIS_DEFINITION[axis]          # 1, 2, or 3
    comp = AXIS_COMPLEMENT[axis]
    dims  = size(C)
    

    ind === :end && (ind = dims[ax] - 1) #end symbol for flux between second last and last slice

    # slices along the chosen axis
    C1 = selectdim(C, ax, ind)
    C2 = selectdim(C, ax, ind + 1)

    m1 = selectdim(img, ax, ind)
    m2 = selectdim(img, ax, ind + 1)

    # flux only through pore voxels
    ΔC = C1 .* (m2 .!= 0) .- C2 .* (m1 .!= 0)

    
    plane_nodes = (dims[comp[1]]) * (dims[comp[2]])
    # sum over the two perpendicular axes
    flux = D * nansum(ΔC)/dx /plane_nodes  #

    return flux  
end
get_flux(C, prob::TransientProblem; ind=:end)=  get_flux(C, prob.D_pore, prob.dx, prob.img, prob.axis; ind=ind)

"""
mass_intake(state)

return the time curve for total mass intake per unit volume since initial-conditions

#Arguments
    state: a DiffusionState holding the solved datapoints

"""
function mass_intake(C_hist, img::AbstractArray)

    voxels = length(img)
    mass_intake = A -> (sum(A)-sum(C_hist[1]))/voxels
    return map(mass_intake, C_hist)
end
mass_intake(C_hist, prob::TransientProblem) = mass_intake(C_hist, prob.img)
mass_intake(sim::TransientState, prob::TransientProblem) = mass_intake(sim.C, prob.img) 