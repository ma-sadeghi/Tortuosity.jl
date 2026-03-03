# adaptation of Mohammad Mehrnia's caverns.py to Tortuosity.jl
# for identifying voxels in a 3D image of a porous material which do not contribute to steady state flux
# i.e. dead ends or 'caverns'


function find_caverns(img::BitArray; vmin = -2, iter = 1, axis::Symbol = :z, reltol = 1e-5, gpu = true )

    N = size(img, AXIS_DEFINITION[axis])

    caverns = falses(size(img))
    kai = zeros(iter+1) #keep track of 'cavernosity' at each iteration
    filled_img = copy(img)
    for i= 1:iter
        filled_img[caverns] .= false

        sim = TortuositySimulation(filled_img; axis=axis, gpu=gpu)
        #scale 'flux' to be resolution independent, maybe better if this could be applied as boundary condition directly
        C  = N.*vec_to_grid(solve(sim.prob, KrylovJL_CG(); verbose=false, reltol=reltol).u, filled_img)

        #C is always on CPU from vec_to_grid, but this shouldn't be a bottleneck
        flux = flux_out(C, filled_img)

        caverns[(log10.(flux) .< vmin) .& filled_img] .= true
        #also trim newly isolated pores
        caverns[img .& .!Imaginator.trim_nonpercolating_paths(img .& .!caverns; axis=axis)] .= true #using the sub-module in the module, not great...

        kai[i+1] = count(caverns) / count(img)
    end

    return caverns, kai
end

#not decided on this vs connectivity list approach
#expects image space form - computation on solids is not a huge deal as this isn't a bottleneck
#sum absolute values of flux over each connection for each voxel -> not the net flux
function flux_out(C::AbstractArray, img::AbstractArray{Bool})
    @assert size(C) == size(img) "size of C must match size of img"

    Fx = (img[1:end-1, :, :] .& img[2:end, :, :]).*
        abs.(C[1:end-1, :, :] .- C[2:end, :, :])

    Fy = (img[:, 1:end-1, :] .& img[:, 2:end, :]).*
        abs.(C[:, 1:end-1, :] .- C[:, 2:end, :])

    Fz = (img[:, :, 1:end-1] .& img[:, :, 2:end]).*
        abs.(C[:, :, 1:end-1] .- C[:, :, 2:end])

    F = similar(C)
    fill!(F, 0)

    # X-direction edges
    F[1:end-1, :, :] .+= Fx
    F[2:end,   :, :] .+= Fx

    # Y-direction edges
    F[:, 1:end-1, :] .+= Fy
    F[:, 2:end,   :] .+= Fy

    # Z-direction edges
    F[:, :, 1:end-1] .+= Fz
    F[:, :, 2:end]   .+= Fz

    return F        
end