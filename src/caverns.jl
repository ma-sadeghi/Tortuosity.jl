# adaptation of Mohammad Mehrnia's caverns.py to Tortuosity.jl
# for identifying voxels in a 3D image of a porous material which do not contribute to steady state flux
# i.e. dead ends or 'caverns'

"""
    find_caverns(img; vmin=-2, iter=1, axis=:z, reltol=1e-5, gpu=true)

Identify low‑flux “cavern” regions in a 3D porous medium by iteratively solving a
steady diffusion problem and thresholding the resulting flux field.

This routine repeatedly:
1. Removes voxels already classified as caverns.
2. Solves a steady diffusion problem on the remaining pore space.
3. Computes the local flux magnitude at each pore voxel.
4. Marks voxels with `log10(flux) < vmin` as caverns.
5. Removes any newly isolated, non‑percolating pore clusters.

The process is repeated for `iter` iterations, and the fraction of voxels
classified as caverns is recorded after each step.

# Arguments
- `img::BitArray`: 3D boolean mask of the pore space (`true` = pore).

# Keyword Arguments
- `vmin::Real = -2`: Log‑flux threshold. Voxels with `log10(flux) < vmin` are
  classified as caverns.
- `iter::Int = 1`: Number of refinement iterations to perform.
- `axis::Symbol = :z`: Transport axis used for the diffusion simulation.
- `reltol::Real = 1e-5`: Relative tolerance for the diffusion solution.
- `gpu::Bool = true`: Whether to run the diffusion solve on the GPU.

# Returns
- `caverns::BitArray`: A 3D boolean mask marking cavern voxels (`true`).
- `kai::Vector{Float64}`: The cavern fraction at each iteration, including the
  initial value of 0.

# Notes
- Caverns are defined as pore voxels with extremely low through‑flux relative to
  the imposed gradient.
- Non‑percolating pore clusters are removed at each iteration to prevent
  artificially isolated regions from being misclassified.
"""
function find_caverns(img::BitArray; vmin = -2, iter = 1, axis::Symbol = :z, reltol = 1e-5, gpu = true )

    N = size(img, axis_dim(axis))

    caverns = falses(size(img))
    kai = zeros(iter+1) #keep track of 'cavernosity' at each iteration
    filled_img = copy(img)
    for i= 1:iter
        filled_img[caverns] .= false

        sim = SteadyDiffusionProblem(filled_img; axis=axis, gpu=gpu)
        #scale 'flux' to be resolution independent, maybe better if this could be applied as boundary condition directly
        C  = N.*reconstruct_field(solve(sim.prob, KrylovJL_CG(); verbose=false, reltol=reltol).u, filled_img)

        #C is always on CPU from reconstruct_field, but this shouldn't be a bottleneck
        flux = flux_out(C, filled_img)

        caverns[(log10.(flux) .< vmin) .& filled_img] .= true
        #also trim newly isolated pores
        caverns[img .& .!Imaginator.trim_nonpercolating_paths(img .& .!caverns; axis=axis)] .= true #using the sub-module in the module, not great...

        kai[i+1] = count(caverns) / count(img)
    end

    return caverns, kai
end

"""
    flux_out(C::AbstractArray, img::AbstractArray{Bool})

Compute the total absolute flux at each voxel by summing `|ΔC|` over all
face-connected neighbors. Operates in image space (full 3D grid). Solid
voxels contribute zero flux. Does not account for non-uniform diffusivity.
"""
function flux_out(C::AbstractArray, img::AbstractArray{Bool})
    @assert size(C) == size(img) "size of C must match size of img"

    #find flux of each connection with image mask. note that in Julia, false * NaN = 0.0
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