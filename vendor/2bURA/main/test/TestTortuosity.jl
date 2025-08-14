using Tortuosity
using Images, FileIO
using Krylov  # Needed for KrylovJL_CG()
using Tortuosity: Imaginator, TortuositySimulation, vec_to_grid, tortuosity, formationfactor # Needed for trimming, if necessary
using ColorTypes

# Load the image
img = load("porous_slice.png")

# Convert to grayscale if not already
if ndims(img) == 3  # color image
    img = channelview(img)[1, :, :]  # convert to grayscale
end

# Binarize: assume pores are white (high pixel value)
binary_mask = Float32.(img .> 0.5)   # `true` = pore, `false` = solid

# Optionally ensure the image percolates in x-direction
binary_mask = Imaginator.trim_nonpercolating_paths(binary_mask, axis = :x)
binary_mask_3d = reshape(binary_mask, size(binary_mask)..., 1)


# Create the Tortuosity simulation
sim = TortuositySimulation(binary_mask_3d; axis = :x, gpu = false)

# Solve the steady-state PDE
sol = solve(sim.prob, KrylovJL_CG(); verbose = false, reltol = 1e-5)

# Map the solution vector back to grid
c_grid = vec_to_grid(sol.u, binary_mask)

# Compute tortuosity
τ = tortuosity(c_grid, axis = :x)
println("τ = $τ")
