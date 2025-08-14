using Tortuosity
using Images, FileIO
using Krylov
using ColorTypes
using Tortuosity: Imaginator, TortuositySimulation, vec_to_grid, tortuosity

for i in 1:5
    filename = "porous_slice_$(i).png"
    println("Processing $filename...")

    # Load the image
    img = load(filename)

    # Convert to grayscale if needed
    if ndims(img) == 3
        img = channelview(img)[1, :, :]
    end

    # Binarize and convert to Float32
    binary_mask = Float32.(img .> 0.5)
    binary_mask = Imaginator.trim_nonpercolating_paths(binary_mask, axis = :x)

    # Reshape to 3D
    binary_mask_3d = reshape(binary_mask, size(binary_mask)..., 1)

    # Create and solve the simulation
    sim = TortuositySimulation(binary_mask_3d; axis = :x, gpu = false)
    sol = solve(sim.prob, KrylovJL_CG(); verbose = false, reltol = 1e-5)

    # Convert vector solution to grid
    c_grid = vec_to_grid(sol.u, binary_mask_3d)

    # Compute tortuosity
    τ = tortuosity(c_grid, axis = :x)
    println("τ[$filename] = $τ")
end
