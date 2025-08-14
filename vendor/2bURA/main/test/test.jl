println("Hello Julia!")

using Tortuosity
using Tortuosity: vec_to_grid, tortuosity
using Images, FileIO, Plots

# Step 1: Create a porous medium (3D binary image)
img = Imaginator.blobs(; shape=(64, 64, 1), porosity=0.6, blobiness=0.5, seed=123)

# Step 2: Ensure the structure is percolating
img = Imaginator.trim_nonpercolating_paths(img, axis=:x)

# Step 3: Simulate diffusion
sim = TortuositySimulation(img; axis=:x)
sol = solve(sim.prob, KrylovJL_CG(); verbose=false, reltol=1e-5)

# Step 4: Convert solution to a grid
c_grid = vec_to_grid(sol.u, img)

# Step 5: Calculate tortuosity
τ = tortuosity(c_grid, axis=:x)
println("Tortuosity τ = $τ")

# Step 6: Visualize and Save
img_slice = img[:, :, 1]  # Take 2D slice
heatmap(img_slice, aspect_ratio=:equal, title="Pore Structure Slice")

# Optional: Save image to file
save("porous_slice.png", img_slice)
