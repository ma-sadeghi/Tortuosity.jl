using Tortuosity: Imaginator
using Images, FileIO

# Parameters for different slices
params = [
    (porosity=0.50, blobiness=0.3, seed=1),
    (porosity=0.55, blobiness=0.4, seed=2),
    (porosity=0.60, blobiness=0.5, seed=3),
    (porosity=0.65, blobiness=0.6, seed=4),
    (porosity=0.70, blobiness=0.7, seed=5)
]

for (i, p) in enumerate(params)
    # Generate the image
    img = Imaginator.blobs(; shape=(64, 64, 1), porosity=p.porosity, blobiness=p.blobiness, seed=p.seed)
    img = Imaginator.trim_nonpercolating_paths(img, axis=:x)

    # Extract and save the slice
    img_slice = img[:, :, 1]  # 2D slice
    filename = "porous_slice_$(i).png"
    save(filename, img_slice)
    println("Saved: $filename")
end


# Ax + b ,_- A sparse is the problem
# Start with CPU -> THen GPU 
# Assume A dense first. (Differential Equations package) 
# Maybe some examples for sparse <-- We're trying to solve heat equtions 


# Diagonal should be sum of row,
# Start with dense -> U-L Triangles

#Goal for now <-- GEt the dots from the transient eq of the heat eq. 
#1. Dense, GPU
#2. Sparse (Hopefully DiffEq.jl can handle sparse on GPU) 
#3. 


#Note: INitial + Boundary conditions do matter. (Right boundary can be 0 or no flux)
# Differnet analyistca soltuonis based on differenet boundary conditions. 


#pmeal server  username: anaconda 
# password: lab5009

# Towards ends: Pores become solid (No longer interacts)