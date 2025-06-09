# Variable diffusivity

You can use `Tortuosity.jl` to compute the tortuosity factor of a porous medium (or non-porous for that matter!) with variable diffusivity, i.e., you can assign a custom diffusivity value to each voxel in the image. This is useful when you have a heterogeneous medium, where the diffusivity varies across the domain, like a fractured rock, or a non-porous medium made of different materials, e.g., a bubbly mixture.

The workflow is similar to the basic usage, but you need to define a custom diffusivity field. Here are two examples:

## Entire image as domain

Assume you have a binary image where the true voxels are 5x more conductive than the false voxels. Note that tortuosity is ill-defined in this case, since we're not really dealing with a porous medium. In such cases, the concentration field is the important quantity to compute, but you can still compute the tortuosity factor for the sake of comparison!

```@example
using Tortuosity
using Tortuosity: tortuosity, vec_to_grid

USE_GPU = true

# Generate a test image
img = Imaginator.blobs(; shape=(64, 64, 1), porosity=0.65, blobiness=0.5, seed=2)

# Define the diffusivity field
D = zeros(size(img))
D[img] .= 1.0       # More conductive phase
D[.!img] .= 0.2     # Less conductive phase
domain = D .> 0     # Define domain as only-conducting voxels (i.e., the entire image)

# Define the simulation
sim = TortuositySimulation(domain; axis=:x, D=D, gpu=USE_GPU);

# Solve the system of equations
sol = solve(sim.prob, KrylovJL_CG())

# Convert the solution vector to an Nd grid
c = vec_to_grid(sol.u, domain)
# Compute the tortuosity factor
τ = tortuosity(c; axis=:x, D=D)
println("τ = $τ")

# Visualize the concentration field
using Plots
heatmap(img[:,:,1]; aspect_ratio=:equal, clim=(0, 1))
savefig("img-plot.svg"); nothing # hide
heatmap(c[:, :, 1]; aspect_ratio=:equal, clim=(0, 1))
savefig("c-plot.svg"); nothing # hide
```

```@raw html
<figure>
    <img src="./img-plot.svg"
         alt="Original binary">
    <figcaption>Original binary image used to assign diffusivity</figcaption>
</figure>

<figure>
    <img src="./c-plot.svg"
         alt="Concentration field">
    <figcaption>Concentration field for the entire domain</figcaption>
</figure>
```

Note that we passed `domain` and not `img` to the `TortuositySimulation` constructor, since we are using the entire image (both true and false voxels) as the domain, and we only use the `img` to generate the diffusivity field.

## Subdomain as domain

This example is similar to the first one, but we only consider a subdomain of the image as the domain. Assume we have a porous medium (described by a binary image) with a variable diffusivity field. Just to demonstrate the variable diffusivity, let's assume a random diffusivity field.

```@example
using Tortuosity
using Tortuosity: tortuosity, vec_to_grid

USE_GPU = true

# Generate a test image
img = Imaginator.blobs(; shape=(64, 64, 1), porosity=0.65, blobiness=0.5, seed=2)

# Define the diffusivity field
D = rand(size(img)...)      # Random diffusivity field
D[.!img] .= 0               # Non-conducting  
domain = D .> 0             # Define domain as only-conducting voxels

# Define the simulation
sim = TortuositySimulation(domain; axis=:x, D=D, gpu=USE_GPU);

# Solve the system of equations
sol = solve(sim.prob, KrylovJL_CG())

# Convert the solution vector to an Nd grid
c = vec_to_grid(sol.u, domain)
# Compute the tortuosity factor
τ = tortuosity(c; axis=:x, D=D)
println("τ = $τ")

# Visualize the concentration field
using Plots
heatmap(img[:,:,1]; aspect_ratio=:equal, clim=(0, 1))
savefig("img-plot-partial-domain.svg"); nothing # hide
heatmap(c[:, :, 1]; aspect_ratio=:equal, clim=(0, 1))
savefig("c-plot-partial-domain.svg"); nothing # hide
```

```@raw html
<figure>
    <img src="./img-plot-partial-domain.svg"
         alt="Binary image">
    <figcaption>Original binary image used to assign diffusivity</figcaption>
</figure>
<figure>
    <img src="./c-plot-partial-domain.svg"
         alt="Concentration field for the subdomain">
    <figcaption>Concentration field</figcaption>
</figure>
```

Note that in this case, `domain` is essentially the same as `img`, but we still define it separately to demonstrate how to use a variable diffusivity field. The diffusivity field `D` is defined only for the conducting voxels, and the non-conducting voxels are set to zero.
