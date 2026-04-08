# Variable Diffusivity

By default, `Tortuosity.jl` assumes uniform diffusivity (`D=1`) across all conducting voxels. You can override this by passing a custom diffusivity array that assigns a value to each voxel. This is useful for heterogeneous media such as fractured rock, or non-porous domains composed of different materials (e.g., a bubbly mixture).

The workflow is similar to the [basic usage](index.md), but you define a custom `D` array and pass it to the simulation. The effective diffusivity (and hence tortuosity) is then computed as a flux-weighted quantity that accounts for the spatially varying `D`.

## Entire image as domain

Assume you have a binary image where the `true` voxels are 5x more conductive than the `false` voxels. Here, we use the entire image as the computational domain by setting all voxels as conducting.

!!! note
    Tortuosity is ill-defined when the entire image is the domain, since there is no distinct void phase. The concentration field is the primary quantity of interest in this case, but you can still compute a tortuosity factor for comparison.

```@example
using Tortuosity

USE_GPU = false

# Generate a test image
img = Imaginator.blobs(; shape=(64, 64, 1), porosity=0.65, blobiness=0.5, seed=2)

# Define the diffusivity field
D = zeros(size(img))
D[img] .= 1.0       # More conductive phase
D[.!img] .= 0.2     # Less conductive phase
domain = D .> 0      # The entire image is the domain

# We pass `domain` (not `img`) to the constructor since all voxels are conducting
sim = TortuositySimulation(domain; axis=:x, D=D, gpu=USE_GPU);

# Solve the system of equations
sol = solve(sim.prob, KrylovJL_CG(); verbose=false)

# Convert the solution vector to an Nd grid
c = vec_to_grid(sol.u, domain)
τ = tortuosity(c, domain; axis=:x, D=D)
println("τ = $τ")

# Visualize the concentration field
using Plots
heatmap(img[:,:,1]; aspect_ratio=:equal, clim=(0, 1))
savefig("img-plot.svg"); nothing # hide
heatmap(c[:, :, 1]; aspect_ratio=:equal, clim=(0, 1))
savefig("c-plot.svg"); nothing # hide
```

```@example
HTML("""<figure><img src=$(joinpath(Main.buildpath,"img-plot.svg"))><figcaption>Binary image used to assign diffusivity</figcaption></figure>""") # hide
```

```@example
HTML("""<figure><img src=$(joinpath(Main.buildpath,"c-plot.svg"))><figcaption>Concentration field for the entire domain</figcaption></figure>""") # hide
```

## Subdomain as domain

This example uses only the pore phase as the computational domain, with a random diffusivity field assigned to each pore voxel. For illustration, we draw diffusivities from a uniform distribution.

```@example
using Tortuosity

USE_GPU = false

# Generate a test image
img = Imaginator.blobs(; shape=(64, 64, 1), porosity=0.65, blobiness=0.5, seed=2)

# Define the diffusivity field
D = rand(Float64, size(img))  # Random diffusivity for all voxels
D[.!img] .= 0                 # Zero out non-conducting voxels
domain = D .> 0                # Conducting voxels only

sim = TortuositySimulation(domain; axis=:x, D=D, gpu=USE_GPU);
sol = solve(sim.prob, KrylovJL_CG(); verbose=false)

# Convert the solution vector to an Nd grid
c = vec_to_grid(sol.u, domain)
τ = tortuosity(c, domain; axis=:x, D=D)
println("τ = $τ")

# Visualize
using Plots
heatmap(img[:,:,1]; aspect_ratio=:equal, clim=(0, 1))
savefig("img-plot-partial-domain.svg"); nothing # hide
heatmap(c[:, :, 1]; aspect_ratio=:equal, clim=(0, 1))
savefig("c-plot-partial-domain.svg"); nothing # hide
```

```@example
HTML("""<figure><img src=$(joinpath(Main.buildpath,"img-plot-partial-domain.svg"))><figcaption>Binary image used to assign diffusivity</figcaption></figure>""") # hide
```

```@example
HTML("""<figure><img src=$(joinpath(Main.buildpath,"c-plot-partial-domain.svg"))><figcaption>Concentration field for the pore subdomain</figcaption></figure>""") # hide
```

In this case, `domain` is essentially the same as `img`, but we define it separately to show how the workflow generalizes. `D` is initialized for all voxels but set to zero for non-conducting ones, and `domain` is derived from where `D > 0`.
