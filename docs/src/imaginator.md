# Imaginator.jl

`Imaginator.jl` is a submodule within `Tortuosity.jl` to generate synthetic 3D voxel images of porous media plus some utilities to manipulate them. The main goal of `Imaginator.jl` is to provide a simple way to generate 3D images of porous media for testing and benchmarking purposes. If you're familiar with the [`PoreSpy`](https://porespy.org) package in Python, you can think of `Imaginator.jl` as a stripped-down `generators` module in `PoreSpy`.

## Example usage

```@example
using Plots
using Tortuosity

img = Imaginator.blobs(shape=(64, 64, 1), porosity=0.5, blobiness=0.5, seed=2)
eps = Imaginator.phase_fraction(img)
@info "Phase fraction:\n$eps"

# Visualize the generated image
heatmap(img[:, :, 1]; aspect_ratio=:equal, clim=(0, 1))
savefig("blobs-init.svg"); nothing # hide

# Remove non-percolating paths along the x-axis
img = Imaginator.trim_nonpercolating_paths(img, axis=:x)
eps = Imaginator.phase_fraction(img)
@info "Phase fraction after trimming:\n$eps"

# Visualize the trimmed image
heatmap(img[:, :, 1]; aspect_ratio=:equal, clim=(0, 1))
savefig("blobs-trimmed.svg"); nothing # hide
```

```@raw html
<figure>
    <img src="./blobs-init.svg"
         alt="Initial blobs image">
    <figcaption>Initial blobs image</figcaption>
</figure>
<figure>
    <img src="./blobs-trimmed.svg"
         alt="Trimmed blobs image">
    <figcaption>Blobs image after trimming non-percolating paths</figcaption>
</figure>
```
