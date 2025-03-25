# Imaginator.jl

`Imaginator.jl` is a submodule within `Tortuosity.jl` to generate synthetic 3D voxel images of porous media plus some utilities to manipulate them. The main goal of `Imaginator.jl` is to provide a simple way to generate 3D images of porous media for testing and benchmarking purposes. If you're familiar with the [`PoreSpy`](https://porespy.org) package in Python, you can think of `Imaginator.jl` as a stripped-down `generators` module in `PoreSpy`.

## Example usage

```julia
using Tortuosity: Imaginator

img = Imaginator.blobs(shape=(100, 100, 100), porosity=0.4, blobiness=1.5)
eps = Imaginator.phase_fraction(img)
@info "Phase fraction: $eps"

# Remove non-percolating paths along the x-axis
img = Imaginator.trim_nonpercolating_paths(img, axis=:x)
eps = Imaginator.phase_fraction(img)
@info "Phase fraction after trimming: $eps"
```
