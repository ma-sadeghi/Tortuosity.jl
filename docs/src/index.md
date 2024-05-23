# Tortuosity.jl

`Tortuosity.jl` is a GPU-accelerated solver to compute the tortuosity factor ($\tau$) of voxel images of porous media. You can think of `Tortuosity.jl` as the equivalent of [TauFactor](https://www.mathworks.com/matlabcentral/fileexchange/57956-taufactor) toolbox in MATLAB, or [taufactor](https://github.com/tldr-group/taufactor) in Python, but faster and [more reliable](taufactor.md).

## Installation

To install `Tortuosity.jl`, simply run the following command in the Julia REPL:

```julia
using Pkg
Pkg.add("Tortuosity")
```

## Basic usage

To compute the tortuosity factor of a voxel image, you can use the `tortuosity` function as follows:

```julia
using Tortuosity
using Tortuosity: Imaginator

# Create a random 3D voxel image using Gaussian noise
img = Imaginator.blobs(shape=(100, 100, 100), porosity=0.4, blobiness=1.5)

# Compute the tortuosity factor
τ = tortuosity(img)
@info "Tortuosity factor: $τ"
```
