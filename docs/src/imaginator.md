# Imaginator

`Imaginator` is a submodule for generating and manipulating synthetic 3D voxel images of porous media. If you are familiar with [PoreSpy](https://porespy.org) in Python, `Imaginator` is analogous to its `generators` module.

## Image generation

### `blobs`

```julia
Imaginator.blobs(; shape, porosity, blobiness, seed=nothing)
```

Generates a random binary image by thresholding Gaussian-blurred noise. Returns a `BitArray` where `true` = pore, `false` = solid.

- **`shape`** — tuple specifying the image dimensions, e.g., `(64, 64, 64)` for 3D or `(64, 64, 1)` for a 2D slice.
- **`porosity`** — target void fraction (0 to 1).
- **`blobiness`** — spatial correlation (0 to 1). Higher values produce larger, smoother features; lower values produce finer, grainier textures.
- **`seed`** — random seed for reproducibility.

```@example imag
using Plots
using Tortuosity

p1 = heatmap(Imaginator.blobs(; shape=(64,64,1), porosity=0.5, blobiness=0.2, seed=1)[:,:,1];
    aspect_ratio=:equal, clim=(0,1), title="blobiness = 0.2")
p2 = heatmap(Imaginator.blobs(; shape=(64,64,1), porosity=0.5, blobiness=0.5, seed=1)[:,:,1];
    aspect_ratio=:equal, clim=(0,1), title="blobiness = 0.5")
p3 = heatmap(Imaginator.blobs(; shape=(64,64,1), porosity=0.5, blobiness=0.8, seed=1)[:,:,1];
    aspect_ratio=:equal, clim=(0,1), title="blobiness = 0.8")
plot(p1, p2, p3; layout=(1,3), size=(900, 300))
savefig("blobiness.svg"); nothing # hide
```

```@example imag
HTML("""<figure><img src=$(joinpath(Main.buildpath,"blobiness.svg"))><figcaption>Effect of blobiness on pore structure (porosity = 0.5 for all)</figcaption></figure>""") # hide
```

## Image analysis

### `phase_fraction`

```julia
Imaginator.phase_fraction(img)          # Dict of all phase fractions
Imaginator.phase_fraction(img, label)   # fraction of voxels equal to label
```

Returns volume fractions. For a binary image, `phase_fraction(img, true)` gives the porosity.

### `trim_nonpercolating_paths`

```julia
Imaginator.trim_nonpercolating_paths(img; axis)
```

Removes pore clusters that do not form a connected path between the inlet and outlet faces along `axis`. This is essential before solving — isolated clusters would make the linear system singular.

```@example imag
img = Imaginator.blobs(; shape=(64, 64, 1), porosity=0.5, blobiness=0.5, seed=2)
println("Before trimming: porosity = $(Imaginator.phase_fraction(img, true))")

img_trimmed = Imaginator.trim_nonpercolating_paths(img, axis=:x)
println("After trimming:  porosity = $(Imaginator.phase_fraction(img_trimmed, true))")

p1 = heatmap(img[:,:,1]; aspect_ratio=:equal, clim=(0,1), title="Before trimming")
p2 = heatmap(img_trimmed[:,:,1]; aspect_ratio=:equal, clim=(0,1), title="After trimming")
plot(p1, p2; layout=(1,2), size=(600, 300))
savefig("trimming.svg"); nothing # hide
```

```@example imag
HTML("""<figure><img src=$(joinpath(Main.buildpath,"trimming.svg"))><figcaption>Non-percolating pore clusters (dead ends) are removed</figcaption></figure>""") # hide
```

## Image processing utilities

### `denoise`

```julia
Imaginator.denoise(img, kernel_radius)
```

Morphological denoising via closing followed by opening. Removes small isolated features (both pore and solid) smaller than `kernel_radius`.

### `disk` / `ball`

```julia
Imaginator.disk(r)   # 2D circular structuring element
Imaginator.ball(r)   # 3D spherical structuring element
```

Structuring elements for morphological operations.

### `faces`

```julia
Imaginator.faces(shape; inlet=nothing, outlet=nothing)
```

Creates boolean masks for the inlet and outlet faces of an image. Used internally for boundary condition application.
