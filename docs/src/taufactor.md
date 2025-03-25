# Why not TauFactor?

For the most part, [taufactor](https://github.com/tldr-group/taufactor) is a great package to compute the tortuosity factor of porous media. However, it has two main drawbacks, which are addressed by `Tortuosity.jl`:

## Inaccurate concentration field

First, `taufactor` doesn't properly solve the Laplace equation. Instead, it solves it enough such that the inlet and outlet fluxes match to a prescribed precision. This is not a problem per se, but it can lead to inaccurate results especially when there's strong heterogeneity in the medium (or more technically, when the size of the image is comparable to the [representative elementary volume](https://en.wikipedia.org/wiki/Representative_elementary_volume)). Also, even when the computed tortuosity factor is accurate, the concentration field is not guaranteed to be accurate.

## Sparse vs. matrix representation

Second, while `taufactor` is GPU-accelerated, it always considers the entire image as the working domain. This is not a problem when porosity (i.e,. void fraction) is high and close to 1, but it starts to slow down the computation at lower porosities. In contrast, `Tortuosity.jl` only considers the void space, which can lead to significant speedups up to 5x at low porosities.

That being said, `taufactor` is a great package and has been an inspiration for `Tortuosity.jl`!
