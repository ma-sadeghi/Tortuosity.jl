---
title: 'Tortuosity.jl: GPU-accelerated diffusion solving and tortuosity computation for porous media'
tags:
  - Julia
  - porous media
  - tortuosity
  - diffusion
  - GPU computing
  - image-based simulation
authors:
  - name: Mohammad Amin Sadeghi
    orcid: 0000-0002-3477-4959
    corresponding: true
    affiliation: 1
  - name: Sawyer Hossfeld
    affiliation: 1
  - name: Jeff T. Gostick
    orcid: 0000-0001-7736-7124
    affiliation: 1
affiliations:
  - name: Department of Chemical Engineering, University of Waterloo, Waterloo, Canada
    index: 1
date: 8 April 2026
bibliography: paper.bib
---

# Summary

`Tortuosity.jl` is a Julia package for solving diffusion equations on voxel
images of porous media and extracting transport properties such as the
tortuosity factor. Given a binary or labeled 3D image of a porous material,
the package assembles and solves a steady-state or transient diffusion
problem on the pore space. The primary output is the tortuosity factor
($\tau$), a scalar that quantifies how much the pore geometry impedes
diffusive transport relative to free diffusion. The package also supports
spatially varying diffusivity, which enables simulation of heterogeneous
systems such as bubbly mixtures or composite electrodes. Both CPU and CUDA
GPU execution are supported, and the package includes a built-in synthetic
image generator for testing and benchmarking.

# Statement of need

Image-based transport simulation is central to the study of batteries,
fuel cells, geological formations, filtration membranes, and other porous
materials. The tortuosity factor links the effective diffusivity
$D_\text{eff}$ of the medium to the intrinsic diffusivity $D_0$ and the
porosity $\varepsilon$ through

$$D_\text{eff} = D_0 \, \frac{\varepsilon}{\tau}$$

Computing $\tau$ from a voxel image requires discretizing the Laplace
equation on the pore space, solving the resulting linear system, and
computing the ratio of actual to ideal flux. Existing tools---TauFactor
[@cooper2016taufactor] in MATLAB, taufactor [@kench2022taufactor] in
Python, and PuMA [@ferguson2018puma; @ferguson2021puma]---have made this
workflow accessible, but each has limitations that `Tortuosity.jl` addresses.

TauFactor is CPU-only. taufactor adds GPU support through PyTorch but uses
a successive over-relaxation (SOR) solver that operates on the full image
grid, including solid voxels, so memory and compute scale with the total
domain size rather than the pore volume. PuMA offers multiple solvers
including finite volume, explicit jump, and random walk methods, but is
also CPU-only (parallelized via OpenMP) and is written in C++ with a Python
interface (`pumapy`), which makes the solver internals harder to inspect or
modify.

`Tortuosity.jl` takes a different approach. The package assembles a sparse
linear system that includes only pore voxels, so memory use and solver cost
scale with the pore count, not the image size. This is particularly
advantageous for high-porosity media. The solver delegates to Krylov methods
from `LinearSolve.jl` [@rackauckas2024linearsolve], which converge faster
than SOR for well-conditioned Laplacian systems. GPU offloading requires a
single keyword argument (`gpu=true`), with no user-managed memory transfers.
Because Julia compiles to native code, per-iteration overhead is lower than
in Python-based solvers, which matters most for smaller domains where the
iteration count is low and the interpreter overhead of a Python loop becomes
a significant fraction of total runtime.

# State of the field

TauFactor [@cooper2016taufactor] introduced the image-based tortuosity
workflow for the porous media community and remains widely cited. Its Python
successor, taufactor [@kench2022taufactor], added PyTorch-based GPU
acceleration, batch processing, and microstructural metrics (volume
fraction, surface area, triple-phase boundaries). PuMA
[@ferguson2018puma; @ferguson2021puma], developed at NASA, is a C++ toolkit
with Python bindings (`pumapy`) that computes tortuosity, thermal and
electrical conductivity, permeability, and mechanical properties using
finite volume, explicit jump, and random walk methods.

`Tortuosity.jl` focuses specifically on diffusion-based tortuosity
computation and provides capabilities that complement these existing tools:
(1) a pore-only sparse formulation that avoids allocating memory for solid
voxels, (2) Krylov solvers instead of relaxation-based iteration, (3)
transient diffusion for studying time-dependent uptake and breakthrough
experiments, and (4) continuously varying per-voxel diffusivity fields
rather than the discrete per-phase labels supported by taufactor. The
package is written entirely in Julia, so both the high-level API and the
low-level kernels are accessible in a single language without a compiled
extension layer.

# Software design

The workflow proceeds in three steps:

1. **Pore extraction and numbering.** Only pore voxels participate in the
   linear system. Solid voxels are excluded entirely, reducing memory use
   and solver time proportionally to the solid fraction.

2. **Sparse system assembly.** A graph Laplacian is assembled from a
   face-connectivity stencil over the pore voxels. Dirichlet boundary
   conditions (concentration of 1 and 0) are applied on opposite faces
   along the transport axis; remaining boundaries are insulated. When a
   per-voxel diffusivity field is supplied, harmonic-mean weighting is
   used at each interface.

3. **Solve and post-process.** The sparse linear system is solved with a
   Krylov solver (conjugate gradient by default) from `LinearSolve.jl`
   [@rackauckas2024linearsolve]. The solution vector is mapped back onto
   the image grid, and the tortuosity factor is computed from the mean
   flux.

GPU acceleration is implemented through CUDA.jl [@besard2019cuda]. When
`gpu=true`, the sparse matrix and vectors are transferred to device memory,
and the Krylov solver operates entirely on the GPU.

For transient problems, the package integrates with `OrdinaryDiffEq.jl`
[@rackauckas2017diffeq] to solve the time-dependent diffusion equation.
Users can specify flexible stop conditions (target time, average
concentration, or periodicity) and fit the resulting concentration curves
to analytical slab-diffusion solutions.

# Performance comparison

To quantify the performance differences, we benchmark `Tortuosity.jl`
against taufactor [@kench2022taufactor] on synthetic 3D images generated
by the Imaginator submodule with domain sizes from 50$^3$ to 200$^3$ and
porosities of 0.3, 0.5, 0.7, and 0.9. Both solvers use a convergence
parameter of $10^{-5}$, though the criteria differ: `Tortuosity.jl` uses
the algebraic residual norm (`reltol`), while taufactor measures flux
uniformity across slices (`conv_crit`). A reference solution from
`Tortuosity.jl` at `reltol` $= 10^{-8}$ serves as ground truth.

![Solve time (left), tortuosity agreement (center), and relative error
(right) for $\varepsilon \approx 0.5$. Tortuosity.jl (circles) scales
smoothly across all sizes. taufactor (squares) becomes slower at larger
domains and exhibits higher relative error, particularly at low
porosity where SOR struggles to converge within 10,000
iterations.\label{fig:benchmark}](benchmark_time.png)

\autoref{fig:benchmark} shows that `Tortuosity.jl` achieves relative
errors below $10^{-4}$ across all cases, while taufactor errors range
from $10^{-3}$ at high porosity to $10^{-1}$ at low porosity. At
$N = 200$ and $\varepsilon = 0.3$, `Tortuosity.jl` is roughly 5$\times$
faster. The advantage stems from two factors: CG converges in fewer
iterations than SOR for Laplacian systems, and the pore-only sparse
formulation avoids computation on solid voxels.

# Research impact statement

`Tortuosity.jl` is registered in the Julia General registry and installable
via the built-in package manager. The package has been developed over
roughly three years (2023--2026) with contributions from multiple
developers and over 500 commits. It is used within the Gostick research
group at the University of Waterloo for characterizing electrode
microstructures in battery and fuel cell research. The Imaginator
submodule, which generates synthetic porous geometries with tunable
porosity and feature size, provides reproducible test cases for
benchmarking across tools.

# AI usage disclosure

Claude (Anthropic) was used during development for code suggestions,
documentation drafting, and code review. All AI-generated content was
reviewed, tested, and validated by the human authors. Core algorithmic
design and scientific decisions were made entirely by the authors.

# Acknowledgements

We acknowledge the developers of TauFactor and taufactor, whose work
inspired the design of this package. We also thank the Julia community for
the excellent ecosystem of packages that `Tortuosity.jl` builds on,
including LinearSolve.jl, CUDA.jl, and OrdinaryDiffEq.jl.

# References
