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
date: 17 August 2026
bibliography: paper.bib
---

# Summary

`Tortuosity.jl` solves diffusion equations on voxel images of porous media and extracts transport properties. Its main output is the tortuosity factor $\tau$, a scalar that measures how much the pore geometry impedes diffusion relative to free diffusion. The package reads a binary or labeled 3D image and solves a steady-state or transient diffusion problem on the pore space. It accepts a diffusivity value for every voxel, so it can model heterogeneous materials such as composite electrodes. The same code runs on the CPU and on the GPU. A matrix-free operator and a two-level preconditioner together make $1000^3$ images practical on a single workstation card. A built-in image generator produces reproducible test geometries.

# Statement of need

Image-based transport simulation is central to research on batteries, fuel cells, geological formations, and filtration membranes. The tortuosity factor links the effective diffusivity $D_\text{eff}$ to the intrinsic diffusivity $D_0$ and the porosity $\varepsilon$:

$$D_\text{eff} = D_0 \, \frac{\varepsilon}{\tau}$$

To compute $\tau$ from an image, a solver must discretize the Laplace equation on the pore space, solve the linear system, and compare the actual flux to the ideal flux. Tomography now routinely produces images whose pore space holds hundreds of millions of voxels. At that scale the linear solve, not the imaging, decides how long a study takes. `Tortuosity.jl` serves researchers who need that solve to be fast, to fit on one workstation GPU, and to stay readable in a single language.

# State of the field

TauFactor [@cooper2016taufactor] introduced this workflow to the porous media community and remains widely cited, but it runs only on the CPU. Its Python successor, taufactor [@kench2023taufactor], added GPU support through PyTorch. That tool solves by successive over-relaxation (SOR) across the full image grid, including the solid voxels. Its memory and its compute therefore scale with the total domain size rather than with the pore volume. PuMA [@ferguson2018puma; @ferguson2021puma], from NASA, is a C++ toolkit with Python bindings for tortuosity, conductivity, permeability, and mechanical properties. PuMA also runs only on the CPU, and its C++ internals are harder to inspect or modify.

`Tortuosity.jl` restricts the linear system to pore voxels alone, so memory use and solver cost scale with the pore count rather than the image size. This matters most for low-porosity media, where a full-grid method spends most of its work on voxels that carry no transport. At high porosity little solid remains to exclude, and the advantage disappears. The solver calls Krylov methods from `LinearSolve.jl` [@rackauckas2024linearsolve], which reach a given residual in far fewer iterations than SOR. Moving a simulation to the GPU takes one keyword argument and no manual memory transfers.

# Software design

The workflow has three steps. A prefix sum over the mask numbers the pore voxels and drops the solid ones. A face-connectivity stencil over those voxels then forms a graph Laplacian. Dirichlet conditions of 1 and 0 apply on opposite faces along the transport axis, and the remaining faces are insulated. A per-voxel diffusivity field combines by harmonic mean at each interface. A Krylov method solves the system, with conjugate gradient as the default, and $\tau$ follows from the mean flux.

The assembly and stencil kernels are written once against KernelAbstractions.jl [@churavy2024ka]. They dispatch to CUDA.jl [@besard2019cuda], Metal.jl, or AMDGPU.jl, according to which package the user loads. The GPU backends are package extensions, as are the optional image generation, curve fitting, and HDF5 features, so a solver-only install carries none of them.

The operator takes one of two interchangeable forms, selected by a keyword. The default assembles a sparse matrix and hands the matrix-vector product to the vendor library. The matrix-free form instead stores one `Int32` index array over the grid and recomputes each row's stencil weights inside the apply kernel. That cuts operator storage from about 40 to 14 bytes per grid voxel. Both forms give the same pore numbering, right-hand side, and iteration count, and they agree on $\tau$ to within $2\times10^{-4}$.

The matrix-free form buys memory rather than speed. Its apply is faster in isolation, at 15.7 ms against 29.1 ms for an $800^3$ image. Across a whole solve that lead narrows, because both forms share the same preconditioner. The memory saving does not narrow, and it decides which problems fit at all. The assembled form widens its sparse offsets from `Int32` to `Int64` once the nonzero count outgrows the narrower type. That costs four bytes per offset but removes any hard size limit.

Krylov iteration counts on a 3D Laplacian normally grow with the linear size of the image. That growth, rather than the voxel count alone, is what makes large images expensive. The package therefore includes a two-level preconditioner. It groups pore voxels into cubic blocks, builds a coarse operator from those blocks, and factorizes it once per solve. A fixed ratio between levels holds the coarse problem at a constant fraction of the fine one, so the iteration count stops tracking image size. At $\varepsilon = 0.5$ the unpreconditioned count climbs from 1044 at $200^3$ to 4805 at $1000^3$, while the preconditioned count stays between 91 and 148.

The coarse solve pays for itself only above a threshold size, so `solve(sim)` applies the preconditioner above $10^5$ pore voxels and picks a tolerance from the element type.

The preconditioner restricts a residual to the coarse grid on every iteration. An earlier version scattered those contributions with atomic floating-point additions. Thread blocks arrive in no fixed order and floating-point addition is not associative, so the same solve returned a slightly different $\tau$ on each run. The restriction now gathers over a fixed coarse-to-fine adjacency built once during setup, which fixes the summation order and pays that cost once rather than per iteration.

For transient problems the package integrates with `OrdinaryDiffEq.jl` [@rackauckas2017diffeq], supports flexible stop conditions, and fits concentration curves to analytical slab-diffusion solutions.

# Performance comparison

We compare `Tortuosity.jl` against taufactor and PuMA on synthetic images from the built-in generator. The grid spans edge lengths $N \in \{200, 400, 600, 800, 1000\}$, five porosities from 0.2 to 0.95, and three feature sizes. All measurements come from one machine, with a Quadro RTX 8000 (48 GB) and a Xeon Silver 4110 (8 cores).

The three tools stop on different quantities: an algebraic residual, the flux uniformity across slices, and a solver residual. Equal settings therefore do not give equal accuracy. We instead sweep each tool's iteration cap over 18 logarithmically spaced values from 1 to 20,000. We then take the wall time of the fastest run that reaches a given relative error in $\tau$. Each time is the median of three repeats, measured against a `Float64` CPU reference for the same image.

Released taufactor places its Dirichlet boundaries half a voxel outside the domain and overrides the user's convergence criterion with a fixed threshold. We therefore benchmark a [13-line fork](https://github.com/ma-sadeghi/taufactor/commit/d05aa2e) that adopts the node-centered discretization used by both `Tortuosity.jl` and PuMA. The fork changes no solver logic.

![Wall time to reach 0.1% relative error in $\tau$. Panels (a) and (c) show scaling with edge length, as a geometric mean over porosity. Panels (b) and (d) resolve the same data by size and porosity. Blue marks a case where `Tortuosity.jl` is faster, red where the other tool is. Each device has its own axes: taufactor runs on the GPU, PuMA on the CPU.\label{fig:benchmark}](benchmark_summary.png)

At the 0.1% target `Tortuosity.jl` is faster than taufactor on the GPU in most cases, and its margin widens with image size. Over the cases where both tools converge, the geometric mean advantage rises from $1.6\times$ at $200^3$ to about $9\times$ at $1000^3$. Excluding solid voxels helps most when there is solid to exclude. The advantage therefore reaches two orders of magnitude near $\varepsilon = 0.2$, and falls to parity at $\varepsilon = 0.95$. A flat iteration count is what makes the margin grow with size, because taufactor needs more SOR sweeps on a larger grid while our solver does not.

The ranking depends on how much accuracy is demanded, so we resolve the same data at three targets. At a 10% target taufactor is usually faster, because its linear starting profile already sits close to the answer for an open medium. As the target tightens that head start stops helping, and the convergence rate decides the outcome. The case for this package is the accurate end of that range.

![Device memory held during the solve, by edge length and porosity, for the two operator forms. The vertical gap between the curves is the result. Each panel marks where the assembled form exhausts the card.\label{fig:memory}](benchmark_memory_gpu.png)

Memory separates the two operator forms more sharply than time does. The assembled form holds several nonzeros per pore voxel, while the matrix-free form holds one `Int32` per grid voxel. Porosity therefore sets the ratio between them, and edge length barely moves it. The matrix-free form is about 2.1 to 2.7 times leaner, and uses less device memory than taufactor at four of five porosities. It completes every case at $1000^3$ on a 48 GB card, where the assembled form exhausts the card at high porosity.

The same solver on the GPU is roughly $36\times$ faster than its own CPU path. PuMA is CPU-only, so we compare it against our CPU path, where `Tortuosity.jl` wins every case both tools solve to 0.1%, by a geometric mean of about $15\times$. That it rests on $200^3$ alone is itself a result: PuMA reached the target in none of the larger images within our budget.

One caveat bounds the CPU numbers. The comparison is not core-matched, because neither tool saturates the machine. Sampled during active solving, our CPU path occupies about two cores and PuMA about one.

# Research impact statement

`Tortuosity.jl` is registered in the Julia General registry and installs through the built-in package manager. Four contributors developed it over three years, from 2023 to 2026, in more than 650 commits. The Gostick research group at the University of Waterloo uses the package to characterize electrode microstructures for battery and fuel cell research.

# AI usage disclosure

We used Claude (Anthropic) during development for code suggestions, documentation drafts, code review, and parts of the benchmark harness. The human authors reviewed, tested, and validated all AI-generated content. The authors alone made the core algorithmic and scientific decisions.

# Acknowledgements

We thank the developers of TauFactor and taufactor, whose work informed this package. We also thank the Julia community for the ecosystem it builds on, including LinearSolve.jl, CUDA.jl, and OrdinaryDiffEq.jl.

# References
