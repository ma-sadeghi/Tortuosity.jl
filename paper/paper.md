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
date: 21 August 2026
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

The operator takes one of two interchangeable forms, selected by a keyword. The default assembles a sparse matrix and hands the matrix-vector product to the vendor library. The matrix-free form instead stores one `Int32` index array over the grid and recomputes each row's stencil weights inside the apply kernel. That cuts operator storage from about 40 to 14 bytes per grid voxel. Both forms use the same pore numbering and right-hand side. They agree on $\tau$ to within $5\times10^{-5}$ for a typical case and $1\times10^{-3}$ at worst, with 53 of 59 paired cases inside $3\times10^{-4}$. The largest disagreements fall on the most tortuous geometries, where single-precision conditioning is least forgiving.

The matrix-free form buys memory rather than speed. Its apply is faster in isolation, at 15.7 ms against 29.1 ms for an $800^3$ image. Across a whole solve that lead narrows, because both forms share the same preconditioner. The memory saving does not narrow, and it decides which problems fit at all. The assembled form widens its sparse offsets from `Int32` to `Int64` once the nonzero count outgrows the narrower type. That costs four bytes per offset but removes any hard size limit.

Krylov iteration counts on a 3D Laplacian normally grow with the linear size of the image. That growth, rather than the voxel count alone, is what makes large images expensive. The package therefore includes a two-level preconditioner. It groups pore voxels into cubic blocks, builds a coarse operator from those blocks, and factorizes it once per solve. A fixed ratio between levels holds the coarse problem at a constant fraction of the fine one, which nearly arrests that growth without eliminating it. Measured to a fixed relative residual of $10^{-6}$, the unpreconditioned count grows as $L^{0.91}$ in the edge length $L$, essentially the textbook linear rate, while the preconditioned count grows as $L^{0.13}$ over the same range. At $\varepsilon = 0.2$ the unpreconditioned count climbs from 2721 at $200^3$ to 7099 at $600^3$, while the preconditioned count goes from 136 to 177, reaching 205 at $1000^3$. The benefit therefore widens with image size, from $12.5\times$ fewer iterations at $200^3$ to $29.6\times$ at $600^3$.

The coarse solve pays for itself only above a threshold size, so `solve(sim)` applies the preconditioner above $10^5$ pore voxels and picks a tolerance from the element type.

The preconditioner restricts a residual to the coarse grid on every iteration. An earlier version scattered those contributions with atomic floating-point additions. Thread blocks arrive in no fixed order and floating-point addition is not associative, so the same solve returned a slightly different $\tau$ on each run. The restriction now gathers over a fixed coarse-to-fine adjacency built once during setup, which fixes the summation order and pays that cost once rather than per iteration.

For transient problems the package integrates with `OrdinaryDiffEq.jl` [@rackauckas2017diffeq], supports flexible stop conditions, and fits concentration curves to analytical slab-diffusion solutions.

# Performance comparison

We compare `Tortuosity.jl` against taufactor and PuMA on synthetic images from the built-in generator. The grid spans edge lengths $N \in \{200, 400, 600, 800, 1000\}$, five porosities from 0.2 to 0.95, and three feature sizes. All measurements come from one machine, with a Quadro RTX 8000 (48 GB) and a Xeon Silver 4110 (8 cores).

The three tools stop on different quantities: an algebraic residual, the flux uniformity across slices, and a solver residual. Equal settings therefore do not give equal accuracy. We instead sweep each tool's iteration cap over 18 logarithmically spaced values from 1 to 20,000. We then take the wall time of the fastest run that reaches a given relative error in $\tau$. Each time is the median of three repeats, measured against a `Float64` CPU reference for the same image.

Released taufactor places its Dirichlet boundaries half a voxel outside the domain and overrides the user's convergence criterion with a fixed threshold. We therefore benchmark a [fork](https://github.com/ma-sadeghi/taufactor/tree/node-centered-bc) carrying two patches, pinned as a submodule of the benchmark harness so that the exact code measured here can be recovered. The first, of 13 lines, adopts the node-centered discretization used by both `Tortuosity.jl` and PuMA and honours the caller's convergence criterion. The second, of 46 lines, reads tortuosity at chosen iteration counts without stopping the solve, which is what lets one run trace the whole accuracy-versus-time frontier, and assigns the final tortuosity when a run ends on its iteration cap rather than on convergence. Neither patch alters the SOR update itself. We apply the same instrumentation to our own solver and to PuMA, so all three tools are swept the same way.

![Wall time to reach 0.1% relative error in $\tau$. Panels (a) and (c) show scaling with edge length, as a geometric mean over porosity. Panels (b) and (d) resolve the same data by size and porosity. Blue marks a case where `Tortuosity.jl` is faster, red where the other tool is. Each device has its own axes: taufactor runs on the GPU, PuMA on the CPU.\label{fig:benchmark}](benchmark_summary.png)

At the 0.1% target `Tortuosity.jl` is faster than taufactor on the GPU in most cases. Over a fixed set of cases that both tools solve at every size, the geometric mean advantage rises steeply with edge length and then levels off: $1.1\times$ at $200^3$, $4.6\times$ at $400^3$, $8.2\times$ at $600^3$, and $8.4\times$ at $1000^3$. It does not keep widening. The mechanism is that taufactor needs more SOR sweeps on a larger grid while our iteration count grows far more slowly, and most of that difference is already spent by $600^3$.

Excluding solid voxels helps most when there is solid to exclude, so porosity moves the result more than size does. Pooled across sizes, the advantage is $58\times$ near $\varepsilon = 0.2$ and $1.8\times$ at $\varepsilon = 0.95$. At $200^3$ the two tools are close on average and taufactor is faster on the easiest images. Two cases at $1000^3$ are absent from these means because taufactor did not reach the target on them at all, after roughly forty minutes each, where our solver reached it in 21 s and 72 s; excluding them makes the reported means conservative.

The ranking depends on how much accuracy is demanded, so we resolve the same data at three targets. At a 10% target taufactor is usually faster, because its linear starting profile already sits close to the answer for an open medium. As the target tightens that head start stops helping, and the convergence rate decides the outcome. The case for this package is the accurate end of that range.

![Device memory held during the solve, by edge length and porosity, for the two operator forms. The vertical gap between the curves is the result. Each panel marks where the assembled form exhausts the card.\label{fig:memory}](benchmark_memory_gpu.png)

Memory separates the two operator forms more sharply than time does. The assembled form holds several nonzeros per pore voxel, while the matrix-free form holds one `Int32` per grid voxel. Porosity therefore sets the ratio between them, and edge length barely moves it. The matrix-free form is about $1.8\times$ leaner, ranging from $1.5\times$ to $2.4\times$. It uses less device memory than taufactor at four of five porosities at $200^3$, and at three of five from $400^3$ upward, because taufactor holds dense arrays over the whole grid at a flat 28.06 bytes per voxel while our footprint scales with the pore count. It completes every case at $1000^3$ on a 48 GB card, though the highest-porosity case reaches 97.9% of it, and there the assembled form exhausts the card. The figure reports the solve itself; a single-precision solve is also refined against a double-precision residual before it is returned, which costs a further 20 bytes per pore node.

The same solver on the GPU is roughly $21\times$ faster than its own CPU path, rising from $10\times$ at $200^3$ to about $27\times$ at $800^3$. PuMA is CPU-only, so we compare it against our CPU path, where `Tortuosity.jl` wins every case both tools solve to 0.1%, by a geometric mean of $31\times$ and by as much as $388\times$ on the least porous image. That comparison rests on $200^3$. PuMA is one of the established tools in this field and belongs in the comparison, but its cost at larger sizes was prohibitive for this campaign, so we did not sweep it above $200^3$ and do not report it there.

Neither tool is single-threaded, and neither saturates the machine. Sampled on the benchmark host during active solving, PuMA occupies a median of 6.4 of 8 cores and our CPU path 5.5, so the comparison is close to core-matched and the difference is not an artifact of one tool using more of the machine than the other. Our figure is taken at $600^3$, where the solve dominates the process; at $200^3$ our solve finishes in about three seconds and the median instead describes Julia's startup.

# Research impact statement

`Tortuosity.jl` is registered in the Julia General registry and installs through the built-in package manager. Four contributors developed it over three years, from 2023 to 2026, in more than 650 commits. The Gostick research group at the University of Waterloo uses the package to characterize electrode microstructures for battery and fuel cell research.

# AI usage disclosure

We used Claude (Anthropic) during development for code suggestions, documentation drafts, code review, and parts of the benchmark harness. The human authors reviewed, tested, and validated all AI-generated content. The authors alone made the core algorithmic and scientific decisions.

# Acknowledgements

We thank the developers of TauFactor and taufactor, whose work informed this package. We also thank the Julia community for the ecosystem it builds on, including LinearSolve.jl, CUDA.jl, and OrdinaryDiffEq.jl.

# References
