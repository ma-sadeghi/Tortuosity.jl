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
    orcid: 0000-0002-6756-9117
    corresponding: true
    affiliation: 1
  - name: Sawyer Hossfeld
    affiliation: 1
  - name: Harry Kim
    affiliation: 1
  - name: Jeff T. Gostick
    orcid: 0000-0001-7736-7124
    affiliation: 1
affiliations:
  - name: Department of Chemical Engineering, University of Waterloo, Waterloo, Canada
    index: 1
    ror: 01aff2v68
date: 21 August 2026
bibliography: paper.bib
---

# Summary

`Tortuosity.jl` solves diffusion on voxel images of porous media and reports the transport properties that follow. Its main output is the tortuosity factor $\tau$, which measures how much the pore geometry slows diffusion relative to an open medium. The package takes a binary 2D or 3D image, builds a linear system over the pore voxels alone, and solves either a steady-state or a transient problem on it. Diffusivity can be set per voxel, so heterogeneous materials such as composite electrodes are in scope. The same code runs on the CPU and the GPU. A matrix-free operator and an aggregation-based coarse-space preconditioner together bring $1000^3$ images within reach of a single 48 GB card.

# Statement of need

Image-based transport simulation is central to research on batteries, fuel cells, geological formations, and filtration membranes. The tortuosity factor links the effective diffusivity $D_\text{eff}$ to the intrinsic diffusivity $D_0$ and the porosity $\varepsilon$:

$$D_\text{eff} = D_0 \, \frac{\varepsilon}{\tau}$$

To compute $\tau$ from an image, a solver must discretize the Laplace equation on the pore space, solve the linear system, and compare the actual flux to the ideal flux. Tomography now routinely produces images whose pore space holds hundreds of millions of voxels. At that scale the linear solve, not the imaging, decides how long a study takes.

`Tortuosity.jl` is for researchers who need that solve to be fast, to fit on one workstation GPU, and to stay open to change: the discretization, the operator, and the preconditioner are all plain Julia, so swapping a solver or a boundary condition does not mean leaving the language the analysis is written in.

# State of the field

Throughout, $\tau$ is the tortuosity factor of @cooper2016taufactor, not the geometric path-length ratio, and it is not squared [@tjaden2016tortuosity].

TauFactor [@cooper2016taufactor] brought this workflow to the porous media community and remains widely cited, but it is a MATLAB application and runs only on the CPU. Its Python successor, taufactor [@kench2023taufactor], is GPU-accelerated through PyTorch. It solves by successive over-relaxation (SOR) across the full grid including the solid voxels, so its memory and its compute scale with the domain size rather than the pore volume. PuMA [@ferguson2018puma; @ferguson2021puma], from NASA, spans tortuosity, conductivity, permeability, and mechanical properties. The finite-volume path we benchmark solves on the CPU with SciPy's conjugate gradient over the full grid, though PuMA can export a workspace to the CUDA code `chfem` [@lopes2023chfem]. PoreSpy [@gostick2019porespy] computes $\tau$ by finite-difference diffusion, also on the CPU.

`Tortuosity.jl` restricts the linear system to pore voxels alone, so iteration cost and most of the memory scale with the pore count rather than the image size. The only residual grid-sized term is a four-byte-per-voxel index map. This matters most at low porosity, where a full-grid method spends most of its work on voxels that carry no transport. At high porosity little solid remains to exclude and the advantage disappears. The solver calls Krylov methods from `LinearSolve.jl` [@kimmerer2024linearsolve], which reach a given residual in far fewer iterations than SOR.

# Software design

A prefix sum over the mask numbers the pore voxels and drops the solid ones, and a face-connectivity stencil over those voxels forms a graph Laplacian. Dirichlet values of 1 and 0 apply on opposite faces along the transport axis, the remaining faces are insulated, and a per-voxel diffusivity field combines by harmonic mean at each interface. Conjugate gradient solves the system by default, and $\tau$ follows from the mean flux through a cross-section.

The kernels are written once against `KernelAbstractions.jl` [@churavy2024ka] and dispatch to whichever backend the user loads. CUDA.jl [@besard2019cuda] is the one we develop and benchmark against, and the only one that routes the assembled product to cuSPARSE. Metal.jl and AMDGPU.jl use the portable kernels throughout. Moving to the GPU takes one `using` statement and no manual transfers. The GPU backends are package extensions, as are image generation, curve fitting, and HDF5 export.

The operator takes one of two interchangeable forms sharing the same pore numbering and right-hand side. The default assembles a sparse matrix for the vendor library, holding about 59 bytes per pore voxel and widening its indices to `Int64` past $3\times10^8$ of them, which removes any hard size limit. The matrix-free form stores one `Int32` index array over the grid — four bytes per voxel — and recomputes each row's weights inside the apply kernel. It buys memory, not speed: its apply is about twice as fast alone, but both forms share a preconditioner, so that lead narrows over a whole solve. The memory saving does not narrow, and it decides which problems fit.

Krylov iteration counts on a 3D Laplacian normally grow with the image's linear size, and that growth is what makes large images expensive. The package therefore includes an aggregation-based coarse-space preconditioner, which groups pore voxels into cubic blocks of a fixed eight-voxel edge and forms a Galerkin coarse operator over them. Fixing that edge rather than scaling it with the image bounds the fine-to-coarse ratio, which nearly arrests the growth. The coarse operator is factorized directly when small enough. Past roughly $3\times10^4$ coarse unknowns a V-cycle over coarser levels ends in that same factorization, so the method is multilevel in depth though two-level in shape.

Measured on the CPU in `Float64` and pooled over five porosities, the unpreconditioned count tracks the edge length at the textbook rate, while the preconditioned count grows only $1.16\times$ from $200^3$ to $600^3$. That is a saving of $12.5\times$ in iterations at $200^3$ and $29.6\times$ at $600^3$. At $\varepsilon = 0.2$ it runs 136, 177, then 205 at $1000^3$, against 2721 rising to 7099 unpreconditioned. `solve(sim)` applies it above $10^5$ pore voxels. Its restriction gathers over a fixed adjacency built at setup, pinning the summation order that an atomic scatter would leave to chance.

GPU solves run in `Float32`, where on strongly tortuous images our preconditioned conjugate gradient stagnates near $10^{-3}$ relative error in $\tau$ — short of what the format can represent, because a Krylov method loses orthogonality to accumulated rounding. Every single-precision solve is therefore refined against a double-precision residual before return, at 20 bytes per pore node.

For transient problems the package builds on `OrdinaryDiffEq.jl` [@rackauckas2017diffeq], offers stop conditions for steady state, flux balance, saturation, and periodic state, and fits concentration curves to analytical slab-diffusion solutions.

# Accuracy

Speed matters only if the answer is right. Where a closed form exists the test suite pins the solver to it: a straight duct returns $\tau = 1$ and $D_\text{eff} = \varepsilon$ to within $10^{-9}$, parallel ducts add their conductances, a dead-end pocket raises porosity without raising $D_\text{eff}$, and a layered medium reproduces the exact one-dimensional series-resistor conductance to a relative $10^{-8}$. Porous geometries have no closed form, so there we rely on two independently written codes: taufactor and PuMA reproduce our reference $\tau$ within the 0.1% benchmark target on 126 of 131 and 15 of 15 shared cases. Where a tool's ladder overshoots that threshold rather than landing on it, agreement tightens to a few parts in $10^6$, including on the most tortuous image in the set, at $\varepsilon = 0.16$ and $\tau = 33.9$. The two operator forms agree to every recorded digit in all 74 `Float64` CPU cases, and to $5\times10^{-5}$ typically on the GPU in `Float32`, where the worst six of 59 pairs are all tortuous geometries and so reflect single-precision conditioning rather than the operators.

# Performance comparison

We compare against taufactor 1.2.1 and PuMA 3.2.2, both current releases, over edge lengths $N \in \{200, \ldots, 1000\}$, five porosities from 0.2 to 0.95, and three feature sizes, on one machine with a Quadro RTX 8000 (48 GB) and an eight-core Xeon. The three tools stop on different quantities, so equal settings do not give equal accuracy. We instead sweep each tool's iteration cap and take the wall time of the fastest run reaching a given relative error in $\tau$, charging setup costs to all three. Two caveats run against the comparison: the reference $\tau$ is our own `Float64` CPU solve, which is why the cross-code agreement above matters, and both competitors start from a supplied initial guess while we start from zero. The documentation gives the protocol, the patch each tool needed, and the command that reproduces the campaign.

![Wall time to reach 0.1% relative error in $\tau$. Panels (a) and (c) average over the same five porosities at every size, with the exponent of a fitted power law in the legend. Solid segments join measurements. A dashed segment and a hollow marker mean one porosity there was projected from its own power law rather than measured, which happens only for taufactor: three CPU points it was never run at, and the GPU point at $\varepsilon = 0.2$ and $1000^3$, where it timed out at 0.86% error. The assembled series stops where the card fills. Panels (b) and (d) resolve the measured data alone by size and porosity, blue where `Tortuosity.jl` is faster and red where the other tool is.\label{fig:benchmark}](benchmark_summary.png)

At the 0.1% target `Tortuosity.jl` is faster than taufactor on the GPU in most cases (\autoref{fig:benchmark}). The two tools also scale differently, which is why the margin widens with the image: a power law fitted over the size sweep gives taufactor $N^{3.3}$ on the GPU and $N^{4.6}$ on the CPU, against $N^{2.5}$ and $N^{3.1}$ for `Tortuosity.jl`. An exponent of 3 is linear in voxel count, so ours stays near a fixed number of iterations per solve while taufactor's rises with the edge length. Over twelve cases both tools solve at every size, the geometric mean advantage rises steeply and then holds near $8\times$: $1.1\times$, $4.6\times$, $8.2\times$, $7.5\times$, and $8.4\times$ from $200^3$ to $1000^3$. Porosity moves the result more than size does: pooled across sizes it is $58\times$ near $\varepsilon = 0.2$ and $1.8\times$ at $\varepsilon = 0.95$, and both means understate it, since they drop the four cases taufactor never solved within its budget — three at $\varepsilon = 0.2$ and one at $\varepsilon = 0.4$. The spread is wide — at $200^3$ the margin runs from $0.39\times$ to $13\times$ — so this is a large-image, low-porosity advantage, not a uniform one. Demanding more accuracy widens it without reversing it: $4.2\times$, $5.2\times$, and $6.6\times$ at 10%, 1%, and 0.1% targets, with taufactor faster only at $200^3$ and only on 10 of 74 paired cases even at the loosest.

![Peak device memory during the solve, for the two operator forms and taufactor. Hatched bars mark where the assembled form exhausts the card.\label{fig:memory}](benchmark_memory_gpu.png)

Memory separates the two operator forms more sharply than time does (\autoref{fig:memory}). The solve itself holds 32.0 bytes per pore node plus 4.00 bytes per grid voxel (the Krylov vectors and the `Int32` index map), a two-term model fitted at $800^3$ that reproduces all five $1000^3$ points to within 0.02%. Porosity therefore sets the ratio to the assembled form, which runs from about $1.75\times$ at $\varepsilon = 0.2$ to $2.55\times$ at $\varepsilon = 0.95$, stepping to $3.2\times$ wherever the assembled operator crosses the `Int32` bound and widens. The figure adds the refinement pass, at a further 20 bytes per pore node, which dilutes the ratio to about $1.8\times$. On that same basis taufactor's flat 28.06 bytes per grid voxel leaves us using less device memory than it at the two lowest porosities and more at the three highest. The matrix-free form completes every $1000^3$ case on the card, though at the highest porosity it reaches 46.2 of the 47.3 GiB available, at which point the refinement pass runs out of room and returns the unrefined solution, since refining there would need 49.8 GiB. The assembled form runs out above $\varepsilon = 0.4$ at $1000^3$.

On the GPU the solver is roughly $21\times$ faster than its own CPU path, rising from $10\times$ at $200^3$ to about $27\times$ at $800^3$ and holding there. PuMA computes tortuosity on the CPU, so we compare it there, where `Tortuosity.jl` wins all fifteen cases both tools solve to 0.1%, by a geometric mean of $31\times$ and a range of $2.1\times$ to $388\times$. That comparison covers $200^3$ only, since one case there costs PuMA about nineteen minutes. Neither CPU tool saturates the machine: PuMA occupies a median of 6.4 of the 8 physical cores while our median at that size is 1.0, so the margin is won on less of the machine, not more.

# Limitations

The coarse hierarchy is assembled and applied on the host, so every iteration pays a device round trip that will eventually bound scaling. Its coarse operator is accumulated with atomic additions at setup, so bit-for-bit equality across runs is not guaranteed on the GPU, even though repeated solves now agree on $\tau$. Disconnected pore clusters are left at zero rather than trimmed, which inflates $\tau$ by their stagnant volume, and the warning about them is disabled above 50 million voxels. Anisotropy needs one solve per axis, and only CUDA is exercised in continuous integration. The benchmark uses one machine, one image generator, and a reference computed by this package. PuMA appears at a single size, and the accuracy ladder cannot resolve margins below about $2\times$.

# Research impact statement

`Tortuosity.jl` is registered in the Julia General registry and installs through the built-in package manager. Four contributors developed it from 2023 to 2026, across nearly 400 commits on the main branch, seven tagged releases, 58 issues, and 29 pull requests. The Gostick research group at the University of Waterloo uses it to characterize electrode microstructures for battery and fuel cell research. The benchmark campaign reported here ships as reproducible material: pinned environments, per-image checksums, the raw result tables, and a documented command that regenerates every number and figure.

# AI usage disclosure

We used Claude (Anthropic) during development for code suggestions, code review, documentation drafts, parts of the benchmark harness, and drafting and editing of this manuscript. The human authors reviewed, tested, and validated all AI-generated content, and made the core algorithmic and scientific decisions themselves.

# Acknowledgements

We thank the developers of TauFactor and taufactor, whose work informed this package, and the Julia community for the ecosystem it builds on. This work received no external funding.

# References
