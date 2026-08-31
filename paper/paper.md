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

`Tortuosity.jl` solves diffusion on voxel images of porous media and reports the transport properties that follow. Its main output is the tortuosity factor $\tau$, which measures how much the pore geometry slows diffusion relative to an open medium. The package takes a binary 2D or 3D image, builds a linear system over the pore voxels alone, and solves either a steady-state or a transient problem on it. Diffusivity can be set per voxel, so heterogeneous materials such as composite electrodes are in scope. The same script runs on a CPU or a GPU, and loading a backend package is the only thing the user changes. On a single 48 GB card a $1000^3$ image, a billion voxels, solves in about half a minute.

# Statement of need

Wherever a pore space controls how something moves through a material, the same question is asked: how much does the geometry slow that transport down? It is asked of battery electrodes and fuel cell layers, of rock, soil and cement, of separation membranes and filters, of packaging films and tissue scaffolds, and the literature answering it is spread across those fields and others still [@ghanbarian2013tortuosity; @fu2021tortuosity]. Tomography and serial sectioning now put the pore space itself in hand, so the question can be put to the image directly. The tortuosity factor links the effective diffusivity $D_\text{eff}$ to the intrinsic diffusivity $D_0$ and the porosity $\varepsilon$:

$$D_\text{eff} = D_0 \, \frac{\varepsilon}{\tau}$$

Throughout, $\tau$ is the tortuosity factor of @cooper2016taufactor, not the geometric ratio of path lengths, and it is not squared [@tjaden2016tortuosity].

To compute $\tau$ from an image, a solver must discretize the Laplace equation on the pore space, solve the linear system, and compare the flux it carries with the flux the same domain would carry with no solid in it at all. Those images routinely hold hundreds of millions of pore voxels, and at that size a single solve already costs a considerable amount of time. A study that sweeps samples or conditions needs many of them.

`Tortuosity.jl` is for the research groups and industrial users who need that solve to be fast and to fit on one workstation GPU. Its discretization, its operator, and its preconditioner are all plain Julia, so swapping a solver or a boundary condition does not mean leaving the language the analysis is written in.

# State of the field

The established tools differ in language, in hardware, and in how much of porous media they set out to cover. They differ most in one decision that none of them advertises: how much of the image ends up in the linear system.

Most of them keep all of it. TauFactor [@cooper2016taufactor] brought this workflow to the porous media community and remains widely cited. The same group later rewrote it in Python as taufactor [@kench2023taufactor], trading MATLAB for PyTorch and gaining the GPU with it. What the rewrite kept is the method. Both versions sweep the full grid by successive over-relaxation (SOR), so memory and compute follow the size of the image rather than the volume of the pore space. Breadth is the priority elsewhere. PuMA [@ferguson2018puma; @ferguson2021puma], from NASA, treats tortuosity as one property among conductivity, permeability, and mechanics, and the finite-volume path we benchmark solves on the CPU with SciPy's conjugate gradient, again over the full grid, though a workspace can be handed to the CUDA code `chfem` [@lopes2023chfem].

PoreSpy [@gostick2019porespy] takes the other route, and it is the closest thing here to what we do. Its `tortuosity_fd` numbers the pore voxels alone, builds a network over them, and solves Fickian diffusion on that by algebraic multigrid. Restricting the system and preconditioning it multilevel are the two ideas this package also rests on. What it pays for them is the network: the pore space becomes an object graph and an assembled matrix before a single iteration runs, and there is no GPU path.

`Tortuosity.jl` keeps the pore-only system and drops the structure around it. Solid voxels stay out, so iteration cost and most of the memory scale with the pore count rather than with the image size, and the only grid-sized term left is a four-byte-per-voxel index map. That saving is largest at low porosity, where a full-grid method spends the most work on voxels that do nothing, and it disappears at high porosity, where little solid remains to exclude. The solver calls Krylov methods from `LinearSolve.jl` [@kimmerer2024linearsolve], which reach a given residual in far fewer iterations than SOR, and the whole path runs on the GPU.

# Software design

Setup is a single pass over the image. The pore voxels are numbered in order, the solid ones are skipped, and each pore voxel is coupled to whichever of its six face neighbours is also pore. Those couplings form the Laplace operator discretized on the pore space alone. Concentration is held at 1 and 0 on the two faces the transport runs between, the other four faces are insulated, and where diffusivity varies from voxel to voxel the two values meeting at a face combine by harmonic mean, which is the combination that conserves flux across the interface. Conjugate gradient solves the resulting system by default, and $\tau$ follows from the mean flux through a cross-section.

The compute kernels are written once against `KernelAbstractions.jl` [@churavy2024ka] and compiled for whichever backend the user has loaded. CUDA.jl [@besard2019cuda] is the one we develop and benchmark against, and the only one that hands the sparse matrix-vector product to a vendor library, cuSPARSE. Metal.jl and AMDGPU.jl run the portable kernels throughout. Moving a script to the GPU takes one `using` statement and no manual transfers. GPU support ships as package extensions, which load themselves only when the backend package is present, and so do image generation, curve fitting, and HDF5 export.

What a Krylov method asks of the operator, once per iteration, is the product of the matrix with a vector. The package can compute that product in either of two interchangeable ways, over the same pore numbering and the same right-hand side. The default builds the sparse matrix once and keeps it, at about 59 bytes per pore voxel, widening its indices from `Int32` to `Int64` beyond $3\times10^8$ of them so that no hard size limit remains. The matrix-free form keeps no matrix at all. It stores a single `Int32` index array over the grid, four bytes per voxel, and recomputes the coefficients of each row inside the kernel every time the product is taken. What it buys is memory rather than speed. The product alone is about twice as fast, but both forms share the same preconditioner, so that lead narrows over a whole solve. The memory saving does not narrow, and it is what decides which problems fit.

The number of iterations a Krylov method needs on a three-dimensional Laplacian normally grows with the edge length of the image, and that growth, rather than the cost of any one iteration, is what makes large images expensive. The package therefore solves a smaller copy of the problem alongside the real one and uses it as a preconditioner. Pore voxels are grouped into cubic blocks eight voxels on a side, one unknown is kept per block, and the coefficients between blocks are formed from the fine ones by Galerkin projection, so the coarse problem summarizes the fine one instead of modelling it separately. What it contributes is the long-range part of the correction, which is the part conjugate gradient is slowest to find on its own. Holding the block edge fixed rather than scaling it with the image keeps the coarse problem a fixed fraction of the fine one, and that is what nearly arrests the growth. The coarse system is factorized directly while it is small enough for that. Past roughly $3\times10^4$ coarse unknowns it is coarsened again in a V-cycle that ends in the same direct factorization.

Measured on the CPU in `Float64` and pooled over five porosities, the unpreconditioned count tracks the edge length at the textbook rate, while the preconditioned count grows by only $1.16\times$ from $200^3$ to $600^3$. That is $12.5\times$ fewer iterations at $200^3$ and $29.6\times$ fewer at $600^3$. At $\varepsilon = 0.2$ it runs 136, then 177, then 205 at $1000^3$, against 2721 rising to 7099 without it. `solve(sim)` turns it on above $10^5$ pore voxels. Each coarse block gathers its contributions from a list of its own voxels, prepared once at setup, rather than each voxel adding into its block whenever it happens to be ready. That fixes the order the contributions are summed in, and since floating-point addition is not associative, a fixed order is what lets repeated runs return the same answer.

GPU solves run in `Float32`. On strongly tortuous images the iteration then stalls near $10^{-3}$ relative error in $\tau$, well short of the accuracy the format itself can hold, because conjugate gradient depends on its successive search directions staying independent and accumulated rounding gradually spoils that. Every single-precision solve is therefore corrected before it is returned, against a residual recomputed in double precision, at a further 20 bytes per pore node.

For transient problems the package builds on `OrdinaryDiffEq.jl` [@rackauckas2017diffeq], offers stop conditions for steady state, flux balance, saturation, and periodic state, and fits concentration curves to analytical slab-diffusion solutions.

# Accuracy

Speed matters only if the answer is right. Ducts, layered media, and dead-end pockets have closed-form answers, and the solver reproduces them to a relative $10^{-8}$ or better. Porous geometries do not, so we take taufactor, PuMA and PoreSpy as independent references there. They agree with our $\tau$ within the 0.1% benchmark target on 126 of 131, 15 of 15 and 15 of 15 shared cases, and to a few parts in $10^6$ wherever their ladders land well inside that target rather than on it. The two operator forms agree to every recorded digit across the 74 `Float64` CPU cases, and to about $5\times10^{-5}$ on the GPU in `Float32`, where the few larger gaps are all strongly tortuous images and so reflect single precision rather than the operators.

# Performance comparison

We compare against taufactor 1.2.1, PuMA 3.2.2 and PoreSpy 3.0.4, all current releases, over edge lengths $N \in \{200, \ldots, 1000\}$, five porosities from 0.2 to 0.95, and three feature sizes, on one machine with a Quadro RTX 8000 (48 GB) and an eight-core Xeon. The four tools stop on different quantities, so equal settings do not give equal accuracy. We instead sweep each tool over whichever of its own settings traces that trade-off — an iteration cap for the three Krylov and SOR solvers, the solver tolerance for PoreSpy's multigrid, which has no cap to sweep — and take the wall time of the fastest run reaching a given relative error in $\tau$, charging setup costs to all four. Two caveats run against the comparison: the reference $\tau$ is our own `Float64` CPU solve, which is why the cross-code agreement above matters, and the two full-grid competitors start from a supplied initial guess while `Tortuosity.jl` and PoreSpy both start from zero. PoreSpy comes from our own group, so we ran it through its documented entry point at its own default solver and report its accuracy beside its time. The documentation gives the protocol, the patch each tool needed, and the command that reproduces the campaign.

![Wall time to reach 0.1% relative error in $\tau$. Panels (a) and (c) average over the same five porosities at every size, with the size exponent in the legend. Solid segments join measurements; a dashed segment and a hollow marker mean at least one porosity there was projected rather than measured. For taufactor the projection is its own fitted power law, over three CPU points it was never run at and the GPU point at $\varepsilon = 0.2$ and $1000^3$, where it timed out at 0.86% error. PuMA and PoreSpy were run at $200^3$ alone, so their whole CPU curves above that size are projected from an exponent measured separately on a matched $200^3$/$400^3$ pair, marked "est." in the legend. The assembled series stops where the card fills. Panel (b) resolves the measured data alone by size and porosity, blue where `Tortuosity.jl` is faster and red where taufactor is. Panel (d) is the two CPU-only tools at $200^3$, with the factor over ours above each bar.\label{fig:benchmark}](benchmark_summary.png)

At the 0.1% target `Tortuosity.jl` is faster than taufactor on the GPU in most cases (\autoref{fig:benchmark}). The two tools also scale differently, which is why the margin widens with the image: a power law fitted over the size sweep gives taufactor $N^{3.3}$ on the GPU and $N^{4.6}$ on the CPU, against $N^{2.5}$ and $N^{3.1}$ for `Tortuosity.jl`. An exponent of 3 is linear in voxel count, so ours stays near a fixed number of iterations per solve while taufactor's rises with the edge length. Over twelve cases both tools solve at every size, the geometric mean advantage rises steeply and then holds near $8\times$: $1.1\times$, $4.6\times$, $8.2\times$, $7.5\times$, and $8.4\times$ from $200^3$ to $1000^3$. Porosity moves the result more than size does: pooled across sizes it is $58\times$ near $\varepsilon = 0.2$ and $1.8\times$ at $\varepsilon = 0.95$, and both means understate it, since they drop the four cases taufactor never solved within its budget — three at $\varepsilon = 0.2$ and one at $\varepsilon = 0.4$. The spread is wide — at $200^3$ the margin runs from $0.39\times$ to $13\times$ — so this is a large-image, low-porosity advantage, not a uniform one. Demanding more accuracy widens it without reversing it: $4.2\times$, $5.2\times$, and $6.6\times$ at 10%, 1%, and 0.1% targets, with taufactor faster only at $200^3$ and only on 10 of 74 paired cases even at the loosest.

![Peak device memory during the solve, for the two operator forms and taufactor. Hatched bars mark where the assembled form exhausts the card.\label{fig:memory}](benchmark_memory_gpu.png)

Memory separates the two operator forms more sharply than time does (\autoref{fig:memory}). The solve itself holds 32.0 bytes per pore node plus 4.00 bytes per grid voxel (the Krylov vectors and the `Int32` index map), a two-term model fitted at $800^3$ that reproduces all five $1000^3$ points to within 0.02%. Porosity therefore sets the ratio to the assembled form, which runs from about $1.75\times$ at $\varepsilon = 0.2$ to $2.55\times$ at $\varepsilon = 0.95$, stepping to $3.2\times$ wherever the assembled operator crosses the `Int32` bound and widens. The figure adds the refinement pass, at a further 20 bytes per pore node, which dilutes the ratio to about $1.8\times$. On that same basis taufactor's flat 28.06 bytes per grid voxel leaves us using less device memory than it at the two lowest porosities and more at the three highest. The matrix-free form completes every $1000^3$ case on the card, though at the highest porosity it reaches 46.2 of the 47.3 GiB available, at which point the refinement pass runs out of room and returns the unrefined solution, since refining there would need 49.8 GiB. The assembled form runs out above $\varepsilon = 0.4$ at $1000^3$.

On the GPU the solver is roughly $21\times$ faster than its own CPU path, rising from $10\times$ at $200^3$ to about $27\times$ at $800^3$ and holding there. PuMA and PoreSpy compute tortuosity on the CPU alone, so we compare them there, and `Tortuosity.jl` wins all fifteen cases at $200^3$ against each: by a geometric mean of $31\times$ over PuMA, ranging $2.1\times$ to $388\times$, and of $10\times$ over PoreSpy, ranging $4.2\times$ to $18\times$. The two lose in opposite regimes, and that is the useful part of running both. PuMA's gap is widest on the densest image and nearly closes on the most open one, $236\times$ down to $3.5\times$, because its conjugate gradient has the hardest system where the pore space is most tortuous. PoreSpy's runs the other way, $6.0\times$ up to $13\times$, because its cost follows how many pore voxels there are rather than how hard they are to connect: it builds a network and a multigrid hierarchy over the pore space before any iteration runs. Restricting the system to the pore space is therefore not by itself what produces the margin. Both comparisons stop at $200^3$: one case there already costs PuMA about nineteen minutes, and sweeping either tool over the size grid would have cost days to settle a gap the smallest size decides. PuMA occupies a median of 6.4 of the 8 physical cores against our 1.0 at that size, so at least that margin is won on less of the machine, not more.

# Limitations

The coarse hierarchy is assembled and applied on the host, so every iteration pays a device round trip that will eventually bound scaling. Its coarse operator is accumulated with atomic additions at setup, so bit-for-bit equality across runs is not guaranteed on the GPU, even though repeated solves now agree on $\tau$. Disconnected pore clusters are left at zero rather than trimmed, which inflates $\tau$ by their stagnant volume, and the warning about them is disabled above 50 million voxels. Anisotropy needs one solve per axis, and only CUDA is exercised in continuous integration. The benchmark uses one machine, one image generator, and a reference computed by this package. PuMA and PoreSpy are timed at a single size, and the accuracy ladder cannot resolve margins below about $2\times$.

# Research impact statement

`Tortuosity.jl` is registered in the Julia General registry and installs through the built-in package manager. Four contributors developed it from 2023 to 2026, across nearly 400 commits on the main branch, seven tagged releases, 58 issues, and 29 pull requests. The Gostick research group at the University of Waterloo uses it to characterize electrode microstructures for battery and fuel cell research. The benchmark campaign reported here ships as reproducible material: pinned environments, per-image checksums, the raw result tables, and a documented command that regenerates every number and figure.

# AI usage disclosure

We used Claude (Anthropic) during development for code suggestions, code review, documentation drafts, parts of the benchmark harness, and drafting and editing of this manuscript. The human authors reviewed, tested, and validated all AI-generated content, and made the core algorithmic and scientific decisions themselves.

# Acknowledgements

We thank the developers of TauFactor and taufactor, whose work informed this package, and the Julia community for the ecosystem it builds on. This work received no external funding.

# References
