---
title: 'Tortuosity.jl: GPU-accelerated tortuosity calculations from porous-media images'
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
date: 31 August 2026
bibliography: paper.bib
---

# Summary

`Tortuosity.jl` is a Julia package for simulating diffusion directly on segmented two- and three-dimensional images of porous materials. It computes effective diffusivity and the associated tortuosity factor from steady-state solutions, and also supports transient diffusion and spatially varying diffusivity. The package restricts the numerical problem to the transporting phase and provides both assembled and matrix-free operators. The same interface runs on CPUs and on NVIDIA, AMD, or Apple GPUs through Julia's GPU ecosystem. This combination enables image-based transport calculations ranging from small exploratory studies to volumes containing one billion voxels on a single workstation GPU.

# Statement of need

Transport through a porous material depends on the connectivity and geometry of its pore space. Quantifying this effect is important in fields including electrochemical energy storage, geoscience, filtration, and membrane science [@ghanbarian2013tortuosity; @fu2021tortuosity]. Three-dimensional imaging allows researchers to estimate transport properties from a resolved microstructure rather than from an empirical porosity correlation. For diffusion through a single transporting phase, the tortuosity factor $\tau$ is commonly defined by

$$D_\text{eff} = D_0 \, \frac{\varepsilon}{\tau}$$

where $D_\text{eff}$ is the effective diffusivity, $D_0$ is the intrinsic diffusivity of the transporting phase, and $\varepsilon$ is its volume fraction. This transport-based quantity is distinct from geometric definitions based on path length [@tjaden2016tortuosity]. Computing it requires solving a diffusion equation over a voxelized domain and evaluating the resulting flux. Modern images can contain hundreds of millions of voxels, making memory use and solver convergence central practical constraints, especially when a study compares many samples or operating conditions. Existing packages support parts of this workflow, but none combines a pore-only formulation, matrix-free GPU execution, and integration with Julia's numerical ecosystem. `Tortuosity.jl` targets academic and industrial researchers and engineers who need these calculations to run efficiently on available CPU or GPU hardware while retaining control over the numerical workflow.

# State of the field

Several established packages calculate tortuosity from voxel images. TauFactor introduced an accessible MATLAB implementation based on successive over-relaxation [@cooper2016taufactor], and its Python successor, taufactor, uses PyTorch to support GPU execution [@kench2023taufactor]. Both update the full image grid, including solid voxels. PuMA provides a broader suite of porous-material properties and solves its finite-volume diffusion problem on the full grid [@ferguson2018puma; @ferguson2021puma]. PoreSpy instead constructs and solves a pore-only network using algebraic multigrid, but its finite-difference tortuosity solver is CPU-only [@gostick2019porespy].

`Tortuosity.jl` combines a pore-only formulation, matrix-free operation, preconditioned Krylov solvers, and portable GPU execution. Restricting the unknowns to the transporting phase is particularly beneficial at low porosity, while the matrix-free path reduces the memory required for large, open structures. Adding this combination to TauFactor or taufactor would require replacing their central full-grid relaxation method; adding it to PoreSpy would require a GPU-compatible operator and solver stack outside its network-based model. Implementing the approach as a Julia package also permits direct integration with Julia's linear-solver and differential-equation ecosystems [@kimmerer2024linearsolve; @rackauckas2017diffeq].

# Software design

The architecture separates porous-media problem construction from numerical solution and post-processing (\autoref{fig:architecture}). Users provide a segmented image and simulation choices; the problem constructors validate and index the geometry, apply boundary conditions, and select CPU or GPU storage. The steady path creates a linear problem, while the transient path creates a diffusion right-hand side for time integration. Both paths return to a common analysis layer for transport properties and concentration-field measurements. This separation keeps image and transport logic within `Tortuosity.jl` while allowing established Julia packages to supply solvers and hardware backends.

![High-level architecture of `Tortuosity.jl`. Solid arrows show the flow from user inputs to simulation outputs; dotted arrows show interfaces to principal numerical dependencies and optional GPU backends.\label{fig:architecture}](software_architecture.png)

Two interchangeable operator representations expose the main design trade-off. The assembled representation stores a sparse matrix and can use vendor sparse-linear-algebra libraries. The matrix-free representation stores only a four-byte index map over the image and reconstructs each row during a matrix-vector product, exchanging repeated arithmetic for lower storage and data movement. A geometric coarse-space preconditioner supplies long-range corrections that conjugate gradient otherwise recovers slowly; over the measured $200^3$ to $600^3$ range, it reduced iteration counts by factors of 12.5 to 29.6.

The numerical engines are integrated through their native interfaces rather than reimplemented. Steady problems use `LinearSolve.jl`, and transient problems use `OrdinaryDiffEq.jl` with composable callbacks [@kimmerer2024linearsolve; @rackauckas2017diffeq]. Data-parallel kernels are implemented with `KernelAbstractions.jl` [@churavy2024ka], while package extensions register CUDA, AMDGPU, or Metal without making them core dependencies. CUDA additionally uses cuSPARSE for assembled matrix-vector products [@besard2019cuda]. Other extensions keep image filtering, curve fitting, and HDF5 export optional. GPU calculations use single precision for performance and portability, with iterative refinement against a double-precision residual when additional accuracy is needed.

# Validation and performance

Validation combines analytical tests with comparisons against independent software. The test suite covers geometries with closed-form solutions, including straight ducts, layered media, and dead-end pores; steady-state results agree with these solutions to relative error of $10^{-8}$ or better on the CPU. Cross-code comparisons against taufactor, PuMA, and PoreSpy show agreement at approximately the benchmark's 0.1% accuracy scale, with tighter agreement when the comparator's available settings resolve that target. The assembled and matrix-free operators agree to recorded precision in double precision, and typically differ by approximately $5\times10^{-5}$ in single-precision GPU calculations.

Performance was evaluated on synthetic images spanning $200^3$ to $1000^3$ voxels, five porosities, and three characteristic feature sizes. Each tool was timed from image input through solution setup and execution, and results were compared at matched relative error in $\tau$. After averaging over the three feature sizes, taufactor's GPU advantage was confined to $N=200$ at the two highest porosities, where both tools completed in approximately one second or less. For the reference microstructure in \autoref{fig:benchmark}, the only such cell was $N=200$ and $\varepsilon=0.95$: $\tau$ was 1.04, and `Tortuosity.jl` took 0.96 s compared with 0.45 s. At $N=1000$, every comparable porosity favored `Tortuosity.jl`; at $\varepsilon=0.4$, for example, the times were 22 s and approximately 12 minutes. Across twelve case families solved by both tools at every size, the geometric-mean advantage at $1000^3$ was $8.4\times$. On the CPU at $200^3$, `Tortuosity.jl` was faster in all shared cases, with geometric-mean advantages of $31\times$ over PuMA and $10\times$ over PoreSpy. The matrix-free representation completed every percolating $1000^3$ test image within 48 GiB of device memory.

![Time to reach 0.1% relative error in $\tau$. Solid lines show measured size sweeps; dashed lines and hollow markers denote projections. Panel (b) shows measured GPU speed ratios against taufactor, and panel (d) compares the CPU-only tools at $200^3$.\label{fig:benchmark}](benchmark_summary.png)

# Limitations

The main input-related limitation is that disconnected pore clusters are retained as stagnant volume; users who want transport only through the percolating phase must trim these clusters before solving. The performance evidence is also deliberately bounded: the benchmarks use one machine, one family of synthetic microstructures, and reference values computed by the package's double-precision CPU solver. Only the CUDA backend was benchmarked, and PuMA and PoreSpy were swept only at $200^3$; their larger dashed values in \autoref{fig:benchmark} are projections calibrated by a separate size probe. The results should therefore be interpreted as representative comparisons rather than universal performance rankings.

# Research impact statement

`Tortuosity.jl` demonstrates credible near-term research significance through its benchmarked ability to analyze billion-voxel images on a single workstation GPU while substantially reducing run time in the difficult regimes where large-image calculations become costly. It is registered in Julia's General registry, has been developed publicly by multiple contributors since 2023, and provides tutorials, API documentation, automated tests, and contribution guidelines. The complete benchmark workflow is distributed with the package, including pinned software environments, image checksums, raw result tables, and scripts that regenerate the reported analyses and figures. Together, these provide both a practical tool for large image-based transport studies and inspectable evidence of its correctness and performance.

# AI usage disclosure

Generative AI tools were used for code suggestions and review, test and benchmark scaffolding, documentation, and manuscript drafting and editing. The tools were Claude (versions not recorded) and Codex (GPT-5). The human authors reviewed, edited, tested, and validated all AI-assisted outputs and made the core algorithmic, scientific, and editorial decisions.

# Acknowledgements

We thank the developers of TauFactor and taufactor, whose work informed this package, and the Julia community for the ecosystem it builds on. This work received no external funding.

# References
