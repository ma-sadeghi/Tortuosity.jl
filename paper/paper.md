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

`Tortuosity.jl` is a Julia package for simulating diffusion directly on segmented two- and three-dimensional images of porous materials. It computes effective diffusivity and the associated tortuosity factor from steady-state solutions and supports transient diffusion and spatially varying diffusivity. The package restricts the numerical problem to the transporting phase and provides both assembled and matrix-free operators. The same interface runs on CPUs and NVIDIA, AMD, or Apple GPUs through Julia's GPU ecosystem. This combination enables image-based transport calculations ranging from small exploratory studies to volumes containing one billion voxels on a single workstation GPU.

# Statement of need

Transport through a porous material depends on the connectivity and geometry of its pore space. Quantifying this effect is important in fields including electrochemical energy storage, geoscience, filtration, and membrane science [@ghanbarian2013tortuosity; @fu2021tortuosity]. Three-dimensional imaging allows researchers to estimate transport properties from a resolved microstructure rather than from an empirical porosity correlation. For diffusion through a single transporting phase, the tortuosity factor $\tau$ is commonly defined by

$$D_\text{eff} = D_0 \, \frac{\varepsilon}{\tau}$$

where $D_\text{eff}$ is the effective diffusivity, $D_0$ is the intrinsic diffusivity of the transporting phase, and $\varepsilon$ is its volume fraction. This transport-based quantity is distinct from geometric definitions based on path length [@tjaden2016tortuosity]. Computing it requires solving a diffusion equation over a voxelized domain and evaluating the resulting flux. Modern images can contain hundreds of millions of voxels, making memory use and solver convergence central practical constraints, especially when a study compares many samples or operating conditions. Existing packages support parts of this workflow, but none combines a pore-only formulation, matrix-free GPU execution, and integration with Julia's numerical ecosystem. `Tortuosity.jl` targets researchers and engineers in academia and industry who need these calculations to run efficiently on available CPU or GPU hardware while retaining control over the numerical workflow.

# State of the field

Several established packages calculate tortuosity from voxel images. TauFactor introduced an accessible MATLAB implementation based on successive over-relaxation [@cooper2016taufactor], and its Python successor, taufactor, uses PyTorch to support GPU execution [@kench2023taufactor]. Both update the full image grid, including solid voxels. PuMA provides a broader suite of porous-material properties and solves its finite-volume diffusion problem on the full grid [@ferguson2018puma; @ferguson2021puma]. PoreSpy instead constructs and solves a pore-only network using algebraic multigrid, but its finite-difference tortuosity solver is CPU-only [@gostick2019porespy].

`Tortuosity.jl` combines a pore-only formulation, matrix-free operation, preconditioned Krylov solvers, and portable GPU execution. Restricting the unknowns to the transporting phase is particularly beneficial at low porosity, while the matrix-free path reduces the memory required for large, open structures. Adding this combination to TauFactor or taufactor would require replacing their central full-grid relaxation method; adding it to PoreSpy would require a GPU-compatible operator and solver stack outside its network-based model. Implementing the approach as a Julia package also permits direct integration with Julia's linear-solver and differential-equation ecosystems [@kimmerer2024linearsolve; @rackauckas2017diffeq].

# Software design

The architecture separates porous-media problem construction from numerical solution and post-processing (\autoref{fig:architecture}). Users provide a segmented image and simulation choices; the problem constructors validate and index the geometry, apply boundary conditions, and select the compute backend. The steady path creates a linear problem, while the transient path creates a diffusion right-hand side for time integration. Both paths feed a common analysis layer for transport properties and concentration-field measurements. This separation keeps image and transport logic within `Tortuosity.jl` while allowing established Julia packages to supply solvers and hardware backends.

![High-level architecture of `Tortuosity.jl`. Solid arrows show the flow from user inputs to simulation outputs; dotted arrows show interfaces to principal numerical dependencies and optional GPU backends. Colors distinguish inputs and outputs, shared stages, steady and transient paths, the operator choice, and external dependencies.\label{fig:architecture}](software_architecture.png)

Two interchangeable operator representations expose the main design trade-off. The assembled representation stores a sparse matrix and can use vendor sparse-linear-algebra libraries. The matrix-free representation stores only a four-byte index map over the image and reconstructs each row during a matrix-vector product, exchanging repeated arithmetic for lower storage and data movement. A geometric coarse-space preconditioner supplies long-range corrections that conjugate gradient would otherwise recover slowly; over the measured $200^3$ to $600^3$ range, it reduced iteration counts by factors of 12.5 to 29.6.

The numerical engines are integrated through their native interfaces rather than reimplemented. Steady problems use `LinearSolve.jl`, and transient problems use `OrdinaryDiffEq.jl` with composable callbacks [@kimmerer2024linearsolve; @rackauckas2017diffeq]. Data-parallel kernels are implemented with `KernelAbstractions.jl` [@churavy2024ka], while package extensions register CUDA, AMDGPU, or Metal without making them core dependencies. The CUDA path additionally uses cuSPARSE for assembled matrix-vector products [@besard2019cuda]. Other extensions keep image filtering, curve fitting, and HDF5 export optional. GPU calculations use single precision for performance and portability, with iterative refinement against a double-precision residual when additional accuracy is needed.

# Validation and performance

Validation combines analytical tests with comparisons against independent software. The test suite covers geometries with closed-form solutions, including straight ducts, layered media, and dead-end pores; steady-state results agree with these solutions to a relative error of $10^{-8}$ or better on the CPU. Cross-code comparisons against taufactor, PuMA, and PoreSpy show agreement at approximately the benchmark's 0.1% accuracy scale, with tighter agreement when the comparator's available settings resolve that target. The assembled and matrix-free operators agree within the recorded precision in double-precision calculations and typically differ by about $5\times10^{-5}$ in single-precision GPU calculations.

Performance was evaluated on synthetic images ranging from $200^3$ to $1000^3$ voxels and covering five porosities and three characteristic feature sizes. Timings included the complete workflow from image input through problem construction and solution, and tools were compared at a common target of at most 0.1% relative error in $\tau$. On the GPU, taufactor was faster only for the smallest images ($N=200$) at the two highest porosities after averaging over the three feature sizes; end-to-end times for both tools were about one second or less. For the reference feature size shown in \autoref{fig:benchmark}, taufactor was faster in only one cell: at $N=200$ and $\varepsilon=0.95$, the reference tortuosity was 1.04 and the end-to-end times were 0.45 s for taufactor and 0.96 s for `Tortuosity.jl`. At the largest size, $N=1000$, `Tortuosity.jl` was faster at every porosity where both tools reached the accuracy target; at $\varepsilon=0.4$, for example, it required 22 s, compared with approximately 12 minutes for taufactor. Across the twelve porosity–feature-size combinations for which both tools reached the target at all five image sizes, `Tortuosity.jl` was faster by a geometric mean of $8.4\times$ at $1000^3$. On the CPU at $200^3$, `Tortuosity.jl` was faster in every shared case, with geometric-mean speedups of $31\times$ over PuMA and $10\times$ over PoreSpy. Its matrix-free representation also completed every percolating $1000^3$ test image on a GPU with 48 GiB of memory.

![Time to reach at most 0.1% relative error in $\tau$ for the reference feature size. The scaling panels retain the same five porosities at every size; missing target times are projected from the same case family's measured size trend so that nonconverged cases do not bias the geometric mean downward. Solid bars show measured size sweeps, while hatched bars denote projections. For the assembled GPU method, hatched bars at the two largest sizes are hypothetical timing projections for cases that exceeded device memory; they preserve the geometric-mean assembled-to-matrix-free timing ratio at the shared measured sizes. Legend exponents describe empirical scaling with the recorded number of transporting voxels, $n_{\mathrm{tr}}$, with “est.” marking exponents derived from size probes rather than fitted sweeps. Panels (b) and (d) show measured speed ratios against taufactor on the GPU and CPU, respectively. Each cell is taufactor's time divided by the `Tortuosity.jl` time, so values above one favor `Tortuosity.jl` and values below one favor taufactor. An em dash indicates that no measured matched-accuracy ratio was available because taufactor was not run or did not reach the target before the timeout.\label{fig:benchmark}](benchmark_summary.png)

# Limitations

The main input-related limitation is that disconnected pore clusters are retained as stagnant volume; users who want transport only through the percolating phase must trim these clusters before solving. The performance evidence is also deliberately bounded: the benchmarks use one machine, one family of synthetic microstructures, and reference values computed by the package's double-precision CPU solver. Among the GPU backends, only CUDA was benchmarked. The timing sweeps for PuMA and PoreSpy were limited to $200^3$; their larger dashed values in \autoref{fig:benchmark} are projections calibrated by separate size probes. The results should therefore be interpreted as representative comparisons rather than universal performance rankings.

# Research impact statement

The benchmark results show that `Tortuosity.jl` can analyze billion-voxel images on a single workstation GPU while reducing run time relative to the benchmarked alternatives in regimes where large-image calculations become costly. The package is registered in Julia's General registry, has been developed publicly by multiple contributors since 2023, and provides tutorials, API documentation, automated tests, and contribution guidelines. The complete benchmark workflow is distributed with the package, including pinned software environments, image checksums, raw result tables, and scripts that regenerate the reported analyses and figures. Together, these resources provide a practical tool for large image-based transport studies and inspectable evidence supporting its correctness and performance.

# AI usage disclosure

Generative AI tools (Claude Opus 5 and GPT-5.6 Sol) were used for code suggestions and review, test and benchmark scaffolding, documentation, and manuscript drafting and editing. The human authors reviewed, edited, tested, and validated all AI-assisted outputs and made the core algorithmic, scientific, and editorial decisions.

# Acknowledgements

We thank the developers of TauFactor, whose work informed this package, and the Julia community for maintaining the ecosystem on which `Tortuosity.jl` depends. This work received no external funding.

# References
