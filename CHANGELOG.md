# Changelog

All notable changes to Tortuosity.jl are recorded here. Versions follow [Julia's SemVer rules](https://pkgdocs.julialang.org/v1/compatibility/), under which a change to the leftmost non-zero version component is breaking.

## Unreleased

A maintainability pass over the package, the benchmark harness and the documentation. No solver, preconditioner or numerical result changed, and every published benchmark figure regenerates byte-identical from the unchanged result tables.

### Changed

- GPU auto-detection uses one crossover of 100,000 pore voxels on every backend. `SteadyDiffusionProblem` previously took 20,000 on CUDA while `TransientDiffusionProblem` took 100,000 everywhere, which meant the same image could be placed on different devices by the two constructors. On CUDA, a steady problem built from an image with between 20,000 and 100,000 pore voxels now runs on CPU by default; pass `gpu=true` to keep it on the device. The threshold is `Tortuosity.GPU_MIN_NODES`.
- `Tortuosity.find_caverns` defaults to `gpu=nothing`, matching every other entry point. It previously defaulted to `gpu=true`, which errors outright in a session with no GPU backend registered — so the call could not be made with its own defaults on a CPU-only machine.

### Added

- `Tortuosity.find_caverns` and `Tortuosity.flux_out` appear in the API reference. Neither is exported and neither changed behaviour; they were simply undiscoverable.
- A Theory page in the documentation, covering what the tortuosity factor is and how it is computed. The text already existed but sat outside the documentation source tree and was never published.

### Removed

- The unused `src/plottools.jl`, and the internal helpers `args_to_dict`, `format_args_dict`, `find_true_indices` and `build_reverse_lookup`. None was exported, documented or called from anywhere in the package.

## v0.2.0 — 2026-09-04

The steady-state pipeline rebuilt around a host solver written for it. Time to reach 0.1% relative error in `τ` on the CPU falls by a geometric mean of **3.25×** on the matrix-free path and **4.29×** on the assembled path, measured over 74 cases spanning `200³` to `1000³` and five porosities. The GPU path gained kernel, preconditioner and refinement tuning in the same span; its published benchmark figures are being re-measured and are not restated here.

### Breaking

`SteadyDiffusionProblem` carries four type parameters instead of one — `SteadyDiffusionProblem{A,P,R,F}` rather than `SteadyDiffusionProblem{A}` — and two new fields, `D0` and `flux`. The `prob` field is now concretely typed instead of held as an abstract `LinearProblem`, which is what lets the observables reduce on the solution backend without a dynamic dispatch per element.

Code that constructs the struct positionally, or that dispatches on `SteadyDiffusionProblem{SomeArrayType}`, must be updated. Construction through `SteadyDiffusionProblem(img; axis, ...)` and every exported entry point are unchanged, and no export was added or removed in this release.

### Added

- `Tortuosity.HostCG`, a threaded conjugate gradient registered as a `LinearSolve.jl` algorithm. Because the host steady iteration is limited by memory bandwidth rather than arithmetic, it fuses the vector kernels of the standard preconditioned iteration: the solution and residual updates are combined, and the preconditioner prolongation is folded into the residual inner product. It is selected automatically for `Float64` host systems of at least 10,000 unknowns; smaller host problems and all GPU problems continue to use `KrylovJL_CG`. It is not exported — reach it as `Tortuosity.HostCG()`.
- Transport-property readouts that take the solution and the problem: `tortuosity(sol.u, sim)`, `effective_diffusivity(sol.u, sim)` and `formation_factor(sol.u, sim)`. The problem now retains a compact inlet-edge map and reference diffusivity, so these reduce physical flux on the solution backend instead of copying and allocating an image-sized concentration field. The `(c, img; axis)` forms are unchanged.

### Performance

On the host: the conjugate-gradient kernels are fused; the uniform and variable steady stencils are specialized separately; pore numbering, sparse column-pointer construction and the assembled products are threaded; dense matrix-free products are fused; and the preconditioner's restriction and prolongation are parallelized at size. Automatic preconditioning now also pays off on solves down to about eight thousand nodes.

On CUDA: steady kernel workgroups are tuned, the auto-selection crossover and the direct-solve ceiling for three-dimensional problems are lowered, the coarse-space ceiling scales with the device, redundant hot-path synchronization is removed, and refinement corrections are neither oversolved nor left unadapted at large sizes.

On both devices: transport scalars are computed without reconstructing the concentration field, percolating voxels are counted without trimming the image, and diffusivity validation no longer allocates a full-grid temporary at construction.

### Fixed

- Automatic preconditioning is no longer applied across short transport axes. At 17 voxels it made the measured CPU solve about four times slower, so the conservative threshold is held through four default coarse blocks.
- Iterative refinement enforces its tolerance contract: loose corrections continue while the true residual improves, success is derived from the returned vector under Krylov's combined tolerances, and `retcode` is decided on the `Float64` iterate that `_refine` controls — so rounding back to `Float32` can no longer turn a converged answer into a failure.
- A steady checkpoint over an empty boundary face yields `NaN`, matching the field readout, instead of throwing inside the callback.
- AMDGPU synchronization barriers are retained.

## v0.1.0 — 2026-08-22

The first release since v0.0.7 in April 2026, and the version the JOSS paper describes. It adds a second way to represent the steady operator, a preconditioner whose iteration count no longer tracks image size, and a rebuilt assembly path. Together those take an 800³ image from `OutOfGPUMemoryError` to a 20.7 s end-to-end solve on a 24 GiB card.

### Added

- A matrix-free form of the steady operator, selected with `SteadyDiffusionProblem(img; axis, matrixfree=true)`. It stores one full-grid `Int32` index array instead of the assembled sparse matrix's row indices and values, and recomputes the seven stencil weights of a row on every apply. Same pore numbering, byte-identical right-hand side and the same `τ` as the assembled path, so `reconstruct_field`, `tortuosity`, `effective_diffusivity` and `formation_factor` work unchanged.
- `solve(sim, alg=KrylovJL_CG(); precond=:auto, reltol=nothing, ...)`, a package-owned entry point that chooses the preconditioner and the tolerance for you: a two-level coarse-space preconditioner once the problem is large enough to pay for the coarse solve, `reltol=1e-10` on a `Float64` system and `1e-6` on a `Float32` one. `solve(sim.prob, alg)` is unchanged — it remains the unopinionated form that takes LinearSolve's defaults.
- `SteadyDiffusionProblem` accepts a scalar `D`, which the tutorial had documented all along. It previously ran `atleast_3d` on the scalar and rejected the resulting 1×1×1 array as not matching the image size. A scalar now rides the same path as the uniform-diffusivity default, so the operator still holds no diffusivity array and the edge weight is that value exactly.
- A warning when part of the pore space does not span inlet to outlet. Those voxels carry no steady flux but still count toward porosity, so `τ` silently includes stagnant volume. `warn_nonpercolating` follows the same three-state convention as `gpu`: `nothing` decides by image size, `true` and `false` force it. Nothing is trimmed or altered — the assembled system is byte-identical with the check on or off.
- The assembled path scales past its 32-bit index wall instead of refusing to run there. Above 306,783,378 pore voxels both the host and the device branch widen the CSC offsets to 64-bit, so the default path runs wherever the hardware has the room. Nothing routes to the matrix-free operator on your behalf; it stays a recommendation.

The assembled path stays the default and is a permanently supported peer, not a deprecated one; it is the only CUSPARSE-backed path.

### Breaking

Three dependencies moved into package extensions, so they are no longer installed or loaded by `using Tortuosity`. `Pkg.add` them yourself, then load them before calling the entry points that need them. This is a breaking interface change for callers of those entry points.

| Add and load this | Before calling |
|-------------------|----------------|
| `ImageFiltering` | `Imaginator.blobs`, `Imaginator.apply_gaussian_blur` |
| `LsqFit`         | `fit_effective_diffusivity`, `fit_voxel_diffusivity` |
| `HDF5`           | `Tortuosity.export_to_hdf5` |

Calling one of them without its package raises an error naming the package to load, rather than an `UndefVarError`. Everything else — problem construction, solving, the observables, `Imaginator.trim_nonpercolating_paths` — is unaffected.

`tortuosity` and `formation_factor` return different numbers than before whenever `D` was anything other than 1.0, because they no longer assume the reference diffusivity is 1.0. The values they returned before were wrong; see *Fixed*. The default path is unchanged.

### Performance

- The assembled steady path was rebuilt. The system is now assembled in one fused pass over the image rather than computing the CSC structure twice through an intermediate adjacency matrix, setup-stage device arrays are released as soon as they are dead, and the SpMV reduces instead of scattering when it knows the matrix is symmetric. An 800³ image went from `OutOfGPUMemoryError` to a 20.7 s end-to-end solve on a 24 GiB card, peaking at 20.588 GiB; setup alone went from 384.2 s to 0.409 s.
- The two-level preconditioner no longer loses ground as images grow. Its coarse block edge is fixed rather than sized from the image, so the ratio between the fine and coarse grids stays bounded, and a V-cycle over coarser grids solves the larger coarse space that results. Iteration counts stop tracking the image edge: on GPU at ε≈0.2, 600³ went from 465 iterations to 150. Wall-clock improves at every size and porosity measured — 5% and 34% at 400³, 58% and 35% at 600³ — and images up to about 253³ take the same code path as before, so nothing changes for them. `τ` stays bit-identical across repeats.
- `using Tortuosity` loads 151 packages instead of 212, and takes about 3.5 s instead of about 4.6 s (Julia 1.12.6, Windows 11, warm cache).
- The precompile workloads that cover the moved entry points moved with them into their extensions, so first-call latency is unchanged for anyone who loads those packages.
- The preconditioner's aggregate inversion caps the scratch it reserves output positions with. The two tables were `nc × nchunks`, so they grew with the image and the thread count at once — about a gigabyte of host memory for a 1000³ image on a 64-thread host, and none of it visible in the device-side memory model a large run is sized against. Cell membership still comes out in ascending node order for any chunk count, so nothing a caller can observe changes.
- Reducing the reference diffusivity over the pore space no longer materialises a copy of it — 4.8 GB at 1000³ and ε = 0.6, allocated immediately after a solve that has already filled memory.

### Fixed

- The coarse solve ran only on Julia 1.12. `SparseArrays`' CHOLMOD carries a three-argument `ldiv!` for plain vectors from 1.12 on; below that the generic `LinearAlgebra` fallback delegates to a two-argument method CHOLMOD never defines, so every preconditioned solve raised `MethodError: no method matching ldiv!(::SparseArrays.CHOLMOD.Factor{Float64, Int64}, ::Vector{Float64})` on both 1.10 and 1.11. The package declares `julia = "1.10"`, and CI covers `lts` and `1` but not 1.11, so the middle version was untested in both directions.
- `tortuosity` and `formation_factor` divided by an effective diffusivity computed with the caller's `D` while implicitly assuming the reference was 1.0, so any other value rescaled the answer. On a fully open box, `D=2.0` returned a tortuosity factor of 0.5 and `D=5.0` returned 0.2 — values a tortuosity factor cannot take, since it is bounded below by 1. Both functions now take the reference explicitly: a scalar `D` is itself the intrinsic diffusivity, a per-voxel field is reduced over the pore space so that whatever fills the solid voxels cannot set the scale, and `D0` overrides either.
- A Dirichlet value was never applied to a boundary node with no neighbours. The condition is imposed as `diag[i]*x[i] = diag[i]*val[i]`, which preserves the diagonal and keeps `A` symmetric, but degenerates at zero degree: the row reads `0 = 0`, `dropzeros!` deletes it, and the prescribed value never lands. An isolated pore voxel on the inlet face then held `c = 0` while sitting on a `c = 1` face, which dragged the inlet-slice mean below the imposed drop, inflated `D_eff`, and reported a tortuosity below 1 — measured at 0.817 on a duct plus one isolated inlet voxel — from a solve that reported success.
- `Float32` conjugate gradient stopped on a recursively-updated residual that drifts away from `b - A*x`. On an ill-conditioned image the drift was large enough that the solver reported success while `τ` was wrong by about 2e-3. `Float32` solves now run an outer refinement loop that recomputes the true residual in `Float64` against the `Float32` operator and solves the correction equation until the residual stops shrinking. Refinement is keyed on the working precision — on by default for `Float32`, off for `Float64` — and `refine=false` restores the previous behaviour. It costs 20 bytes per pore node, and each of those allocations sits behind a guard that warns and returns the unrefined solve rather than taking the whole solve down when a large image leaves no room.
- `laplacian(::PortableSparseCSC)` sized its output on the assumption that every column gains a diagonal entry. A column that already held one gained none, leaving one slot per such column uninitialised; the garbage row index then reached the SpMV kernel, which indexes `@inbounds` and writes through an atomic — reproduced as a segfault rather than a wrong number. The size is now computed from a pre-pass over which columns lack a diagonal.
- Degenerate inputs are refused or handled rather than reaching an opaque error. An image whose pore space has no face-connected pairs at all crashed inside chunk-bound arithmetic with `ArgumentError: step cannot be zero`, and `SteadyDiffusionProblem` accepted a single voxel along the transport axis, where the inlet and outlet faces are the same voxels, both Dirichlet values land on the same nodes, and the solve returns a silently meaningless field.
- `reconstruct_field` copied a device mask to the host only when it was a bare device array. A strided `SubArray` or a `PermutedDimsArray` over device memory fell through and reached the logical index still on the device, triggering the scalar-iteration error the guard above it exists to prevent.
- `build_transient_operator` scaled `nzval` in place without first invalidating the matrix's cached symmetry flag. It was the one mutation site in the package that did not, harmless only because the flag happened to be false already and the next line invalidated anyway.
- All five extensions with a precompile workload imported `PrecompileTools.workload_enabled`, which is internal rather than exported and admitted by any `1.x` compat bound. A PrecompileTools release that dropped it would have been a load-time error in all five — and for the GPU extensions that means the backend registration in `__init__` never runs, so `using Tortuosity; using CUDA` would fall back to CPU without saying so.
- A coarse block holding nothing but a pore cluster enclosed within it could be kept in the coarse space on floating-point round-off alone. Its coarse row then carried a diagonal near `1e-16` and the coarse solve amplified along that direction by the same factor — `‖ldiv!‖∞` of 5.6e15 against 6.9 for the same input — and the assembled and matrix-free paths could disagree about whether to keep it. Such a block is now dropped against a round-off floor. Coarse sizes and iteration counts at the default block size are unchanged.
- `set_preferences!(Tortuosity, "precompile_workload" => false)` now also disables the GPU extensions' precompile workloads. `PrecompileTools` resolved that preference against the extension module, whose UUID `set_preferences!` refuses, so the GPU workload kept running. This is a development-time switch only; the workloads remain enabled by default.
