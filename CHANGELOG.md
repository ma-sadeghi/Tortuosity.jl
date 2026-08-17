# Changelog

All notable changes to Tortuosity.jl are recorded here. Versions follow [Julia's SemVer rules](https://pkgdocs.julialang.org/v1/compatibility/), under which a change to the leftmost non-zero version component is breaking.

## Unreleased

### Added

- A matrix-free form of the steady operator, selected with `SteadyDiffusionProblem(img; axis, matrixfree=true)`. It stores one full-grid `Int32` index array instead of the assembled sparse matrix's row indices and values, and recomputes the seven stencil weights of a row on every apply. Same pore numbering, byte-identical right-hand side and the same `τ` as the assembled path, so `reconstruct_field`, `tortuosity`, `effective_diffusivity` and `formation_factor` work unchanged.
- `solve(sim, alg=KrylovJL_CG(); precond=:auto, reltol=nothing, ...)`, a package-owned entry point that chooses the preconditioner and the tolerance for you: a two-level coarse-space preconditioner once the problem is large enough to pay for the coarse solve, `reltol=1e-10` on a `Float64` system and `1e-6` on a `Float32` one. `solve(sim.prob, alg)` is unchanged — it remains the unopinionated form that takes LinearSolve's defaults.

The assembled path stays the default and is a permanently supported peer, not a deprecated one; it is the only CUSPARSE-backed path.

### Breaking

Three dependencies moved into package extensions, so they are no longer installed or loaded by `using Tortuosity`. `Pkg.add` them yourself, then load them before calling the entry points that need them. This is a breaking interface change for callers of those entry points.

| Add and load this | Before calling |
|-------------------|----------------|
| `ImageFiltering` | `Imaginator.blobs`, `Imaginator.apply_gaussian_blur` |
| `LsqFit`         | `fit_effective_diffusivity`, `fit_voxel_diffusivity` |
| `HDF5`           | `Tortuosity.export_to_hdf5` |

Calling one of them without its package raises an error naming the package to load, rather than an `UndefVarError`. Everything else — problem construction, solving, the observables, `Imaginator.trim_nonpercolating_paths` — is unaffected.

### Performance

- The two-level preconditioner no longer loses ground as images grow. Its coarse block edge is fixed rather than sized from the image, so the ratio between the fine and coarse grids stays bounded, and a V-cycle over coarser grids solves the larger coarse space that results. Iteration counts stop tracking the image edge: on GPU at ε≈0.2, 600³ went from 465 iterations to 150. Wall-clock improves at every size and porosity measured — 5% and 34% at 400³, 58% and 35% at 600³ — and images up to about 253³ take the same code path as before, so nothing changes for them. `τ` stays bit-identical across repeats.
- `using Tortuosity` loads 151 packages instead of 212, and takes about 3.5 s instead of about 4.6 s (Julia 1.12.6, Windows 11, warm cache).
- The precompile workloads that cover the moved entry points moved with them into their extensions, so first-call latency is unchanged for anyone who loads those packages.

### Fixed

- A coarse block holding nothing but a pore cluster enclosed within it could be kept in the coarse space on floating-point round-off alone. Its coarse row then carried a diagonal near `1e-16` and the coarse solve amplified along that direction by the same factor — `‖ldiv!‖∞` of 5.6e15 against 6.9 for the same input — and the assembled and matrix-free paths could disagree about whether to keep it. Such a block is now dropped against a round-off floor. Coarse sizes and iteration counts at the default block size are unchanged.
- `set_preferences!(Tortuosity, "precompile_workload" => false)` now also disables the GPU extensions' precompile workloads. `PrecompileTools` resolved that preference against the extension module, whose UUID `set_preferences!` refuses, so the GPU workload kept running. This is a development-time switch only; the workloads remain enabled by default.
