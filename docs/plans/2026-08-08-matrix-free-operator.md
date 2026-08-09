---
title: Matrix-free operator
created: 2026-08-08
updated: 2026-08-09
status: draft
branch: "-"
supersedes: "-"
superseded-by: "-"
related: 2026-08-08-matrix-path-optimization.md
---

> **Status: draft.** Design spec for replacing the assembled sparse Laplacian with a matrix-free operator, so images beyond ~850³ fit in GPU memory. Not started. **Its motivating evidence was superseded on 2026-08-08** by the matrix-path optimization campaign — the 800³ OOM no longer reproduces, bytes-per-pore-voxel fell 198 → ~76, and "setup cost collapses" is retired as a benefit because assembled setup is now 0.409 s. Read the refresh box below before quoting any number from this file. The remaining case is still strong but rests on one argument: the assembled-CSC floor of ~55 B/pore-voxel is structural, and only deleting `rowval`+`nzval` gets past it.

# Matrix-free operator plan

Design notes for replacing the assembled sparse Laplacian with a matrix-free operator, so large images (≥800³) fit in GPU memory. This document is the handoff spec — it records the measured evidence, the proposed design, and the open decisions.

## Motivation: the measured failure

> ## ⚠ REFRESHED 2026-08-08 — the evidence below is superseded
>
> **The assembled-path optimization campaign (`2026-08-08-matrix-path-optimization.md`) has landed, and the premise of this document has changed.** The original motivating measurements are kept below for the historical record, struck through where they no longer hold. **Read this box before quoting any number from this file.**
>
> | N | peak device memory: as-measured 2026-08-08 (was) | outcome |
> | --- | --- | --- |
> | 200³ | **1.718 GiB** (was 3.25) | solves |
> | 400³ | **4.156 GiB** (was 13.78) | solves |
> | 600³ | **10.750 GiB** (never previously completed) | solves, e2e 71.8 s |
> | 800³ | **21.637 GiB of 23.89** (was OOM) | **solves, e2e 207.8 s, τ 1.884** |
>
> **The 800³ OOM no longer reproduces.** Setup at 800³ went from 384 s (and, in the originally reported session, a throw) to **0.39 s**. `dropzeros!` is off the steady path entirely — boundary conditions are now applied *during* assembly (`src/assembly.jl`), so the temporaries that used to throw are never allocated.
>
> **What this does to the case for matrix-free:**
>
> - **Bytes per pore voxel: ~198 → ~85** (21.637 GiB peak less the 1.375 GiB CUDA context, over 254.6 M pore voxels). The matrix-free target of ~29 B/voxel is now a **~2.9× improvement, not ~6.8×**. Still a large win, but it must be argued on the new baseline.
> - **"Setup cost collapses" is no longer a selling point.** Assembled setup at 800³ is 0.39 s — matrix-free cannot meaningfully beat that. Delete this from the case.
> - **The ceiling claim needs restating.** This document said the assembled path caps out "around 700³"; it now completes 800³ with 2.25 GiB of headroom. The assembled-CSC floor of ~55 B/pore-voxel still caps it near 850³, so matrix-free's move to ~1200³ stands — but it is a jump from 850³, not from 700³.
> - **The real remaining case for matrix-free is unchanged and still strong**: the ~55 B/voxel assembled floor is structural, and only deleting `rowval`+`nzval` gets past it.
> - **The solve, not assembly, is now the bottleneck** — 99.0–99.8 % of end-to-end at 600–800³ (800³: setup 0.39 s, solve 205.7 s). Matrix-free changes SpMV bandwidth per iteration, so its speed case should be made on *per-iteration cost*, which is now where all the time is.
> - Several code shapes this document anticipated needing **already exist**: fused assembly, inline edge weights, and BCs applied during assembly all landed in `src/assembly.jl`. The "an edge contributes only when both endpoints are non-BC" rule described below is implemented there and is bit-identical to the old pipeline across ~30 verified image configurations.

~~An 800³ image exhausts a 23.89 GiB GPU during problem construction.~~ Reproduced with `Imaginator.blobs(shape=(800,800,800), porosity=0.5, blobiness=1.0, seed=42)`, `axis=:x`, `gpu=true`, on an RTX PRO 5000 Blackwell.

*Historical — superseded by the table above:*

| N | pore voxels | edges | peak device memory | outcome |
| --- | --- | --- | --- | --- |
| 200³ | 4.01 M | 22.8 M | ~~3.25 GiB~~ | solves |
| 400³ | 31.9 M | 186 M | ~~13.78 GiB~~ | solves |
| 800³ | 254.6 M | 1.503 B | ~~23.89 GiB (100 %)~~ | ~~`CUDA.OutOfGPUMemoryError`~~ |

~~The throw site is `dropzeros!` (`src/kernels/sparse.jl:252`) called from `apply_dirichlet_bc_fast!` (`src/pdetools.jl:90`), requesting 6.548 GiB.~~ CUDA reported 40.46 GiB of pool reserved against a 23.89 GiB device, meaning stages 2–4 only "succeeded" by spilling into host memory over PCIe — **and that fragmentation is now believed to be why the OOM was seen at all**, since a clean session completes the same construction. `apply_dirichlet_bc_fast!` no longer has any production caller.

Where the memory goes at 800³ (`nnodes` = 254.6 M, `nedges` = 1.503 B, `nnz_L` = 1.758 B):

| allocation | size |
| --- | --- |
| `conns` — COO edge list, `Int32[nedges, 2]` | 11.2 GiB |
| adjacency matrix — `rowval` + `nzval` + `colptr` | 12.2 GiB |
| Laplacian — `nnz + n` entries | 14.0 GiB |
| `dropzeros!` temporaries (`flags`, `flags_Ti`, `scan_inclusive`, `new_rowval`, `new_nzval`) | ~28 GiB transient |

That is roughly 198 bytes per pore voxel.

## The core observation

The operator is a **7-point stencil on a regular Cartesian grid, masked by the pore image**. Every matrix entry is recoverable from the pore mask and the six neighbour offsets in O(1) — the assembled matrix caches a value that costs three flops to recompute.

Neither consumer needs the matrix as a matrix. Both need only its action:

- `SteadyDiffusionProblem` hands it to `KrylovJL_CG`, which calls `mul!`.
- `TransientDiffusionProblem` uses it as `dc/dt = A*c` for `ROCK4`, which calls `mul!` (and estimates the spectral radius by power iteration on the same action).

There is no random entry access, no factorization, and no preconditioner that inspects entries.

## Proposed design

Introduce an operator type — `MaskedLaplacian{T} <: AbstractMatrix{T}` — implementing exactly the interface surface `PortableSparseCSC` already implements and that LinearSolve/Krylov require: `size`, `eltype`, and `LinearAlgebra.mul!(y, A, x)`. Because `PortableSparseCSC` is already accepted by `LinearProblem` with only the 3-argument `mul!`, this is a proven drop-in path.

Stored state — no edges, no `nzval`, no `rowval`, no `colptr`:

- `idx::AbstractArray{Int32,3}` — grid position to compact pore index, `0` for solid. This doubles as the pore mask (`idx > 0` means pore), so the boolean image need not be resident on the device at all.
- `D` — only when diffusivity is variable, kept as a full-grid field rather than per-edge weights.
- `bc_mask::AbstractVector{Bool}` — length `nnodes`. For the steady inlet/outlet case the BC nodes are exactly the pore voxels on the two faces normal to `axis`, so a kernel could test `i == 1 || i == nx` with zero storage; the mask is kept for generality at negligible cost.

`mul!` is a single row-parallel kernel. Launch over grid voxels; a thread skips solids, otherwise owns output node `p = idx[i,j,k]`, reads its six neighbours from `idx`, accumulates in a register, and writes `y[p]` exactly once:

```
y[p] = diag[p] * x[p] - Σ_{q ∈ pore neighbours of p} w[p,q] * x[q]
```

Edge weights are recomputed inline rather than looked up: `w = 1` for uniform `D`, and the harmonic mean `2·D_a·D_b/(D_a + D_b)` for variable `D`. `diag[p] = Σ w[p,q]`, also accumulated inline.

### Boundary conditions

The current path maintains symmetry by zeroing BC rows, zeroing BC columns, restoring the diagonal, then compacting with `dropzeros!`. Matrix-free replaces all of that with one rule: **an edge contributes only when both endpoints are non-BC**. That is symmetric by construction, which matters because CG requires it, and it is much harder to get wrong than the zero-rows/zero-cols/drop dance. For a BC node `p` the row reduces to `y[p] = diag[p] * x[p]`, matching the existing convention of preserving the original diagonal. The RHS contribution `w[p,q] · val[p]` for interior `q` adjacent to BC `p` is folded in once at setup.

## Expected memory

At 800³ with uniform `D`:

| allocation | size |
| --- | --- |
| `idx` (`Int32`, full grid) | 1.91 GiB |
| ~5 Krylov working vectors (`Float32`, `nnodes`) | 4.77 GiB |
| `bc_mask` (`Bool`, `nnodes`) | 0.24 GiB |
| **total** | **~6.9 GiB** |

That is ~29 bytes per pore voxel against the current **~85** (~~198~~ — see the refresh box at the top; the assembled path was optimized on 2026-08-08 and now peaks at 21.637 GiB at 800³, of which 1.375 GiB is CUDA context). It moves the ceiling on a 24 GiB card from roughly **850³** (~~700³~~ — the assembled path now completes 800³ with 2.25 GiB of headroom, and its structural ~55 B/pore-voxel CSC floor is what caps it near 850³) to roughly **1200³**.

## Secondary benefits

**Less SpMV memory traffic, and no atomics on the portable path.** Note that on CUDA `mul!` is already overridden to call CUSPARSE (`ext/TortuosityCUDAExt.jl:63`), so the atomic KA kernel `_spmv_kernel!` (`src/sparse_type.jl:100`) is *not* the CUDA hot path — that kernel, which performs `Atomix.@atomic y[r] += v` for every nonzero (1.758 B atomic adds per SpMV at 800³), is what Metal, AMDGPU, and CPU use. Matrix-free removes those atomics entirely on those backends. On CUDA the win is bandwidth: assembled SpMV reads `rowval` + `nzval` + gathered `x` at ~12 bytes per nonzero (~83 bytes per row at 6.9 nnz/row), while matrix-free reads six `idx` lookups plus six gathered `x` values (~48 bytes per row), and it skips any CSC-to-CSR handling inside CUSPARSE. Treat the CUDA speedup as "expected but must be measured", not assumed.

~~**Setup cost collapses.** No `findall`, no histogram pass, no exclusive scan, no COO scatter, no separate Laplacian assembly, no `dropzeros!`. Construction reduces to a single prefix sum building `idx`.~~ **RETIRED as a benefit.** The assembled path already did all of this on 2026-08-08: `findall`, the histogram pass, the COO scatter, the separate Laplacian assembly and `dropzeros!` are all gone from the steady path, and construction at 800³ takes **0.39 s**. Matrix-free cannot meaningfully improve on that. Do not carry this argument forward.

**The `Int32` ceiling moves out of the way.** Index overflow risk currently lives on `nedges`/`nnz`, which are ~5.9× `nnodes` (measured 5.90 edges per pore voxel at ε = 0.5). Deleting the edge list makes `nnodes` the largest index, moving the `Int32` wall from ~900³ to ~1600³ — past the ~1200³ memory wall. Memory becomes the binding constraint again, which makes deferring the `Int32` work a deliberate choice rather than a gamble. It still needs fixing eventually.

## Scope and compatibility

The public API is unchanged. `sol.u` remains a length-`nnodes` pore-ordered vector, so `reconstruct_field`, `tortuosity`, `effective_diffusivity`, and `formation_factor` keep working unmodified.

`PortableSparseCSC` stays in the codebase. It stops being the production path and becomes the parity reference for tests.

Files affected: `src/simulations.jl` (construction site), a new operator source file, and `src/transient.jl` if the transient path is migrated. `bench/gpu_bench.jl` and `bench/cpu_bench.jl` reach into `build_connectivity_list`, `build_adjacency_matrix`, `laplacian`, and `apply_dirichlet_bc_fast!` directly and will need updating.

## Verification

`test/test_impl_parity.jl` and `test/test_gpu_parity.jl` already provide the harness. The assembled path becomes the executable specification for the new one: assert that matrix-free `mul!` matches `PortableSparseCSC` `mul!` to `Float32` tolerance on small images, across uniform and variable `D`, all three axes, and with boundary conditions applied. End-to-end, assert that `tortuosity` agrees with the existing CPU `Float64` reference values.

## Status of this document

This is a design sketch produced from reading, not from measuring. Whoever implements it has full authority to depart from it: pursue approaches it does not mention, rewrite parts that turn out to be wrong, add code and abstractions where they genuinely serve the goal, and contradict any analysis here that measurement disproves. The same decision heuristic and staff-engineer agency described in `2026-08-08-matrix-path-optimization.md` apply. The design below is a starting point and a record of what was already investigated — not a boundary.

## Open decisions

1. **Steady-only first, or steady and transient together?** The transient operator needs the same treatment but has a wrinkle: `bc_inlet`/`bc_outlet` may be `f(t)`, so the operator and RHS are time-dependent. Recommendation: land steady first, transient as a follow-up.
2. **Benchmark suite.** Update `bench/` in the same change, or keep it pinned to the assembled path as a comparison baseline?
3. **Jacobi preconditioning.** The diagonal is available for free in the matrix-free formulation (one `Float32` vector, 0.95 GiB at 800³, or recomputed inline). Worth evaluating for CG convergence, but it is a separate concern from memory.

## Relationship to assembled-path optimization

Optimizing the existing assembled implementation is a prerequisite, not competing work. The redundancies it removes — fused connectivity-to-CSC assembly, inline edge weights, boundary conditions applied during assembly, atomic-free symmetric SpMV — are the same code shapes the matrix-free operator needs, so that effort carries forward rather than being discarded.
