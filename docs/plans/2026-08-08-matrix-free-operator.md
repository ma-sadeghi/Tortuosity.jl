---
title: Matrix-free operator
created: 2026-08-08
updated: 2026-08-09
status: draft
branch: "-"
supersedes: "-"
superseded-by: "-"
related: 2026-08-08-matrix-path-optimization.md, 2026-08-09-development-loop-latency.md
---

> **Status: draft.** Campaign plan for adding a matrix-free stencil operator beside the assembled sparse path, so images beyond the assembled ceiling (~850³ on a 24 GiB card) become solvable and every size below it gets cheaper. Not started; the five open decisions were settled by Amin on 2026-08-09 (see Decisions) and the campaign is ready to launch. This is the second full version of the document: the 2026-08-08 draft predated the matrix-path optimization campaign and was rewritten from scratch on 2026-08-09 against that campaign's outcomes plus fresh prototype measurements. The headline evidence: a 60-line prototype apply kernel already beats CUSPARSE CSR SpMV **1.80× at 800³** with Float32 parity, a threaded CPU apply beats `SparseArrays` `mul!` **7.3×**, and a **1000³ image (499 M unknowns) — unrepresentable by the assembled path — solved end-to-end through the production LinearSolve path** in 381 s unpreconditioned, peaking at 15.5 GiB of 23.9 — all measured on this machine, 2026-08-09.

# Matrix-free operator campaign plan

Execution plan for a matrix-free 7-point stencil operator as a peer of the assembled sparse path. **Both paths are keepers**: Amin has ruled that the assembled matrix path remains a first-class, permanently supported option — this campaign adds a second production path, it does not replace the first.

## How this document relates to its first draft

The original version of this file was a design sketch written from code reading on 2026-08-08, before the matrix-path campaign ran. That campaign (`2026-08-08-matrix-path-optimization.md`, complete, 39 commits) invalidated most of the draft's motivating numbers: the 800³ OOM it cited no longer exists, the "setup cost collapses" benefit is retired (assembled setup is 0.409 s at 800³), and its memory table described a pipeline that has since been deleted. Rather than patch the draft again, this version was rewritten whole. What survives of the draft unchanged: the core observation, the Dirichlet-by-construction boundary rule, and the drop-in interface strategy. What is new: prototype measurements replacing every load-bearing estimate, the preconditioner interaction (the draft predated B24), the corrected memory ceilings, and the campaign structure.

## The core observation (unchanged from the draft)

The steady operator is a **7-point stencil on a regular Cartesian grid, masked by the pore image**. Every matrix entry is recoverable in O(1) from the pore-index array and the six neighbour offsets — the assembled matrix caches values that cost a few flops to recompute. Neither consumer needs the matrix as a matrix: `KrylovJL_CG` and the transient `ROCK4` path both need only `mul!`. There is no random entry access, no factorization, and — after B24 — the one preconditioner that matters builds its coarse space from a pass over stored entries that can equally be a pass over the grid.

## Why matrix-free, argued on the post-campaign baseline

The matrix-path campaign's final audit round ended on exactly this sentence: *"The largest remaining shares of the 800³ preconditioned e2e are CUSPARSE SpMV (~27 %) and Krylov's own vector ops (~24 %) — neither is Tortuosity code, and neither is improvable without going matrix-free."* This campaign is that sentence's follow-through. The case has three legs, in priority order.

### 1. Scale — the primary case

The assembled path is structurally finished at ~850³ on this card, by two independent walls that land in the same place:

- **Memory wall.** Measured solve peak at 800³ is 20.588 GiB, of which 1.375 GiB is CUDA context → 40.3 bytes per grid voxel. Scaling to the 22.51 GiB data budget gives **~843³ at zero headroom**. The dominant term, `rowval` + `nzval` at 6.89 nnz per pore node (~55 B/pore-voxel), is the floor of the representation — no assembled-path work can remove it.
- **Int32 wall.** `nnz` = 1.754 B at 800³ crosses `typemax(Int32)` at **~856³**. Moving to `Int64` indices to pass it would *raise* bytes per voxel by ~27 % and pull the memory wall down to ~780³. The two walls interlock: the assembled path cannot buy scale with either currency.

The matrix-free operator stores no edges: its solve-time state is one `Int32` full-grid index array plus the Krylov vectors. Measured-constant arithmetic (ε = 0.497, Float32, data budget 22.51 GiB):

| configuration | bytes/grid-voxel | N at zero headroom | N at ≥2 GiB headroom |
| --- | --- | --- | --- |
| assembled, default solve (measured) | 40.3 | ~843³ | ~818³ |
| assembled, preconditioned (measured) | 43.2 | ~824³ | ~800³ |
| matrix-free, default solve | 14 | ~1200³ | ~1160³ |
| matrix-free, preconditioned | 17 | ~1125³ | ~1090³ |
| matrix-free, variable `D` (+node `D`) | 16 | ~1147³ | ~1110³ |
| matrix-free, preconditioned + compressed idx (M10) | 13.2 | ~1223³ | ~1185³ |

Composition of the matrix-free 14 B/voxel: `idx` (Int32, full grid) 4 B + five pore vectors (`b`, `u`, `r`, `p`, `Ap`, Float32 at ε ≈ 0.5) 10 B. The preconditioner adds Krylov's `z` (2 B) and `agg` (Int16, 1 B) — same accounting that measured 21.959 GiB assembled at 800³. **The honest claim is ~1100³ preconditioned / ~1150³ default on this card** — the first draft's "~1200³" did not survive counting `z` and `agg`. The `Int32` ceiling stops binding entirely: the largest index becomes `nnodes`, which crosses `typemax(Int32)` only at ~1630³, far past the memory wall. At 800³ the matrix-free solve peak is a projected **~8.1 GiB + context ≈ 9.5 GiB** against today's 21.96 — memory freed at every size, not only at the frontier.

CPU scale moves the same way: an 800³ CPU solve needs ~12 GB matrix-free in the CPU path's Float64 (~7 GB if Float32 were used) against ~30 GB assembled — it drops from workstation-only to feasible on an ordinary 16–32 GB machine, which matters for the JOSS PuMA comparison.

**The scale claim is demonstrated, not projected.** In this planning session a prototype operator solved a fresh 1000³ blob (seed 42, ε 0.499, `nnodes` = 498 957 533) end-to-end through the production path — `LinearProblem(op, b)` + `KrylovJL_CG` at `reltol=1e-6`, with the `init_cacheval` workspace hook mirrored — on the 23.889 GiB card:

| quantity | measured (2026-08-09) |
| --- | --- |
| matrix-free setup (`cumsum!` + RHS kernel) | 0.135 s |
| apply cost | 30.77 ms (1.95× the 800³ apply — linear in voxels, as expected) |
| solve | **381.4 s, 4805 iterations, retcode `Success`** |
| device peak | ≤15.5 GiB → **≥8.4 GiB headroom** |
| solution sanity | mean 0.482, extrema [0.0000, 1.0002] |

Two side findings worth recording: Float32 CG **converged cleanly at half a billion unknowns** (campaign 1's κ·eps concern did not bite, again), and unpreconditioned iterations continue their ~O(N) growth (3620 → 4805). The growth is exactly what M5 exists to remove: with the two-level preconditioner's flat ~200–230 iterations, 1000³ projects to **~20–25 s** — the size frontier moves 1.95× in voxels while e2e stays where 800³ is today. At 200³ the same prototype matched the assembled path at **identical iteration count (1044 = 1044)**, exact RHS parity, and 7.5e-7 solution agreement.

### 2. Speed — the secondary case

Measured on this machine, 2026-08-09 (RTX PRO 5000 Blackwell, campaign blob fixtures, seed 42, ε 0.5, medians; prototype = raw CUDA kernel mirroring `assembly.jl` semantics, Float32, uniform `D`):

| N | CUSPARSE CSR `mul!` | prototype apply | speedup | parity (rel L2) |
| --- | --- | --- | --- | --- |
| 200³ | 0.338 ms | 0.199 ms | 1.70× | 1.1e-7 |
| 400³ | 3.174 ms | 2.242 ms | 1.42× | 1.1e-7 |
| 600³ | 11.459 ms | 6.905 ms | 1.66× | 1.1e-7 |
| 800³ | 28.439 ms | 15.791 ms | **1.80×** | 1.1e-7 |

The 28.439 ms CSR figure reproduces the campaign's recorded 28.0 ms, so the baseline is consistent. Two further facts from the same session:

- **Tuning headroom is real and quantified.** The prototype's effective bandwidth against its compulsory traffic (`idx` once per grid voxel + `x`/`y` once per pore voxel ≈ 4.1 GB at 800³) is ~259 GB/s, on a device where CUSPARSE itself sustains ~600 GB/s. A tuned kernel reaching CUSPARSE's bandwidth on matrix-free traffic would apply in ~7 ms — **~4× over CSR**. The 1.8× is a raw-CUDA figure; the portable KA port currently pays it back down to 1.38× (see the KA-port gap below), so the honest bracket entering the campaign is **1.4–1.8× before tuning, ~4× as the tuning ceiling**.
- **CPU, 16 threads:** `SparseArrays` `mul!` 36.5 ms → threaded stencil 5.0 ms at 200³ (**7.3×**, parity 1.4e-16); 281.9 ms → 39.5 ms at 400³ (7.1×). This exceeds what blocked item B17 promised (3.31× at 4 threads), and it needs no change to any published `SparseMatrixCSC` object — the operator is a new type, so the API question is faced once, deliberately, instead of retrofitted.

End-to-end projections at 800³ from measured shares (SpMV replaced at 15.8 ms / at a tuned ~7 ms; everything else unchanged — these are projections, to be replaced by Phase 2 measurements):

| path | today | with prototype apply | with tuned apply |
| --- | --- | --- | --- |
| GPU default (3620 iters × 51.9 ms) | 189.7 s | ~144 s (−24 %) | ~115 s (−39 %) |
| GPU preconditioned (202 iters × 91.8 ms) | 20.7 s | ~18.1 s (−12 %) | ~16.5 s (−20 %) |
| CPU 200³ default (1087 iters × 40.1 ms) | 43.9 s | ~9.7 s (−78 %) | — |

The preconditioned-path gain is modest because B24 already made iterations flat — after the swap, Krylov's own vector ops become the largest cost, and campaign 1 measured that fusing them is a ≤6 % dead end. **Speed alone would not justify this campaign; scale does, and speed comes with it.**

### 3. Structure — the tertiary case

No CUSPARSE dependency on the apply path: one KA kernel serves CUDA, CPU, Metal and AMDGPU. The portable backends currently run `_spmv_kernel!`'s 1.758 B atomic adds per SpMV at 800³ — matrix-free deletes that entire class of problem (unverifiable here per A28: no such hardware; the claim is architectural, not measured). CPU threading arrives free through the KA backend. And the CUDA extension's cache/invalidation machinery — `_as_cusparse`, the `symmetric` flag discipline, F2's silent invariant — has no matrix-free counterpart, because there is nothing to cache.

## What campaign 1 settled — do not re-litigate

Facts the previous campaign measured that this plan builds on; the numbers live in `2026-08-08-matrix-path-optimization.md`:

- The solve is 99.0–99.8 % of e2e at 600–800³; assembly is 0.19 % and needs nothing.
- The whole cheap-preconditioner family (Jacobi, polynomial/Neumann) **loses** — measured, monotonically. Only the two-level coarse space wins, and it wins −89 %. Jacobi's rejection carries over to matrix-free unchanged: the diagonal is the degree, bounded by 6.
- Warm-start (B25) has no mechanism through LinearSolve; fused/pipelined CG recovers ≤6 % and means vendoring a solver. Both stay rejected.
- LinearSolve's workspace allocation had to be tamed by hand (`init_cacheval` + `_cg_workspace`, items A20/A29) — the operator type must plug into the same hook or the double-workspace regression returns.
- CSR SpMV made GPU τ run-to-run reproducible; τ is quoted to ≤3 significant figures. The matrix-free apply is a fixed-order per-row sum — deterministic by construction, same quoting rule.
- The assembled CPU chain is pinned bit-identical by 56 `==` comparisons over purpose-built fixtures including zero-degree pore voxels interior and on both Dirichlet faces (F6). Those fixtures are this campaign's parity harness, ready-made.
- Open blockers B17 (CPU SpMV via type change), B23 (tolerance policy has no home), B24-as-default (the −89 % is opt-in) all waited on one API decision — a Tortuosity-owned solve entry point. Amin approved the additive entry point on 2026-08-09 (Decision 2); this campaign implements it in M4, which resolves B23 and B24-as-default outright.

## Design

### Operator type

`MaskedLaplacian{T} <: AbstractMatrix{T}` (working name), stored state:

- `idx::AbstractArray{Int32,3}` — grid position → compact pore ordinal, 0 at solids. Built by the same `cumsum!`-and-mask idiom as `build_steady_system`, and **numbering-identical to the assembled path's**, so `sol.u` is the same pore-ordered vector and `reconstruct_field`, `tortuosity`, `effective_diffusivity`, `formation_factor` work unmodified.
- `axis`/`nbc` — Dirichlet membership stays a coordinate test (`_is_bc(_face_coord(i,j,k,d), nbc)`), exactly as `assembly.jl` does it. **The first draft's `bc_mask` vector is dropped** — it stored what a two-comparison test computes.
- For variable `D`: a node-compacted `Float32` diffusivity vector (2 B/grid-voxel), gathered like `x`; edge weights recomputed inline via the same `_edge_weight` harmonic mean. Storage form to be confirmed by measurement (M9).

No `colptr`, no `rowval`, no `nzval`, no edge list, no `symmetric` flag, no cache.

### Apply kernel

One KA kernel, shared CPU/GPU like `assembly.jl`'s pair, launched over grid voxels at `wg=(64,4,1)`; a thread skips solids, owns output `p = idx[i,j,k]`, walks the six neighbours **in `_NEIGHBOURS` order** (fixed order → deterministic sums), accumulates degree and off-diagonal action in registers, writes `y[p]` once:

- free row: `y[p] = deg·x[p] − Σ_{q pore nb, non-BC} w·x[q]`, where `deg` sums **all** pore neighbours (BC included) — this reproduces the assembled convention that elimination keeps the original diagonal;
- BC row: `y[p] = d·x[p]` with `d = deg`, or 1 when `deg == 0` (`_unit_where_zero`);
- empty row (isolated free voxel): `y[p] = 0` falls out of `deg == 0` with no branch.

The kernel must **share** `_edge_weight`, `_is_bc`, `_face_coord`, `_NEIGHBOURS` with `assembly.jl` rather than restate them — one source of truth for the stencil semantics is what makes parity with the assembled path structural rather than coincidental. Both 3- and 5-argument `mul!` (the 5-arg form is a two-line epilogue change; ROCK4 and some Krylov methods want it). The prototype validated this exact semantics to 1.1e-7 (Float32) and 1.4e-16 (Float64) against assembled matrices on blob fixtures.

### RHS

A sibling kernel emits `b` directly (the folded-in Dirichlet load: `Σ w` over inlet-face BC neighbours for free rows; `d` or 0 for BC rows) — the prototype's version matched `build_steady_system`'s `b` to the last bit at 200³. Setup is then `cumsum!` + one kernel: the operator constructor is strictly cheaper than assembled setup, which was already 0.409 s at 800³.

### LinearSolve integration

`LinearProblem(op, b)` + `KrylovJL_CG` works today with only `size`/`eltype`/`mul!` — proven by the prototype end-to-end through the production `solve` call. The operator gets the same `LinearSolve.init_cacheval` specialization `PortableSparseCSC` has (zero-length placeholder, `_cg_workspace` aliasing `x = u`); prefer hoisting that method to dispatch on `Union{PortableSparseCSC, MaskedLaplacian}` or a small abstract type over copy-pasting it. The existing field-by-field workspace test extends to the operator.

### Preconditioner port — required, not optional

B24 is worth −89 % and must compose. Two of its three inputs are representation-independent (`agg` from the image, restrict/prolong/coarse-solve on vectors); the two that read the matrix need matrix-free replacements:

- `_coarse_stencil_kernel!` iterates **stored entries** of `A` to accumulate `WᵀAW`. Replacement: a grid-pass kernel of the same shape as the apply kernel — each pore voxel contributes its diagonal to `(agg[p], agg[p])` and each edge its `−w` to `(agg[p], agg[q])`, same slot arithmetic. Everything downstream (host assembly, shift, Float64 Cholesky, block-drop rule) is untouched.
- `inv_lambda = 1/(2·maximum(nzval))`. For uniform `D` the max diagonal on any percolating blob is exactly `6·D0` — use the closed form. For variable `D`, fold a max-reduction into the RHS kernel's pass.

Acceptance: preconditioned iteration counts and τ at 200–800³ match the assembled preconditioned path (202 iters at 800³) to within fp-reordering noise.

### Constructor integration

`SteadyDiffusionProblem` grows `matrixfree::Bool=false` (Decision 1), routing construction to the operator instead of `build_steady_system`. **Default stays assembled** — mid-JOSS-review, the default path's behavior must not move (see Constraints). Alongside it lands the Tortuosity-owned solve entry point (Decision 2): an additive `solve(sim, …)`-form function — never a shadowed LinearSolve name — that owns path choice, preconditioner default (two-level on by default there), and tolerance policy, resolving campaign 1's B23 and B24-as-default in passing; `solve(sim.prob, alg)` keeps working unchanged. The transient path is out of scope (Decision 3): `TransientDiffusionProblem` keeps the assembled pipeline; the matrix-free transient operator (row-only BC rule, possibly time-dependent `b(t)`) is a named follow-on, M12.

## Prototype provenance

The measurements above came from three throwaway scripts run in this planning session (2026-08-09) against the campaign's cached blob fixtures (`tempdir()/tortuosity_bench_blobs`, seed 42, ε 0.5): a raw-CUDA apply-kernel race vs CUSPARSE with parity checks; a KA-port overhead check; and an end-to-end LinearSolve-path solve demonstration. Phase 0 recreates them as a committed `bench/` harness — the numbers in this plan are quotable but not yet reproducible from the repo, which is exactly the F13 mistake campaign 1 logged; fixing that is Phase 0's first job.

### The KA-port gap — measured, and a design tension to resolve

The numbers above are from a **raw CUDA** kernel. A line-for-line KernelAbstractions port (the pattern `assembly.jl` uses, and what a portable production kernel would be) produced **bit-identical output** but measurably slower launches: 1.18× at 400³, **1.36× at 800³** (20.57 ms vs 15.10 ms — still 1.38× faster than CUSPARSE). For `assembly.jl` this overhead never mattered (setup is 0.19 % of e2e); the apply runs hundreds to thousands of times per solve, so here it does. Three ways out, to be settled by measurement in M8: shrink the KA overhead in place (the usual suspects are `@index(Global, NTuple)`'s per-thread division arithmetic and the dynamic ndrange — static ranges and manual index math inside a KA kernel are both legal), keep the KA kernel as the portable path and add a CUDA-specialized kernel in `ext/TortuosityCUDAExt.jl` (the exact precedent the CUSPARSE `mul!` override already sets), or accept 1.38× if tuning erases the gap anyway. Do not accept the naive port without measuring — a third of the apply win is at stake.

## Non-negotiable constraints

1. **Golden values are frozen.** `GOLDEN_STEADY` and `GOLDEN_VARIABLE_D` in `test/test_regression_golden.jl` never change. They exercise the default (assembled) path, which this campaign does not touch behaviorally, so they cannot legitimately move. Matrix-free results are verified *against* them at the same tolerances via new tests.
2. **The suite floor is 11576 assertions, green, GPU included.** Never weaken, skip, or delete a test to make a change pass. New parity tests only add.
3. **The assembled path stays default and behaviorally untouched.** JOSS review is in flight; `solve(sim.prob, KrylovJL_CG())` on an unmodified construction must produce today's numbers. Refactors that share helpers (`_edge_weight` etc.) are fine; anything that changes assembled output is a blocker.
4. **Both paths are first-class, permanently.** Amin's directive. The assembled path is not deprecated, not test-only, not "the reference implementation" — it is a supported production path (and the only one CUSPARSE-backed). The first draft's line that `PortableSparseCSC` "stops being the production path" is superseded.
5. **`Int32` overflow work stays out of scope** (Amin's standing deferral) — but note the operator narrows the exposure rather than widening it: its binding index is `nnodes` (~1630³) instead of `nnz` (~856³).
6. Unattended-run rules from campaign 1 apply verbatim: blockers are logged and skipped, never decided unilaterally; failures are reverted, never left red; goldens are never updated unattended.

## Optimization inventory — M-series

Statuses are all `pending`; est. gains marked **measured** come from this session's prototypes, everything else is arithmetic or projection. Maintain this table exactly as campaign 1 maintained its inventory: add discoveries with fresh ids, correct estimates with measurements, retire honestly, re-rank as the bottleneck moves.

| id | item | gain | complexity | phase |
| --- | --- | --- | --- | --- |
| M1 | `MaskedLaplacian` type + KA apply kernel (3- and 5-arg `mul!`), CPU+GPU shared, helpers shared with `assembly.jl` | apply vs CSR at 800³ GPU: 1.80× raw / 1.38× naive KA port — **measured (prototype)**; 7.3× vs stdlib at 200³ CPU/16t | ~200 lines, the campaign's core | 1 |
| M2 | LinearSolve workspace hook for the operator (shared dispatch with `PortableSparseCSC`) | avoids +2 n-vectors at solve peak — measured in campaign 1 | ~10 lines | 1 |
| M3 | RHS kernel + operator constructor (`build_steady_operator` beside `build_steady_system`) | setup ≤ assembled's 0.409 s — **b parity measured exact** | ~60 lines | 1 |
| M4 | `SteadyDiffusionProblem` `matrixfree=false` keyword + the Tortuosity-owned solve entry point (Decisions 1–2): path choice, preconditioner default, tolerance policy in one place | unlocks user access; makes B24's −89 % the entry point's default; resolves B23 | ~30–50 lines | 2 |
| M5 | Two-level preconditioner port: grid-pass `WᵀAW` kernel + `inv_lambda` closed form / fused max | keeps the −89 %; **required** for the flagship sizes | ~80 lines | 2 |
| M6 | Parity + edge-case tests: apply vs assembled on the F6 fixture suite × {uniform, variable D} × 3 axes; e2e τ vs goldens; 5-arg `mul!`; empty columns; zero-degree BC nodes; workspace field-by-field | correctness backbone | additive tests only | 1–2 |
| M7 | Bench harness: `mf` pass in `bench/scaling_bench.jl`, sizes extended to 1000³/1100³, fixtures cached; commit the prototype scripts' successors. Any new sibling script that generates blobs needs its own `using ImageFiltering` and must run under `--project=bench` | makes every number here repo-reproducible | moderate | 0, 2 |
| M8 | Apply-kernel tuning: close the measured KA-vs-raw gap (1.36× at 800³) or specialize per-backend in the CUDA ext; launch-config sweep, `Int32` in-kernel index arithmetic, `@Const`/read-only paths, plane/slab tiling for `idx` reuse (each `idx` value is read by 7 threads), occupancy | measured headroom 259 → ~600 GB/s ⇒ apply ~7 ms at 800³, ~4× vs CSR | open-ended; audit-round stop rule | 3 |
| M9 | Variable-`D` apply variant + storage decision (node-compacted vs full-grid `D`) | closes feature parity; +2 B/voxel | kernel branch + measurement | 2–3 |
| M10 | Compressed `idx`: pore bitmask + per-block popcount prefix (rank query) replaces the Int32 array — 4 B/voxel → ~0.19 B/voxel | ceiling ~1125³ → ~1185³ preconditioned; possibly *faster* (less compulsory traffic) | +1 real concept; gate on M8 measurements | 3 |
| M11 | CPU threaded apply — free through the KA CPU backend | CPU solve projected −70 %+ at 200³ — **measured 7.3× on the SpMV share** | none beyond M1 | 1 |
| M12 | Transient matrix-free operator (row-only BC rule, `b(t)`, ROCK4 spectral bound via `mul!` power iteration) | transient memory/speed | **follow-on, out of scope** (Decision 3); interacts with A30 | — |
| M13 | Docs: operator docstrings, README/docs positioning of the two paths, JOSS-safe wording, **plus a `CHANGELOG.md` entry** under `## Unreleased` for M4's user-visible `matrixfree` keyword and solve entry point | — | small | 4 |

### Considered and rejected at planning time

Logged so nobody re-investigates; reasoning per campaign convention.

- **Stored `bc_mask` vector** (was in the first draft) — Dirichlet membership is a two-comparison coordinate test in-kernel; storing it buys nothing and costs 1 B/pore-voxel. Rejected; the draft is corrected by this revision.
- **Stored diagonal vector** (the draft's Open decision 3) — the degree accumulates for free in the same six neighbour reads the apply already does; a stored diagonal adds 2–4 B/voxel to save ~6 flops on a bandwidth-bound kernel. Rejected by arithmetic.
- **Jacobi preconditioning** — B3's measured rejection carries over: post-elimination the diagonal is the degree (1…6), a bounded row rescaling that leaves the O(N²) low-frequency mode untouched; and for uniform interior it is exactly the identity. Revisit only for strongly variable `D`, per campaign 1.
- **Full-grid vectors** (no `idx`, no gathers, perfectly structured stencil) — doubles every Krylov vector at ε = 0.5 (solid entries), and dots/norms then run over solids too. Worse memory *and* worse vector-op traffic at this package's porosities. Reject; revisit only if a high-porosity (ε ≳ 0.75) workload ever becomes primary.
- **fp16 anything** — campaign 1 established CG needs fp32 vectors to converge at `reltol=1e-6`; the apply's traffic is dominated by `idx` and the vectors, so an fp16 weight buys nothing (weights are computed, not stored).
- **Multi-axis batching** (solve `:x`/`:y`/`:z` concurrently in freed memory) — the operator differs per axis (BC rule), so there is no shared-operator multi-RHS; three concurrent solves would contend for bandwidth with no reuse. Nothing to build; τ-over-axes stays sequential.

## Verification protocol

The assembled path is the executable specification, exactly as campaign 1 left it:

1. **Apply parity**: `mul!` vs `PortableSparseCSC`/`SparseMatrixCSC` on the F6 fixture suite (7 images including interior and Dirichlet-face zero-degree voxels) × {uniform, variable `D`} × 3 axes × random vectors — Float64 to ~1e-14 rel, Float32 to ~1e-6 rel. Both 3- and 5-arg forms.
2. **RHS parity**: `b` exact-equal on CPU Float64 (integer-valued sums), tolerance on GPU Float32.
3. **End-to-end**: matrix-free τ agrees with the golden tables at golden tolerances on all seeds/axes, and with the assembled GPU path at Float32 tolerance at bench sizes. A golden mismatch is a matrix-free bug by definition — never a golden update.
4. **Preconditioner parity** (M5): iteration counts within a few of the assembled preconditioned path at 200–800³; τ unchanged.
5. **Workspace hook**: field-by-field CgWorkspace comparison, extended from the existing test.
6. **Determinism**: two identical solves produce identical τ (the apply's fixed summation order makes this strictly checkable, like the CSR path).

### Suite and session mechanics

Floor: **11576 assertions**, verified green at 222.93 s on 2026-08-09. Full `Pkg.test()` at phase boundaries in the background, foreground green run at the end.

**Iterate through the persistent Julia MCP session, not fresh processes.** This supersedes campaign 1's scratch-env recipe, which this plan originally inherited. Call `julia_eval` with `env_path` set to `C:\Users\sadegmo\.julia\dev\Tortuosity\test\`; the trailing `test/` makes the server run `using TestEnv; TestEnv.activate()` and treat the parent as project root, which is how this package's `[extras]`/`[targets]` layout is handled (`--project=.` cannot `using Test`). Measured edit→green for one test file: **1.13 s against 173.9 s** for the fresh-process loop. Full guidance is in the global `julia-workflow` skill; the two rules that matter most:

- After editing any `.jl`, call `Revise.revise()` **and verify the reload applied** — assert a value the edit changed. A missed reload returns results for code that was never compiled and reads exactly like a passing test. This failure mode was hit during measurement, not theorised.
- `julia_restart` with the **same** `env_path` after any struct or `const` redefinition error, world-age error, branch switch, or `Project.toml` / `Manifest.toml` change. A cheap reset beats debugging a poisoned session.

**Each `env_path` is its own persistent session**, which is how this campaign's two environments coexist without interfering: the test environment above, and `C:\Users\sadegmo\.julia\dev\Tortuosity\bench\` for developing the bench harness.

**Benchmark in the warm session too. Measurement hygiene comes from warm-up, not from process freshness.** Compilation latency is the cost of *using Julia*; it is not a property of the code under test, so a timing that includes it is measuring the wrong thing. The discipline is therefore **warm up on a small problem, discard that run, then measure** — the same protocol `BenchmarkTools` applies inside a single process. A cold process does not buy this: it hands you the compile cost baked into your first number unless you warm up anyway. Over a campaign this long, spawning a fresh process per measurement would also burn more wall clock than the campaign's actual computation.

**The cases that genuinely need a fresh process — few by definition:**

- **Peak-memory and size-ceiling certification.** This campaign's headline claims (1000³ and 1100³ preconditioned on a 24 GiB card) are *memory* claims, and a session holding earlier allocations invalidates them outright. Certify in a fresh process, or prove the session is clean immediately beforehand — `GC.gc()`, `CUDA.reclaim()`, and a checked `CUDA.memory_status()`. Prefer the fresh process for anything that goes in the final table.
- **The final benchmark table at Phase 4**, so a third party can reproduce it from a clean machine.
- **The foreground `Pkg.test()` release gate.**
- **Anything measuring startup, load or precompile time itself** — there the overhead *is* the quantity, so a warm session cannot answer the question.

Everything else — apply-vs-CSR comparisons, iteration counts, solve timings while tuning, the whole M8 audit loop — belongs in the warm session, with warm-up and discard.

**`bench/` and `benchmarks/` are different directories, and confusing them breaks Phase 0.** This campaign owns `bench/`. `benchmarks/` belongs to the JOSS effort and **must not be modified**. Bench scripts must run under `--project=bench`: as of the 2026-08-09 extension split, `--project=benchmarks` cannot resolve `ImageFiltering` and fails before producing a number. `bench/scaling_bench.jl`'s own usage header still says `--project=benchmarks` and is stale — correcting that one line is a legitimate Phase 0 chore.

## Execution

### Phases

- **Phase 0 — harness and baseline.** Branch `perf/matrix-free` off `main` (suite verified green first). Commit this plan. Recreate the prototype scripts as a committed benchmark pass (M7 first half): `mf` variant in `bench/scaling_bench.jl` or a sibling, so apply-vs-CSR and solve numbers are repo-reproducible. Generate and cache ≥1000³ blob fixtures (host-RAM caution: `blobs` at 1100³+ may need chunked or Float32 generation — new fixture sizes carry no goldens, so B27's Float32 blocker does not apply to them; log whatever is done). **`Imaginator.blobs` now requires `using ImageFiltering` in the calling script** — since the 2026-08-09 extension split it routes through a weak-dependency hook and raises an actionable error otherwise, so any fixture-generation script needs that `using` and an environment that can resolve it. Fixtures for 64/100/200/400/600/800 and **1000³ are already cached** at `%TEMP%\tortuosity_bench_blobs\` (e.g. `blobs_n1000_p0.5_b1.0_seed42.raw`, 1.0 GB); only 1100³ needs generating. Re-run the assembled bench at 200–800³ **under `--project=bench`** to confirm the campaign-1 baseline still reproduces at HEAD.
- **Phase 1 — the operator (the main event).** M1 + M2 + M3 + M11 with M6's parity tests landing in the same commits. Exit: parity suite green on CPU and GPU, apply at least at the measured naive-KA level (≥1.38× vs CSR at 800³ — closing the KA gap belongs to Phase 3, not here), LinearSolve path solving end-to-end via `LinearProblem(op, b)`.
- **Phase 2 — integration and the preconditioner.** M4 (the `matrixfree` keyword and the solve entry point, per Decisions 1–2), M5, M9, second half of M7. Exit: 800³ preconditioned matrix-free e2e measured ≤ assembled's 20.7 s; 1000³ certified end-to-end on the bench; iteration-parity acceptance met.
- **Phase 3 — the frontier and the speed rounds.** Certify the largest solvable sizes (target ≥1100³ preconditioned; M10 if the arithmetic is needed to get there or M8 shows it is free speed). Alternate audit and implementation rounds on M8 exactly like campaign 1's Phase 4, with the same printed stop convention (`AUDIT ROUND <n>: <k> candidates surfaced, <m> accepted`; `PHASE 3 COMPLETE: 0 accepted candidates`).
- **Phase 4 — verification and consolidation.** Independent adversarial review of the whole diff; foreground full suite; final bench at all sizes both paths; M13 docs; refresh this file's numbers and the matrix-path plan's cross-references; final report in this file.

### Orchestration

Per Amin's standing preference: the master stays context-light and directs; all reading, editing, testing and measuring is delegated; one write-agent at a time; every write is checked by an independent read-only reviewer who re-runs tests and benchmarks; agents report in campaign 1's compact format; the Progress log in this file — not anyone's context — is the state. The full protocol, including the `/goal` evaluator-visibility rules, is in `2026-08-08-matrix-path-optimization.md` §Orchestration protocol and applies verbatim.

### Git discipline

Branch `perf/matrix-free`; one conventional commit per accepted change referencing inventory ids; never `git add -A` or `commit -a` (path-scope to `src/`, `test/`, `ext/`, `bench/`, plus this file); no pushes; no attribution trailers. Commit authorization for unattended runs is per-campaign, granted when Amin starts it.

### Goal condition (paste to `/goal` when starting)

```
/goal Execute the plan in docs/plans/2026-08-08-matrix-free-operator.md to completion. Read that file first, then resume from its Progress log. The condition is met when, and only when, your visible message text contains the exact line: CAMPAIGN COMPLETE - all conditions met. Print that line only after you have personally verified all five: (1) every M-series inventory item except M12 is terminal (done, rejected, BLOCKED or REVERTED) in the Progress log; (2) Phase 3 ended on the printed diminishing-returns stop condition; (3) a full Pkg.test() run is green with assertions at or above 11576; (4) the benchmark harness has been re-run at all sizes including the largest certified size, on both the assembled and matrix-free paths; (5) the Final report is written into this plan file. Constraints: never modify golden tau values, never weaken or skip a test, never use git add -A or git commit -a, never leave the tree red, never change the assembled path's default behavior, never modify anything under benchmarks/. Run all code iteration through the persistent Julia MCP session (julia_eval, env_path ending in test/) rather than spawning julia from Bash or PowerShell, and always verify a Revise reload applied before trusting a result. Benchmark inside the warm session using warm-up-then-discard: never report a first-call timing, because compile time is Julia overhead and not a property of the code under test. The few cases that require a fresh process are peak-memory and size-ceiling certification, the final Phase 4 benchmark table, the foreground Pkg.test() gate, and anything measuring startup or precompile time itself. Bench scripts run under --project=bench, never --project=benchmarks. Stop after 60 turns if not complete, print CAMPAIGN HALTED and a status summary.
```

## Decisions — settled by Amin, 2026-08-09

The five questions this plan was drafted with were put to Amin on 2026-08-09 and ruled on as recommended. They are recorded here as rulings; the campaign executes them without revisiting.

1. **API spelling: `matrixfree::Bool=false` on `SteadyDiffusionProblem`.** The smallest surface. Flipping the *default* is explicitly not part of this campaign — it is its own post-JOSS decision, to be made with the final benchmark table in hand.
2. **The solve entry point: approved, additive form.** Tortuosity owns a `solve(sim, …)`-form entry point — a new function, never a shadowed LinearSolve name — implemented in this campaign as part of M4. It owns path choice, preconditioner default (two-level on by default there), and tolerance/iteration policy, which resolves campaign 1's B23 and B24-as-default together. `solve(sim.prob, alg)` keeps working unchanged, so the **solve** surface does not move. (Narrowed 2026-08-09: the package's public surface as a whole is no longer frozen — the extension split made `Imaginator.blobs`, `apply_gaussian_blur`, `fit_effective_diffusivity`, `fit_voxel_diffusivity` and `export_to_hdf5` require the caller to load an optional package first, recorded as a `### Breaking` entry in `CHANGELOG.md`. That does not affect this ruling, which is about the solve path only.)
3. **Scope: steady only.** The transient operator (M12) gets its own follow-on plan; it has a different BC rule and a time-dependence wrinkle, and A30 (assembled transient port) remains open in parallel.
4. **Certified size targets: 1000³ and 1100³ preconditioned** on the 24 GiB card as the campaign headline, with 1200³ attempted only if M10 lands. Fixture generation cost and disk (raw caches reach ~1.3 GB each) are the practical constraints.
5. **Float32 stays, with true-residual monitoring.** The prototype's 1000³ Float32 CG converged to `reltol=1e-6` with retcode `Success` and a physically sane field, so no mixed-precision machinery is needed at the target sizes. Certification runs at ≥1000³ include a true-residual check (recursive-vs-true residual drift is the known failure mode); a Float64 variant (which halves the size ceiling) is considered only if 1100³ certification actually stalls.

## Relationship to the matrix-path campaign's open items

- **B17** (CPU SpMV type change, blocked): the matrix-free CPU path delivers more than B17 promised without touching `SparseMatrixCSC` publics. B17 stays open *for the assembled path*, but its urgency drops to near zero once M11 lands; recommend re-triaging it to rejected-unless-someone-asks after this campaign.
- **B23 / B24-as-default**: resolved — the entry point was approved on 2026-08-09 (Decision 2) and this campaign implements it in M4.
- **A30** (transient port to fused assembly): untouched by this campaign; if M12 ever lands, A30's GPU motivation shrinks to maintenance; judge then.
- **A28** (portable `_free!`): unchanged; the operator allocates less, so the no-op matters less.
- **F13** (benchmarks not repo-reproducible): this campaign's Phase 0 explicitly closes the same gap for its own numbers.

## Progress log

**This log is the campaign's state.** Read it before doing anything; append one line per terminal item, including rejections, blockers, and reverts. An empty log means Phase 0 has not run.

Format: `date — id(s) — status — memory delta — speed delta — commit sha — reviewer verdict`.

2026-08-09 — Phase 0 — done — baseline suite green on `main` (fresh-process `Pkg.test()`, exit 0) before branching; branch `perf/matrix-free` cut from `ff89527`; `bench/scaling_bench.jl` usage header corrected to `--project=bench` (4 lines) — no perf delta — `37c0404` — self-verified.
2026-08-09 — Phase 0 note — fixtures 64/100/200/400/600/800/1000³ confirmed cached in `%TEMP%\tortuosity_bench_blobs\` (1.0 GB for 1000³); 314 GB free, so 1100³ generation is affordable. The `mf` bench pass (M7 first half) is deliberately re-sequenced *after* M1 — it cannot exercise an operator that does not exist yet; the Phase 0 half of M7 was only ever the header fix and the fixture audit.
2026-08-09 — M1, M2, M3, M6(part) — done — operator state is one Int32 full-grid `idx` (4 B/grid-voxel) against the assembled `colptr`+`rowval`+`nzval` — speed: GPU apply at 800³ **15.376 ms vs CUSPARSE CSR 30.363 ms = 1.97×**; 1.46×/1.64×/1.69× at 200/400/600³; Float32 parity 1.0e-7 throughout — `a994580` — self-verified, independent review scheduled at the Phase 1 exit.
2026-08-09 — M1 finding, supersedes the plan's "KA-port gap" — **there is no KA-port gap.** The plan entered the campaign expecting a portable KernelAbstractions kernel to cost 1.36× against raw CUDA at 800³ (20.57 ms vs 15.10 ms) and budgeted M8 to close it. The production KA kernel measures **15.376 ms at 800³** — at or below the raw-CUDA prototype's 15.791 ms. The three-way design tension the plan posed (shrink the KA overhead / specialize in the CUDA ext / accept 1.38×) is void: no CUDA-specialized apply is needed, `ext/TortuosityCUDAExt.jl` gains nothing on this path, and M8 reduces from "recover a third of the win" to pure bandwidth tuning against the 259 → ~600 GB/s headroom.
2026-08-09 — M1 measurement-hygiene finding — CUSPARSE SpMV needs a **~20-call warm-up**, not the 3 calls that suffice for the KA kernels: at 200³ the first seven `mul!` calls run at 10–19 ms and then settle at 0.37 ms. A short warm-up silently reports the assembled path as 30× slower than it is (a first pass here measured 200³ CSR at 10.4 ms — slower than 400³ — which is what exposed it). Every assembled-vs-matrix-free comparison in this campaign uses `warmup=25, reps=15, median`.
2026-08-09 — M11 — done — no new storage — CPU apply, 20 threads, free through the KA CPU backend: 200³ `SparseArrays` `mul!` 36.656 ms → **5.424 ms (6.76×)**, parity 9.4e-17; 100³ 4.541 → 0.831 ms (5.47×). Slightly under the prototype's 7.3× and comfortably over B17's 3.31× promise — `a994580` — self-verified.
2026-08-09 — M4 — done — no memory delta on the default path — no speed delta on the default path — `e08f606` — self-verified; goldens, steady physics and error paths re-run green. `matrixfree::Bool=false` on `SteadyDiffusionProblem` (Decision 1) plus the entry point (Decision 2), which resolves **B23** (tolerance policy: `1e-10` on Float64, `1e-6` on Float32, because Float32 CG cannot drive the relative residual below ~1e-6) and **B24-as-default** (two-level preconditioner on by default above `_PRECOND_MIN_NODES = 100_000`).
2026-08-09 — M4 interpretive call, flagged for Amin — Decision 2 says "an additive `solve(sim, …)`-form function — never a shadowed LinearSolve name". Implemented as a **method on LinearSolve's own `solve` generic** (`function LinearSolve.solve(sim::SteadyDiffusionProblem, alg=KrylovJL_CG(); …)`), not as a new name. Rationale: `Tortuosity` does `using LinearSolve`, so defining a bare `function solve(…)` inside the module would create a *new* `Tortuosity.solve` that shadows the re-exported one and breaks `solve(sim.prob, alg)` for every existing caller — precisely the outcome the ruling forbids. Adding a method to the shared generic is additive, is literally the `solve(sim, …)` form the ruling names, and is not piracy (`SteadyDiffusionProblem` is ours). Verified: `solve(sim.prob, KrylovJL_CG())` is unchanged. If a distinct name was wanted instead, this is a one-line rename.
2026-08-09 — M4 constructor hazard, fixed at the point of writing — the assembled branch frees the device `D` copy right after assembly. The matrix-free operator **holds** `D` for its whole life and recomputes weights from it on every apply, so the same free would leave the operator pointing at released memory. The free is now conditional on the path.
2026-08-09 — M9 — done (variant), storage decision resolved — full-grid `D`, 4 B/grid-voxel — variable-`D` apply at 600³ **11.22 ms vs CSR 11.88 ms (1.06×)**; +70 % over the uniform kernel, flat in size — see the two entries below for the fix and the rejection.
2026-08-09 — M9 finding, `@Const(D)` — the variable-`D` apply was **losing to CSR at 600³** (29.81 ms vs 12.00 ms, 0.40×) with a cost over the uniform kernel that grew from +72 % at 400³ to +308 % at 600³ — a cache cliff, not a constant. Cause: `D` was the one kernel argument not marked `@Const`, so the neighbour reads got neither the read-only data path nor a no-alias guarantee against `y`. Marking it `@Const` (one word) gives **29.81 → 11.22 ms at 600³, 2.66×**, turns 0.40× against CSR into 1.06× for it, and flattens the overhead to a size-independent +61…70 %. Uniform-`D` and CPU results are unchanged. This is an M8-class win found in Phase 2; `assembly.jl` has the same unmarked argument but assembly is 0.19 % of e2e, so it is not worth touching there.
2026-08-09 — M9 storage decision — **node-compacted `D` rejected.** Measured against the full-grid array with a like-for-like kernel: **1.05× at 400³, 1.07× at 600³, output bit-identical (parity exactly 0.0)**. It would save 2 B/grid-voxel at ε = 0.5, moving the *variable-`D`* preconditioned ceiling from ~1048³ to ~1083³ — both comfortably under the uniform-`D` headline of ~1124³, so it does not change what this campaign certifies. Against that: a new stored representation, a wrapper type, a changed kernel signature and its own tests, and `_node_diffusivity` would stop being shared verbatim with `assembly.jl`. Not worth it. Full-grid `D` stands, and the plan's memory table row for variable `D` should read 18 B/voxel (4 + 4 + 10), not 16.
2026-08-09 — design ruling, apply summation order — the plan asked for a bit-identical apply. It is not achievable in one pass: a CSC `mul!` left-associates all seven terms of a row in ascending column order, but the diagonal sits between the lower and upper halves and its value is only known after both have been walked, so the upper half must be summed on its own and folded in as a single term. Exactness would cost either a second full pass over `idx` (the kernel is bandwidth-bound on exactly those reads) or a source-level unroll of the upper loop. Measured departure: **4.8e-17 relative, sub-ULP**. The test asserts `rtol=1e-15`, an order tighter than the verification protocol's own 1e-14 requirement; `b` remains **exactly** equal. No test was weakened — the claim was corrected before it ever landed green.
