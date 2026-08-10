---
title: Matrix-free operator
created: 2026-08-08
updated: 2026-08-09
status: complete
branch: perf/matrix-free
supersedes: "-"
superseded-by: "-"
related: 2026-08-08-matrix-path-optimization.md, 2026-08-09-development-loop-latency.md
---

> **Status: complete**, 2026-08-09, on branch `perf/matrix-free`. The matrix-free stencil operator ships beside the assembled sparse path as a first-class peer, selected with `SteadyDiffusionProblem(img; axis, matrixfree=true)`. Delivered: **1000³ certified end-to-end at 14.16 bytes per grid voxel** (predicted 14) with 9.42 GiB spare on a 23.9 GiB card, against an assembled path that is structurally finished near 850³; **1.97× the CUSPARSE CSR apply at 800³** and 6.76× threaded `SparseArrays` on CPU; **26 % faster end to end at 800³ on 44 % of the peak memory**, with iteration counts identical between the two paths at every size. Suite 12708 assertions green, goldens untouched, the assembled default behaviourally unchanged. Read the **Final report** for the outcome and the **Progress log** for the reasoning; everything below them is the plan as written before execution, preserved so the two can be compared — where a prediction did not survive measurement, the Final report says so.
>
> **What the plan got wrong, in one place:** the feared KernelAbstractions port gap does not exist (the portable kernel matches raw CUDA, so M8's main premise evaporated); the "~4× tuning ceiling" was arithmetic on a DRAM-bound model that M10 disproved by making the kernel *slower* while cutting index storage 21×; and the variable-`D` memory row below should read 18 B/voxel, not 16, now that full-grid `D` is the decided storage.

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
| matrix-free, variable `D` (+node `D`) | 16 → **18 measured**, see M9 | ~1147³ | ~1110³ |
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

End-to-end projections at 800³ from measured shares (SpMV replaced at 15.8 ms / at a tuned ~7 ms; everything else unchanged — these are projections, to be replaced by Phase 2 measurements). **Outcome: the first column of projections held and the second is unreachable.** Measured at 800³, GPU default came in at 144.3 s against the projected ~144, and preconditioned at 17.2 s against the projected ~18.1 — both slightly better than projected. The "tuned apply" column never materialised, because M8 found no tuning left to do; see the Final report.

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

> **Void.** The production KernelAbstractions kernel measures 15.4 ms at 800³, at or below the raw-CUDA prototype's 15.79 ms — there is no gap to close, no CUDA-specialized apply was written, and `ext/TortuosityCUDAExt.jl` gained nothing on this path. The section is kept because the reasoning was sound given the prototype measurement; only the premise was wrong. A plausible explanation is that the prototype's advantage came from launch parameters the KA version happens to match, but the campaign did not chase it once the portable kernel won.

The numbers above are from a **raw CUDA** kernel. A line-for-line KernelAbstractions port (the pattern `assembly.jl` uses, and what a portable production kernel would be) produced **bit-identical output** but measurably slower launches: 1.18× at 400³, **1.36× at 800³** (20.57 ms vs 15.10 ms — still 1.38× faster than CUSPARSE). For `assembly.jl` this overhead never mattered (setup is 0.19 % of e2e); the apply runs hundreds to thousands of times per solve, so here it does. Three ways out, to be settled by measurement in M8: shrink the KA overhead in place (the usual suspects are `@index(Global, NTuple)`'s per-thread division arithmetic and the dynamic ndrange — static ranges and manual index math inside a KA kernel are both legal), keep the KA kernel as the portable path and add a CUDA-specialized kernel in `ext/TortuosityCUDAExt.jl` (the exact precedent the CUSPARSE `mul!` override already sets), or accept 1.38× if tuning erases the gap anyway. Do not accept the naive port without measuring — a third of the apply win is at stake.

## Non-negotiable constraints

1. **Golden values are frozen.** `GOLDEN_STEADY` and `GOLDEN_VARIABLE_D` in `test/test_regression_golden.jl` never change. They exercise the default (assembled) path, which this campaign does not touch behaviorally, so they cannot legitimately move. Matrix-free results are verified *against* them at the same tolerances via new tests.
2. **The suite floor is 11576 assertions, green, GPU included.** Never weaken, skip, or delete a test to make a change pass. New parity tests only add.
3. **The assembled path stays default and behaviorally untouched.** JOSS review is in flight; `solve(sim.prob, KrylovJL_CG())` on an unmodified construction must produce today's numbers. Refactors that share helpers (`_edge_weight` etc.) are fine; anything that changes assembled output is a blocker.
4. **Both paths are first-class, permanently.** Amin's directive. The assembled path is not deprecated, not test-only, not "the reference implementation" — it is a supported production path (and the only one CUSPARSE-backed). The first draft's line that `PortableSparseCSC` "stops being the production path" is superseded.
5. **`Int32` overflow work stays out of scope** (Amin's standing deferral) — but note the operator narrows the exposure rather than widening it: its binding index is `nnodes` (~1630³) instead of `nnz` (~856³). **~~Standing deferral~~ LIFTED 2026-08-09**, after this campaign closed: both paths now guard the bound rather than launching past it — `_operator_index_type` for the operator (landed in this campaign) and `_assembled_index_type` for `build_steady_system` (landed in the matrix-path review follow-up). Neither has a 64-bit device index path; both widen on the host and raise on the device. Do not read this constraint as licence to leave a new `Int32` wall unguarded.
6. Unattended-run rules from campaign 1 apply verbatim: blockers are logged and skipped, never decided unilaterally; failures are reverted, never left red; goldens are never updated unattended.

## Optimization inventory — M-series

*(Written before execution. For how each item actually ended, see the disposition table in the Final report — M10 was rejected by measurement, M12 stayed out of scope, everything else is done.)* Statuses are all `pending`; est. gains marked **measured** come from this session's prototypes, everything else is arithmetic or projection. Maintain this table exactly as campaign 1 maintained its inventory: add discoveries with fresh ids, correct estimates with measurements, retire honestly, re-rank as the bottleneck moves.

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

## Final report

**Outcome: the campaign's primary case — scale — is delivered and certified, and the secondary case — speed — beat its own plan.** Both paths remain first-class; the assembled path's default behaviour did not move.

### What the operator costs and buys, measured at HEAD

| quantity | assembled | matrix-free |
| --- | --- | --- |
| bytes per grid voxel, default solve | 40.3 (measured, campaign 1) | **14.16 (measured at 1000³)** |
| largest cube certified on a 23.889 GiB card | ~850³ (structural ceiling) | **1000³ with 9.42 GiB spare; 1100³ preconditioned with 0.00** |
| GPU apply, 800³, Float32 | 30.4 ms (CUSPARSE CSR) | **15.4 ms (1.97×)** |
| GPU apply, 600³ / 400³ / 200³ | 11.8 / 3.50 / 0.373 ms | 6.98 / 2.13 / 0.255 ms (1.69× / 1.64× / 1.46×) |
| CPU apply, 200³, 20 threads, Float64 | 36.7 ms (`SparseArrays`) | **5.42 ms (6.76×)** |
| variable-`D` apply, 600³ | 11.9 ms | 11.2 ms (1.06×) |
| apply parity | — | 1.0e-7 Float32, sub-ULP Float64 |

### The benchmark table

Quiet card, GPU, Float32, seed 42, ε 0.5, `reltol=1e-6`, `bench/results/matrixfree.csv` (254 rows, regenerated from empty). `e2e` sums build + preconditioner + solve + post; `peak` is the maximum over those stages.

| N | apply (ms) | | solve e2e (s) | | preconditioned e2e (s) | | peak, preconditioned (GiB) | | iters (plain / precond) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | asm | **mf** | asm | **mf** | asm | **mf** | asm | **mf** | both paths |
| 200³ | 0.38 | **0.22** | 0.74 | **0.63** | 0.30 | **0.28** | 1.78 | **1.53** | 1044 / 82 |
| 400³ | 3.44 | **2.11** | 14.33 | **11.32** | 2.42 | **2.26** | 4.50 | **2.47** | 2094 / 168 |
| 600³ | 11.90 | **6.96** | 66.26 | **52.74** | 9.27 | **7.91** | 11.97 | **5.03** | 2983 / 223 |
| 800³ | 29.10 | **15.74** | 195.09 | **144.27** | 21.15 | **17.22** | 22.44 | **9.94** | 3620 / 202 |
| 1000³ | *fails* | **29.19** | *fails* | **368.86** | *fails* | **66.65** | *fails* | **18.89** | — / 4805, 363 |
| 1100³ | — | **37.40** | — | **538.52** | — | **79.62** | — | **23.89** | — / 5059, 298 |

**Iteration counts are identical between the two paths at every size where both run** — two different operators reaching the same Krylov trajectory. τ agrees to 4–5 significant figures. At 800³ the matrix-free path is **26 % faster end to end unpreconditioned, 19 % preconditioned, on 44 % of the peak memory**; the plan projected −24 % and −12 %, so both came in slightly better.

The 1000³ and 1100³ *assembled* cells say **fails**, not "out of memory": the assembled path aborts with `ERROR_ILLEGAL_ADDRESS` from the unchecked `Int32` overflow described in the Progress log, at a point where it has allocated only 11.6 GiB of a 23.9 GiB card. That is the campaign's scale case stated as plainly as it can be — the ceiling is not memory, it is an overflow that faults.

The 14.16 B/voxel figure lands on the plan's predicted 14 — the memory model was right. The 1.97× apply beat the plan's 1.38× working assumption because **the KernelAbstractions port gap the plan budgeted M8 to close does not exist**: the portable kernel matches the raw-CUDA prototype. No CUDA-specialized apply was written, and `ext/TortuosityCUDAExt.jl` was not touched.

### What the campaign learned that the plan had wrong

1. **The apply is not bandwidth-bound.** The plan projected a ~4× tuning ceiling from a 259 GB/s effective bandwidth against 4.1 GB of compulsory traffic. M10 tested that directly — a bitmask-plus-rank index at **0.19 B/voxel instead of 4.00**, a 21× cut in index storage with bit-identical output — and it ran **21–25 % slower**. The reads were already being served from cache. Counting the accesses that actually happen puts the kernel near 1.35 TB/s of cache traffic at 800³, which is the real wall. The ~4× ceiling was arithmetic on a wrong model.
2. **The one real tuning win was a missing `@Const`.** Variable `D` was *losing* to CUSPARSE at 600³ (0.40×) with an overhead that grew with size — a cache cliff. One word fixed it: 29.81 → 11.22 ms, 2.66×.
3. **Bit-identical apply is not achievable in one pass** and was never needed; the departure is 4.8e-17 and the tests assert 1e-15, tighter than the protocol's 1e-14.
4. **CUSPARSE needs a ~20-call warm-up.** A three-call warm-up reports the assembled path 30× slower than it is. Every comparison here uses `warmup=25, reps=15, median`.
5. **`reltol` is a request, not a guarantee, at half a billion Float32 unknowns.** At 1000³ CG returns `Success` at `reltol=1e-6` while the recomputed true residual is 3.11e-5. Decision 5 anticipated exactly this; it is now measured, and `certify_frontier.jl` prints it on every run.

### Inventory disposition

| id | status |
| --- | --- |
| M1 operator + KA apply | **done** — `a994580` |
| M2 workspace hook | **done** — `a994580` |
| M3 RHS kernel + constructor | **done** — `a994580`, `b` exactly equal to the assembler's |
| M4 `matrixfree` keyword + solve entry point | **done** — `e08f606`; resolves B23 and B24-as-default |
| M5 two-level preconditioner port | **done** — `0cbf7ad` |
| M6 parity and edge-case tests | **done** — 823 operator + 384 preconditioner assertions |
| M7 bench harness | **done** — `0cbf7ad` (`matrixfree_bench.jl`, `certify_frontier.jl`) |
| M8 apply tuning | **done** — Phase 3 closed on `PHASE 3 COMPLETE: 0 accepted candidates` after two audit rounds |
| M9 variable `D` + storage decision | **done** — full-grid `D` kept, node-compaction rejected at 1.05–1.07× |
| M10 compressed `idx` | **rejected** — 21× less index memory but 0.79× the speed |
| M11 CPU threaded apply | **done** — 6.76× at 200³ |
| M12 transient operator | **out of scope** by Decision 3 |
| M13 docs | **done** — operator and entry-point docstrings, README two-path section, `CHANGELOG.md` `## Unreleased` entry |

### Defects found by review and fixed

The independent review of the diff found two real defects, both since fixed and covered by tests (`4584b28`):

- **`mul!` did no dimension checking.** The kernel body is `@inbounds`, so a short `y` was written past its end and the call returned quietly, where the assembled path raises `DimensionMismatch`. A memory-safety hole, not an accuracy one.
- **`_free!` leaked the device `D` copy the package itself allocated.** The operator holds `D` for its whole life, so the constructor deliberately skips freeing it — and nothing else ever did. `MaskedLaplacian` gained `owns_D`.

Fixing the second one also proved a neighbouring comment false: `_gpu_adapt` allocates a fresh device array even for a `CuArray` that is already the right element type, so `D_dev === D` holds on the CPU path alone. The comment is corrected; no behaviour depended on it.

A third finding — `solve(sim)` failing for `matrixfree=true` above `_PRECOND_MIN_NODES` — was real between `e08f606` and `0cbf7ad` and is closed by M5. It had no test, which is why nothing caught it; there is one now, at 170k nodes.

### Second review pass — one medium, four low, all fixed

A second review found nothing wrong with the kernel mathematics (it re-derived the apply against `build_steady_system` branch by branch and the coarse stencil against the stored-entry kernel, `-0.0` included) but did find five real defects, all since fixed:

- **Medium — the operator had the same unguarded `Int32` as the assembled path.** `Ti = (on_gpu || nnodes + 1 <= typemax(Int32)) ? Int32 : Int` bound-checks the host branch and short-circuits past the check on GPU. Past `nnodes > typemax(Int32)` the `cumsum!` into `idx` wraps to `typemin` — Julia's `Int32` does not saturate — so `c0 > 0` goes false for most voxels and `y` comes back partly unwritten, with no error. Unreachable on this 24 GiB card (it needs ~1625³, about 56 GiB at 14 B/voxel) but reachable on an 80–96 GiB one, which is exactly the regime the operator exists for. Now `_operator_index_type` widens to `Int` on the host and **throws** on GPU, tested through the predicate since the image cannot be built here. This is the one place the campaign's own new code repeated the defect it had logged against `build_steady_system`.
- **Low** — the large-`solve(sim)` test asserted *exact* PCG iteration parity while the same file documents the two applies as deliberately not bit-identical; a residual near the tolerance can cross it one iteration apart, so it was a tripwire for floating-point reassociation rather than for behaviour. Relaxed to `≤ 1`, matching the plan's own "within a few" acceptance criterion and the sibling assertion in `test_matrixfree_precond.jl`.
- **Low** — `certify_frontier.jl` accepted any existing fixture without checking its size and wrote straight to the final path, so an interrupted 1 GB write wedged the script permanently. Now checks `filesize == n^3` and writes to a `.partial` file it renames.
- **Low** — `matrixfree_bench.jl` validated flag *values* but never flag *keys*, so `--path=matrixfree` was silently ignored and the sweep ran on defaults. This is the harness that produces the README numbers. It now errors and lists the valid keys.
- **Low** — the `Int32` skip row used a stage name of its own, which is never a pass's terminal stage, so `completed_cells` never marked the cell settled and every resume re-attempted it and appended a duplicate. It now writes the pass's terminal stage, and for `apply` its last repeat.

### Carried forward

- **B17** (CPU SpMV via type change) should be re-triaged to rejected-unless-asked: M11 delivers more without touching any `SparseMatrixCSC` public.
- **M10** remains the right tool for anyone who needs past ~1150³ and will trade ~25 % of the apply for it — it would take the default configuration to 10.2 B/voxel and the ceiling to ~1258³. Decision 4 makes 1200³ conditional on M10 landing, so 1200³ was not attempted.
- The plan's memory table row for variable `D` should read **18 B/voxel**, not 16, now that full-grid `D` is the decided storage.
- **Flag for Amin:** Decision 2's "never a shadowed LinearSolve name" was implemented as a *method on* LinearSolve's `solve` generic rather than a new name, because a bare `function solve(...)` inside the module would have shadowed the re-export and broken `solve(sim.prob, alg)`. One-line rename if a distinct name was intended.

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
2026-08-09 — **M7 trap, closes an F13-shaped hole the campaign nearly repeated** — `bench/` had **no `Manifest.toml`**, so `--project=bench` resolved `Tortuosity v0.0.5 from the registry` rather than the working tree. A bench run in that state silently measures the *released* package. It was fixed with `Pkg.develop(path=<repo root>)` in the bench environment, and the 1000³ certification is unaffected — it ran after the fix, and `matrixfree=true` does not exist in v0.0.5, so a registry resolution would have errored rather than lied. But `Manifest.toml` is gitignored, so **a fresh clone, or deleting `bench/Manifest.toml`, reintroduces this silently**. Any future run must assert `pathof(Tortuosity)` points into the repo before trusting a number; that check belongs in both bench scripts.
2026-08-09 — **M7 measurement hazard — do not wrap the apply timing in the peak-memory sampler.** With the 1 kHz device-memory sampler running, the 200³ assembled apply measures 1.98 ms; with the sampler moved off the timed region it measures 0.395 ms, matching the independently quoted 0.37 ms. The sampler now covers construction plus the 25 warm-up applies, which reach the same steady-state usage because an apply allocates nothing. Sits alongside the CUSPARSE warm-up finding as the second way this campaign's numbers can be quietly wrong by an order of magnitude.
2026-08-09 — M7 small-size results (both paths, GPU, seed 42, ε 0.5) — iteration counts **identical** on every configuration (200³: 1044 = 1044 unpreconditioned, 82 = 82 preconditioned), τ agreeing to ~5 significant figures. 200³ apply 0.395 ms assembled vs 0.215–0.29 ms matrix-free; setup 0.014 s vs 0.005 s; peak device 1.78 GiB vs 1.53 GiB. CPU 64³, 20 threads: apply 1.012 → 0.299 ms, solve 1.101 → 0.561 s, preconditioned 0.530 → 0.272 s.
2026-08-09 — **The assembled path does not fail cleanly past its Int32 wall — it corrupts memory.** The bench sweep completed 200/400/600/800/1000³ on **both** paths (369 CSV rows) and then died at 1100³ assembled, in the apply pass, with `CUDA error: an illegal memory access was encountered (code 700, ERROR_ILLEGAL_ADDRESS)` surfacing asynchronously in `cuMemGetInfo`. Cause: `build_steady_system` selects `Ti = Int32` **unconditionally when `on_gpu`** (`assembly.jl`: `Ti = (on_gpu || 7 * nnodes + 1 <= typemax(Int32)) ? Int32 : Int`), and at 1100³ `7 * nnodes` is 4.66e9 against a `typemax(Int32)` of 2.15e9. The plan predicted this wall at ~856³ and treated it as a *capacity* limit; it is worse than that — the overflow produces out-of-range offsets and a faulting kernel, not an error message. This strengthens the campaign's scale argument (the matrix-free operator's binding index is `nnodes`, ~1630³) and it is a **pre-existing assembled-path defect, untouched by this campaign** and out of scope by constraint 5. It deserves its own issue: at minimum `build_steady_system` should raise when `7 * nnodes + 1 > typemax(Int32)` on GPU rather than launch.
2026-08-09 — bench sweep, **status: rows collected but not quotable.** 369 rows cover 200–1000³ on both paths, but the run overlapped the 1100³ certification and the full test suite on the same card, and the harness's author documents that under contention `peak_dev_bytes` goes negative and apply medians swing 10×. The **contention-independent** contents are trustworthy — iteration counts, τ, `nnz`, retcodes, and the 1100³ assembled crash. The timings are not. `bench/results/matrixfree.csv` must be deleted and the sweep re-run on a quiet card before any number from it is published.
2026-08-09 — **Suite gate: PASSED.** Foreground fresh-process `Pkg.test()` at the final HEAD: **12717 assertions, 0 failures, 0 errors, 3m51s, exit 0**, GPU included — against the campaign floor of 11576, so this campaign adds **1141 assertions** and breaks nothing. Goldens unmoved throughout. (Three gate runs were taken as the branch settled: 12677 before the first review's fixes, 12708 after them, 12717 after the second review's.)
2026-08-09 — **Benchmark sweep: complete, on a quiet card, both paths, 200–1100³.** `bench/results/matrixfree.csv` was deleted and regenerated from scratch (254 rows, exit 0 on both invocations). Iteration counts are **identical between the two paths at every size** — 1044 / 2094 / 2983 / 3620 unpreconditioned and 82 / 168 / 223 / 202 preconditioned at 200–800³ — which is the strongest correctness signal the campaign has: two different operators, two different arithmetic orders, the same Krylov trajectory. τ agrees to 4–5 significant figures throughout. The table is in the Final report.
2026-08-09 — **Assembled at 1100³, re-run alone on a quiet card: same fault, no row.** Exit 1, `ERROR_ILLEGAL_ADDRESS` raised from the apply pass, and the harness writes **nothing** — an asynchronous CUDA fault poisons the context, so no `try` in the measurement path can catch it and no row survives. (At 1000³ the same overflow surfaces early enough to be caught and is recorded as an `ERROR` row; at 1100³ it is not.) The harness now **guards** the case: when `7 * nnodes + 1` exceeds `typemax(Int32)` on GPU it skips the assembled path and emits a `skipped` row noting the reason, so `bench/matrixfree_bench.jl 200 400 600 800 1000 1100` completes in one invocation instead of dying part-way. The guard lives in the bench harness, not in `src/` — constraint 5 keeps `Int32` work out of the package, and the underlying `build_steady_system` defect is logged above for its own issue.
2026-08-09 — **Correction to the 1100³ certification's timing.** The certification reported a 314.5 s solve; the clean sweep measures the same configuration at **74.9 s (79.6 s end to end)**. The certification ran while the bench sweep and the full test suite were both on the card, so its *timing* was contaminated — exactly the hazard the harness's author flagged. Its *memory* figure is unaffected and reproduces: peak **23.889 GiB, the entire card**, in both runs. So the caveat about 1100³ narrows: the "1.05 s per iteration" observation was an artefact of contention (the clean figure is 0.251 s/iteration, most of it the preconditioner's per-iteration host round-trip), but **zero headroom and the −0.0004 concentration stand**. 1000³ remains the number to quote.
2026-08-09 — **Phase 4 independent review — verdict: constraints hold, two defects open.** Reviewed `37c0404..976d5f4` (so *before* `0cbf7ad`), re-derived the stencil from `assembly.jl` and ran ~100 extra parity configurations: 2-voxel-thick axes, single-pore-voxel images at inlet/interior/outlet, `1×4×4`, random non-cubic shapes on all three axes, uniform and variable `D`, Float32/Float64 mixing. **No stencil defect found** — `b` exactly equal in every case, apply ≤1e-13 relative. Goldens 23/23, no existing test modified, `runtests.jl` purely additive, the `_free!` guard provably reduces to the old expression when `matrixfree=false`, `solve(sim.prob, KrylovJL_CG())` still dispatches to LinearSolve's own method.
2026-08-09 — **OPEN DEFECT 1 — `mul!` does no dimension checking** (`src/matrixfree.jl:168-186`). The kernel body is `@inbounds` and nothing validates `length(y)`/`length(x)` against `nnodes`, so `mul!(zeros(184), op, randn(216))` writes 32 elements past the end and returns quietly. The assembled path raises `DimensionMismatch` for the same call. This is both a divergence from the executable specification and a memory-safety hole. Fix: a `DimensionMismatch` check in the 5-argument `mul!` before the launch, plus tests for both the short-`y` and short-`x` cases. **Must land before this branch merges.**
2026-08-09 — **OPEN DEFECT 2 — `_free!(::MaskedLaplacian)` leaks the package's own `D` copy** (`src/matrixfree.jl:59-61`). The comment claims `D` belongs to the caller; that is false on the constructor path, where `simulations.jl` deliberately skips freeing `D_dev` because the operator holds it — so with `gpu=true, matrixfree=true` and a host `D`, the operator owns a package-made, grid-sized device array that `_free!` never releases. Verified on a 60³ variable-`D` GPU sim. Fix: have the constructor record ownership (or free `D` when the operator made the copy) and correct the comment.
2026-08-09 — review defect 3, **already resolved** — the reviewer found `solve(sim)` broken for `matrixfree=true` at `976d5f4`, because `_PRECOND_MIN_NODES` routes every ≥100k-node problem into `two_level_preconditioner`, which began with `nnz(A)` and raised `MethodError` on a `MaskedLaplacian`. The M5 commit `0cbf7ad` adds the operator method and lands the combination; the reviewed range predates it. A committed test for that exact combination at ≥100k nodes is still missing and should be added with the defect-1 fix.
2026-08-09 — review, lesser items logged not fixed — no `mul!` for `A'`/`transpose`, and `issymmetric` errors (irrelevant to CG); `solve(sim; precond=P, Pl=Q)` silently lets `Pl` win via the duplicate splatted keyword, and `precond` is unvalidated; `show(::SteadyDiffusionProblem)` gained `, assembled` on the default path — user-visible, though nothing asserts the old string; the two apply-parity testsets are the same loop at 1e-14 and 1e-15, so the looser one cannot fail alone (redundant, not vacuous).
2026-08-09 — **1000³ certified** (fresh process, `bench/certify_frontier.jl 1000 --no-precond`) — `nnodes` 498 957 533, ε 0.4990 — peak device **14.468 GiB of 23.889, 9.421 GiB headroom**, base 1.281 GiB → **14.16 bytes per grid voxel, against the plan's predicted 14** — setup 7.808 s, solve **373.6 s, 4805 iterations, retcode `Success`**, τ = 1.894130, `u` mean 0.4820, extrema [0.0000, 1.0002]. Reproduces the planning-session prototype (381.4 s, same 4805 iterations) at HEAD, from a committed script.
2026-08-09 — **1100³ preconditioned certified, but at exactly zero headroom** (fresh process, `bench/certify_frontier.jl 1100`) — `nnodes` 665 593 077, ε 0.5001 — setup 9.825 s, preconditioner 3.379 s, solve **314.5 s, 298 iterations, retcode `Success`**, τ = 1.851586. Peak device **23.889 GiB of 23.889 — headroom 0.000**, i.e. 18.24 bytes per grid voxel against the plan's predicted 17. Three caveats that make this a *reached* frontier rather than a *comfortable* one, and all three should be stated wherever the number is quoted: (a) the card was completely full, so this does not survive a slightly denser image, a second process, or a smaller card; (b) the solve took 1.05 s per iteration where the apply alone should be ~40 ms, which is what running the allocator at 100 % occupancy costs; (c) `u` extrema are **[-0.0004, 1.0002]** — the first physically out-of-range value the campaign has seen, against [0.0000, 1.0002] at 1000³ — and the true residual is 9.54e-6 against a requested 1e-6, the same ~10× drift. **The defensible headline is 1000³; 1100³ is reached, not comfortable.** Decision 4's 1100³ target is met on the letter and worth re-reading in light of (a)–(c). The plan's ≥2 GiB-headroom column is the honest one to quote, and by this measurement it sits nearer 1060³ than 1090³.
2026-08-09 — **Decision 5's true-residual check fired.** At 1000³ the solve returns `Success` at `reltol=1e-6` but the **true relative residual is 3.11e-5** — a 31× drift between the residual CG tracks recursively and the one you get by recomputing `b - A·u`. This is precisely the failure mode Decision 5 named, and it is now measured rather than hypothesised. It does not invalidate the result: the field is physically sane and τ is unaffected at the 3 significant figures the campaign quotes. It does mean **`reltol` at half a billion Float32 unknowns is a request, not a guarantee**, and anyone needing a certified residual at that size must check it explicitly — which is why `certify_frontier.jl` recomputes it and prints it every run. Not a blocker; recorded as a property of Float32 CG at this scale.
2026-08-09 — Phase 3, **AUDIT ROUND 1: 4 candidates surfaced, 1 accepted.** Candidates: (a) `@Const` on the kernel's `D` argument — **accepted**, 2.66× on the variable-`D` apply at 600³, landed as `976d5f4` under M9; (b) launch-configuration sweep, 8 configurations at 400³ and 800³ — **rejected**, the current `(64,4,1)` is at the optimum; the best alternative `(32,8,1)` is +3.9 % at 800³ and −0.5 % at 400³, which does not justify diverging from `assembly.jl`'s shared convention, and every configuration produced bit-identical output; (c) node-compacted `D` — **rejected**, 1.05–1.07×, see the M9 entry; (d) M10 compressed `idx` — **rejected by measurement**, see below.
2026-08-09 — M10 — rejected — would save 4.00 → **0.19 B/grid-voxel** (a 21× cut in index storage, measured on a real prototype, output bit-identical) — but it is **slower: 0.75× at 400³, 0.79× at 800³** — no commit — self-verified. A pore bitmask plus a per-64-voxel exclusive prefix count, with the pore ordinal recovered as `pre[b] + popcount(word & mask)`. The result is the campaign's most informative negative: **the apply is not index-bandwidth-bound.** Trading DRAM traffic for two loads plus popcount/shift arithmetic loses, which means `idx` reads were already being served from cache. As a memory lever it is still real — it would take the default configuration from 14 to 10.2 B/voxel and the preconditioned ceiling from ~1124³ to ~1258³ — so it is the right tool for anyone who needs past ~1150³ and will pay ~25 % of the apply for it. It is not needed for this campaign's targets, and Decision 4 makes 1200³ conditional on it landing, so 1200³ is not attempted.
2026-08-09 — Phase 3, **AUDIT ROUND 2: 2 candidates surfaced, 0 accepted.** (a) Interior fast path — hoist the six per-neighbour bounds tests into one `(1 < i < nx) & (1 < j < ny) & (1 < k < nz)` test, with the boundary voxels taking the checked path: **rejected, 0.91× at 400³ and 0.97× at 800³**, bit-identical output. The branch divergence and the duplicated loop body cost more than the comparisons they remove. (b) Shared-memory halo tiling for `idx` reuse (each `idx` value is read by seven threads): **rejected without implementation**, on the bracketing evidence of round 1(d) and round 2(a) — a candidate that removed memory traffic and added arithmetic lost 21–25 %, and a candidate that removed arithmetic and added divergence lost 3–9 %. The kernel sits in a local optimum between those two failures, and tiling adds *both* shared-memory traffic and a barrier. Recorded as the one untried idea should a future round want it.
2026-08-09 — **PHASE 3 COMPLETE: 0 accepted candidates.** M8 ends here. The apply stands at **15.4 ms at 800³, 1.97× CUSPARSE CSR**, against the plan's entry expectation of 1.38× for a portable KA kernel. The plan's "~4× tuning ceiling" projection — derived from a 259 GB/s effective bandwidth against 4.1 GB of compulsory traffic — **does not survive contact**: that arithmetic assumed the kernel was DRAM-bound, and M10 disproved it directly. Counting the reads that actually happen (seven `idx` per grid voxel, six `x` gathers per pore voxel) puts the kernel at roughly 1.35 TB/s of cache traffic at 800³, which is the wall it is actually against. The honest ceiling is close to where the kernel already is.
2026-08-09 — design ruling, apply summation order — the plan asked for a bit-identical apply. It is not achievable in one pass: a CSC `mul!` left-associates all seven terms of a row in ascending column order, but the diagonal sits between the lower and upper halves and its value is only known after both have been walked, so the upper half must be summed on its own and folded in as a single term. Exactness would cost either a second full pass over `idx` (the kernel is bandwidth-bound on exactly those reads) or a source-level unroll of the upper loop. Measured departure: **4.8e-17 relative, sub-ULP**. The test asserts `rtol=1e-15`, an order tighter than the verification protocol's own 1e-14 requirement; `b` remains **exactly** equal. No test was weakened — the claim was corrected before it ever landed green.
2026-08-09 — **The `build_steady_system` Int32 defect logged above is fixed; constraint 5's deferral is lifted.** The post-merge code review of the *matrix-path* campaign (`23d0704..443814d`) reached the same defect from the other direction and Amin lifted the deferral rather than carrying it: 307M pore voxels is a 900³ blob at half porosity, inside the package's advertised range, so the wall is reached in ordinary use. `_assembled_index_type` now applies the `7 * nnodes` bound to both branches — host widens to `Int`, device raises and names `matrixfree=true`. So the entries above that read "it deserves its own issue" and "logged above for its own issue" are **closed in `src/`, not in the tracker**; the bench harness guard from `b48fa44` is now belt-and-braces rather than the only guard. What remains open is only the 64-bit *device* index path, which would let the assembled path actually run past the wall instead of refusing, and which is filed as its own issue.
