---
title: GPU preconditioner determinism
created: 2026-08-16
updated: 2026-08-16
status: complete
outcome: τ is bit-identical across repeats (spread exactly 0.0, not merely small); wall-clock improves at 400³ and costs 4–8% at 200³, which is an open call for Amin.
branch: main
supersedes: "-"
superseded-by: "-"
related: 2026-08-16-coarse-space-mesh-independence.md
---

> **Status: complete (executed 2026-08-16).** Repeated GPU solves of the same image returned a different τ every run, because the two-level preconditioner's restriction accumulated floats through `Atomix.@atomic` on every CG iteration. Inverting the aggregation once at build time into a coarse→fine adjacency, and gathering over it in fixed order, removes the atomic. **τ is now bit-identical across three repeats on all four spot-check cases — spread exactly `0.0`** — with iteration counts unchanged and `Pkg.test()` green at 12717 / 12717.
>
> **Outcome:** D1, D2, D3 done · D4 **not needed** (the build-time atomics turned out not to be τ-visible) · D5 **resolved by reasoning, no change** · P1, P2 done. The wall-clock bar was met at 400³ (−7.7% and +0.4%) and **missed at 200³ (+7.9% and +4.3%)**, entirely because of the host-side inversion the fix adds. That trade-off is deliberately left open for Amin rather than resolved unilaterally — see [Open items](#open-items).
>
> **How to read this document.** Sections down to *Fix direction* are the plan **as handed over on 2026-08-16**, kept so the reasoning survives. From *Work items* onward is what actually happened. Where execution contradicted the design, the contradiction is marked inline rather than silently overwritten — there are three, and two of them are the interesting part of this document.

# GPU preconditioner determinism plan

This plan was written from the JOSS benchmark campaign (200³–800³, both devices), measured on `pmeal-hpc` (2× Quadro RTX 8000, Xeon Silver 4110, 250 GB RAM). Execution ran on the local **RTX PRO 5000 Blackwell Laptop, 23.9 GiB**.

It is Phase 1 of two. Phase 2 — the coarse space is not mesh-independent, which is what costs us ground against taufactor as problems grow — is [`2026-08-16-coarse-space-mesh-independence.md`](2026-08-16-coarse-space-mesh-independence.md) and was deliberately **not** started here.

## The defect, as measured in the campaign

Repeated runs of the *same* GPU solve on the *same* image returned different τ. The CPU path was bit-identical across repeats; so was the GPU path with the preconditioner disabled. **The preconditioner was the sole source.**

Five repeats per image, `reltol=1e-6`, four ε≈0.2 geometries. Spread is `(max τ − min τ)/median τ`:

| path | N=100 | N=200 | N=300 | N=400 |
|---|---|---|---|---|
| matrix-free + two-level preconditioner | 0.008% | 0.019% | 0.035% | **0.094%** |
| assembled, **preconditioner off** | 0.000% | 0.000% | 0.000% | 0.000% |

It grew with N and did not level off. At 600³ the three ε≈0.2 cases recorded `tau_spread` of **2.27e-3 to 2.75e-3** — larger than the **1e-3** target the benchmark selects on, so those cases could not reliably reach the target however long they iterated; one burned the full 20 000-iteration ladder chasing a target inside its own noise band.

Convergence, GPU matrix-free: 15/15 at 200³ → 14/15 at 400³ → **12/15 at 600³ and 800³**. CPU was 15/15 at every size 200³–800³.

**Why this was urgent rather than cosmetic:** the noise floor was on track to exceed the accuracy target for most cases at 1000³. Left unfixed, the outcome is not "results wobble slightly" but "we cannot report GPU numbers at large sizes at all."

## The cause

Float accumulation through `Atomix.@atomic`. Thread-block arrival order is not fixed between launches, so a non-associative float sum gives a different answer each run.

| site | line (pre-fix) | runs |
|---|---|---|
| `_restrict_kernel!` — `rc[a] += x[i]` | `src/preconditioner.jl:535` | **every CG iteration** |
| `_coarse_stencil_kernel!` | `:138`, `:142` | once, at build |
| `_coarse_grid_stencil_kernel!` | `:207`, `:219` | once, at build |
| `Atomix.replace!` CAS loop | `:153`–`:156` | once, at build |

`:535` was identified as the one that matters, being in the application path. The confirmation sat directly below it: the CPU overload `_restrict!` is a plain serial loop with **no atomic** — exactly why CPU results were bit-reproducible and GPU ones were not.

## Fix direction

Invert `agg` **once at build time** into a coarse→fine adjacency (CSR-style: sorted `fine` ids plus per-coarse-cell offsets), then have each coarse cell **gather** its contributions in a fixed order — one thread per coarse cell, no atomics, deterministic. The ordering cost is paid once instead of every iteration.

**Keep the signatures of `_restrict!` and `_prolong!` unchanged.** Phase 2 calls them per level in a recursive V-cycle, and a stable interface is what lets Phase 2 build on this instead of reworking it.

---

## Work items

Statuses: `pending`, `done`, `rejected`, `not-needed`, `deferred`.

| id | item | status | result |
|---|---|---|---|
| D1 | Invert `agg` at build time into a CSR coarse→fine adjacency | **done** | new `Aggregation` type; verified against a serial reference |
| D2 | Rewrite `_restrict_kernel!` as a fixed-order gather | **done** | atomic removed from the application path |
| D3 | Preserve the signatures of `_restrict!` / `_prolong!` | **done** | new methods dispatch on `Aggregation`; no parameters added |
| D4 | Make the build-time stencil kernels deterministic too | **not-needed** | τ is bit-identical without touching them — see below |
| D5 | Check whether the `Atomix.replace!` CAS loop is order-sensitive | **done (no change)** | `max` is associative and exact, so it is order-insensitive by construction |
| P1 | Thread the inversion without making it schedule-dependent | **done** | chunked counting sort, 315 ms → 180 ms at 400³ p095 |
| P2 | Give the gather an explicit workgroup size | **done** | the backend default was 3× worse at 200³ |

### D1 / D2 / D3 — what was built

`Aggregation` carries the forward map plus its inverse (`fwd`, `offsets`, `fine`; each cell's slice ascending). It is built **after** the remap, so the adjacency is indexed by the coarse numbering that survived rather than the one the blocks started with. `P.agg` holds one on device; the host path keeps the plain vector, because its `_restrict!` is a serial loop that was already reproducible and already faster than the scatter (the comment at `_restrict!` records that measurement).

Because the adjacency travels inside `agg` rather than as extra parameters, `_restrict!(rc, agg, x)` and `_prolong!(y, agg, xc, x, inv_lambda)` keep their exact signatures, as Phase 2 requires.

Correctness was checked against a host reference before any timing: gather matches to `0.0`, the adjacency covers every mapped node, every cell's slice is ascending, and every node is filed under its own cell.

Two consequences worth knowing:

- The generic GPU `_restrict!` scatter method is **gone**, not merely bypassed. A GPU call with a raw vector `agg` now raises `MethodError` rather than silently running the nondeterministic path. That is deliberate.
- `TwoLevelPreconditioner`'s `Vi` type parameter lost its `<:AbstractVector` bound, since `Aggregation` is not an array.

### D4 — the build-time atomics did not need fixing. **This contradicts the design.**

The handover expected the build-time sites to be second-order but still real, and the same inversion was offered as their fix. In execution they turned out **not to be τ-visible at all**: `:138`, `:142`, `:207` and `:219` were left exactly as they were, and τ is still bit-identical.

That is not an argument that they are harmless in principle — they are order-sensitive float sums, and the coarse stencil is `Float64`. The most likely reason it does not surface is that the coarse correction is narrowed to `Float32` when it is copied back to the device, which absorbs perturbations that small.

The obvious objection is that the four spot-check cases all run a **uniform** diffusivity, where the edge weights being summed are equal and the sum is order-independent anyway. So this was tested directly: a **non-uniform `D`** field at 200³ p095, where the weights genuinely differ, is also bit-identical across three repeats, spread `0.0`.

**Do not read this as "the build-time atomics are fine."** Read it as: they were measured, on both uniform and non-uniform `D`, and did not move τ. If a future change narrows less aggressively or widens the coarse correction to `Float64` end-to-end, they become live again.

### D5 — `_atomic_max!` is order-insensitive, by construction

The handover flagged the CAS loop at `:153`–`:156` as "check whether its op is order-sensitive". It is not, and this needs no measurement: `max` is associative, commutative and exact for floats, so any arrival order yields the same result. No change was made.

### P1 — threading the inversion, without letting the schedule into the answer

The inversion is a counting sort on the host. Done serially it cost 315 ms at 400³ p095, which is the single largest cost this fix adds.

It is now chunked: the nodes are split into contiguous ranges, each range counts how many nodes it gives every cell, and those counts are run into a reserved position for each range within each cell **before anything is written**. A chunk then files only into its own slice, in ascending node order, and chunks are laid down in ascending order within each cell.

That reservation is what keeps the result off the thread schedule — the output is bit-identical to a serial reference, verified, not assumed. 315 ms → 180 ms with 20 threads. The speedup is well short of linear because the filing pass is memory-bound on scattered writes.

### P2 — the backend's default launch was leaving 2–3× on the table

There is one work item per coarse cell, and there are far fewer coarse cells than pore voxels, so the workgroup size KernelAbstractions picks for a grid-sized launch leaves the device idle:

| case | default launch | groupsize 128 |
|---|---|---|
| 200³ p020 (nc = 8186) | 0.083 ms | **0.028 ms** |
| 400³ p095 (nc = 29745) | 1.535 ms | **1.333 ms** |

32, 64, 128 and 256 were all within noise of each other at 200³; 128 was best at 400³, so 128 it is.

*Noted and not acted on:* other kernels in the package also launch with the default workgroup size. Whether any of them are in the same position was not investigated — it is outside this plan.

---

## Measurement method — and why the obvious method does not work here

**Cross-session wall-clock on this card is not comparable.** The same code on the same cached image gave **2.09 s, 2.42 s and 2.67 s** for 400³ p095 in three different sessions; 200³ end-to-end repeats ranged 0.207–0.317 s. The card also drives a display, and a session restart resets the CUDA pool. The drift is larger than the effect being measured, so the natural loop — measure baseline, edit, measure again — **cannot resolve anything here and will invent or hide regressions.**

Everything below was therefore measured as a **same-session interleaved A/B**: the pre-fix scatter `ldiv!` was rebuilt in a throwaway wrapper struct and alternated against the new one on the *same* built preconditioner, with the added build step timed separately. Images were cached to disk so before and after ran on byte-identical geometry.

**Isolated kernel microbenchmarks mislead in the other direction** and were not trusted on their own: the gather looks 1.8–2.3× faster than the scatter standalone, but is worth about 1 ms in a real solve, because restriction is only ~13% of `ldiv!` — the host Cholesky solve and the D2H/H2D round trip dominate it.

## Results

### Determinism — the goal, met exactly

τ across three repeats, GPU matrix-free, `reltol=1e-6`:

| case | before | after |
|---|---|---|
| 200³ p020 (ε=0.175, n=1.40M) | 3.588e-4 | **0.0** |
| 200³ p095 (ε=0.952, n=7.62M) | 3.029e-6 | **0.0** |
| 400³ p020 (ε=0.186, n=11.9M) | 4.300e-4 | **0.0** |
| 400³ p095 (ε=0.949, n=60.7M) | 7.035e-6 | **0.0** |

Bit-identical, not close. Non-uniform `D` at 200³ p095 is also `0.0`. Iteration counts are unchanged in every case (145 / 60 / 221 / 95), as they must be — the preconditioner is mathematically the same operator.

### Wall-clock — met at 400³, missed at 200³

Same-session interleaved A/B, per-case minimum of five (200³) or three (400³) alternating repeats:

| case | solve, gather | solve, scatter | inversion added | net | % of solve+build |
|---|---|---|---|---|---|
| 200³ p020 | 88.8 ms | 89.6 ms | 8.2 ms | +7.5 ms | **+7.9%** |
| 200³ p095 | 217.8 ms | 216.7 ms | 11.9 ms | +13.0 ms | **+4.3%** |
| 400³ p020 | 807.7 ms | 898.1 ms | 19.9 ms | −70.5 ms | **−7.7%** |
| 400³ p095 | 1654.4 ms | 1757.7 ms | 110.8 ms | +7.4 ms | **+0.4%** |

Against full case wall-clock (which also carries image upload, RHS build and the τ reduction) the 200³ costs are ~2–4%.

**The handover predicted a 200³ regression and predicted it for the right reason** — "setup is charged inside the timed region, and your build-time inversion lands there" — but not by the expected mechanism. The gather kernel is a win at every size. What decides the outcome is the host-side inversion: at 400³ the per-iteration saving outruns it, at 200³ the solve is too short to absorb it.

### Test suite

`Pkg.test()`: **12717 / 12717 passed, 3m19.8s**, zero failures. No test was weakened, loosened or skipped. The CPU path was separately smoke-tested and is untouched — `P.agg` there is still a `Vector{Int16}` reaching the original serial loop.

---

## Open items

**O1 — the 200³ cost is Amin's call, and is deliberately unresolved.** Three options, in the state they were left:

1. **Ship as-is.** ~4% at 200³ buys exact reproducibility at 400³ and above, which is where the campaign was actually blocked. The determinism payoff is at 600³–1000³, where the noise was exceeding the accuracy target; 200³ never had a reproducibility problem worth the name (0.008–0.019%).
2. **Build the adjacency on device**, by walking each block's grid positions in fixed order instead of counting-sorting on the host. This drops the inversion from 180 ms to an estimated ~3 ms and would turn all four cases into wins. It needs `A.idx`: free for the matrix-free operator, which owns it, but **the assembled path deliberately frees its `idx` before `_two_level_from_aggregates`**, and keeping it alive costs ~256 MB at 400³ and breaks a documented peak-memory property. So it is a matrix-free-only path, i.e. a second implementation to maintain.
3. **Leave it and revisit** if 200³ turns out to matter in the re-measured numbers.

**O2 — the build-time atomics remain order-sensitive in principle.** Not τ-visible today on either uniform or non-uniform `D` (D4 above). If the coarse correction ever stops being narrowed to `Float32`, re-test before assuming reproducibility survives.

**O3 — `benchmarks/` was not run**, by instruction. The published numbers are Amin's to re-measure. Every figure in this document is a spot check on two porosities at two sizes.

## Rejected

**Reporting a wall-clock regression rather than investigating it — rejected.** The first working version was 7.8% slower at 400³ p095. Reporting that as the trade-off would have been wrong: the cost was the serial inversion (P1) and a default workgroup size (P2), both of which were fixable, and neither of which was intrinsic to the determinism fix. The remaining 200³ cost is intrinsic to doing the inversion on the host, which is why it is an open item rather than another round of tuning.

**A device-side sort to build the adjacency — rejected for now.** Sorting `(agg, node)` pairs on device would be deterministic and fast, but KernelAbstractions has no generic sort, so it would pin this code to CUDA and break the Metal/AMDGPU story the rest of the file maintains. The grid-walk in O2 gets the same result without that cost, and is the better version of this idea if the work is greenlit.

## Progress log

**2026-08-16.** Baseline captured on four cases; defect reproduced (spreads 3.0e-6 to 4.3e-4). `Aggregation` + gather implemented and verified against a host reference. τ went to spread `0.0` on all four cases immediately, and the build-time atomics were confirmed unnecessary (D4), including under non-uniform `D`. First timing pass showed a 7.8% regression at 400³ p095; component breakdown attributed it to the serial host inversion, which was then chunked and threaded (P1). A second regression at 200³ was traced to the default workgroup size (P2). Cross-session timing was found to be unusable at this effect size, and the performance question was re-answered with a same-session interleaved A/B, which is what the results table reports. Full suite green.
