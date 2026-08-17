---
title: Coarse space mesh independence
created: 2026-08-16
updated: 2026-08-16
status: complete
outcome: The block edge is fixed and a V-cycle carries the coarse space down to the direct solve. Iterations no longer track the image edge (600³ p020: 465 → 150), wall-clock improves at every size measured, 200³ is bit-identical, and τ spread stays exactly 0.0.
branch: worktree-precond-h-dependence
supersedes: "-"
superseded-by: "-"
related: 2026-08-16-gpu-preconditioner-determinism.md
---

> **Status: complete (executed 2026-08-16).** The coarsening ratio grew with the image because the block edge was sized to keep the coarse grid under a fixed ceiling. The block edge is now **fixed**, so the coarse space grows with the image instead, and a **V-cycle over coarser grids** carries it down to the same direct solve as before.
>
> **Iterations no longer track the image edge.** GPU matrix-free, ε≈0.2: 200³/400³/600³ went 151 / 167 / 465 before and 151 / 125 / 150 after. Wall-clock improves at **every** size and porosity measured — −2.8% and −4.5% at 200³ (an identical code path, so this is noise), −5.3% and −34.3% at 400³, **−57.7% and −34.5% at 600³**. τ is bit-identical across repeats on all six cases, spread exactly `0.0`, so Phase 1's guarantee survives. `Pkg.test()` green.
>
> **How to read this document.** Sections down to *Working constraints* are the plan **as designed**; every number in them comes from the JOSS campaign. From [*Work items*](#work-items) onward is what actually happened. Where execution departed from the design it is marked inline — there are **three**, and the first is the substantive one.

# Coarse space mesh independence plan

Written from the JOSS benchmark campaign (200³–800³, both devices), measured on `pmeal-hpc` (2× Quadro RTX 8000, Xeon Silver 4110, 250 GB RAM).

This is Phase 2 of two. Phase 1 made the GPU preconditioner deterministic and is complete; its `_restrict!` / `_prolong!` interface was kept stable specifically so this plan can build on it.

## The defect

Iterations to reach the 1e-3 accuracy target grow with N instead of staying flat. Median across 15 cases:

| | 200³ | 400³ | 600³ | 800³ |
|---|---|---|---|---|
| GPU matrix-free | 59 | 106 | 106 | 189 |
| GPU assembled | 59 | 106 | 189 | 189 |

Roughly **3.2× more iterations for a 4× increase in N**. A mesh-independent preconditioner holds these constant. (The benchmark ladder is log-spaced — …33, 59, 106, 189, 339… — so medians snap to rungs; read the trend, not the exact values.)

**This is not the GPU saturating — internalise that before starting.** Control experiment at a fixed 100 iterations: our GPU path does **8× the work in only 2.5–3.8× the time**, while taufactor on the same card scales **dead-linear at 8.2–8.4×**. Our per-iteration cost is not degrading with size; it is *improving*. Every second lost at scale is the iteration count.

Consequence, paired against taufactor over cases where both converged (GPU, geomean over a fixed case set):

| 200³ | 400³ | 600³ | 800³ |
|---|---|---|---|
| 1.62× | **4.56×** | 3.86× | 2.48× |

The same shape appears on CPU. **Fixing this is the highest-value change available to the package.**

The defect is already plainly visible at the two sizes Phase 1 used for its spot check, so it needs no large run to reproduce: at fixed porosity, iteration counts went **145 → 221** (ε≈0.18) and **60 → 95** (ε≈0.95) for a 2× size step.

## The cause

`src/preconditioner.jl`:

```julia
const DEFAULT_MAX_COARSE = 32_000
```

`_choose_block` grows the block edge until the coarse grid fits under that ceiling. The ceiling is a **fixed constant**, so the coarse problem stays at roughly 31³ unknowns no matter how large the fine grid gets:

| fine grid | block edge | coarse grid | coarsening ratio |
|---|---|---|---|
| 200³ | 8 | 25³ | 8 |
| 800³ | ~26 | ~31³ | 26 |
| 1000³ | ~33 | ~31³ | 33 |

A two-level method is mesh-independent only when the coarsening ratio is **bounded** (typically 2–3). Here it grows linearly with N, so the coarse space resolves a steadily smaller fraction of the low-frequency error and the method degenerates toward unpreconditioned CG as N → ∞. The table above is exactly that degeneration.

**Do not simply raise the constant.** The comment above it records why it exists: the coarse solve is a **direct Cholesky** applied once per CG iteration, and a 50³ coarse grid already costs 1.7 s to factorise and 47 ms to solve, against 0.44 s and 1.9 ms at 25³. A direct solve on a coarse space that grows with N is unaffordable — which is exactly why the fix must be structural.

## Fix direction

**Go multilevel.** Keep the coarsening ratio bounded per level (~2) and recurse, so no single level's ratio grows with N while the coarsest grid stays small enough for the existing direct solve.

In rough order of effort:

1. **Recursive aggregation V-cycle.** Reuse the existing `agg`, `_restrict!` and `_prolong!` per level; replace the single direct solve with a recursive call terminating in the existing Cholesky at the coarsest level. Smallest change that preserves the machinery.
2. **Smoothed aggregation** — a damped-Jacobi smoothing pass on the tentative prolongator. Materially better convergence than piecewise-constant for elliptic problems, at modest extra setup.
3. **AlgebraicMultigrid.jl on the CPU path** as a correctness and iteration-count baseline to target, even if the GPU path needs a bespoke implementation.

Do **1** before reaching for **2**. If iteration counts go flat across 200³–800³, the diagnosis is confirmed and everything after is tuning.

**Call `_restrict!` and `_prolong!` through their existing signatures; do not reimplement or inline them.** Phase 1 rewrote their internals to be deterministic. Going around them reintroduces the defect that was just paid for.

### What Phase 1 leaves you, concretely

- `_restrict!(rc, agg, x)` and `_prolong!(y, agg, xc, x, inv_lambda)` — unchanged signatures, deterministic on device.
- On device, `agg` is an `Aggregation` carrying the forward map **and** its inverse as a CSR coarse→fine adjacency. A recursive hierarchy needs one per level; building one currently costs a host-side counting sort, which is the open cost question in Phase 1 (its `O1`) and which a multilevel hierarchy will multiply by the number of levels. **Read Phase 1's `O1` before designing the setup phase** — if this plan is greenlit, building the adjacency on device likely stops being optional.
- The gather kernel launches at an explicit workgroup size of 128; the backend default was 2–3× worse.

## The point is to get faster — and not to get slower anywhere

**Fewer iterations is not the goal. Less wall-clock is.** This change has a real regression risk in two places:

- **A V-cycle costs more per iteration than one coarse solve.** The iteration reduction has to more than pay for it. A change that halves iterations and doubles per-iteration cost is worth nothing.
- **Multilevel setup is more expensive than a single aggregation, and setup is charged inside the timed region.** At 200³ the current preconditioner's fixed setup is already a large share of the solve. **Small sizes are where this change is most likely to lose**, because setup dominates and there is little iteration growth left to recover. Do not buy a large-N win with a 200³ regression — the package is used at both ends. Phase 1 hit exactly this and left it unresolved; do not repeat it by accident.

**This is a spot check, not a benchmark campaign. Do not run the `benchmarks/` harness.** Amin will re-measure the published numbers himself once the fixes are in.

**Compare per-case, never a mean over cases** — if the change alters which cases converge, a mean covers a different case set and the comparison is invalid. That mistake produced a bogus result earlier in the campaign.

**Cross-session GPU wall-clock on the local card is not comparable at this effect size.** Phase 1 measured up to 25% drift between sessions on identical code and input. Use a same-session interleaved A/B, as Phase 1 did; the method is written up in its *Measurement method* section.

## What "done" looks like

Record a baseline before touching code, on the machine you are actually using; the numbers above are shape, not targets.

1. **Iteration counts flatten.** Measure at 200³ and 400³ on the same porosity and blobiness. The defect is plainly visible across just those two. Flat is the goal. 600³ matrix-free is a useful third point if convenient.
2. **Wall-clock improves** on those same cases, **including at 200³** — where it must at least not regress.
3. **τ stays bit-identical across repeats.** Phase 1 achieved spread exactly `0.0`; a multilevel hierarchy that reintroduces an order-dependent reduction gives that back. Check it, do not assume it.
4. **`Pkg.test()` green** — one run at the end, ~3.5 minutes for 12 717 assertions. Use targeted test files while iterating. Never weaken, loosen or skip a test to make a change pass.

**Measure convergence rate on the CPU path.** The h-dependence is a property of the *method*, not the device, so a fix that flattens CPU iteration counts flattens GPU ones — and CPU needs no GPU memory and gives a clean signal. Use GPU only for the wall-clock question.

## Hardware limits — the local GPU has 24 GB

The local card is an **RTX PRO 5000 Blackwell Laptop, 23.9 GiB** — not the 48 GB card the campaign ran on. Measured peak device memory:

| | 200³ | 400³ | 600³ | 800³ |
|---|---|---|---|---|
| matrix-free | 0.07–0.22 | 0.52–1.71 | 1.98–5.98 | 4.68–14.19 GiB |
| assembled | 0.18–0.68 | 1.21–4.85 | 4.16–16.38 | 9.86–**44.90 GiB** |

**Work at 200³ and 400³**, which is enough to see both the defect and the fix. 600³ matrix-free fits comfortably. **Do not attempt 800³ assembled — it needs 33–45 GiB and will OOM**, and 600³ assembled at p095 (16.4 GiB) is tight on a card also driving a display. A multilevel hierarchy adds device memory of its own, so leave headroom.

## Working constraints

- **Run Julia through the persistent MCP session**, never `julia` from a shell. Pass `env_path` as `<repo>/test/` — this package declares test deps via `[extras]`/`[targets]`, so `--project=.` cannot `using Test`. Load the `julia-workflow` skill. Restart the session after any struct/`const` redefinition, world-age error, or `Project.toml` change.
- **Benchmark in the warm session**, warming up and discarding the first run. A timing that includes compile latency measures the wrong thing.
- **Check whether `pmealsrv1` is free** before using it; campaign work has repeatedly owned that box for days at a time, and anything else running there corrupts both.
- **Branch `joss` carries a large uncommitted working tree** (the `benchmarks/` rewrite and the JOSS paper). Work in a separate worktree; do not commit on `joss` without asking Amin.
- `benchmarks/` is the only benchmark harness — do not add a second one.

---

## Work items

Statuses: `done`, `rejected`, `not-needed`.

| id | item | status | result |
|---|---|---|---|
| W1 | Stop the block edge growing with the image | **done** | `_choose_block` deleted; `DEFAULT_COARSE_BLOCK = 8` at every size |
| W2 | Carry the coarse space down to the direct solve | **done** | `_coarse_hierarchy`, ratio 2 per edge, Galerkin products on the host |
| W3 | Replace the single direct solve with a recursive call | **done** | `_vcycle!`, symmetric V(1,1), damped Jacobi, ending in the existing Cholesky |
| W4 | Reuse `_restrict!` / `_prolong!` per level, unchanged signatures | **done** | both host methods take the per-level `parent` map exactly as they take `agg` |
| W5 | Additive (BPX-style) multilevel instead of a cycle | **rejected on measurement** | 425 iterations at 256³ where the V-cycle gives 152 — see below |
| W6 | Thread the coarse apply | **done** | needed: without it 400³ p020 gained nothing (see *The one case that nearly failed*) |
| W7 | Build the adjacency on device (Phase 1's `O1`) | **not-needed** | the hierarchy did not multiply it — see below |
| W8 | Drop a coarse block on a round-off floor, not on `> 0` | **done** | out of plan, added on Amin's instruction; a pre-existing defect worth 1e15 in `‖ldiv!‖∞` — see [O1](#open-items) |

### W1/W2/W3 — what was built, and **the one place it departs from the design**

The design said "recursive aggregation V-cycle … replace the single direct solve with a recursive call". That is what was built, with one deliberate restriction: **the V-cycle covers the coarse grids only. The fine level is untouched and stays additive.**

An application is still exactly

    y = W · (coarse inverse) · Wᵀx  +  x / λmax

with one restriction and one prolongation over the fine grid, as before. Only the *coarse inverse* changed, from a direct solve to a V-cycle. Nothing in the preconditioner ever applies the fine operator, so **no extra fine SpMV is introduced** — which is the first of the two regression risks the plan named. The whole hierarchy is `1/block³` of the fine grid, so the cycle costs a fixed few percent of an application at any size: measured **2.1% of `ldiv!`** at 256³.

Because the block edge is fixed at the value `_choose_block` already returned for every image up to ~253³, **images at or below that size take a bit-identical code path** — same block, same coarse size, same iteration count, no levels built. The 200³ no-regression requirement is therefore met structurally rather than by tuning, which is the reason it is met at all.

`TwoLevelPreconditioner` keeps its name and its public shape. Two-level is the shape of the preconditioner; the hierarchy is how the coarse inverse is applied. No API changed, and `max_coarse` keeps its meaning — the largest problem worth solving directly — while `block` no longer grows.

### W5 — the additive variant, rejected on measurement. **This is the second departure.**

The plan's fix direction was ordered "1 before 2", and item 1 read as a straight recursion over the existing pieces. The *purely additive* form of that recursion is the one the existing code invites, because `_prolong!(y, agg, xc, x, θ)` already computes `W xc + θ·x` — exactly a BPX level. It was built first and measured first:

| 256³ replica, ε≈0.5 | 64³ | 128³ | 256³ |
|---|---|---|---|
| before (block grows) | 222 | 362 | 798 |
| additive multilevel | 222 | 315 | **425** |
| V-cycle over the coarse grids | 222 | 204 | **152** |
| block fixed at 8, coarse solved exactly | 222 | 198 | 128 |

The additive form removes about half the growth and no more. A sloppy coarse inverse is punished twice over: the two-level bound is multiplied by the coarse solver's own condition number, so an inexact coarse solve costs roughly its square root in iterations. The V-cycle lands within 19% of an exact coarse solve; the additive form is 3.3× off it. **Do not revisit the additive form** — it is not a tuning question, it is the wrong shape.

### W7 — the hierarchy did **not** multiply Phase 1's `O1`. **Third departure.**

The plan predicted: "building [the adjacency] currently costs a host-side counting sort … which a multilevel hierarchy will multiply by the number of levels. Building the adjacency on device likely stops being optional."

It did not happen, because only the **fine → coarse** aggregation ever needs an `Aggregation`. Every level below is host-side, and its transfers are the serial host `_restrict!` / `_prolong!` over a plain `parent` vector — no inversion, no counting sort, no device memory. The count of counting sorts is still exactly one.

The inversion did get *bigger*, because the coarse space is now larger (400³ p095: 29 741 → 123 888 cells). That was more than paid for by the direct solve shrinking: setup at 400³ p095 went **0.359 s → 0.293 s**. Setup is slightly more expensive at the other sizes and is inside every total reported below. **`O1` can stay open.**

### W6 — the one case that nearly failed

The first working version improved iterations everywhere but bought **nothing** at 400³ p020: 167 → 125 iterations and wall-clock +0.9%. Per-iteration cost had risen 45%, and a component breakdown of `ldiv!` found all of it in one place:

| 400³ p020 | restrict | D2H | coarse | H2D | prolong | `ldiv!` |
|---|---|---|---|---|---|---|
| before | 0.410 | 0.016 | **0.233** | 0.012 | 0.159 | 1.119 ms |
| after, serial cycle | 0.323 | 0.029 | **0.860** | 0.019 | 0.194 | 1.592 ms |

The device gather got *faster*; the coarse solve got 3.7× slower. Inside it, the two sparse applies were 0.609 ms of 0.888 — memory-bound and serial.

`mul!` on a `SparseMatrixCSC` is a column scatter, which cannot be threaded without an atomic — and an atomic float sum is precisely what Phase 1 removed. But **every operator in the hierarchy is symmetric to the last bit**, so a column *is* a row and the product can be a gather of independent dot products instead: threadable with no atomic, and each output summed in one fixed order, so the result does not depend on the schedule. It is also bit-identical to `mul!`, since both walk a row in ascending index order — verified, not assumed.

That required making the symmetry true rather than probable. `_coarse_operator` already averages opposite stencil slots; `_coarse_hierarchy` now averages each Galerkin product the same way (`(B + Bᵀ)/2`, exactly symmetric because float addition is commutative and halving is exact).

Result: 0.295 ms → 0.034 ms per apply, and 400³ p020 went from +0.9% to −5.3%. Below 4096 rows the ~11 µs of thread startup is the whole cost, so smaller levels run serially.

---

## Measurement method

Phase 1's method, for its reasons: **same-session interleaved A/B**, because cross-session GPU wall-clock on this card drifts more than the effect being measured.

One simplification over Phase 1: the previous behaviour needs no throwaway wrapper, because `block=` reproduces it exactly. `_choose_block(N,N,N,32_000)` returned 8 / 13 / 20 at 200³ / 400³ / 600³, so passing those values *is* the old preconditioner, built on the same `sim` in the same session and alternated against the new one. Every figure is the best of 5 (200³) or 3 (400³, 600³) alternating repeats, and **the preconditioner build is inside the timed region**, as the solver charges it.

Convergence was measured on the CPU path, as instructed. To iterate on designs at CPU speed, the defect was replicated in miniature: with `max_coarse = 512`, `_choose_block` grows the block 8 → 16 → 32 across 64³ / 128³ / 256³, which is the same mechanism and the same ratios as 200³ → 800³ at the released ceiling. Every design decision below was made on that replica and then confirmed at released settings.

`benchmarks/` was **not** run, by instruction.

## Results

### Iterations no longer track the image edge — the goal

GPU matrix-free, `reltol=1e-6`:

| | 200³ | 400³ | 600³ |
|---|---|---|---|
| ε≈0.2, before | 151 | 167 | **465** |
| ε≈0.2, after | 151 | **125** | **150** |
| ε≈0.95, before | 54 | 88 | 116 |
| ε≈0.95, after | 54 | **60** | **69** |

CPU, `reltol=1e-8`, ε≈0.5: 200³ **144 → 144** (identical), 400³ **254 → 179**.

The residual growth at ε≈0.95 (54 → 69) is the V-cycle's inexactness, not the coarsening ratio: at 200³ the coarse space fits under the ceiling and is solved exactly, and each size above it adds a level. That is a cost in the number of levels, not in the image edge.

### Wall-clock — improves at every size and porosity

Same-session interleaved A/B, best of N, build included:

| case | iters | total before | total after | change |
|---|---|---|---|---|
| 200³ p020 | 151 → 151 | 0.127 s | 0.124 s | −2.8% |
| 200³ p095 | 54 → 54 | 0.306 s | 0.293 s | −4.5% |
| 400³ p020 | 167 → 125 | 0.735 s | 0.696 s | −5.3% |
| 400³ p095 | 88 → 60 | 1.911 s | 1.255 s | **−34.3%** |
| 600³ p020 | 465 → 150 | 5.821 s | 2.460 s | **−57.7%** |
| 600³ p095 | 116 → 69 | 5.575 s | 3.653 s | **−34.5%** |

**The two 200³ rows run an identical code path** — same block, same coarse size, same iteration count — so read them as "no regression", not as a 3–4% win. That is the plan's 200³ bar, met by construction.

The gain grows with N because what it removes grows with N: the old block edge was 8 at 200³ (no change possible), 13 at 400³, and 20 at 600³.

### Determinism — Phase 1's guarantee survives

τ spread across repeats, GPU matrix-free, all six cases: **exactly `0.0`**. The cycle runs entirely on the host, and every step of it — including the threaded gather — sums each output element in one fixed order.

Every wall-clock and iteration figure above was measured before [O1](#open-items) was fixed. The fix cannot move them, and that was checked rather than assumed: `nc` and the iteration counts at the default block are identical with and without the floor on all four 200³/400³ cases (9824/151, 15620/54, 49642/125, 123888/60).

### Test suite

`Pkg.test()`: **13022 / 13022 passed, 3m54.5s**, zero failures — 305 assertions more than Phase 1's 12717. No test was weakened, loosened or skipped. Two testsets were **replaced rather than removed**: the one pinning `_choose_block`'s growth now pins that the ceiling is met by depth instead, and new sets cover the Galerkin product at every level, SPD-ness of the hierarchical apply on all seven fixtures, assembled/matrix-free parity of the hierarchy, and — the property this plan exists for — that doubling the image edge does not double the iteration count.

## Open items

**O1 — a pre-existing defect, investigated and fixed on Amin's instruction.** Reported first as an assembled/matrix-free parity nit: at `block=2` the two paths disagreed on the coarse size, 4009 cells against 4008, **with no hierarchy involved and `max_coarse` infinite**. Measured properly it was not a parity nit.

`_coarse_operator` dropped a block on `diagonal > 0`. That diagonal is **exactly** zero for a block holding nothing but a cluster enclosed within it — the degrees and couplings cancel term for term — so in floating point it lands on a residue whose sign is whichever way the threads raced. The assembled path reached `+2.22e-16` where the matrix-free path reached `0.0`, so one kept the block and the other dropped it. What the keeping path got was a coarse row with a 1e-16 diagonal, and a coarse solve that divides by it:

| 32³ blob seed=7, variable `D`, `block=2` | `nc` | `‖ldiv!‖∞` |
|---|---|---|
| assembled (kept the residue) | 4009 | **5.6e15** |
| matrix-free (dropped it) | 4008 | 6.9 |

The two paths disagreed by 18 orders of magnitude on the same residual. Kept diagonals otherwise ran from `0.76` to `98`, so there is nothing legitimate within fifteen orders of magnitude of the residue.

**Fixed** by `_coarse_diagonal_floor(bs, maximum_diagonal)`: a block is kept when its diagonal clears the round-off bound for the sum that produced it — at most `_COARSE_SLOTS` terms per voxel, each no larger than the biggest diagonal of `A`, times `eps`, times a slack constant. The same floor decides it at every level of the hierarchy, so one rule governs the whole coarse space. Both paths now return 4008 with identical aggregates and agree to `rtol=1e-10`, and `‖ldiv!‖∞` is 6.9 on both.

**Nothing changes at released settings**, verified rather than assumed: `nc` and iteration counts are identical at the default block on 200³ and 400³ at both porosities (9824/151, 15620/54, 49642/125, 123888/60). The floor removes numerically-null cells and only those.

Its regression test is a direct one on `_coarse_operator` — a hand-built stencil carrying the measured `2.22e-16` residue, which the floor rejects and `> 0` keeps (`remap` `[1,0,0,2]` against `[1,2,0,3]`), checked for both signs of the residue — plus the `block=2` parity case that exposed it. A third test guards the property end to end but does **not** discriminate, and says so: on that fixture the residue happens to come out negative, so `> 0` would have dropped the block anyway. Which side of zero a residue falls on is not something a fixture can pin down, which is exactly why the threshold could not stay there.

**O1b — the documentation build was already failing on `main`, and is fixed here.** Not part of this plan, found because the new type needed a cross-reference. `docs/make.jl` sets `warnonly=[:missing_docs]`, so an unresolvable `@ref` **fails the build**, and `Documentation.yml` runs on every pull request. Three references inside *rendered* docstrings pointed at bindings that are not in the manual: `MaskedLaplacian` and `build_steady_operator` both reference `build_steady_system` (pre-existing), and `TwoLevelPreconditioner` references `Aggregation` (added by Phase 1). Fixed by documenting `Aggregation` and `CoarseLevel` in `docs/src/api.md` — they are both named in `TwoLevelPreconditioner`'s field list, so a reader had two type names with nothing to click — and demoting the internal-helper references to plain code spans. `include("docs/make.jl")` now exits 0 with no cross-reference errors. A `@ref` inside a docstring that is *not* rendered is never checked, so the many others in `src/preconditioner.jl` are untouched and harmless.

**O2 — device memory during setup grew, by a small absolute amount.** The coarse stencil is `7 · nblocks` `Float64` on device while the coarse operator is built. With the block edge fixed, `nblocks` grows with the image: ~1.7 MB before at any size, against 24 MB at 600³ and ~56 MB at 800³. It is transient and freed before the solve, and negligible beside the multi-GiB peaks at those sizes, but it is no longer a constant.

**O3 — the coarse hierarchy is on the host.** That is what keeps it simple, deterministic and free of Phase 1's `O1`, and at these sizes it costs 2–5% of an application. It scales with the image, though: level 2 is ~1M cells at 800³ and ~2M at 1000³. If 1000³ becomes routine, the top coarse level is the thing to move to the device — not the adjacency.

**O4 — `benchmarks/` was not run**, by instruction. Every number here is a spot check on two porosities at three sizes.

## Rejected

**Raising `DEFAULT_MAX_COARSE` — rejected, as the plan required.** Confirmed rather than assumed: with the block fixed at 8, a 400³ coarse space is 125 000 unknowns, and the plan's own measurement (1.7 s to factorise 50³, 47 ms per solve) rules out a direct solve on it. The ceiling keeps its released value and its meaning.

**The additive multilevel form — rejected on measurement.** See W5. 425 iterations against the V-cycle's 152.

**A larger coarsening ratio — rejected.** Ratio 3 and 4 need fewer levels and cheaper setup, and cost iterations: 187 and 194 against 152 at ratio 2, on the 256³ replica.

**`ω = 1.0` for the smoother — rejected as unsafe.** It is the best-measured value (150 iterations against 152 at 0.8) and it sits exactly on the stability boundary: every coarse operator here is a weakly diagonally dominant M-matrix, so `λmax(D⁻¹A) ≤ 2` and the symmetric cycle is SPD only for `ω < 1`. Measured `λmax` at both coarse levels of a 256³ case: 1.996. At `ω = 1.2` CG rejects the preconditioner outright. 0.8 keeps the margin for two iterations.

## Progress log

**2026-08-16.** Worktree fast-forwarded onto Phase 1. Defect replicated in miniature on CPU (222 / 362 / 798 across 64³–256³) and the target established by fixing the block and solving the coarse space exactly (222 / 198 / 128), which confirmed the coarsening ratio as the whole cause before any code was written. Additive multilevel built first and measured short (425); the coarse-only V-cycle built next and landed at 152. Ratio, smoother weight and coarsest size swept on the replica; ω pinned below the stability bound the M-matrix property provides, verified numerically at 1.996. Implemented, reproducing the prototype exactly. Confirmed at released settings on CPU (200³ identical, 400³ 254 → 179). GPU interleaved A/B found 400³ p020 flat despite 34% fewer iterations; traced to two serial sparse applies inside the cycle, fixed by making the Galerkin products exactly symmetric and gathering columns as rows across threads. Re-measured: every case improves. τ spread `0.0` on all six. Suite green at 13006/13006. Documenting the new type surfaced a documentation build that was already failing on `main` for three unresolvable cross-references; fixed, and `docs/make.jl` now exits 0.
