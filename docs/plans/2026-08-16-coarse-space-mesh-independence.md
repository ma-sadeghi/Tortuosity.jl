---
title: Coarse space mesh independence
created: 2026-08-16
updated: 2026-08-16
status: draft
branch: "-"
supersedes: "-"
superseded-by: "-"
related: 2026-08-16-gpu-preconditioner-determinism.md
---

> **Status: draft — not started, and not to be started without Amin's explicit greenlight.** Iterations to reach the accuracy target grow with N instead of staying flat (200³ → 800³: 59 → 189 on GPU matrix-free), because the coarse space is capped at a fixed size while the fine grid grows. This is the defect that costs the package its lead over taufactor as problems get larger: our paired advantage peaks at **4.56× at 400³** and decays to **2.48× at 800³**. Extrapolated, it crosses 1.0.
>
> Its prerequisite, [`2026-08-16-gpu-preconditioner-determinism.md`](2026-08-16-gpu-preconditioner-determinism.md), is **complete** — you cannot cleanly measure a convergence-rate improvement against a solver that returns a different answer every run, and now it does not.
>
> **This document is the plan as designed on 2026-08-16.** Nothing below has been executed. Every number in it comes from the JOSS benchmark campaign, not from a run against the current tree.

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
