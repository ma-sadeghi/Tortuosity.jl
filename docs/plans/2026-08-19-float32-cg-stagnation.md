---
title: Float32 CG stagnation on the GPU path
created: 2026-08-19
updated: 2026-08-20
status: fix landed, gated and measured at scale; open items are the GPU campaign re-run and paper.md — see "Where this stands"
outcome: "Float32 CG stagnation traced to finite-precision Krylov breakdown, not to the format, the geometry or the preconditioner. Fixed by iterative refinement against a Float64 residual: 200 cubed grid 1/15 failures to 0/15, 1000 cubed 1.5e-3 to 4.1e-6, 800 cubed assembled 1.2e-3 to 4.1e-8. Time-to-target unchanged for 14 of 15 cases, so the taufactor margin holds. Memory guard measured on a real 950M-node image: it fires, warns and returns the unrefined solution. Refinement costs 20 B per pore node. Suite green at 12,644 assertions."
branch: joss
supersedes: "-"
superseded-by: "-"
related: 2026-08-16-coarse-space-mesh-independence.md
---

# Float32 CG stagnation on the GPU path

## The defect

On ill-conditioned (low-porosity, high-$\tau$) images the GPU `Float32` path returns an answer wrong by ~1.4e-3 in $\tau$ **while reporting success**. The recursively-updated residual that CG stops on keeps shrinking; the true residual `b - A*x` does not. `sol.resid` is the recurrence residual in the M-norm (`LinearSolve/iterative_wrappers.jl:370`), so it can read 1e-6 while `‖b-Ax‖₂/‖b‖` sits at 2.6e-6 and the forward error in $\tau$ sits at 2e-3.

Campaign impact: 3 of the ~60 GPU cases exhaust the ladder — `n200_b050_p020` (matrix-free), `n800_b200_p020` (assembled), `n1000_b100_p020` (matrix-free). Coverage at $1000^3$ is 14/14 CPU, 13/14 GPU.

The package already half-knew this: `src/simulations.jl:288` sets `_default_reltol(Float32) = 1f-6` with the comment "`Float32` CG cannot drive the relative residual much below `1e-6`".

## Working rule for this investigation

Every hypothesis gets a test that can refute it, **before** it is written down as a cause. Three explanations were asserted earlier without one and all three were wrong (voxel-count accumulation, dead-end pore volume, concentration dynamic range). Two constraints must both hold before this is called fixed:

1. the failing cases reach the 0.1% target, and
2. the speed margin over taufactor is roughly preserved.

## Hypotheses

| id | hypothesis | status | evidence |
|---|---|---|---|
| H0 | `Float32` cannot represent the answer | **refuted** | `Float64` solution rounded to `Float32` evaluates to rel_err 7.15e-6; taufactor reaches 8.0e-6 in `Float32` on the same image and card |
| H0b | the two-level preconditioner causes it | **refuted** | unpreconditioned CG is *worse* (7.9e-3 vs 1.4e-3) |
| H1a | `Float32` residual replacement (restart) | **confirmed, partial** | 1.4e-3 → 3.0e-5, but non-monotone: degrades to 4.9e-4 by round 6 |
| H1b | `Float64` residual, iterative refinement | **confirmed** | 1.4e-3 → 5.95e-7 in 3 rounds / 195 iters, stable |
| H2 | restructure the matvec to difference first, removing cancellation | **refuted** | only 1.85× more accurate residual; ~200× needed |

### Why H2 failed, and what it proves

The differenced form $\sum_j w_{ij}(x_i-x_j) + bc_i x_i$ was tested against the standard $\deg_i x_i - \sum_j w_{ij}x_j$ on the converged solution, both in `Float32` against a `Float64` ground truth: error 1.07e-4 vs 1.99e-4. **Rounding `x` into `Float32` is itself the floor**, so no rearrangement of the arithmetic recovers the digits. This is the direct argument for why H1b needs a `Float64` *iterate*, not merely `Float64` accumulation.

## The fix (H1b)

An outer refinement loop. Residual replacement cannot be done inside `Krylov.jl`'s `cg!` — the recurrence scalars `γ` and `pNorm²` are locals of `cg!`, not workspace fields, so a callback that rewrote `workspace.r` would leave `β = γ_next/γ` reading a stale value. An outer loop of repeated `cg!` calls is the supported composition, and `cg!` zeroes `x` on entry, so each call must solve the *correction* equation.

The `Float64` residual is free of any `Float64` operator: `MaskedLaplacian`'s kernel keys its accumulator off `eltype(y)` (`src/matrixfree.jl:113-114, 125-127, 137, 161`), and out-of-place `*` promotes via `promote_type` (`src/matrixfree.jl:208-209`, `src/sparse_type.jl:245-246`). So `A::{Float32} * x::CuArray{Float64}` is a genuine `Float64` matvec with `Float32` storage.

**The trap, not taken:** `mul!(y64, A32, x32)` rounds each product to `Float32` before widening and recovers nothing.

Extra memory: 24 bytes per pore node — see the measured section below.

## Measurements, $200^3$, blobiness 0.5, seed 42 (local RTX PRO 5000)

Solver-declared convergence, `reltol=1f-6`:

| $\varepsilon$ | $\tau$ | plain iters | plain err | refined iters | rounds | refined err | cost |
|---|---|---|---|---|---|---|---|
| 0.16 | 33.93 | 84 | **2.00e-3** (fails) | 194 | 2 | 5.09e-7 | 1.63× |
| 0.40 | 2.92 | 95 | 6.36e-5 | 205 | 2 | 1.04e-8 | 1.78× |
| 0.61 | 1.55 | 83 | 3.56e-5 | 185 | 2 | 7.86e-9 | 1.84× |
| 0.95 | 1.04 | 55 | 7.85e-5 | 120 | 2 | 7.61e-10 | 1.79× |

Assembled and matrix-free behave identically (194 iters, 5.09e-7 vs 5.67e-7).

### The residual does not discriminate

After round 1 every case sits at true `‖r‖/‖b‖` ≈ 2.4–2.9e-6, but the $\tau$ error spans 3.6e-5 to 2.0e-3. The ratio *is* the conditioning amplification (758× at $\varepsilon=0.16$, 12–33× elsewhere). **So a residual-based trigger cannot single out the case that needs refinement** — that idea is closed.

### Tolerance schedule

Round 1 at `r1`, correction rounds at `rk`. Cost is relative to plain; "OK" means err ≤ 1e-4.

| r1 | rk | worst err | max cost |
|---|---|---|---|
| 1e-6 | 1e-2 | 2.24e-5 | 1.83× |
| 1e-4 | 1e-2 | 7.32e-6 | 1.92× |
| **1e-4** | **1e-1** | **6.41e-5** | **1.57×** |
| 1e-3 | 1e-2 | 5.71e-4 | 1.42× |
| 1e-2 | 1e-1 | 3.47e-3 (fails) | 1.20× |

Superseded by the full-grid result below: `r1=1e-4, rk=1e-1` costs more (1.57×) than leaving round 1 alone. See the next section.


## Full $200^3$ grid, matrix-free, local RTX PRO 5000 (2026-08-19)

Images regenerated locally reproduce the campaign **exactly** — all 15 node counts match `references.csv`, and the plain arm fails exactly the one case the campaign recorded (`b050_p020`, 2.0e-3). That match is what licenses using local results at all.

Chosen schedule: round 1 at `reltol=1f-6` (**identical to today**, so round 1 cannot regress), correction rounds at `reltol=1f-1`.

| metric | plain | refined |
|---|---|---|
| failures at 0.1% | 1/15 | **0/15** |
| worst error | 2.0e-3 | 6.7e-5 |
| cost to full convergence | 1× | median 1.29×, max 1.40× |

Schedule comparison over the same grid: `r1=1e-4, rk=1e-1` costs median 1.44× / max 1.58× and made 4 cases slightly worse; `r1=1e-6, rk=1e-1` costs median 1.29× / max 1.40× and made 1 case slightly worse (7.7e-6 → 1.2e-5, both far inside target). The second is chosen — a looser round 1 buys nothing.

### The harness metric is unchanged — measured, not argued

Replicating the campaign's own rule (evaluate $\tau$ only at ladder rungs; pad rungs above an early exit with the final iterate), the rung at which each case first meets 0.1%:

| | result |
|---|---|
| unchanged rung | **14/15** |
| newly fixed | 1/15 (`b050_p020`, now rung 106) |
| regressed to a later rung | **0/15** |

Same rung means same iteration count means same time. **So the published time-to-target is untouched for every case that already worked**, and the case that did not now enters the comparison set — where taufactor needs 11169 iterations and 18.09 s. The 1.29× is the honest cost of a *full* solve to `reltol`, which is the user-facing contract, not the benchmarked quantity.

A first attempt at this measurement reported 3 spurious failures because the plain solve converges at 55 iterations, between rungs 33 and 59, so no rung ever recorded the converged answer. The harness pads such rungs with the final iterate; adding that padding removed the artefact. Recorded because the uncorrected version would have overstated the fix.


### The trigger must key on precision, not on the residual ratio

Measured on CPU `Float64` with the default `reltol=1e-10`:

| $\varepsilon$ | iters | true `‖r‖/‖b‖` | overshoot vs reltol | $\tau$ error |
|---|---|---|---|---|
| 0.20 | 190 | 2.40e-10 | 2.4× | 6.9e-8 |
| 0.95 | 118 | 1.82e-10 | 2.4× | 1.3e-9 |

The `Float32` path overshoots its own `reltol` by 2.6×, the `Float64` path by 2.4×. **The ratios are indistinguishable**, so a rule of the form "refine when the true residual exceeds `reltol`" — with or without a slack factor — would fire on the CPU as well and charge it for nothing, since `Float64` already delivers 1e-8 in $\tau$.

Refinement is therefore gated on the working precision being `Float32`, and on that path it runs unconditionally, because the earlier result stands: nothing observable at round 1 distinguishes the cases that need it.

## Speed context (campaign data, server Quadro RTX 8000)

Time to target, GPU:

| case | ours | taufactor | margin |
|---|---|---|---|
| `n800_b200_p020` | 14.01 s (mf, 189 it) | 1202.35 s (11169 it) | 86× |
| `n800_b200_p040` | 14.09 s (106 it) | 214.97 s (1945 it) | 15.3× |
| `n200_b050_p020` | fails | 18.09 s (11169 it) | — |

A 1.6× cost is trivially affordable at 800³. The pinch point is $200^3$, where the published margin is 1.58×.

**Open question, to be answered with numbers, not argument:** the harness measures time to reach 0.1% along the ladder and stops at the first rung that meets it. For the 13 healthy cases that rung is reached during round 1, before refinement engages, so their published times should be unchanged; and the currently-failing case would *enter* the comparison set at ~40–90× margin over taufactor, which would raise the geomean rather than lower it. This has not been measured yet and must not be asserted until it is.


## At-scale validation, server Quadro RTX 8000 (2026-08-19)

Both cases the campaign recorded as `ladder_exhausted`, run through the shipped code path. The plain arm reproduces the campaign failure in each, which is what licenses the rest of the row.

| case | operator | nodes | arm | iters | rounds | time | rel_err |
|---|---|---|---|---|---|---|---|
| `n1000_b100_p020` | matrix-free | 184 M | plain | 205 | — | 22.0 s | **1.499e-3** (fails) |
| | | | refined | 587 | 2 | 62.4 s | **1.25e-8** |
| `n800_b200_p020` | assembled | 97 M | plain | 318 | — | 21.0 s | **1.160e-3** (fails) |
| | | | refined | 864 | 2 | 55.8 s | **4.08e-8** |

Cost 2.84× and 2.66×. Both are bit-identical across warm-up and two timed repeats, so the determinism guarantee survives at scale as well as at $200^3$.

Residual trace on the $1000^3$ case: 6.28e-6 after the main solve, 5.57e-9 after two rounds.


### Cost grows with size, and the $200^3$ figure understates it

Adding the fourth at-scale case, and separating the prototype from the shipped code:

| case | nodes | arm | iters | time | rel_err | cost |
|---|---|---|---|---|---|---|
| `n800_b200_p040` mf (healthy) | 204 M | plain | 145 | 18.78 s | 1.178e-4 | — |
| | | refined (prototype) | 534 | 67.02 s | 3.04e-8 | 3.57× |
| `n1000_b100_p020` mf (shipped) | 184 M | plain | 205 | 22.52 s | 1.499e-3 | — |
| | | refined (shipped) | 631 | 60.65 s | 4.098e-6 | **2.69×** |

The $200^3$ grid gave a median of 1.29×. At 184–204 M nodes it is 2.7–3.6×. **The cost ratio grows with image size**, because the base solve gets relatively cheaper — 205 iterations at $1000^3$ against 84 at $200^3$ — while the correction rounds do not shrink with it. Quoting 1.3× as the cost would be wrong; the honest range is **1.3× at $200^3$ rising to ~2.7× at $1000^3$** for a full solve to convergence.

This does not touch the benchmark metric, which is time to reach 0.1% and is unchanged for every case that already converged. Against taufactor at $800^3$ the margin was 15–86×; at 2.7× it is 5.5–32×.

The prototype's 3.57× is an overestimate of the shipped cost: it solved each correction to the full `reltol=1e-6`, where the package uses `1e-1`. Its accuracy figures are correspondingly better than the package's for the opposite reason — it returned the `Float64` iterate (1.25e-8) where the package returns a narrowed `Float32` one (4.10e-6). Both clear the 1e-3 target by more than two orders of magnitude.

`n800_b200_p040`'s memory numbers are **discarded**: they were sampled while a second job held the card, and `peak_pool` measures the whole device.

### Memory is the binding constraint, not time

| case | nodes | plain peak live | refined peak live | extra | per node |
|---|---|---|---|---|---|
| `n800_b200_p020` | 97 M | 8.38 GiB | 12.11 GiB | 3.73 GiB | 41 B |
| `n1000_b100_p020` | 184 M | 9.22 GiB | 16.77 GiB | 7.55 GiB | 44 B |

The plain figure at 184 M nodes matches the campaign's memory stage (9.2 GB at $\varepsilon=0.18$), which cross-checks the probe.

~42 B/node is more than the three `Float64` vectors the algorithm needs (24 B/node). The rest is a second Krylov workspace: each round builds a fresh `LinearProblem`, so the correction solve allocates its own vectors instead of reusing the ones the main solve already has.

**This does not scale to high porosity.** At $\varepsilon=0.95$, $1000^3$ carries 950 M nodes and the plain solve already peaks near 30 GiB; 42 B/node of refinement on top is ~40 GiB, well past the 47.3 GiB card. Reusing the existing `LinearCache` (`cache.b = r; solve!(cache)` is supported — `LinearSolve/common.jl:270`) removes the second workspace and should bring the extra to ~20 B/node, which fits. **Unverified — being measured now on `n1000_b100_p095`, the 950 M-node case, before any of this is treated as settled.**

Worth noting the shape of the risk: the cases that *need* refinement are low-porosity and therefore small, while the cases that are large are well-conditioned and do not need it. That is a happy accident, not a design, and it must not become the thing the fix silently relies on.


## Implementation

In `src/simulations.jl`, on the package-owned `solve(sim, alg; ...)`:

1. **`refine` keyword**, defaulting to on for a `Float32` system and off for a `Float64` one (`_refines_by_default`). `refine=false` restores the previous behaviour exactly.
2. **`_refine`** runs correction rounds against a `Float64` residual, reusing the existing `LinearCache` (`cache.b = r; solve!(cache)`) so the corrections run on the Krylov vectors and preconditioner already resident.
3. **`sol.resid` is now the true relative residual** and `sol.iters` counts every iteration spent. Reporting the recursive residual is what let the defect hide, so the repaired path does not repeat it.
4. **`abstol` now defaults to zero** rather than being left to LinearSolve. This was a second, pre-existing defect: LinearSolve defaults `abstol` to `sqrt(eps(T))`, which on `Float32` is 3.5e-4 — loose enough to stop the solve before the `reltol` the package deliberately chose is anywhere near met. It truncated the $200^3$ worst case at 75 iterations instead of 84 and left 3.2e-3 rather than 2.0e-3. The benchmark harness already worked around it by passing `abstol=0.0` explicitly and says why; ordinary callers did not.


### The reported residual must describe the vector returned

The regression test caught an inconsistency in the first implementation. Refinement carries a `Float64` iterate internally but narrows it back to `Float32` on the way out, and the first version reported the *iterate's* residual (3.3e-10) rather than the *returned vector's* (1.1e-7). Those differ by the `Float32` storage precision alone.

Reporting the better of the two would have been the same defect this whole exercise exists to remove: a number attached to an answer it does not describe. `_refine` now narrows first and recomputes the residual from the narrowed vector, at the cost of one extra matvec. Verified against an oracle built from the **CPU `Float64` operator**, which shares no arithmetic with the device path: reported and oracle agree to all printed digits (1.147e-7 on both, seed 1; 1.052e-7, seed 42).

**No earlier measurement is affected.** Every tortuosity figure above was computed from `sol.u`, the narrowed vector, so the accuracy numbers stand; only the reported `resid` changed.

Returning the `Float64` iterate instead would deliver ~6e-7 in tortuosity rather than ~8e-6, but it changes the element type callers get from `reconstruct_field` on the GPU path. 8e-6 is already 126× inside the 1e-3 target, so the API churn is not worth it.

### Memory, measured

Extra device memory over the plain solve is **24 bytes per pore node** — `x64` (8) + `r64` (8) + a `Float32` correction right-hand side (4) + the narrowed `Float32` vector returned (4). A 2 ms sampler on the laptop reported 20 B/node across 1.29 M, 3.19 M and 7.59 M nodes; a 10 ms sampler on the server, taking 8494 samples on a 184 M-node case, caught 24. The higher figure is the one to plan against — the laptop sampler was missing the transient peak at the final narrowing. The first implementation cost 41–44 B/node because each round built a fresh `LinearProblem` and so allocated a second Krylov workspace; reusing the cache removed that.

**20 B/node is still not always enough room.** At $1000^3$ and $\varepsilon = 0.95$ the image carries 950 M pore nodes and the plain solve peaks at 32.078 GiB of a 47.27 GiB card, measured. Refinement needs 17.7 GiB more; the prototype died there with "Out of GPU memory trying to allocate 3.541 GiB", which is exactly the `Float32` buffer, so the two `Float64` vectors had already gone in.

`_refine` therefore **guards its allocations**: if they fail it emits a `@warn` naming the consequence and returns the unrefined solution instead of throwing. A loud degradation, not a silent one — the whole point of this work is that a wrong answer must not be reported as a success.

The shape of the risk is worth stating plainly: the cases that *need* refinement are low-porosity and therefore small, and the cases too large to refine are well-conditioned and do not need it (`n1000_b100_p095` plain is already 2.24e-5). That is a fortunate correlation, not a design, and the guard exists so that nothing depends on it silently.

**The way to remove the constraint** — not done, and deliberately not attempted yet — is to let the matvec accumulate in the wider of the operand types and store the residual in `Float32`. `_steady_apply_kernel!` currently keys its accumulator off `eltype(y)` (`src/matrixfree.jl:113-114`), so a `Float32` output forces `Float32` accumulation. Widening the accumulator only when the operand types differ would leave the common path untouched and cut refinement to 12 B/node, which fits at 950 M nodes with room to spare. It is a change to a hot kernel and needs its own before/after timing.




### The harness change is validated end to end

Run locally through `run/campaign.sh`, not by calling the pieces:

- `--grid=smoke --sizes=100 --tools=tortuosity --devices=gpu --stages=timings` — exit 0 on both operators, 15 of 15 cases `target_reached`, rows and timings sane. This exercises the ordinary path, where nothing stalls and refinement never enters the trace.
- `--grid=full --cases=n200_b050_p020 --devices=gpu --stages=timings --overwrite` — the case the campaign recorded as `ladder_exhausted`. Both operators now report **`target_reached`**, matrix-free at rel 7.47e-6 in 0.205 s, assembled at 7.28e-6 in 0.153 s. The stalled trajectory is still visible below it (1.40e-3 at rung 106), with the refined result padding the rungs above. That is the branch that had to be right before a campaign re-run, and it is.

Local `benchmarks/results/` was backed up before these runs and restored afterwards.

## Gate status (re-run 2026-08-20)

`Pkg.test()` **green** — 12,644 assertions across 20 top-level testsets, exit 0, zero failures and zero errors, with the GPU path exercised. Re-run after the 20 B/node change, which is bit-identical by construction and changed nothing. Three new testsets:

- `Float32 solves are refined against a true residual` (seeds 1, 42) — 4 assertions each. Pins that `sol.resid` equals the true relative residual computed from the **CPU `Float64` operator**, an oracle sharing no arithmetic with the device path; that refinement never returns a worse residual than the solve it repairs; that the extra iterations are counted rather than hidden; and that the physics still agrees.
- `refinement is Float32-only` — 2 assertions on the precision gate.

### Two bugs the tests and probes caught in my own fix

1. **`sol.resid` described the wrong vector.** Refinement carries a `Float64` iterate but returns a narrowed `Float32` one; the first version reported the iterate's residual (3.3e-10) rather than the returned vector's (1.1e-7). Fixed by narrowing first and recomputing. Caught by the new regression test on its first run.
2. **`cache.reltol = correction_reltol` was a `TypeError` on a `Float64` system.** The cache's tolerance field is typed to the problem, so a `Float32` literal could not be assigned. It only fires when a caller passes `refine=true` on the CPU — a path the default never takes and no test covered. Caught by exercising the override deliberately. Fixed with `oftype`. Verified: CPU `refine=true` now runs, 470 → 800 iterations, true residual 9.9e-11 → 4.5e-16, $\tau$ unchanged to 9 digits.

## A harness integration defect this fix introduces

`benchmarks/bench_tortuosity.jl:188` traces the ladder by passing a callback into the package solve:

```julia
sol = quiet_solve(sim, KrylovJL_CG(; callback=cb); precond=precond, ..., abstol=0.0)
```

Refinement reuses the solve's `LinearCache`, and that cache carries the algorithm — **including the callback**. So with refinement on, `cb` fires during the correction rounds too, where `workspace.x` is the correction `δ`, not the solution. `trace_case` would then compute a tortuosity from `δ` and push it as a ladder row. Those rows would be nonsense.

It cannot be fixed by swapping the algorithm inside `_refine`: `LinearCache` is parameterised on the algorithm type, and a callback-free `KrylovJL` is a different type, so the field cannot be reassigned.

The right fix is on the harness side, and it is needed anyway: `trace_case` must pass **`refine=false`** and run the refinement rounds itself, so that ladder rungs count iterations cumulatively across rounds. That is exactly the shape already validated in the $200^3$ ladder measurement above — evaluate $\tau$ at rungs against `base + workspace.x`, where `base` is the accumulated iterate at the start of the current round.

Until that lands, **any benchmark number produced with refinement on is invalid**. The measurements in this document were taken through direct calls, not through `trace_case`, so they are unaffected.


## The outage, and a diagnosis that had to be corrected twice

**Diagnosis corrected twice.** I first read the outage as a dropped VPN, on the strength of `tun0` being up but `pmealsrv1` timing out during banner exchange. I then corrected that to "host down" on the strength of `uwaterloo.ca` answering "through the same tunnel" — **and that evidence was itself invalid**: `uwaterloo.ca` resolves to IPv6 and routes over the wifi interface, not `tun0`, so it never tested the tunnel at all. Forcing `ping -4 -I tun0` at it also fails, because the VPN is split-tunnel and only carries UW-internal subnets.

The valid test is a UW-internal address. The VPN pushes routes for `10.40.0.0/18`, `10.200.0.0/16`, `129.97.0.0/16` and, specifically, `129.97.2.1` / `129.97.2.2` — UW's internal DNS:

| check | result |
|---|---|
| openconnect session | CSTP + DTLS connected, `10.40.32.191`, valid to Sep 2 |
| route to `129.97.161.145` | `dev tun0` — correct |
| ping `129.97.2.1` (internal DNS) | **0% loss, 56 ms** |
| `dig @129.97.2.1 pmealsrv1.uwaterloo.ca` | returns `129.97.161.145` — address confirmed |
| ping `129.97.161.145` | **100% loss** |

The tunnel carries internal traffic and UW's own DNS resolves the host. **`pmealsrv1` itself was down** and needed someone on site. It came back on 2026-08-20 with an uptime of 20 h, which puts the reboot inside the outage window and confirms the diagnosis.

A watcher polled it every two minutes for 150 minutes and it never answered; the outage ran roughly four hours, with `uwaterloo.ca` still replying at 4 ms through the same tunnel throughout. Watching was stopped — it was a fault someone had to clear physically, not one that resolves itself.

**The lesson is the one this whole document is about.** Both wrong diagnoses were stated as conclusions before anything tested them, and the second was *correct by luck* on evidence that could not have supported it: `uwaterloo.ca` never crossed `tun0` at all. A right answer from an invalid test is worse than a wrong one, because nothing prompts you to look again.

**The server's copy of the file, resolved 2026-08-20.** `/tmp/simulations.jl.orig` did not survive the reboot, as expected. It did not matter: git holds the pristine version, and the pre-existing ` M` was line endings only, content identical to `b070198`.

What the reboot did expose is that the copy left on the server was **the first version of the fix, before the three defects below were found** — it still carried the bare `cache.reltol` assignment, still reported the un-narrowed iterate's residual, and never restored the cache. Anything measured with it would have been measured against code that no longer exists. The current version was put in its place, and `git -C ~/Tortuosity.jl checkout -- src/simulations.jl` restores the tree once the measurements are done.

One further thing about that checkout: its `joss` branch sits five commits behind at `b070198`, so the *history* is missing the coarse-space and Int32-index work. The *working tree* is not — `src/preconditioner.jl`, `src/matrixfree.jl`, `src/kernels/sparse.jl`, `src/pdetools.jl` and `Project.toml` are byte-identical to the local tree once line endings are normalised, because the tree was synced by content rather than by pull. Check that before trusting a server number; the branch name alone will mislead you.


### What the package-level run did confirm before the host went

`n1000_b100_p020`, matrix-free, 184,187,030 nodes, through the shipped code path:

| arm | iters | best-of-2 | tau | rel_err | peak live |
|---|---|---|---|---|---|
| plain | 205 | 22.482 s | 13.1856858513 | **1.498592e-03** | 9.900 GB |
| refined | 631 | 60.648 s | 13.1660094046 | **4.098043e-06** | 14.320 GB |

Extra memory **24.0000 B/node exactly** (4,420,487,618 / 184,187,030), confirming the corrected figure rather than the 20 my laptop sampler reported. `resid` came back as a plain number on the refined arm and a `RefValue` on the plain one, as designed. **No `@warn` fired, nothing thrown, exit 0.**

Cases 2–5 never ran, so the memory guard was still unexercised at 950 M nodes when the host went away. It has since been measured, and the section below replaces this gap.


### The memory guard, measured (2026-08-20, server Quadro RTX 8000)

This was the one claim in the document with nothing behind it. It now has a model, four predictions recorded before anything ran, and four measurements.

**The model.** The campaign's own memory stage already contained the answer and nobody had read it that way. Fitting `bytes = a·nodes + b·voxels` to the two extreme porosities of the $800^3$ matrix-free GPU sweep gives $a = 32.02$ B per pore node and $b = 4.003$ B per voxel, and those two numbers then reproduce **all five** measured porosities to $\pm 0.01\%$:

| case | nodes | predicted | measured | error |
|---|---|---|---|---|
| `n800_b100_p020` | 94,937,803 | 5.0893 GB | 5.0893 GB | +0.00% |
| `n800_b100_p040` | 202,068,918 | 8.5198 GB | 8.5205 GB | −0.01% |
| `n800_b100_p060` | 306,846,383 | 11.8750 GB | 11.8758 GB | −0.01% |
| `n800_b100_p080` | 411,032,289 | 15.2111 GB | 15.2117 GB | −0.00% |
| `n800_b100_p095` | 487,445,980 | 17.6580 GB | 17.6580 GB | +0.00% |

Neither coefficient is a fitted artefact: 32 B/node is eight `Float32` vectors, and 4 B/voxel is the single `Int32` index map over the full grid that the paper already claims for the matrix-free form. **That claim is now measured rather than asserted.**

**The out-of-sample test.** The campaign's memory stage covers $1000^3$ as well — a fact I initially missed by grepping `^n1000` when `case_id` is the *fifth* CSV field, the same column error made earlier in this investigation. Those five points were not used in the fit, and they sit a 2.1x jump in voxel count away from it:

| case | nodes | predicted | measured | error |
|---|---|---|---|---|
| `n1000_b100_p020` | 184,187,030 | 9.9009 GB | 9.9000 GB | +0.009% |
| `n1000_b100_p040` | 395,251,340 | 16.6593 GB | 16.6598 GB | −0.003% |
| `n1000_b100_p060` | 601,833,672 | 23.2743 GB | 23.2752 GB | −0.004% |
| `n1000_b100_p080` | 805,226,521 | 29.7872 GB | 29.7877 GB | −0.002% |
| `n1000_b100_p095` | 950,636,264 | 34.4433 GB | 34.4432 GB | +0.000% |

**What the server runs added, stated precisely.** Two of them re-measured the plain solve at $1000^3$ and returned 29.7877 and 34.4432 GB — the campaign's own values to six significant figures, on a different day and against changed code. That is a reproducibility check, not a new data point, and it should not be presented as one. The allocation pattern of a Krylov solve is deterministic, so this is what a correct measurement *ought* to look like; it is worth recording because it is also the check that would have caught a contaminated run.

**The genuinely new predictions were all about refinement**, because nothing had ever measured that. The card reports `CUDA.total_memory()` = 47.27 GiB = 50.74 GB, not the 48 GiB on the box — which matters, because the margins are about 1 GB wide.

| case | predicted | measured |
|---|---|---|
| `n1000_b100_p080` refined, 24 B/node | 49.12 GB | **49.1131 GB** — 1.63 GB spare |
| `n1000_b100_p095` refined | needs 22.8 GB, has 16.3 → **the guard must fire** | **`guard_fired=true`**, exit 0 |
| `n1000_b100_p080` refined, 20 B/node | 45.89 GB | **45.8922 GB** — 4.85 GB spare |

**What the guard did.** On `n1000_b100_p095` the `@warn` fired, `status=ok`, the process exited 0, and the returned solution was the unrefined one — `resid = 1.709e-3` on both arms, identical to the last digit, which is the signature of the fallback returning `sol` untouched rather than a half-refined vector.

The peak also identifies *which* allocation failed, and the arithmetic is exact:

| case | refined peak − plain peak | per node |
|---|---|---|
| `n1000_b100_p080` (refined) | 19.3254 GB | **24.0000 B/node** — the full cost |
| `n1000_b100_p095` (guard fired) | 15.2101 GB | **16.0000 B/node** — two `Float64` vectors, nothing else |

So `x64` and `r64` allocated and `correction_rhs` (3.803 GB) failed with 1.09 GB free. That is the third of the three allocations inside the `try`, which is where a guard is supposed to catch it. The 24.0000 B/node on `p080` also confirms at 805 M nodes what the earlier run measured at 184 M nodes — the same figure to four decimal places, at 4.4x the size.

**Refinement works at this scale.** On `n1000_b100_p080` the true residual went from 3.431e-3 unrefined to **5.577e-7** refined, a factor of 6150, on an image of 805 million pore nodes.

**A hole the measurement exposed, now closed.** Refinement made a *fourth* allocation — narrowing the `Float64` iterate back to `Float32` — and that one sat **outside** the `try`. On a card where the three guarded allocations succeed and the fourth does not, the guard would have promised a warning and thrown instead. The window is about 3.8 GB wide out of 50, but it is real, and it is precisely the failure mode the guard exists to prevent.

It is also unnecessary. By that point `sol.u` holds nothing but the last correction, so narrowing into it costs no allocation at all. Refinement now needs **20 B/node** — two `Float64` vectors and one `Float32` — instead of 24; every allocation it makes is inside the guard; and the vector it returns aliases the cache exactly as the non-refined path's already does (verified: `refined.u === refined.cache.u`). The values are unchanged bit for bit, so nothing measured for accuracy or determinism needed re-running.

**Measured after the change, not assumed.** The prediction was recorded first — `n1000_b100_p080` refined should peak at 29.7877 + 16.1045 = 45.89 GB, a delta of exactly 20.0000 B/node — and then re-run on the server against the changed code:

| case | peak, 24 B/node | peak, 20 B/node | delta per node | card margin |
|---|---|---|---|---|
| `n1000_b100_p080` | 49.1131 GB | **45.8922 GB** | **20.0000 B/node** | 1.63 GB → **4.85 GB** |
| `n1000_b100_p095` | guard fired, 49.6533 GB | guard fired, 49.6533 GB | — | still does not fit |

`iters = 527` and `resid = 5.577e-7` came back identical on both, which is the check that the change is value-neutral: it moves where a vector lives, not what is in it. `p095` peaks at the same figure either way, because the guard still fires on the same allocation — `correction_rhs` does not change size.

That also moves the ceiling. At $1000^3$ on this card, refinement fits up to $\varepsilon \approx 0.83$ at 24 B/node and up to $\varepsilon \approx 0.90$ at 20 B/node, and `p080`'s margin goes from 1.63 GB to 4.85 GB.

**The constraint binds where it does not matter, and that is structural rather than lucky.** Refinement is needed on ill-conditioned images, which are the *low*-porosity ones; memory runs out on high-porosity images, because that is where the pore nodes are. The two run in opposite directions:

| $\varepsilon$ | worst GPU error anywhere in the campaign | $1000^3$ nodes (b100) | refinement at $1000^3$, 20 B/node |
|---|---|---|---|
| 0.20 | $1.29\times10^{-3}$ — **the only band that fails** | 184 M | 3.7 GB needed, 40.8 GB free — **11x over** |
| 0.40 | $4.92\times10^{-4}$ | 395 M | fits |
| 0.60 | $7.91\times10^{-4}$ | 602 M | fits |
| 0.80 | $8.05\times10^{-4}$ | 805 M | fits, 4.9 GB spare |
| 0.95 | $5.76\times10^{-4}$ — already inside target | 951 M | **does not fit** |

All three cases the campaign recorded as `ladder_exhausted` — `n200_b050_p020`, `n800_b200_p020`, `n1000_b100_p020` — are `p020`, and each has an order of magnitude more headroom than refinement asks for. The band where refinement cannot run is the band that never needed it.

**A negative result worth keeping.** The guard could not be exercised on the laptop by holding ballast on the card so the refinement buffers would fail while the base solve still fitted. CUDA.jl runs a full GC and `reclaim()` when an allocation fails, which collects the ballast and lets the solve through — across free-memory targets from 0.45 GiB down to 0.012 GiB against a case needing 0.170 GiB, every attempt refined normally. That window cannot be held open from Julia. Exercising it took a real 950 M-node image on a real 48 GB card, which is why this measurement waited for the server.

**One instrument note.** A first attempt at the local $300^3$ check read `CUDA.memory_stats().live` after the solve returned and got 18.0 B/node for something that is exactly 20. The buffers are garbage by then but not yet collected, so that reading measures GC timing, not the algorithm. Peak sampling during the solve is the only correct instrument here, and it is what `with_peak_sampling` in the harness already does.


### GPU campaign re-run: timings (2026-08-20, both stages exit 0)

Both timing sweeps completed. `timings_tortuosity_gpu_matrixfree` 12:56–14:36, `timings_tortuosity_gpu_assembled` 14:36–15:32.

**All three cases the campaign recorded as `ladder_exhausted` now reach the target.** These were the entire accuracy failure of the GPU path:

| case | operator | before | after |
|---|---|---|---|
| `n200_b050_p020` | matrix-free | 1.289e-3, `ladder_exhausted`, 0.361 s | **7.594e-6**, rung 189, 0.372 s |
| `n1000_b100_p020` | matrix-free | 1.455e-3, `ladder_exhausted`, 54.35 s | **4.183e-6**, 72.01 s |
| `n800_b200_p020` | assembled | 1.636e-3, `ladder_exhausted`, 49.74 s | **3.313e-6**, rung 1086, 74.77 s |

None of the three "before" times was a time-to-target — those solves never reached the target at any rung — so the time column is not a regression, it is the first measurement of something that previously had no value.

**Nothing else moved.** At $1000^3$, thirteen of the fourteen matrix-free cases returned a relative error **bit-identical** to the archived run — 4.920e-04, 7.914e-04, 2.965e-04, 8.593e-05, 8.695e-05, 7.067e-04, 1.168e-04, 7.459e-05, 9.111e-04, 9.524e-06, 1.141e-04, 8.367e-05, 6.192e-05 — every digit, with wall clocks 1–4% apart. Determinism is intact and the fix perturbs nothing it was not meant to touch.

The assembled operator's out-of-memory cases are also unchanged: the same four cases at $800^3$, requesting the same 1.541 / 1.819 / 1.531 / 1.816 GiB at 99.98% of 47.268 GiB. Those failures happen during assembly, before any solve, so refinement cannot reach them.

**Coverage, cases reaching the 0.1% target:**

| | $200^3$ | $400^3$ | $600^3$ | $800^3$ | $1000^3$ |
|---|---|---|---|---|---|
| matrix-free, before | 14 | 15 | 15 | 15 | 13 |
| matrix-free, after | **15** | 15 | 15 | 15 | **14 of 14** |
| assembled, after | 15 | 15 | 15 | 8 (7 OOM) | 5 (9 OOM) |

**The advantage over taufactor rises, and the mechanism is the recovered cases.** Paired geometric mean over cases both tools solve to 0.1%, GPU, with the paired count in brackets:

| | $200^3$ | $400^3$ | $600^3$ | $800^3$ |
|---|---|---|---|---|
| matrix-free, before | 1.58x (14) | 7.01x (15) | 10.28x (14) | 10.24x (14) |
| matrix-free, after | **2.05x (15)** | 7.12x (15) | 10.40x (14) | **10.55x (14)** |
| assembled, before | 1.96x (15) | 6.37x (15) | 9.00x (14) | 14.72x (7) |
| assembled, after | **2.05x (15)** | 6.44x (15) | 9.12x (14) | **15.25x (8)** |

Where the fix recovered a case the mean rises — 30% at $200^3$ matrix-free, and $800^3$ assembled gains a pairing. Everywhere else the change is 1–3%, inside the cross-session drift measured above. This is the speed constraint answered: repairing the accuracy defect **increased** the measured margin rather than spending it.

**One thing the paper currently gets wrong about the shape of that curve.** It says the margin "widens with image size". Matrix-free goes 2.05, 7.12, 10.40, 10.55 — it rises steeply to $600^3$ and is then flat to $800^3$ (+1.4%, inside drift). Whether it resumes rising at $1000^3$ cannot be answered from any existing data, because taufactor has no $1000^3$ rows at all. That sweep is queued.

**The log is clean.** Zero guard warnings across all 75 matrix-free cases, and no errors or warnings beyond the routine `max_iters` notice the ladder produces by design.


### GPU campaign re-run: memory (2026-08-20, both stages complete)

The memory stage measures the solve *with* refinement, since `solve_case` does not pass `refine`. Against the archived figures, refinement costs **20.00 B per pore node** at 23 of the 25 matrix-free cases — the algorithm's cost, now measured across five sizes and five porosities rather than derived.

The two exceptions are the guard, on different operators and failing at different allocations:

| case | delta | reading |
|---|---|---|
| `n1000_b100_p095`, matrix-free | 16.00 B/node | fired at the **third** allocation — `x64` and `r64` fit, `correction_rhs` did not |
| `n1000_b100_p040`, assembled | **0.00 B/node** | fired at the **first** — 2.23 GB free against `x64` needing 3.16, so the peak never moved |

The assembled figure was predicted from card headroom before the row was read, and the prediction was that the peak would not move at all. It did not. Together with the earlier standalone measurement this is three independent demonstrations of the guard, at three different allocation sites.

**Do not read "no warnings in the log" as "the guard did not fire".** `bench_tortuosity.jl` wraps every solve in `quiet_solve`, which sets the logger to `Error` to suppress the ladder's per-rung `max_iters` notice — and that suppresses the guard's `@warn` along with it. The stage log shows zero warnings while the guard fired twice. The B/node delta is the evidence; a user outside the harness would see the warning.

**The assembled operator's memory is noisier and the reason is structural.** Its `p020` cases come back at 16.2–16.6 B/node at every size from $400^3$ up, despite having ample room to refine. For the assembled form the *assembly transient* sets the peak, so refinement only appears in the delta to the extent it exceeds that transient — which is exactly where the solve vectors are smallest relative to the assembled structure, at the lowest porosity. Matrix-free has no such transient and reports the flat 20.00.


### The fix raises the measured advantage over taufactor, it does not cost it

The speed constraint was that repairing the accuracy defect must not spend the margin over taufactor. It does the opposite, and the mechanism is worth stating because it is not obvious.

A case only enters the paired comparison if **both** tools reach the 0.1% target. `n200_b050_p020` never did on our side, so it was excluded — and it is the case where taufactor struggles most. Now that it converges, it joins the pairing at 48.6x:

| `n200_b050_p020` | time to 0.1% |
|---|---|
| Tortuosity.jl, GPU matrix-free | 0.372 s at rung 189 |
| taufactor | 18.094 s at 11,169 iterations |

The effect on the $200^3$ headline, measured on the same file:

| paired set | cases | geometric mean |
|---|---|---|
| excluding the recovered case — what the old code could report | 14 | 1.633x |
| including it — what the fixed code reports | 15 | **2.048x** |

So the $200^3$ advantage rises about 25%, from roughly 1.6x to roughly 2.0x. The 1.633x also cross-checks against the archived 1.58x: the two differ by 3%, inside the ~10% cross-session drift measured above, which is what says the re-run is measuring the same thing the campaign did.

**Why an accuracy fix moves a speed number.** Excluding a case for non-convergence is not neutral — it removes precisely the ill-conditioned images where a Krylov method beats a stationary one by the widest margin, so the reported mean was biased *against* us by our own defect. Fixing the defect removes the bias. The same should hold at every size that had a `ladder_exhausted` case, which is $800^3$ assembled and $1000^3$ matrix-free.


### Cross-session drift on the server is small enough to compare across, measured

A worry worth settling before any of the re-run numbers get used: the taufactor timings were measured on 16–18 August and the tortuosity timings are being re-measured on 20 August, so every published ratio is a cross-session comparison. The standing rule from the local card is that GPU wall clock drifts up to 25% between sessions, which would put a 10x advantage anywhere between 7.7x and 12.9x.

That rule does not transfer, and the re-run itself measures why. A case that reaches the target at the same ladder rung in both runs did **identical work** — the traced sweep runs with `refine=false`, so refinement does not enter a rung that the plain solve already carried. Any difference in its wall clock is drift and nothing else. Thirty such cases had completed at the time of writing:

| | value |
|---|---|
| cases at an identical rung | **30 of 30** |
| geometric mean new/old | **0.974** |
| range | 0.901 to 1.064 |

So the server drifts by about 2.6% systematically and 10% at worst, against the 25% measured on the laptop. The difference is explained: the laptop's RTX PRO 5000 also drives the display and its clocks move with whatever else is on screen, while `pmealsrv1`'s Quadro RTX 8000 is a dedicated compute card in a machine reporting zero logged-in users. **Comparing the new tortuosity timings against the archived taufactor timings is therefore sound**, and a 10x advantage stays within about 9x to 11x.

The 30-of-30 identical rungs are a second result in their own right. They say the fix does not perturb convergence on any case that already converged — refinement runs only where the plain solve fell short — which is the mechanism behind the time-to-target claim holding.


## Where this stands (2026-08-20)

Steps 1 and 2 of the previous resume list are done, and doing them turned up two things worth knowing.

**The server was holding the wrong version of the fix.** Not the pristine file, and not the current fix either — the *first* version, written before the three defects below were found. Anything measured against it would have been measured against code that no longer exists. That is the second time in this investigation that a stale artefact would have produced a confident, wrong number, and it is why the sync-and-verify step is written out explicitly below rather than assumed.

**The memory guard is measured**, and the measurement found a hole in it. Both are recorded under "The memory guard, measured" above. Refinement now costs 20 B/node rather than 24, and every allocation it makes sits inside the guard.

### Results are home, and one file is still growing (2026-08-20 18:20)

The tortuosity GPU re-run is complete and **retrieved to the laptop**, verified file by file against the server rather than assumed — all 19 result files plus the manifest match on row count. Local `benchmarks/results/` was stale before this (128 rows against the server's 633), so today's work had a single copy for about four hours. It no longer does.

**`results/timings/taufactor-gpu.csv` is the exception and must be pulled again.** The $1000^3$ sweep was still running when the copy was taken, so the local file holds 787 rows of which the $1000^3$ part (57 rows) is partial. Everything else is final. `results/memory/taufactor-gpu.csv` has no $1000^3$ rows at all yet — that stage runs after the timings.

The pre-fix figures are preserved under `results/archive/pre-refine-2026-08-20/`, locally and on the server, and every before/after comparison in this document was computed against them.

**The VPN dropped at about 18:15 and was restored.** The jobs were unaffected — they run under tmux, which is the whole reason the runbook insists on it. Watch for `Connection timed out during banner exchange` on `pmealsrv1` while `zaboor` still answers: that is the tunnel, not the host, and `~/scripts/uwvpn.sh` on `zaboor` fixes it with a Duo push Amin has to approve within about a minute.

### Still running on the server

One script, `/tmp/run_remaining.sh`, in tmux session `remaining`, logging to `/tmp/remaining.log`:

1. **taufactor $1000^3$**, timings then memory. Started 16:48, projected about 2.1 h over 14 cases, with two expected to hit the 30-minute per-case ceiling and return `timeout`. Predictions are recorded in `paper/NUMBERS-TO-VERIFY.md` before the fact: memory should land at 28.08 GB, and the memory comparison should come out three of five without refinement and two of five with.
2. **`measure_iters.jl`**, which follows it automatically. Iterations to a fixed relative residual, preconditioned and unpreconditioned, over the cached campaign images. This is what settles the sentence at `paper.md` line 55 — and it is the falsification test for the claim that the preconditioner is mesh-independent in the residual while only the target metric grows.

Both sweeps were originally queued as separate scripts waiting on each other through `pgrep`. That deadlocked, and clearing it cost both queues. The single-script form above has no inter-process guard to fail.


### An independent review, and one of its conclusions disproved (2026-08-20 19:00)

An independent reviewer read the whole change at high effort and raised twelve findings. Six it fixed, six it left as judgement calls. Two of its fixes touch the solve path, so both were re-verified here rather than taken on trust.

**Both solve-path fixes are sound, and both are dormant.**

*The zero right-hand-side guard.* `resid = norm(r64) / nb` with `nb == 0` is `NaN`, and `NaN > shrink * prev` is false, so no round would ever break: all eight correction solves would run and the returned `resid` would be `NaN` for an answer that is exactly right. Real images never have a zero right-hand side, so this cannot affect a measured number.

*The rollback of a worsening correction.* The stopping test fires **after** the correction has been added, so a correction that grew the residual was kept. The fix subtracts it back off. Two things had to be checked before trusting it:

1. **Does it ever fire?** Traced round by round on the hardest class of image — $160^3$, blobiness 0.5, `Float32`, GPU — at three porosities:

| $\varepsilon$ | round 1 | 2 | 3 | 4 | 5 | 6 | grew? |
|---|---|---|---|---|---|---|---|
| 0.20 | 2.367e-6 | 2.192e-7 | 2.087e-8 | 1.921e-9 | 3.966e-10 | 2.664e-10 | never |
| 0.40 | 2.168e-6 | 1.878e-7 | 1.561e-8 | 1.502e-9 | 3.245e-10 | 2.222e-10 | never |
| 0.95 | 2.152e-6 | 1.831e-7 | 1.797e-8 | 1.697e-9 | 1.712e-10 | 1.249e-10 | never |

Refinement always stops on **stagnation** — the residual is still falling, just by less than half — and never on growth. The branch is a safety net that no measured case reaches, and no campaign number moves.

2. **Is its aliasing assumption true?** It subtracts `sol.u`, assuming that is the same array as the last correction. A dormant branch is precisely the one nothing else would ever catch, so this was verified directly: `sol.u === cache.u === correction.u` is `true` on the GPU and on the CPU. It holds.

`test_gpu_e2e.jl` is green, all three refinement testsets included.

**The reviewer's parting warning is wrong, and the data says so.** It flagged that its `bench_tortuosity.jl` fix — adding `setup_s` back into the padded GPU rungs — "changes numbers the current GPU campaign already produced". The accounting bug is real: the padded branch started a fresh clock after the problem was built, while every checkpointed rung carries construction. But it changed nothing, because **the padded branch never executed**:

- Zero `(case, tau, time_s)` groups appear more than once in either GPU file. Padding writes one identical row per remaining rung, so a single padded run would leave a duplicate group. There are none.
- Every `target_reached` row in both GPU files is a real checkpoint, whose `elapsed = mark - t0 - excluded[]` already includes construction.

The reason is the `abstol = 0` change itself: with no absolute tolerance the solve no longer gives up early, so no unvisited rungs are left to pad. `n200_b050_p020` reaching 7.594e-6 at rung 189 — the case that lifts the $200^3$ mean from 1.633x to 2.048x — is a genuine checkpoint, not a padded re-solve, and its 0.372 s already includes the build.

The same holds for the reviewer's `bench_taufactor.py` fix, which stopped a failed timing case being published as `oom`: **no `oom` row exists in any timing file**, for any tool. The five `oom` rows in the memory results come from `bench_tortuosity.jl`, which always wrote the right label.

**One finding is blocking and cannot be fixed from here.** `.gitmodules` pins `benchmarks/vendor/taufactor` at `d05aa2e`, but the checkpointing patch that `bench_taufactor.py` depends on — `solve(..., checkpoints=, checkpoint_hook=)`, 46 lines — exists **only as an uncommitted working-tree edit** in the submodule. A reviewer cloning `--recursive` gets a taufactor without those keyword arguments, `run/setup.sh` asserts on it, and not one published taufactor number can be reproduced. The patch has to be committed and pushed to the fork, and the submodule re-pinned, before submission.

### Still open, in order

**Running as of 2026-08-20 12:56 server time.** `./run/campaign.sh --grid=full --tools=tortuosity --devices=gpu --stages=timings,memory --overwrite`, in tmux session `campaign`, log at `/tmp/campaign_gpu_rerun.log`. The pre-fix GPU files are archived under `results/archive/pre-refine-2026-08-20/{timings,memory}/`, verified byte-identical to the originals before the run started.

Note that the memory stage calls `solve` without `refine`, so the new memory numbers include refinement and the archived ones do not. That is deliberate: it leaves both on disk, which is what the reporting decision in `paper/NUMBERS-TO-VERIFY.md` needs.

First result in, and it is the case this whole document exists for. `n200_b050_p020`, GPU matrix-free:

| | rung | $\tau$ | rel err | time | stop_reason |
|---|---|---|---|---|---|
| archived (pre-fix) | 20000 | 33.88715896 | 1.29e-3 | 0.361 s | `ladder_exhausted` |
| re-run | 189 | 33.93181232 | **7.59e-6** | **0.372 s** | `target_reached` |

170x more accurate for 3% more wall clock, and $200^3$ coverage goes 14/15 to **15/15**. The 0.361 s was never a time-to-target — that solve never reached the target at any rung.

**1. Re-run the GPU half of the campaign.** Amin's call, roughly 3.7 h of the 27 h campaign. Every GPU timing predates the fix; CPU is unaffected, because refinement is off at `Float64` by design and by measurement.

Before anything overwrites them, archive the current result files:

```bash
ssh pmealsrv1 'cd ~/Tortuosity.jl/benchmarks/results && mkdir -p archive/pre-refine-2026-08-20 && cp timings/tortuosity-gpu-*.csv memory/tortuosity-gpu-*.csv archive/pre-refine-2026-08-20/'
```

Then sync **both** changed files and confirm they arrived, rather than assuming:

```bash
# from the laptop, normalising line endings
for f in src/simulations.jl benchmarks/bench_tortuosity.jl; do
  tr -d '\r' < $f | ssh pmealsrv1 "cat > ~/Tortuosity.jl/$f"
done
ssh pmealsrv1 'cd ~/Tortuosity.jl && grep -c "u = sol.u" src/simulations.jl && grep -c "refine=false" benchmarks/bench_tortuosity.jl'
```

Driver shape, one size per invocation: `./run/campaign.sh --grid=full --sizes=$s --tools=tortuosity --devices=gpu --stages=timings,memory`. Read `benchmarks/run/ORCHESTRATION.md` first, and serialise the sizes against each other — two GPU jobs overlapping is how `n800_b200_p040`'s wall clock was lost the first time.

**Restore the server file afterwards**, and note that `git checkout` will also drop the pre-existing CRLF-only modification, which is harmless:

```bash
ssh pmealsrv1 'git -C ~/Tortuosity.jl checkout -- src/simulations.jl'
```

**2. Then `paper/paper.md`.** `paper/NUMBERS-TO-VERIFY.md` carries the detail. Three things change, and one thing must *not* be written:

- GPU coverage at $1000^3$ becomes 14 of 14, matching the CPU. (14, not 15, because `n1000_b050_p020` trims to **zero** pore nodes — it does not percolate at all, so there is nothing to solve. It is absent from the CPU sweep for the same reason. That is a degenerate image, not a failure, and the paper should not imply a missing case.)
- The accuracy caveat the paper was going to carry — use `Float64` on the CPU for strongly tortuous images — is no longer true and **must not be written**.
- The memory paragraph's claim that the matrix-free form "holds one `Int32` per grid voxel" is now *measured*: 4.003 B/voxel, plus 32.02 B/pore node, reproducing five porosities to $\pm0.01\%$. It can be stated as a measurement.
- Line 79's sentence, "It completes every case at $1000^3$ on a 48 GB card", needs a qualifier: at $\varepsilon = 0.95$ the refinement buffers do not fit, so that case completes *unrefined*, with a warning. It does not need refining — its error is already $5.8\times10^{-4}$ — but the sentence as written now implies something the code will warn about.

Also recompute the "roughly $36\times$ faster than its own CPU path" figure. It is a geometric mean over the cases both paths solve, and the GPU set gains a case (`n1000_b100_p020`) that previously never reached target.

**3. Optional, and probably not worth it: 16 B/node.** Widening the matvec accumulator so `mul!(y32, A32, x64)` accumulates in `Float64` would drop `r64`, leaving 16 B/node — 15.2 GB at 951 M nodes against 16.3 GB free, so $1000^3$ at $\varepsilon = 0.95$ would refine. But it touches a hot kernel, it risks the speed constraint, and the case it unlocks is well conditioned and does not need refining. The measured anti-correlation above is the reason to leave this alone: memory runs out exactly where accuracy is already fine.

### State of the working tree

Modified and **uncommitted**: `src/simulations.jl` (the fix, plus today's 20 B/node change), `test/test_gpu_e2e.jl` (three regression testsets), `benchmarks/bench_tortuosity.jl` (the `trace_case` fix — note this file also carries unrelated pre-existing staged/worktree differences). New: this document. Nothing has been committed, per standing instruction.

### Checklist

- [x] Root cause established by falsification, not assertion.
- [x] Fix implemented in `src/simulations.jl`, validated on the full $200^3$ grid (1/15 → 0/15 failures) and at $800^3$ / $1000^3$.
- [x] Harness callback defect found and fixed in `trace_case`, validated end to end through `run/campaign.sh` on both a clean grid and the failing case.
- [x] Cache state restored after refinement, verified by re-solving the same problem and reproducing exactly.
- [x] Memory guard measured on a real 950 M-node image: fires, warns, returns the unrefined solution, exits 0.
- [x] The fourth, unguarded allocation found by that measurement and removed — refinement is 20 B/node and fully guarded.
- [x] `Pkg.test()` green with every edit in place — **12,644 assertions across 20 top-level testsets, 0 failures, 0 errors** — including three new regression testsets.
- [ ] Re-run the GPU half of the campaign and recompute the taufactor margin per size. Amin's call.
- [ ] Update `paper/paper.md` itself.

## Orchestration note

Two subagents were run against the same GPU and their jobs overlapped between 02:06 and 02:10 server time. Accuracy, iteration counts and per-process peak memory are unaffected — all are deterministic or per-process — but the wall clock for `n800_b200_p040` in the first run is contaminated and is not used anywhere above. My error: each agent was told to serialise its own jobs, and nothing serialised them against each other.
