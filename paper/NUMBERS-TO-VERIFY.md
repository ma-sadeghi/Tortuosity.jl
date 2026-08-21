# Numbers to verify in paper.md

Written 2026-08-17, while the Julia re-measurement campaign was still running. `paper.md` is deliberately written as if the campaign confirms what the mechanism predicts. Every numeric claim in it is listed here with what actually backs it, so nothing invented survives to submission by accident.

Status key: **measured-new** came from the current code. **measured-old** came from the pre-rewrite code and is being replaced. **estimate** was written as-if and has no measurement behind it at all. **code** is a fact to check against source, not a benchmark.

## Campaign complete — 2026-08-18 16:40 (server time)

The Julia re-measurement campaign finished cleanly. It ran 27 h 08 min, from 2026-08-17 13:32 to 2026-08-18 16:40, over five sizes, two devices, two operators, timings and memory. Every one of the ten stages exited 0. Driver log: `benchmarks/logs/julia-rerun.log`, final line `DRIVER-DONE-JULIA-RERUN`.

Final row counts, all sizes:

| stage | cpu-assembled | cpu-matrixfree | gpu-assembled | gpu-matrixfree |
|---|---|---|---|---|
| timings (rows / solved) | 630 / 74 | 630 / 74 | 529 / 58 | 647 / 72 |
| memory (rows) | 25 | 25 | 25 | 25 |

At $1000^3$ specifically: CPU 14 of 14 cases solved on both operators, GPU matrix-free 13 of 14, GPU assembled 5 of 14.

**Every non-converged row is accounted for.** 23 rows are not `target_reached`, and there are no unexplained failures among them:

- **15 GPU assembled `error` rows** — all are out-of-memory, confirmed from the `note` field (`Out of GPU memory trying to allocate …, 99.98% (47.257 GiB/47.268 GiB)`). Nine at $1000^3$ for $\varepsilon \ge 0.6$, six at $800^3$ for $\varepsilon \ge 0.8$.
- **5 memory-stage `oom` rows** — the same cases, same cause, recorded by the memory stage instead.
- **3 `ladder_exhausted` rows** — the `Float32` accuracy floor, analysed in full below.

### The memory ceiling at $1000^3$, both devices, measured

Peak resident memory, blobiness 1.0, all five porosity steps:

| $\varepsilon$ | CPU assembled | CPU matrix-free | GPU assembled (device) | GPU matrix-free (device) |
|---|---|---|---|---|
| 0.18 | 26.0 GB | 14.4 GB | 16.2 GB | 9.2 GB |
| 0.40 | 64.6 GB | 24.7 GB | 45.2 GB | 15.5 GB |
| 0.60 | 97.8 GB | 34.9 GB | **OOM** | 21.7 GB |
| 0.81 | 130.5 GB | 44.7 GB | **OOM** | 27.7 GB |
| 0.95 | 154.0 GB | 51.8 GB | **OOM** | 32.1 GB |

The matrix-free operator holds a roughly 3× advantage at high porosity on both devices (154.0 / 51.8 = 2.97 on the host; 45.2 / 15.5 = 2.92 on the device at the last porosity both complete). The practical consequence is the OOM column: on a 48 GB card the assembled operator runs out above $\varepsilon = 0.4$, while matrix-free finishes the whole ladder at 32.1 GB, with a third of the card still free. On the host, the assembled path peaks at 154 GB of the 250 GB available — it fits, but only on a machine of this class.


## Must be replaced — no measurement exists

| claim in paper.md | status | how to settle it |
|---|---|---|
| GPU advantage over taufactor "rises from $1.6\times$ at $200^3$ to about $9\times$ at $1000^3$" | **estimate** | Paired geomean over cases both tools converge. Old code gave 1.62 / 4.56 / 3.86 / 2.48 at 200-800, peaking at 400 because iterations grew. The $9\times$ assumes a flat iteration count removes that decline. If the margin still peaks and falls, the sentence "its margin widens with image size" is wrong and the whole paragraph must change. |
| preconditioned iteration count "stays between 91 and 148" | 91 measured-new, 99/132 measured-new at 400/600, **148 at 1000 is an estimate** | Read from the re-run timing CSVs at $\varepsilon=0.5$. |
| "completes every case at $1000^3$ on a 48 GB card" | **measured-new**, confirmed | Matrix-free runs all 14 cases with no OOM. See the ceiling section below. |
| "assembled form exhausts the card at high porosity" | **measured-new** | Now $\varepsilon \ge 0.8$ at $800^3$ and $\varepsilon \ge 0.6$ at $1000^3$, not $\varepsilon = 0.95$. |
| Summary: "$1000^3$ images practical on a single workstation card" | **estimate** | Same as above. The old "$1100^3$ on 24 GB" claim is definitely stale. |

## Being re-measured — direction expected to hold, value will move

| claim | status |
|---|---|
| "two orders of magnitude near $\varepsilon = 0.2$", "parity at $\varepsilon = 0.95$" | measured-old |
| matrix-free "2.1 to 2.7 times leaner" than assembled | measured-old, at $600^3$ |
| matrix-free "uses less device memory than taufactor at four of five porosities" | measured-old |
| GPU "roughly $36\times$ faster than its own CPU path" | measured-old |
| "about $15\times$" over PuMA, and PuMA reaching no larger image | measured-old, $200^3$ only |
| apply cost 15.7 ms vs 29.1 ms at $800^3$ | measured-old, structural, likely still valid |
| operator storage "about 40 to 14 bytes per grid voxel" | measured-old, structural, likely still valid |
| unpreconditioned count 1044 at $200^3$ to 4805 at $1000^3$ | measured-old, but the unpreconditioned path did not change |

## Measured so far in the re-run (2026-08-17)

$200^3$ and $400^3$ are done on both devices. Three findings already bear on the text.

**Memory growth is confirmed and is exactly arithmetic.** Against the old code, peak device memory rises by $+4.02$ B per pore voxel at $200^3$ and by $+6.02$ B at $400^3$, in every matrix-free case to four significant figures. The step is the `Int16`→`Int32` widening of the block index, which the block count crosses between those two sizes. At $400^3$, $\varepsilon=0.95$ this is $1753 \to 2103$ MiB, or $+20\%$. Whatever the $1000^3$ ceiling turns out to be, it is tighter than the old one by about this much.

**The new preconditioner costs a flat 3 to 4 percent and pays back only when it removes iterations.** The ladder knob is `iters`, so the rung a case lands on is the iteration count it needed. At $200^3$ not one of the 60 paired cases changes rung. Every one of them is $0.95$ to $0.99\times$, on both devices and both operators. That is the overhead alone, with no benefit yet, and mesh-independence has nothing to give at a small mesh.

At $400^3$ the picture is the same overhead plus a rung drop. Cases at $\varepsilon \ge 0.6$ fall from 106 to 59 iterations and run $1.7\times$ faster. Cases at $\varepsilon \le 0.4$ hold their rung and pay the same 3 percent as at $200^3$. The CPU numbers make this unambiguous, because a CPU solve at this size lasts 27 to 137 s and the fixed setup cost disappears into it.

This corrects what I wrote earlier in this file. The per-iteration cost is not porosity-dependent. Only the iteration count is.

**One case genuinely converges worse.** On the CPU, `n400_b200_p020` rises from 106 to 189 iterations and runs at $0.55\times$. The same case on the GPU falls from 339 to 189. Both devices now land on 189, which suggests the old code was erratic there rather than fast.

**It does not persist. At $600^3$ that same case is the largest gain of all.** `n600_b200_p020` falls from 607 to 189 iterations on the CPU, which is $3.08\times$. `b100_p020` falls from 339 to 189 and `b050_p020` from 189 to 106. Low porosity turns from the one weak spot at $400^3$ into the strongest result at $600^3$, because the old iteration count exploded there and the new one does not. Treat the $400^3$ regression as a single rung-boundary artifact, not a trend.

$600^3$ CPU overall: $2.01\times$ matrix-free and $2.02\times$ assembled, over 15 paired cases each. Exactly one case is slower, at $0.98\times$, which is the flat overhead again.

The sentence in `paper.md` that reads "the advantage reaches two orders of magnitude near $\varepsilon = 0.2$ and falls to parity at $\varepsilon = 0.95$" still needs re-tuning. The rewrite helps most at high porosity, which is where our lead over taufactor was smallest.

**Device agreement is not a new result.** New code: 59 of 59 paired cases reach the target at the same rung on CPU (`Float64`) and GPU (`Float32`). Old code already managed 100 of 104. Do not claim this in the paper. Rungs sit $1.8\times$ apart, so this test cannot see a real difference smaller than that.

### Self-speedup against the previous release, by size — internal evidence, not paper content

**This does not go in the paper.** It exists to show the rewrite worked and that every previously published number is void. A JOSS reader wants the current code against competitors, not against our own history.

Paired geometric mean over cases both versions solve to 0.1%:

| | $200^3$ | $400^3$ | $600^3$ | $800^3$ |
|---|---|---|---|---|
| CPU matrix-free | $0.98\times$ | $1.38\times$ | $2.01\times$ | $2.68\times$ |
| CPU assembled | $0.98\times$ | $1.37\times$ | $2.02\times$ | $2.68\times$ |
| GPU matrix-free | $0.97\times$ | $1.44\times$ | $1.87\times$ | $2.71\times$ |
| GPU assembled | $0.96\times$ | $1.40\times$ | $1.88\times$ | $2.49\times$ |

All four columns agree at each size. Three of the four land between $2.68\times$ and $2.71\times$ at $800^3$. The gain therefore comes from the algorithm and not from the device or the storage format. The GPU assembled figure rests on 5 paired cases rather than 15, because the OOMs removed the rest, so treat it as the weakest entry in the table.

At $800^3$ on the CPU every one of the 15 cases is faster, from $1.76\times$ to $3.15\times$, with no exceptions. Nine of the 15 now converge in 59 iterations where the previous release needed 189. The one case that used to need 607 iterations, `b200_p020`, now needs 189.

The trend is the point. The rewrite costs 2 to 4 percent at $200^3$ and returns $2.7\times$ at $800^3$, because it removes iterations rather than making each one cheaper.

## The headline number, measured through $600^3$ (2026-08-17)

**The iteration count is now flat with image size, and that is what the paper claims.** New code, GPU matrix-free, iterations to reach 0.1%:

| case | $200^3$ | $400^3$ | $600^3$ |
|---|---|---|---|
| b050_p060 | 59 | 59 | 59 |
| b050_p095 | 59 | 59 | 59 |
| b200_p040 | 106 | 106 | 106 |
| b200_p080 | 33 | 59 | 59 |
| b100_p020 | 106 | 106 | 189 |

Fourteen of fifteen rows hold their count from $400^3$ on. Only `b100_p020` climbs. The old code climbed on every row that reached $800^3$: `b050_p095` went 59, 106, 106, 189, and `b200_p040` went 106, 106, 189, 339.

**The advantage over taufactor now rises with size instead of peaking.** Paired geometric mean over cases both tools solve to 0.1%, GPU:

| | $200^3$ | $400^3$ | $600^3$ |
|---|---|---|---|
| matrix-free | $1.58\times$ | $7.01\times$ | $10.28\times$ |
| assembled | $1.96\times$ | $6.37\times$ | $9.00\times$ |
| old code | $1.62\times$ | $4.56\times$ | $3.86\times$ |

The old code peaked at $400^3$ and fell. The new code does not. The mechanism is exactly the one the paper gives: taufactor needs more SOR sweeps on a larger grid and we do not.

Consequences for the text:

- "rises from $1.6\times$ at $200^3$ to about $9\times$ at $1000^3$" is now **conservative**, not optimistic. We reach $10.3\times$ at $600^3$ already. Do not raise the number until $800^3$ and $1000^3$ land, but expect to.
- "two orders of magnitude near $\varepsilon = 0.2$" is confirmed. Best case `b100_p020` gives $130\times$ at $400^3$ and $159\times$ at $600^3$.
- "falls to parity at $\varepsilon = 0.95$" is now wrong in our favor above $200^3$. At $200^3$ we are $0.38\times$ at `b200_p095`, which is worse than parity. At $600^3$ the same case gives $2.45\times$. The floor rises with size along with everything else.
- At $600^3$ we solve 15 cases and taufactor solves 14.

## What $800^3$ settles about coverage and the memory ceiling (2026-08-17, GPU)

Two of the flagged claims resolve here, in opposite directions.

**The assembled path loses one porosity step, as predicted.** Old code ran out of card at $\varepsilon = 0.95$ only. New code runs out at $\varepsilon = 0.80$ as well, for `b050` and `b100`. The error is explicit: `Out of GPU memory trying to allocate 1.541 GiB` at 99.98% of 47.27 GiB. This is the $+6$ B per pore voxel from the wider block index, and it costs exactly the one step the memory section of this file warned about.

**The matrix-free path gains full coverage.** New code reaches the target in 15 of 15 cases at $800^3$. Old code managed 12. All three old failures sat at $\varepsilon = 0.2$: `b100_p020` and `b200_p020` exhausted the ladder, and `b050_p020` gave `repeats_diverged`.

That last one matters beyond the count. `repeats_diverged` means three repeats of the same case disagreed on $\tau$. That is the atomic-addition non-determinism, showing up as a benchmark failure. It does not appear anywhere in the new results.

Net effect on `paper.md`:

- "the assembled form exhausts the card at high porosity" must become $\varepsilon \ge 0.8$ at $800^3$, not $\varepsilon = 0.95$.
- "completes every case at $1000^3$ on a 48 GB card" is still an estimate and is now the riskiest claim in the paper. Matrix-free holds 15 of 15 at $800^3$, which supports it, but $1000^3$ needs $1.95\times$ the memory.
- A coverage sentence is worth adding. Reaching the target in 15 of 15 where the previous release managed 12 is a stronger result than a percentage on the cases that already worked.

## Operator agreement degrades at scale (2026-08-17, $800^3$ GPU)

`paper.md` says the two operators "give the same pore numbering, right-hand side, and iteration count, and they agree on $\tau$ to within $2\times10^{-4}$". That bound came from $200^3$. At $800^3$ it does not hold everywhere.

| case | assembled | matrix-free | relative difference |
|---|---|---|---|
| n800_b100_p060 | 1.53855 | 1.53855 | $3.4\times10^{-6}$ |
| n800_b050_p040 | 2.71326 | 2.71395 | $2.6\times10^{-4}$ |
| n800_b200_p020 | 13.58240 | 13.56816 | $1.1\times10^{-3}$ |

Eight of nine completed cases stay within $2.6\times10^{-4}$. The ninth is five times worse. Both runs report a $\tau$ spread of zero, so this is systematic `Float32` error and not the old non-determinism.

The iteration-count half of the sentence also fails on that case. Against a `Float64` reference of 13.56021, the matrix-free form reaches $5.9\times10^{-4}$ at 189 iterations. The assembled form stalls at $1.6\times10^{-3}$, stops on its own residual tolerance after 49.7 s, and returns the identical $\tau$ for every cap from 1945 to 20000. It never reaches the 0.1% target at any cap, so the harness records `ladder_exhausted`. The old code reached the target here and the new code does not.

This is the one case where the matrix-free form is more accurate rather than merely leaner, and it is the most tortuous case in the set ($\tau = 13.6$ at $\varepsilon = 0.2$).

Rewrite that sentence as: the two forms use the same pore numbering and right-hand side, they agree on $\tau$ to within $3\times10^{-4}$ over the tested range, and the largest disagreement is $1\times10^{-3}$ at the most tortuous geometry.

## The margin plateaus at $600^3$, it does not keep widening (2026-08-17)

Corrects the section above, which read the trend from three sizes. On a **fixed set of 14 cases** that both tools solve at every size, GPU matrix-free:

| | $200^3$ | $400^3$ | $600^3$ | $800^3$ |
|---|---|---|---|---|
| geometric mean | $1.58\times$ | $6.02\times$ | $10.28\times$ | $10.24\times$ |
| worst case | $0.38\times$ | $1.51\times$ | $2.45\times$ | $2.45\times$ |
| best case | $36.5\times$ | $130\times$ | $159\times$ | $184\times$ |

The margin rises steeply to $600^3$ and then holds. The spread keeps opening, because the best case improves while the worst case does not.

The reason is taufactor, not us. Its sweep count to reach 0.1% is not monotonic in size. `b050_p080` goes 33, 189, 1086, 339. `b100_p040` goes 607, 1086, 1945, 189. It stops on flux uniformity across slices, which is a weaker criterion than a residual and behaves erratically from one geometry to the next. Our own count is flat over the same range.

So the paper must not say the margin widens without bound:

- "rises from $1.6\times$ at $200^3$ to about $9\times$ at $1000^3$" should become **rises to about $10\times$ by $600^3$ and holds there**. Confirm at $1000^3$ before finalizing.
- "A flat iteration count is what makes the margin grow with size" is still the right mechanism, but it explains the rise to $600^3$, not a rise beyond it.
- Reporting the geometric mean alone hides that the worst case is $2.45\times$ and the best is $184\times$. Give the range.

## The $1000^3$ grid has 14 cases, not 15 (2026-08-17)

`n1000_b050_p020` has no reference and never runs. The image exists, but the pore space does not percolate. `generate_images.jl` reports it directly:

```
Warning: [1/15] n1000_b050_p020  porosity=0.0000  nodes=0  (320.2s) — no percolating pore space
```

After the non-percolating paths are trimmed, nothing is left to solve. The same porosity and feature size do percolate at $200^3$, so this is a finite-size effect: a small domain percolates by chance where a large one does not. Every other size has all 15 references.

Any coverage sentence about $1000^3$ must therefore say 14, and must not present the missing case as a solver failure. If a reviewer counts cases across the figure panels, the gap is visible, so the caption needs one clause explaining it.

## The memory ceiling, fully measured (2026-08-18, $1000^3$ GPU)

Per-case detail behind the summary table in **Campaign complete** above. That table gives measured peak memory at blobiness 1.0 on both devices; this section gives the failed allocation sizes from the timings stage at blobiness 0.5.

This settles the riskiest claim in the paper. **The matrix-free form runs all 14 cases at $1000^3$ on a 48 GB card without ever exhausting it.** It reaches the 0.1% target in 13 of them.

The assembled form does not. It runs out of card from $\varepsilon = 0.6$ upward:

| case | requested allocation | outcome |
|---|---|---|
| n1000_b050_p040 | — | target reached |
| n1000_b050_p060 | 15.5 GiB | out of memory |
| n1000_b050_p080 | 42.0 GiB | out of memory |
| n1000_b050_p095 | 49.5 GiB | out of memory |

The ceiling moves with size exactly as the operator storage predicts. The assembled form fails at $\varepsilon \ge 0.8$ at $800^3$, and at $\varepsilon \ge 0.6$ at $1000^3$.

So the memory paragraph can now say something stronger and fully measured: at $1000^3$ the matrix-free operator is not merely leaner, it is the difference between running and not running.

One qualifier is needed. The single matrix-free case that misses the target, `n1000_b100_p020`, does not run out of memory. It converges and stalls at $1.46\times10^{-3}$ relative error, which is the same `Float32` accuracy floor seen at $800^3$. Write "completes every case" for memory and "reaches the target in 13 of 14" for accuracy. Those are different statements and the paper must not merge them.

## Comparison-data gaps that the paper does not currently admit (2026-08-18)

Rows per size in `results/timings/`, with cases that reach 0.1%:

| file | $200^3$ | $400^3$ | $600^3$ | $800^3$ | $1000^3$ |
|---|---|---|---|---|---|
| tortuosity-gpu-matrixfree | 14 ok | 15 ok | 15 ok | 15 ok | 13 ok |
| tortuosity-gpu-assembled | 15 ok | 15 ok | 15 ok | 8 ok | 5 ok |
| taufactor-gpu | 15 ok | 15 ok | 14 ok | 14 ok | **none** |
| taufactor-cpu | 15 ok | 15 ok | 14 ok | 12 ok | **none** |
| puma-cpu | 15 ok | **none** | **none** | **none** | **none** |

Neither gap is a tool failure. Both files hold zero rows at those sizes, so the runs never happened. `config.toml` sets no per-tool size cap, so nothing excluded them by design.

**Closing the taufactor gap: authorized and queued 2026-08-20.** The paper's headline — "the geometric mean advantage rises from $1.6\times$ at $200^3$ to about $9\times$ at $1000^3$" — had **no taufactor data behind the $1000^3$ end of it**. Amin's call was to measure it rather than extrapolate, so a $1000^3$ taufactor GPU sweep is queued behind the tortuosity re-run.

**A correction to the first cost estimate published here.** It said 13.5 h, arrived at by summing the `time_s` column across a case's ladder rungs. That column is **cumulative**, not incremental — `bench_taufactor.py` writes `t_setup + elapsed` where `elapsed` is taufactor's own clock since the start of the solve — so summing it counts the early rungs once per later rung. The correct per-case figure is the **maximum** over its rungs:

| taufactor GPU | $200^3$ | $400^3$ | $600^3$ | $800^3$ |
|---|---|---|---|---|
| total wall clock | 0.8 min | 7.2 min | 42.9 min | 114.8 min |
| worst single case | 0.3 min | 2.4 min | 14.5 min | 34.3 min |

Scaling the last rung gives $n^{3.4}$, so a $1000^3$ sweep projects to about **2.1 h** over its 14 cases, not 13.5.

**The 30-minute ceiling is left where it is.** `sweep.timeout_s = 1800` stops a case's ladder once cumulative solve time crosses it. Two cases already hit it at $800^3$ (`b050_p020` and `b100_p020`, both at 34.3 min — the ladder stops after the rung that crosses, not before it), and scaling puts two of the fourteen $1000^3$ cases over it as well. Raising the timeout for this one size, whether by editing `config.toml` or by passing `--timeout`, would measure $1000^3$ under a different definition from every other size in the table. A `timeout` row is a legitimate result and the paper already reports that shape for PuMA.

Expect, therefore, that the low-porosity end of $1000^3$ may come back as `timeout` rather than `target_reached`. That is not a missing measurement — it is the measurement, and it lands on exactly the cases where our advantage is largest.

**Predictions recorded before the sweep runs**, so the result can contradict them. taufactor's device memory is flat in porosity and exactly proportional to voxels — 28.43, 28.15, 28.09, 28.08 B/voxel at $200^3$ through $800^3$ — which puts $1000^3$ at **28.08 GB**, fitting the 50.74 GB card with 22.7 GB to spare. Against that, our matrix-free path at $1000^3$ is 9.90 / 16.66 / 23.28 / 29.79 / 34.44 GB unrefined and 13.58 / 24.57 / 35.32 / 45.89 / 34.44 refined (the last is unrefined because the guard fires). So the memory comparison at $1000^3$ should come out **three of five without refinement and two of five with it**, matching every smaller size.

On time, the sweep should return `target_reached` for the high-porosity cases and `timeout` for `b100_p020` and `b200_p020`.

**Not a gap: self-speedup is not going in the paper.** The old-versus-new comparison was campaign bookkeeping — it told us the rewrite worked and that the previously published numbers were void. It is not a claim a JOSS reader needs, and no competitor paper reports one. The pre-rewrite archive stops at $800^3$ and will not be extended. Keep the self-speedup table below as internal evidence only; the paper compares the current code against taufactor and PuMA, full stop.

The sweep is now finished, so the constraint in the last paragraph below no longer applies: the machine is idle and either option can start.

**This breaks two sentences in `paper.md`.**

First, "the geometric mean advantage rises from $1.6\times$ at $200^3$ to about $9\times$ at $1000^3$" cannot be supported at all. There is no taufactor measurement at $1000^3$ to compare against. The honest comparison stops at $800^3$.

Second, "PuMA reached the target in none of the larger images within our budget" reads as though PuMA ran and failed. It did not run. A reviewer who asks for the failing rows will find an empty file. Either reword it to say plainly that the PuMA comparison covers $200^3$ only, or run PuMA at the larger sizes.

Two ways forward, both needing Amin:

1. **Run taufactor at $1000^3$** after the CPU sweep finishes. It would complete the headline claim. The risk is that taufactor allocates over the full grid including solid voxels, so $10^9$ voxels in `Float32` may not fit in 48 GB. If it does not fit, that is itself a strong result and belongs in the paper.
2. **Scope the claims to what exists.** Compare against taufactor through $800^3$, present $1000^3$ as `Tortuosity.jl` alone, and say directly that PuMA was measured at $200^3$ only.

Option 2 costs nothing and is defensible. Option 1 is stronger if taufactor either runs or visibly fails. Do not start either while the $1000^3$ CPU sweep is running, because nothing on this machine is pinned and a second tool would corrupt both sets of timings.

## Two separate defects at low porosity: residual mesh dependence, and CG stagnating in `Float32` (2026-08-18, campaign complete)

The objection is the right one to raise: if the low-porosity cases are ill-conditioned, the coarse space plan (`docs/plans/2026-08-16-coarse-space-mesh-independence.md`) was supposed to have dealt with that. The campaign says one root cause produces **two different symptoms**, and that plan addresses only the first.

**Symptom 1 — iterations grow with image size.** This is what the coarse space attacked, and it largely won. CPU `Float64` matrix-free, iterations to reach 0.1%, fixed geometry down each row:

| case | $200^3$ | $400^3$ | $600^3$ | $800^3$ | $1000^3$ |
|---|---|---|---|---|---|
| b050_p020 | 59 | 106 | 106 | 189 | — |
| b100_p020 | 106 | 106 | **189** | **189** | **339** |
| b200_p020 | 189 | 189 | 189 | 189 | 189 |
| b050_p040 | 59 | 106 | 106 | 106 | 106 |
| b100_p040 | 33 | 59 | 59 | 106 | 106 |
| b200_p040 | 106 | 106 | 106 | 106 | 106 |
| all $\varepsilon \ge 0.6$ (nine rows) | 33–59 | 59 | 59 | 59–106 | 59–106 |

Thirteen of fifteen rows are flat or drift by one rung across a 5× size range, which is the ladder's own resolution. **Two rows are not flat**: `b100_p020` climbs 106 → 339, and `b050_p020` climbs 59 → 189. Both sit at the lowest porosity.

**This qualifies the plan's headline claim.** That plan validated flatness at $\varepsilon \approx 0.2$ over $200^3$/$400^3$/$600^3$ on one geometry (151 / 125 / 150) and never measured $800^3$ or $1000^3$ — its own O4 records that `benchmarks/` was not run. Over the full size range the property holds everywhere except the lowest porosity, where a residual $h$-dependence survives. That is a real finding about the preconditioner and belongs on the plan, not only here.

**Symptom 2 — our CG stagnates in `Float32` long before the precision limit.** This is what causes the three GPU failures, and it is **not** a property of `Float32`, of the geometry, or of the preconditioner. It is a property of our solver. Three earlier explanations for it were each proposed here and each refuted by measurement; the record of that is below, because a reviewer may propose the same ones.

Every number in this subsection is for `n200_b050_p020`, regenerated locally from the campaign parameters (seed 42, axis `x`, blobiness 0.5, porosity target 0.2). It reproduces the cached image exactly — 1 291 692 pore nodes, porosity 0.161462, both matching the manifest — and a tight `Float64` solve returns $\tau = 33.93155698$ against the campaign reference $33.93155464$.

**The falsification chain.** Four measurements on that one image, all in `Float32` unless stated:

| what | relative error | iterations |
|---|---|---|
| `Float64` solution rounded to `Float32`, then evaluated | $7.2\times10^{-6}$ | — |
| **taufactor** (SOR), same image, same card | $8.0\times10^{-6}$ | 11 169 |
| our preconditioned CG | $1.4\times10^{-3}$ | 187 (self-terminated) |
| our unpreconditioned CG | $7.9\times10^{-3}$ | 3 768 |
| our preconditioned CG in `Float64` | $1.5\times10^{-5}$ | 59 |

The first row is the ceiling: `Float32` **can** hold this solution to $7\times10^{-6}$, so representation is not the limit. taufactor reaches that ceiling in the same precision on the same hardware, so neither the image nor the precision is the limit. Our preconditioner is not the limit either — removing it makes the floor 5.6 times *worse*. What is left is our CG, which stops about 200 times short of what the format supports.

The standard explanation for this is finite-precision Krylov breakdown: CG maintains conjugacy through short recurrences, orthogonality among the search directions degrades as rounding accumulates, and the attainable error floor grows with $\kappa(A)$. A stationary method such as SOR has no such mechanism — each sweep re-derives the field from its neighbours, so its attainable accuracy is set by the local rounding of one update and is roughly $\varepsilon$ regardless of conditioning. That is exactly the split measured above. **This attribution is the textbook one, not something measured here** — what is measured is where the limitation lives, and it lives in our solver.

**The trade this reveals is worth stating plainly in the paper.** On this case our CG needs 59 iterations in `Float64` where taufactor needs 11 169 — about 190 times fewer. taufactor buys robustness in low precision with a slow stationary method. We buy speed with a Krylov method that gives up early in single precision on the hardest cases. Both are real engineering positions and the paper should say so rather than presenting our floor as a law of nature.

**It is also fixable, which the paper should not pretend otherwise.** Accumulating the CG inner products in `Float64` while keeping the operator apply in `Float32` is the standard remedy and is cheap — the dot products are a negligible share of the work. `Float64` iterative refinement around a `Float32` solve is the other. Neither has been tried yet.

### Three refuted explanations, kept because a reviewer may repeat them

**Refuted 1 — error accumulating over $10^9$ voxels.** The worst floor in the whole campaign belongs to a $200^3$ image with 1.29 M pore nodes, while `n1000_b100_p095` carries 950 M nodes and solves to $10^{-6}$. Node count and floor move in opposite directions.

**Refuted 2 — dead-end pore volume.** The argument was that regions attached to the spanning cluster at one end carry no flux and contribute a near-null subspace. `porespy` 3.1.0 has no dead-end or backbone function — `fill_invalid_pores`, `trim_nonpercolating_paths` and `trim_disconnected_voxels` all remove closed or non-spanning clusters, which `Imaginator.trim_nonpercolating_paths` already does at image-generation time. Dead ends were therefore identified physically, as voxels whose concentration gradient to every pore neighbour vanishes.

The pruning is valid — removing them leaves transport untouched, $D_{\text{eff}}$ moving by under $3\times10^{-11}$ at every threshold. But the dead-end fraction is only **3.4% to 6.8%** of pore volume, not the large share claimed, and removing it changes nothing. Repeats are bit-identical, so these are exact:

| pruned at | nodes | CPU `Float64` iters to 0.1% | GPU `Float32` floor |
|---|---|---|---|
| — (as benchmarked) | 1 291 692 | 57 | $1.43\times10^{-3}$ |
| $10^{-12}$ | 1 247 836 | 59 | $3.34\times10^{-3}$ |
| $10^{-10}$ | 1 229 356 | 59 | $2.37\times10^{-3}$ |
| $10^{-8}$ | 1 204 296 | 57 | $8.09\times10^{-4}$ |

Iteration count is flat, so $\kappa$ is unchanged in any way CG can see, and the floor moves **non-monotonically** — removing 3.4% makes it 2.3 times worse. That is scatter in where rounding lands, not a dose response.

**Refuted 3 — the concentration dynamic range.** The argument was that a long tortuous path spreads a unit concentration drop over many voxels, leaving neighbour differences of $10^{-5}$–$10^{-4}$ that `Float32` cannot resolve. The correlation is real — across a porosity sweep at $200^3$, as the median neighbour gradient falls 55× the floor rises about 35×:

| $\varepsilon$ | pore nodes | $\tau$ | median neighbour $\lvert\nabla c\rvert$ | GPU `Float32` floor | CPU `Float64` iters |
|---|---|---|---|---|---|
| 0.161 | 1 291 692 | 33.93 | $8.8\times10^{-5}$ | $1.43\times10^{-3}$ | 57 |
| 0.399 | 3 192 287 | 2.92 | $1.8\times10^{-3}$ | $1.73\times10^{-4}$ | 30 |
| 0.610 | 4 877 016 | 1.55 | $3.3\times10^{-3}$ | $7.9\times10^{-6}$ | 42 |
| 0.949 | 7 591 813 | 1.04 | $4.9\times10^{-3}$ | $4.0\times10^{-5}$ | 28 |

But correlation is not the mechanism. taufactor resolves the same gradients on the same image in the same precision to $8\times10^{-6}$. The dynamic range tracks conditioning, and conditioning is what degrades CG in finite precision — so this table measures the *symptom's* correlate, not its cause.

**This has now been fixed, and the campaign numbers predate the fix.** See `docs/plans/2026-08-19-float32-cg-stagnation.md`. The GPU path refines a `Float32` solve against a `Float64` residual before returning it, which repairs every case the campaign recorded as `ladder_exhausted`:

| case | operator | plain (as campaigned) | refined |
|---|---|---|---|
| `n200_b050_p020` | matrix-free | 2.0e-3 | 7.9e-6 |
| `n800_b200_p020` | assembled | 1.16e-3 | 4.08e-8 |
| `n1000_b100_p020` | matrix-free | 1.50e-3 | 1.25e-8 |

Over the full $200^3$ grid the failure count goes from 1/15 to **0/15**, worst error 7.9e-6, and no case is made worse. `Pkg.test()` is green and $\tau$ stays bit-identical across repeats, so the determinism guarantee is intact.

**Three consequences for the manuscript, none of them yet reflected in any published number:**

1. **GPU coverage at $1000^3$ becomes 14 of 14**, matching the CPU. (14 and not 15 because `n1000_b050_p020` trims to **zero** pore nodes: it does not percolate at all, so there is nothing to solve, and it is absent from the CPU sweep for the same reason. Degenerate image, not a failure; the paper must not imply a missing case.) The accuracy caveat the paper was going to carry — use `Float64` on the CPU for strongly tortuous images — is no longer true and must not be written.
2. **Every GPU timing in the campaign is now stale.** The fix changes what the GPU path does. CPU numbers are unaffected: refinement is off at `Float64` by design and by measurement. Re-running the GPU half is roughly 3.7 h of the 27 h campaign.
3. **The harness must be fixed before that re-run**, and it has been: `trace_case` passed a callback into the solve, and refinement reuses the solve's cache — including its algorithm — so the callback would have fired on correction rounds and recorded tortuosities computed from correction vectors. It now traces with `refine=false` and pads the rungs above an early exit with a separately timed refined solve, which is what a caller actually receives.

What the paper can say, and could not before, is that the single-precision GPU path reaches the same 0.1% target as the double-precision CPU path on every image in the sweep. The cost is that a full GPU solve is now ~1.3–2.8× slower than the previous, wrong one; the time-to-target the benchmark measures is unchanged for the cases that already converged, because refinement runs only after the target has been passed. Also recompute the "roughly 36× faster than its own CPU path" figure — it is a geometric mean over the cases both paths solve, and the GPU set gains `n1000_b100_p020`, which previously never reached target.


**A memory sentence now needs a qualifier, and the memory claim next to it can now be stated as measured.** Both concern `paper.md` line 79.

*The claim that gets stronger.* The paragraph says the matrix-free form "holds one `Int32` per grid voxel". Fitting `bytes = a·nodes + b·voxels` to the campaign's own $800^3$ GPU memory stage gives

| | fitted | reading |
|---|---|---|
| $a$ | 32.02 B per pore node | eight `Float32` vectors |
| $b$ | 4.003 B per voxel | the `Int32` index map, exactly as claimed |

Those two numbers reproduce the other three measured porosities at $800^3$ to $\pm0.01\%$, and — without being refitted — predict all five $1000^3$ points in the same file to within $0.009\%$, across a 2.1x jump in voxel count:

| $1000^3$, b100 | $\varepsilon$ 0.20 | 0.40 | 0.60 | 0.80 | 0.95 |
|---|---|---|---|---|---|
| predicted, GB | 9.9009 | 16.6593 | 23.2743 | 29.7872 | 34.4433 |
| measured, GB | 9.9000 | 16.6598 | 23.2752 | 29.7877 | 34.4432 |

The refined peaks were the genuinely new predictions, since nothing had measured refinement before: $1000^3$ $\varepsilon=0.80$ predicted at 49.12 GB and measured at 49.1131, and $\varepsilon=0.95$ predicted to exhaust the card, which it did.

The paper can state the per-voxel cost as a measurement rather than as a design claim.

*The sentence that needs the qualifier.* "It completes every case at $1000^3$ on a 48 GB card" remains true of the *solve*. It is not true of the solve **plus refinement**. Refinement costs 20 B per pore node, and at $\varepsilon = 0.95$ the image carries 950 M nodes against a solve already holding 34.44 GB of a card that reports 47.27 GiB = 50.74 GB. Measured on that case: the guard fires, warns, and returns the unrefined solution, exit 0.

That case does not need refining — its campaign error is $7.46\times10^{-5}$, well inside the 0.1% target — and the reason is structural rather than lucky. Refinement is needed on ill-conditioned images, which are the *low*-porosity ones; memory runs out on high-porosity images, because that is where the pore nodes are. All three cases the campaign recorded as `ladder_exhausted` are $\varepsilon = 0.2$, and each has an order of magnitude more headroom than refinement asks for. So the sentence should either stay scoped to the solve, or say plainly that the highest-porosity $1000^3$ case runs unrefined; it must not imply that the refined path fits everywhere.

The refinement ceiling on this card at $1000^3$ is $\varepsilon \approx 0.90$. It was $\varepsilon \approx 0.83$ until the narrowing step was changed to reuse the cache's own solution vector instead of allocating a fourth buffer.

**What the campaign measured, for the record.** As campaigned, coverage at $1000^3$ was 14 of 14 on the CPU and 13 of 14 on the GPU, and the single-precision path stagnated on the most ill-conditioned images. Claiming `Float32` could not reach the target would have been false even then, and taufactor's own numbers in this very campaign contradict it. The paragraph above supersedes this: the limitation was in our CG, and it is now repaired.

**Still open on the code, separately.** The `b100_p020` iteration growth from 189 to 339 between $600^3$ and $1000^3$ — residual mesh dependence at the lowest porosity, which the coarse-space plan validated only to $600^3$. That plan's own O3 flagged the top coarse level reaching ~2 M cells at $1000^3$ while staying on the host, which is the obvious place for the V-cycle to lose ground at sizes it was never measured at. Not a JOSS blocker, and unrelated to the precision defect above.

## The refinement fix dents a memory claim, and that is a decision (2026-08-20, re-measured)

**This needs Amin.** It is the one place where fixing the accuracy defect costs something a reviewer will see.

The memory stage calls `solve_case`, which calls `solve` **without** `refine`, so refinement is on by default at `Float32` and the GPU re-run measures the solve *plus* refinement. That is the right thing to measure, because it is what a user pays. It is also what moves the comparison against taufactor.

**Refinement costs 20.00 B per pore node**, measured, at 23 of the 25 matrix-free cases — not projected from the algorithm but read off the re-run against the archived figures. The two exceptions are both the guard doing its job, on different operators and at different allocations:

| case | delta | reading |
|---|---|---|
| `n1000_b100_p095`, matrix-free | 16.00 B/node | guard fired at the **third** allocation: `x64` and `r64` fit, `correction_rhs` did not |
| `n1000_b100_p040`, assembled | **0.00 B/node** | guard fired at the **first**: 2.23 GB free against `x64` needing 3.16, so the peak never moved |

The second was predicted from the card headroom before it was looked up, and the prediction was that the peak would be *unchanged*. It is.

taufactor's GPU footprint is flat with porosity — it stores dense arrays over the whole grid, 28.08 B per voxel — so the crossover is entirely on our side of the ledger. All figures below are measured, in GB:

| size | | $\varepsilon$ 0.20 | 0.40 | 0.60 | 0.80 | 0.95 | taufactor |
|---|---|---|---|---|---|---|---|
| $200^3$ | solve | 0.081 | 0.128 | 0.177 | 0.225 | 0.260 | 0.227 |
| | + refinement | 0.101 | 0.192 | **0.274** | **0.354** | 0.412 | |
| $400^3$ | solve | 0.632 | 1.065 | 1.491 | 1.911 | 2.205 | 1.801 |
| | + refinement | 0.867 | 1.570 | **2.262** | 2.945 | 3.422 | |
| $600^3$ | solve | 2.152 | 3.632 | 5.023 | 6.403 | 7.440 | 6.068 |
| | + refinement | 2.956 | 5.361 | **7.620** | 9.862 | 11.546 | |
| $800^3$ | solve | 5.089 | 8.521 | 11.876 | 15.212 | 17.658 | 14.375 |
| | + refinement | 6.988 | 12.562 | **18.013** | 23.432 | 27.407 | |
| $1000^3$ | solve | 9.900 | 16.660 | 23.275 | 29.788 | 34.443 | (queued) |
| | + refinement | 13.584 | 24.565 | 35.312 | 45.892 | 49.653 | |

Bold marks a cell that beats taufactor before refinement and loses after.

**Two separate problems, and only one is new.**

*The pre-existing one.* `paper.md` says the matrix-free form "uses less device memory than taufactor at four of five porosities". That is true at $200^3$ and **only** at $200^3$ — where `p080` clears taufactor by 0.002 GB. At $400^3$, $600^3$ and $800^3$ it is three of five. The sentence carries no size, sits beside a figure showing every size, and will read as general. It is wrong in the manuscript today, independent of anything else here.

*The new one.* With refinement always on it becomes **two of five at every size**. The $\varepsilon = 0.6$ band flips from win to loss at all four sizes where a comparison exists.

**Why this is a reporting decision and not a bug.** Refinement is needed on ill-conditioned images, which are the low-porosity ones, and those are exactly where we clear taufactor by 2x or more. It is unnecessary at $\varepsilon \ge 0.6$: the campaign's worst GPU error at $\varepsilon = 0.6$ is $7.9\times10^{-4}$ and at $\varepsilon = 0.95$ is $5.8\times10^{-4}$, both inside the 0.1% target with no refinement at all. The porosities where refinement costs the comparison are the porosities where it buys nothing.

**Do not reach for an adaptive trigger.** Already measured and closed: at a fixed true residual of 2.4–2.9e-6 the $\tau$ error spans 3.6e-5 to 2.0e-3, because the conditioning amplification runs from 12x to 758x across the grid. A residual threshold cannot tell the two apart. Deciding from porosity, or from an estimate of $\kappa$, is unmeasured speculation and has no place in a paper.

**Recommendation, for Amin to accept or overrule.** Report the solve footprint, as the current figure does, and state refinement's cost as the constant it is — one sentence: the single-precision path adds 20 bytes per pore node when it refines, which it does on the ill-conditioned low-porosity images, and which the highest-porosity $1000^3$ case skips for want of room. That keeps the figure comparing operator forms, which is what the figure is for, while disclosing the full cost.

The alternative — publish the refined memory numbers — is more conservative and equally defensible, but it hands a reviewer a 60% memory increase at high porosity in exchange for accuracy those cases already had.

Either way the "four of five" sentence has to change, because it is not right today.

**One more figure for the memory paragraph.** At $1000^3$, $\varepsilon = 0.95$ the peak is 49.653 GB against a card reporting 50.74 GB — **97.9% of it** — and that case completes unrefined. "Completes every case at $1000^3$ on a 48 GB card" is still true, and is closer to the edge than the sentence suggests.

## The iteration-count sentence cannot be sourced from the benchmark data (2026-08-20)

`paper.md` line 55 reads: *"At $\varepsilon = 0.5$ the unpreconditioned count climbs from 1044 at $200^3$ to 4805 at $1000^3$, while the preconditioned count stays between 91 and 148."* Three separate problems, all verified against the data:

1. **$\varepsilon = 0.5$ is not in the campaign.** The porosities are 0.2, 0.4, 0.6, 0.8 and 0.95.
2. **The numbers cannot come from the sweep.** Its knob is a log-spaced iteration ladder — 1, 2, 3, 6, 10, 18, 33, 59, 106, 189, 339, 607 — and `knob_name` is `iters` in every row of every file. Rungs sit 1.8x apart, so the ladder cannot express 91 and cannot resolve a 1.6x range at all. Whatever produced "91 to 148" was an ad-hoc measurement outside the harness, and nothing in `results/` can reproduce it.
3. **Stated unconditionally, the mesh-independence claim is not true at every porosity** — but the reason is not the one the benchmark first suggests. See below.

### Two different questions, one of which the benchmark cannot answer

The sweep reports the rung at which a case first meets the **0.1% tortuosity target**. That is not the same as iterations to a fixed residual, and at low porosity the two diverge sharply. Rung to target, GPU matrix-free, blobiness 1.0, from the re-run:

| $\varepsilon$ | $200^3$ | $400^3$ | $600^3$ | $800^3$ | $1000^3$ | growth |
|---|---|---|---|---|---|---|
| 0.20 | 106 | 106 | 189 | 189 | **607** | 5.7x |
| 0.40 | 33 | 59 | 59 | 106 | 106 | 3.2x |
| 0.60 | 59 | 59 | 59 | 59 | 59 | 1.0x |
| 0.80 | 33 | 59 | 59 | 59 | 106 | 3.2x |
| 0.95 | 59 | 59 | 59 | 59 | 106 | 1.8x |

Measured directly instead, at a fixed relative residual of 1e-6 on the CPU in `Float64` — iteration counts are deterministic given the image and the code, so this needs no quiet machine:

| $\varepsilon$ | preconditioned $200^3$ | $400^3$ | | unpreconditioned $200^3$ | $400^3$ | growth |
|---|---|---|---|---|---|---|
| 0.20 | 136 | **132** | | 2721 | 4958 | 1.8x |
| 0.40 | 119 | **123** | | 1219 | 2465 | 2.0x |
| 0.60 | 83 | **89** | | 989 | 1864 | 1.9x |
| 0.80 | 86 | **80** | | 850 | 1515 | 1.8x |
| 0.95 | 58 | **62** | | 747 | 1377 | 1.8x |

**Preconditioned is flat to within ±7% at every porosity, including $\varepsilon = 0.2$ where the rung table shows 5.7x growth. Unpreconditioned nearly doubles per doubling of edge length**, which is the textbook $O(n)$ behaviour the paper describes.

### What this means, and what it does not yet establish

The two tables are consistent, and reconciling them corrects something recorded earlier in this investigation. Iterations to a fixed **residual** are mesh-independent — that is what the coarse space was built to deliver and it delivers it. What grows with size at low porosity is the **amplification from residual to tortuosity error**, which is the conditioning factor measured at 758x at $\varepsilon = 0.16$ against 12–33x elsewhere. A larger low-porosity image therefore needs a smaller residual to hit the same 0.1% target, and so more iterations — while the solver's convergence rate has not degraded at all.

So "Symptom 1 — iterations grow with image size", recorded earlier as a shortfall of the preconditioner, is better read as a property of the target metric. The preconditioner is not losing ground at $1000^3$.

**This is not yet established, and must not be written up as though it were.** It rests on preconditioned counts at two sizes only. The claim is that 136 / 132 continues flat at $600^3$, $800^3$ and $1000^3$; if it instead climbs there, the reading above is wrong and the earlier one was right. A measurement over all five sizes using the cached campaign images is queued behind the taufactor sweep (`benchmarks/measure_iters.jl`, log at `/tmp/iters_measurement.log`). Nothing here should reach the paper before it returns.

Whatever it returns, the sentence needs rewriting: it must cite a porosity the campaign contains, and give counts that something in `results/` can reproduce.

## $1000^3$ closes the taufactor gap, and settles the scaling claim (2026-08-20, measured)

taufactor now has $1000^3$ data — 14 timing cases and 5 memory cases, exit 0 — so every published comparison is measured rather than extrapolated. Three predictions were written down before the run. All three hold.

### Prediction 1: taufactor memory is flat in porosity at 28.08 B/voxel

Measured **28.0569 B/voxel**, and exactly flat: all five porosities report the identical 28,056,869,888 bytes, not one distinct value between them. The prediction was 0.08% high.

### Prediction 2 and 3: the memory comparison at $1000^3$ is 3-of-5 before refinement and 2-of-5 after

Both confirmed against the 28.057 GB taufactor figure:

| $\varepsilon$ | solve only | refined (as shipped) | taufactor |
|---|---|---|---|
| 0.20 | 9.900 **win** | 13.584 **win** | 28.057 |
| 0.40 | 16.660 **win** | 24.565 **win** | 28.057 |
| 0.60 | 23.275 **win** | 35.312 loss | 28.057 |
| 0.80 | 29.788 loss | 45.892 loss | 28.057 |
| 0.95 | 34.443 loss | 49.653 loss | 28.057 |

This is the same crossover already recorded at the smaller sizes, and it settles the memory decision: at $1000^3$ the shipped path wins on the two lowest porosities.

### The margin plateaus, and the $1000^3$ point confirms it

The 2026-08-17 section predicted "rises to about $10\times$ by $600^3$ and holds there. Confirm at $1000^3$." Confirmed. On a **fixed set of 12 families** that both tools solve at all five sizes, GPU matrix-free, post-fix:

| | $200^3$ | $400^3$ | $600^3$ | $800^3$ | $1000^3$ |
|---|---|---|---|---|---|
| geometric mean | $1.11\times$ | $4.57\times$ | $8.18\times$ | $7.52\times$ | $8.44\times$ |
| worst case | $0.39\times$ | $1.55\times$ | $2.53\times$ | $2.51\times$ | $1.40\times$ |
| best case | $13.4\times$ | $18.4\times$ | $45.2\times$ | $87.9\times$ | $58.6\times$ |

The margin rises steeply to $600^3$ and then holds near $8\times$. It does not keep widening, and the paper must not say it does. Two things a reviewer will notice and the manuscript should state first:

- **At $200^3$ the worst case is $0.39\times$ — taufactor is more than twice as fast on the easiest small images.** The geometric mean of $1.11\times$ is close to parity. Our advantage is a large-image advantage; claiming it at every size invites a reviewer to find the counterexample.
- The **spread keeps opening** while the mean holds flat, so the geometric mean alone hides the shape. Give the range.

This fixed set is a stricter subset than the one used on 2026-08-17, because a family now has to pair at $1000^3$ as well. That excludes the low-porosity families where the margin is largest, which is why every entry sits below the earlier table. Both are honest; only this one supports a five-size scaling claim.

### The stronger claim at $1000^3$ is capability, not speed

Restricting to cases both tools solve is required for a speedup ratio, and it **biases against us**, because the cases taufactor cannot finish are exactly the ones we win biggest. At $1000^3$ taufactor fails to reach the 0.1% target on 2 of its 14 cases after roughly 40 minutes each:

| case | taufactor | Tortuosity.jl |
|---|---|---|
| `n1000_b050_p040` | gave up at 2351.8 s, rel. error 2.18e-3 | target in **20.6 s**, rel. error 4.92e-4 |
| `n1000_b100_p020` | gave up at 2349.2 s, rel. error 8.61e-3 | target in **72.0 s**, rel. error 4.18e-6 |

Neither pairs, so neither appears in any speedup figure. As lower bounds they are $114\times$ and $33\times$, and in both cases taufactor never produced the answer at all. `n1000_b100_p020` is one of the three cases the refinement fix recovered — before the fix we could not solve it either.

### The mechanism, confirmed on the new data

taufactor's sweep count to target is **not monotonic in size**, which is why the ratio wanders rather than trending:

```
b050_p095:  200:106   400:59    600:339   800:339   1000:607
b100_p095:  200:59    400:106   600:189   800:189   1000:59
b200_p020:  200:6238  400:3484  600:6238  800:11169 1000:6238
b050_p040:  200:1086  400:339   600:607   800:3484
```

It stops on flux uniformity across slices, a weaker criterion than a residual, and it behaves erratically from one geometry to the next. Our own iteration count is flat over the same range. The $800^3$ dip to $7.52\times$ is this effect, not a regression in our solver.

## Wrong machine

The core-occupancy claim — "our CPU path occupies about two cores and PuMA about one" — was sampled on a 20-core Windows machine, not on `pmeal-hpc` (Xeon Silver 4110, 8 cores). Either re-sample on the benchmark machine or delete the sentence. Leaving it invites a reviewer to ask which machine produced the rest of the numbers.

## Verified 2026-08-17, no action needed

- Two operators agree on $\tau$ to within $2\times10^{-4}$. Measured today: CPU operators agree exactly, GPU operators differ by 1.98e-4 at $200^3$. An earlier draft of this rewrite said "five significant figures", which the GPU data does not support.
- Four contributors, more than 650 commits, 2023 to 2026. `git shortlog -sne --all` gives 663 commits over four humans; Sawyer Hossfeld and Harry Kim each have two git identities.
- Hardware: Quadro RTX 8000 (48 GB), Xeon Silver 4110, 8 cores.
- The assembled path widens `Int32`→`Int64` rather than refusing (`src/assembly.jl:191`, commit `ab63e7f`). The paper previously claimed the opposite in four places.
- The restriction is now a fixed-order gather (`src/preconditioner.jl`, `Aggregation`). Atomics remain in the once-per-solve coarse-operator assembly, which is why the paper does not claim bit-reproducibility.

## Still open, needs Amin

- **How to report memory now that the GPU path refines.** Refinement adds a flat 20 B per pore node, which flips the taufactor memory comparison from three of five porosities to two of five, at every size. It is unnecessary at exactly the porosities where it costs the comparison. See "The refinement fix dents a memory claim" above for the measured table and a recommendation.
- **The "four of five porosities" sentence is wrong today**, independently of any of this: it holds at $200^3$ only, and is three of five at $400^3$, $600^3$ and $800^3$.
- **Submission date** in the YAML header currently reads 17 August 2026.
- **Zenodo DOI** is a post-review step, not a submission prerequisite. Confirmed against the JOSS review process: "Upon successful completion of the review, authors will make a tagged release ... get a DOI for the archive". Nothing to do now.
- **Figures** `benchmark_summary.png` and `benchmark_memory_gpu.png` are the old ones and must be regenerated from the new results.
