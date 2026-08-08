# Matrix path optimization plan

Execution plan for making the **assembled sparse matrix path** as fast and as memory-efficient as it can reasonably be. This is the orchestration document: it survives context compaction and is the single source of truth for what is done, what is pending, and what was deliberately rejected. Paired with `MATRIX_FREE_PLAN.md`, which describes the *separate, later* matrix-free work.

## How to run this

Designed for unattended execution with `/goal`: Amin starts it and returns when it is finished.

**Do not invoke as `/goal MATRIX_PATH_PLAN.md`.** The argument to `/goal` *is* the completion condition, not a file to execute — that form would set the literal string `MATRIX_PATH_PLAN.md` as the condition and the evaluator would judge nonsense. Paste the condition below instead.

### The goal condition (copy verbatim)

```
/goal Execute the plan in MATRIX_PATH_PLAN.md to completion. Read that file first, then resume from its Progress Log. The condition is met when, and only when, your visible message text contains the exact line: CAMPAIGN COMPLETE - all conditions met. Print that line only after you have personally verified all five: (1) every A-, B- and C-series inventory item is terminal (done, rejected, BLOCKED or REVERTED) in the Progress Log; (2) Phase 4 ended on diminishing returns; (3) a full Pkg.test() run is green with assertions at or above the Phase 0 baseline; (4) bench/scaling_bench.jl has been re-run at all sizes including 800-cubed; (5) the Final report is written into MATRIX_PATH_PLAN.md. If any is unmet, do not print the line - keep working. Constraints: never modify golden tau values, never weaken or skip a test to make a change pass, never use git add -A or git commit -a, never leave the tree red. Stop after 60 turns if not complete, print CAMPAIGN HALTED and a status summary.
```

**Why it keys on one literal string.** The evaluator is a small fast model that cannot run commands, read files, or see inside subagents — it only reads the transcript. Asking it to judge five compound conditions invites a false positive that ends the campaign early. Reducing its job to matching one exact string moves the judgement to the orchestrator, which *can* verify, and leaves the evaluator doing something it cannot get wrong.

Note `CAMPAIGN HALTED` is a distinct marker, so an exhausted turn budget does not read as success.

Roughly 1.2 k characters, inside the 4 000-character limit. Adjust the turn clause to taste.

### Preconditions

- **Branch: work on `perf/matrix-path`. Check it out before doing anything else.** It already exists, created from `main` at `9ba4064` on 2026-08-08 with the full suite verified green including GPU. Do **not** work on `main`, and do **not** work on `joss` — `joss` holds only JOSS paper material and is not the base for this work. Confirm with `git branch --show-current` before Phase 0. If checkout fails because `paper/paper.md` or `paper/paper.bib` have local modifications, that is Amin's in-flight JOSS work: stop and report, do not resolve it yourself. Neither `main` nor `perf/matrix-path` has been pushed; both are 5 commits ahead of their remotes. Full git rules are under **Unattended execution → Git discipline**.
- **Hard gate: the hardened test suite must be committed and present.** It landed on 2026-08-08 and was cherry-picked onto `main` (head `9ba4064`). Verify before Phase 0 that `git ls-files test/` lists `test_assembly.jl` and `test_regression_golden.jl`. If it does not, you are on the wrong branch — stop and report.
- **Working-tree cleanliness is scoped to code paths, not the whole tree.** Amin keeps in-flight JOSS paper work (`paper/`, `benchmarks/`, `docs/`, `JOSS_HANDOFF.md`, root `*.md`) uncommitted in this checkout; that is expected and is not yours to touch. Require only that `git status --short -- src test ext bench` is empty before starting. If it is not, stop and report rather than guessing what belongs to whom.
- **Auto mode must be on**, or the run stalls at the first permission prompt. `/goal` does not change permissions. Already satisfied here — `~/.claude/settings.json` sets `defaultMode: "auto"`.
- `/goal` is unavailable if `disableAllHooks` or `allowManagedHooksOnly` is set at any settings level, and requires an accepted workspace trust dialog. Neither blocker is present here.
- Headless variant: `claude -p "/goal ..."` runs the loop to completion in one invocation. Add `--output-format stream-json --verbose` or nothing prints until it finishes and a long run looks hung.
- A goal still active when a session ends is restored by `--resume`/`--continue`; the turn count and timer reset, the condition carries over.

**First action of every session, including resumed ones: read the Progress Log at the bottom of this file and resume from the first pending item.** The log is the state, not the agent's context. Never restart from Phase 0 because the log is unfamiliar; an empty log is the only thing that means Phase 0 has not run.

See **Unattended execution** below for the rules that apply when nobody is available to answer a question.

## Governing principles

These three rules decide every question below. When in doubt, re-read them.

**1. The matrix path is a first-class citizen, not a stopgap.** We are keeping it. Do not skip, defer, or half-do work here on the grounds that the matrix-free path will eventually supersede it. Both paths will exist; both should be good. The goal is the most efficient, most optimal matrix-path implementation we can reasonably build — *before* matrix-free work begins.

**2. Take every substantial win. Take marginal wins only when they are free.** A substantial optimization is accepted regardless of implementation cost — it is worth the complexity. A marginal optimization is accepted only if it leaves the code as simple or simpler. We would rather have 90 % of the achievable efficiency in a codebase that is easy to reason about than 100 % in one that is ugly and hard to maintain.

**3. You have staff-engineer agency.** The rules below are heuristics, not law. If you have a good reason to violate one, you have the authority to do so — just say so explicitly in your report, with the reasoning. Judgement that is well-argued beats rule-following. This applies to the threshold, the inventory, the phase boundaries, and the scope.

## Scope: what you are authorized to do

The inventory in this document is a *starting point from one reading pass*, not a work order and not a boundary. Explicitly authorized, without asking:

- **Pursue optimizations that are not in the inventory.** Some paths only become visible once earlier work lands. Chase them.
- **Rewrite implementations that are simply bad.** If something is needlessly slow or needlessly memory-hungry, replacing it is in scope even if no inventory item names it.
- **Add GPU paths where none exist.** Several operations currently run on CPU, or copy device data back to the host, purely because nobody wrote the kernel. Writing it is in scope. See the C-series inventory for known cases.
- **Add code, add files, add abstractions** where they genuinely serve principle 1. This is not a minimal-diff exercise.
- **Contradict this document.** If analysis here is wrong — and some of it will be, since it came from reading rather than measuring — say so and correct it.

The only hard limits are the non-negotiable constraints below.

### The decision heuristic

| gain on a headline metric | complexity cost | verdict |
| --- | --- | --- |
| ≥ 15 %, **or** unlocks a problem size that previously failed | any | **accept** |
| < 15 % | simplifying or neutral | **accept** |
| < 15 % | adds a concept, a branch, or > ~30 lines | **reject, and log why** |

Override this when it makes sense (principle 3) — for example, several sub-15 % wins that compose into one coherent change should be judged as a unit, and a marginal win that removes a whole failure mode may be worth more than its percentage suggests.

Rejections are as valuable as acceptances — log them in the Progress Log so nobody re-investigates.

## Memory and speed are one campaign

Both are headline metrics and both are first-class:

- **Memory:** peak device memory during construction; peak during solve; bytes per pore voxel.
- **Speed:** assembly (setup) wall time; solve wall time; end-to-end wall time. Also CPU-path wall time — the JOSS benchmarks compare against PuMA, which is CPU-only, so CPU speed is not a second-class concern.

They are kept in one campaign because on this workload they are largely the same axis: the kernels are bandwidth-bound, so memory not allocated is memory not written and then read back. The largest inventory items improve both simultaneously. Splitting them would mean touching the same kernels twice, against a test suite that pins conventions — double the regression risk for no benefit.

**Conflict rule.** When a change trades one against the other:

- **Until 800³ fits in 23.89 GiB with ≥ 3 GiB of headroom, memory wins.** Memory is a hard constraint — exceeding it is a crash.
- **After that threshold is met, speed wins.** Memory becomes a budget to spend rather than a wall to avoid.

Report both numbers for every change regardless of which it targets, so regressions on the other axis are visible.

## Non-negotiable constraints

**The test suite is the safety net and it was built for exactly this work.** It was expanded from ~1.1 k assertions on 2026-08-06 specifically so that a performance rewrite could be verified, and hardened further before this campaign began. A green suite is the definition of "not broken".

Three pinned conventions that an "optimization" can silently violate. **The file and test names below were accurate on 2026-08-08 but predate the hardening pass — re-verify each against the committed suite during Phase 0 and correct this list if they have moved.** The conventions themselves hold regardless of where the tests live:

1. **`test/test_assembly.jl` — "Dirichlet elimination — exact contract"** states `apply_dirichlet_bc_fast!` as six identities against the pre-elimination Laplacian (free–free block untouched, coupling zeroed symmetrically, original diagonal preserved, RHS folding formula). Any change to boundary handling must reproduce these exactly.
2. **CSC row ordering.** Connectivity rows must stay grouped by column and ascending within a column, because `build_adjacency_matrix` writes CSC arrays directly from that ordering with no `sparse()` call. A reordering "optimization" corrupts the matrix *silently*. CUSPARSE also depends on it (`ext/TortuosityCUDAExt.jl` wraps the arrays as `CuSparseMatrixCSC`). **Verify whether the current kernel actually guarantees ascending order or merely happens to** — the write path uses per-bucket atomics, so this needs checking, not assuming.
3. **`test/test_regression_golden.jl`** holds hard-coded τ values for three blob seeds. A change to these is either a bug or a deliberate improvement. **Never update a golden value during an unattended run** — treat it as a blocker (see below).

If a constraint genuinely blocks a large win, do not work around it silently. Record it as a blocker with the full tradeoff and move on.

**Out of scope:** the `Int32` index overflow (`nedges` crosses `typemax(Int32)` around 900³). Amin has deferred this. Do not fix it here; do not make it worse. Where a wider accumulator is free, use one.

## Test and measurement workflow

**Fast loop (~15 s) — use while iterating.** `Pkg.test()` takes ~3 minutes; do not run it per edit. Build a throwaway env once and include the target file directly:

```bash
julia --project=<scratch> -e 'using Pkg; Pkg.develop(path="C:/Users/sadegmo/.julia/dev/Tortuosity"); Pkg.add(["Test","JLD2","HDF5","SparseArrays","LinearAlgebra","Statistics","Random"])'
julia --project=<scratch> -e 'using Test; @testset verbose=true "x" begin include("test/test_assembly.jl") end'
```

**Do not create `test/Project.toml`.** The package uses `[extras]`/`[targets]`; a real `test/Project.toml` would override the mechanism `Pkg.test()` relies on. (`test/Project.toml.bak` is a leftover from a previous attempt — leave it.)

### Test cadence — do not block on the full suite

The suite is comprehensive and `Pkg.test()` takes ~3 minutes, most of it Julia startup and precompilation rather than assertions. Running it per edit would dominate the campaign's wall clock. Space it out deliberately:

| tier | when | what | cost |
| --- | --- | --- | --- |
| 1 | per edit, while iterating | scratch env, `include` only the test files covering the code you touched | ~15 s |
| 2 | per accepted change, before commit | the affected test files plus the assembly and parity tests | seconds |
| 3 | at each phase boundary | full `Pkg.test()` — **launch in the background and keep working** | ~3 min, not blocking |
| 4 | Phase 6 only | full `Pkg.test()` in the foreground, must be green | ~3 min |

**Tier 3 is the important one.** Start the full run in the background and continue to the next item rather than waiting on it. Treat its result as a gate on *starting the next phase*, not on continuing the current one. Because commits are one-per-accepted-change, a red result identifies a bounded window of commits to bisect — that is precisely why the per-change commit rule exists.

**Never let more than one phase of work accumulate on an unverified suite.** If a Tier 3 run has not come back green, do not begin the phase after next.

Reviewers use Tier 2 by default; they should not each pay for a full run when a Tier 3 is already in flight.

`Pkg.test()` exercises the CUDA GPU tests on this machine, so a green full run covers both CPU and GPU paths. Tier 1 and 2 runs against a scratch env do not — that is the tradeoff being made for speed, and it is why Tier 3 exists.

**Benchmark environment:** `julia --project=benchmarks` has CUDA + Tortuosity wired up with a resolved `Manifest.toml`. **Use it as an environment only — do not add files to, modify, or commit anything under `benchmarks/`.** That directory belongs to the separate JOSS submission effort; its contents are parked in a git stash and will be restored later. This campaign's own harness goes in `bench/` (note the different directory), which is tracked and yours to work in. `benchmarks/data/images.h5` and `benchmarks/taufactor` are left in place as gitignored cache and a vendored checkout respectively; ignore both.

### Baseline reproduction

The bug that motivated this work: an 800³ blob image exhausts the 23.89 GiB RTX PRO 5000 during construction.

```julia
img = Imaginator.blobs(shape=(800,800,800), porosity=0.5, blobiness=1.0, seed=42)
sim = SteadyDiffusionProblem(img; axis=:x, gpu=true)   # OutOfGPUMemoryError
```

Generation costs ~60 s at 800³ (~10 s at 400³, ~29 s at 600³); cache the result to a raw file rather than regenerating per run.

Measured baseline (`nnodes` = 254.6 M, `nedges` = 1.503 B, `nnz_L` = 1.758 B at 800³):

| N | pore voxels | edges | peak device memory | outcome |
| --- | --- | --- | --- | --- |
| 200³ | 4.01 M | 22.8 M | 3.25 GiB | solves |
| 400³ | 31.9 M | 186 M | 13.78 GiB | solves |
| 800³ | 254.6 M | 1.503 B | 23.89 GiB (100 %) | **OOM** |

Throw site: `dropzeros!` (`src/kernels/sparse.jl`) via `apply_dirichlet_bc_fast!` (`src/pdetools.jl`), requesting 6.548 GiB. CUDA reported 40.46 GiB of pool reserved against a 23.89 GiB device — earlier stages only "succeeded" by spilling to host memory over PCIe.

Memory target: **~19 GiB peak at 800³** (~198 → ~79 bytes per pore voxel). No speed baseline exists yet — Phase 0 establishes one. Note the assembled-CSC memory floor is ~55 B/pore-voxel (`rowval` + `nzval` at 6.9 nnz per node), which caps this path around 850³; getting past that is `MATRIX_FREE_PLAN.md`'s job, not this plan's. That floor is a reason to push hard on *speed* here, not to stop at the memory target.

## Orchestration protocol

Amin has asked that the master agent stay context-light and act as a director. Rules for the master:

- **Never read source files directly.** Delegate reading, editing, testing, and measuring. Read only agent reports.
- **One write-agent at a time.** Concurrent edits to this codebase are not worth the merge risk. Read-only auditors may run in parallel with each other and with a write-agent.
- **Every write-agent's work is checked by a separate read-only reviewer agent**, never by the master re-reading the diff. The reviewer independently re-runs the tests and the benchmark rather than trusting the report.
- **Agents report in the compact format below.** Anything longer is a bug in the brief.
- **After each phase, the master appends one line per change to the Progress Log** in this file. That log, not the master's context, is the state.

### Evaluator visibility — mandatory under `/goal`

The `/goal` completion evaluator is a small fast model that **cannot run commands, cannot read files, and cannot see inside subagents.** It judges only what the master has written in its own visible message text. This directly conflicts with delegating everything, so it needs a deliberate counter-rule:

**Every turn, the master must print — in its own text, not by pointing at a subagent report — a short progress block:**

```
PHASE: <n> — <name>
DONE THIS TURN: <ids and one-line outcomes>
PROGRESS LOG: <count> items terminal, <count> pending
NEXT: <what the following turn does>
```

And at the milestones the condition names, print the literal evidence: the tail of the `Pkg.test()` output, the `bench/scaling_bench.jl` results table, the exact line `PHASE 4 COMPLETE: 0 accepted candidates`, and the Final report text. Paraphrasing is not enough — "tests passed" is not evidence, the test output is.

This is the one place where spending master context is required rather than avoided. Keep it to the block above plus the milestone evidence; everything else still goes to subagents.

### Required agent report format

```
STATUS:     done | rejected | BLOCKED | REVERTED
CHANGE:     <inventory ids, or NEW: short name>
FILES:      <paths touched>
TESTS:      Pkg.test() -> pass|fail  (assertions, duration)
MEMORY:     peak @800³  before -> after  (delta %)
SPEED:      assembly / solve / e2e  before -> after  (delta %)
COMPLEXITY: simplifying | neutral | +N lines, +M concepts
OVERRIDES:  any heuristic deliberately violated, and why
NOTES:      <= 5 lines. Surprises, rejected sub-ideas, newly discovered opportunities.
```

The `NOTES` field is where newly discovered optimization paths get surfaced. **The master must then write them into the Optimization inventory as new rows with fresh ids** — see *The inventory is a living document*. A `NOTES` mention alone is not enough; reports scroll out of context, the inventory does not.

## Unattended execution

Amin starts this run and returns when it is done. Nobody is available to answer questions mid-run, so the following rules replace every "ask" with a deterministic action.

### Git discipline

**Work on a dedicated branch, and commit one conventional commit per accepted change.** This overrides Amin's standing "never commit unless explicitly instructed" rule — for this campaign only, he has authorized it. One commit per accepted optimization gives a bisect point per change and makes each reviewer's job concrete.

**Branch: `perf/matrix-path`** — see Preconditions above for the details and the checkout gate. Never commit to `main` or `joss`.

**Do not push.** Leave every commit local; Amin decides when and where anything goes to a remote.

**Commits must be path-scoped. Never `git add -A`, never `git commit -a`.** This checkout carries Amin's uncommitted JOSS paper work alongside your changes, and a blanket add would sweep it into a commit labelled as a performance optimization. Stage explicit paths under `src/`, `test/`, `ext/`, and `bench/` only. Leave `paper/`, `benchmarks/`, `docs/`, `JOSS_HANDOFF.md`, and root `*.md` files alone — except this plan file, which you update as you go.

Commit messages: conventional commits, referencing the inventory id — e.g. `perf: fuse connectivity and Laplacian assembly (A5, A6, A7)`. Never `--no-verify`; never bypass hooks. **Never add `Co-authored-by` trailers, AI attribution, or generated-by notices.**

### Blockers — skip, log, continue

A blocker is anything that would normally warrant asking Amin: a golden τ value would change, a pinned test constraint conflicts with a large win, or a change cannot be made without violating a non-negotiable constraint.

**Never change a golden value and never bend a constraint during an unattended run.** Instead: log the item in the Progress Log with status `BLOCKED`, record the reasoning and the size of the win being forgone, move to the next item, and surface every blocker in the final report. A single blocker must never end the run.

### Failures — revert, log, continue

If a change turns `Pkg.test()` red and the agent cannot fix it within a reasonable effort, **revert that change**, log it as `REVERTED` with the failure mode, and continue with the next item. Never leave the tree red. Never commit a red tree. Never disable, skip, or weaken a test to make a change pass — that is a blocker, not a fix.

### Definition of done

The campaign is complete when all of the following hold:

1. Phases 0–3 and 5 have every inventory item in a terminal state: `done`, `rejected`, `BLOCKED`, or `REVERTED`.
2. **Phase 4 has reached diminishing returns** — a full audit round surfaces no candidate that clears the acceptance rule. This is the stop condition for the open-ended speed work; run to exhaustion, not to a fixed count.
3. `Pkg.test()` is green, with an assertion count at or above the Phase 0 baseline.
4. `bench/scaling_bench.jl` has been re-run at all sizes including 800³, and the results are written into this file.
5. The numbers quoted in `MATRIX_FREE_PLAN.md` have been refreshed so the comparison between the two paths stays honest.

### Final report

On completion, write a concise summary at the end of this document covering: total memory and speed deltas at each size; the list of accepted changes with their measured gains; every rejection with its reasoning; every blocker with the size of the forgone win and what decision Amin needs to make; and any newly discovered work that did not fit this campaign. That report, plus the Progress Log, is what Amin reads when he returns — assume he reads nothing else.

## Phases

Phases are sequential for write work. Phase 2 is read-only and runs concurrently with Phase 1.

### Phase 0 — baseline harness (one agent, must finish first)

**First action: commit `MATRIX_PATH_PLAN.md` and `MATRIX_FREE_PLAN.md` to this branch** (`docs: add matrix path optimization campaign plan`). They arrive untracked; committing them versions the Progress Log so the campaign's state is recoverable rather than living only in an untracked file.

Build `bench/scaling_bench.jl`: one canonical script reporting **peak GPU memory, assembly wall time, solve wall time, and end-to-end wall time** at N ∈ {200, 400, 600, 800}, with per-stage memory and time attribution, resumable, CSV-emitting, with warmup and repeats so the timings are trustworthy. Cover the CPU path too. Every later agent reports deltas against this one script so numbers are comparable.

Also confirm `Pkg.test()` is green at HEAD and **record the assertion count and duration as measured, not as quoted anywhere in this document** — the suite was hardened after this plan was written, so any figure here is stale. That measured count is the floor the campaign must never drop below. While you are here, re-verify the three pinned conventions above against the committed suite and correct their file and test names in this document if the hardening moved them.

### Phase 1 — quick wins (serialized write-agents)

A1–A4, A10. Small, local, low-risk. Expected to make 800³ *nearly* fit on their own.

### Phase 2 — exhaustive audit (parallel, read-only)

Principles 2 and 3 say leave no optimization unchecked and go beyond the list. The inventory is **not** assumed complete. Spawn parallel auditors over disjoint territory, each returning a ranked list of candidates with estimated gain (memory *and* speed) and complexity cost. Auditors are explicitly asked to find things this document missed, to flag anything here that is wrong, and to look for missing GPU paths and for implementations that should simply be rewritten. Suggested split:

- **(a) GPU setup path** — `topotools.jl`, `kernels/graph.jl`
- **(b) Sparse ops and SpMV** — `sparse_type.jl`, `kernels/sparse.jl`, `ext/TortuosityCUDAExt.jl`
- **(c) CPU path** — `_build_connectivity_list_cpu`, the `Array{Int,2}` `build_adjacency_matrix` method, CPU BC application, threading opportunities
- **(d) Solve, postprocess, transient** — Krylov workspace and preconditioning, `reconstruct_field`, `tortuosity`, `transient.jl`

Auditors ignore A1–A4 (already in flight).

### Phase 3 — fused assembly (one agent, the main event)

A5 + A6 + A7 as a single coherent change, folding in whatever Phase 2 surfaced. Bulk of the memory saving.

### Phase 4 — speed campaign (serialized write-agents)

Once the memory target is met the conflict rule flips and speed becomes the objective. B-series inventory plus Phase 2 findings. This phase is expected to grow as work reveals new opportunities.

**Stop condition: diminishing returns.** Alternate between an audit round (parallel read-only agents hunting for candidates, including ones nobody has named yet) and an implementation round. The phase ends when a full audit round surfaces no candidate that clears the acceptance rule. Run to exhaustion rather than to a fixed item count — but if a round produces only rejections, that is the signal to stop, not to lower the bar.

Because the `/goal` evaluator cannot judge "diminishing returns" from a transcript, make it explicit: after each audit round print `AUDIT ROUND <n>: <k> candidates surfaced, <m> accepted`, and when a round yields none, print the exact line `PHASE 4 COMPLETE: 0 accepted candidates`. That literal string is what the completion condition keys on.

### Phase 5 — missing GPU paths and structural gaps

C-series. May be pulled earlier if Phase 2 shows one of these dominates.

### Phase 6 — verification and consolidation

Independent adversarial review of the whole diff; full-suite run; re-run of `bench/scaling_bench.jl` at all sizes including 800³; update the numbers in this file and in `MATRIX_FREE_PLAN.md` so the matrix-free comparison stays honest.

## Optimization inventory

### The inventory is a living document — maintain it

**Every `est. gain` figure below was derived by reading code and doing arithmetic on array sizes. Nothing here was measured.** Treat every number as a hypothesis to be tested, not a fact to be reproduced. Expect some items to be worth far more than stated, some far less, and some to evaporate entirely on contact with a profiler.

Maintaining this table is part of the work, not bookkeeping around it:

- **Add what you discover.** Any new optimization candidate — surfaced by an audit, noticed mid-implementation, or made visible only because an earlier change landed — gets a new row with the next free id (`A11+`, `B8+`, `C5+`). Recording it only in a report's `NOTES` is not enough; reports scroll away, this table does not.
- **Correct estimates with measurements.** When a figure is measured, replace it and mark it as measured. If it was badly wrong, say so — a wrong estimate corrected is information about where the reading-based analysis was weak, which makes the remaining estimates easier to weigh.
- **Retire items honestly.** An item that turns out to be already done, moot after another change, or based on a misreading gets marked as such with the reason. Silent deletion loses the fact that it was considered.
- **Re-rank as you go.** Later work changes what matters. An item parked as marginal may clear the bar once something else lands; an item that looked large may shrink once the bottleneck moves.

A campaign that ends with the inventory unchanged from this starting list did not look hard enough.

Starting inventory from one reading pass. **Incomplete by construction** — extend it. `est. gain` figures are at 800³.

### A-series — memory-dominant

| id | change | est. gain | complexity | phase | status |
| --- | --- | --- | --- | --- | --- |
| A1 | Stop calling `dropzeros!` after Dirichlet BC application. It is the crash site and costs ~28 GiB of temporaries to reclaim ~0.4 % of nnz (BC nodes are 0.25 % of all nodes). Explicit zeros are harmless to CSC SpMV and to CUSPARSE. Also stops needless invalidation of the cached CUSPARSE wrapper, and removes a full compaction pass — a speed win too. | −28 GiB transient, + setup speed | simplifying | 1 | pending |
| A2 | Replace `findall(img)` with a prefix sum when building `idx_gpu`. `findall` on a 3-D Bool `CuArray` returns `CartesianIndex{3}` — 24 B per pore voxel, 5.69 GiB — purely to hand out sequential numbers. | −3.8…5.7 GiB | neutral | 1 | pending |
| A3 | Fuse the `temp_inclusive = similar(out)` allocation inside `exclusive_scan!`. | −0.95 GiB | neutral | 1 | pending |
| A4 | `copyto!(write_offsets, colptr[1:n])` materializes a GPU slice before copying it again. | −0.95 GiB | simplifying | 1 | pending |
| A10 | Same as A1 for the transient path — `zero_rows!` calls `dropzeros!` internally. | transient mem | simplifying | 1 | pending |
| A5 | **Fused connectivity → Laplacian assembly.** The pipeline currently computes the CSC structure twice and materializes a COO list it never needs: the connectivity kernel's histogram *is* the column count, its exclusive scan *is* `colptr − 1`, and `conns[:,1]` *is* `rowval` — then `build_adjacency_matrix` throws that away and recomputes it. Emit the Laplacian directly: one thread per grid voxel, skip solids, own column `j = idx[i,j,k]` entirely, write its contiguous slot range. No COO, no adjacency matrix, no scatter, no atomics in assembly. **Must preserve ascending row order within each column (constraint 2) — a ≤ 7-element per-thread insertion sort is free.** | −42 GiB, large setup speedup | moderate ↑ | 3 | pending |
| A6 | Compute edge weights inline inside A5's kernel. For uniform `D` the current code materializes ~5.6 GiB of a repeated constant; for variable `D`, `interpolate_edge_values` materializes another ~5.6 GiB. Both are three flops from `D[p]`, `D[q]`. | −5.6…11.2 GiB, − a kernel launch | neutral within A5 | 3 | pending |
| A7 | Apply Dirichlet BCs during assembly rather than by mutating afterwards. Deletes `zero_rows_cols!`, `set_diag!`, `dropzeros!`, and the `b .-= A * x_bc` step — the last being a full 1.758 B-nonzero SpMV against a vector that is zero at 99.75 % of entries. **Must satisfy the six identities in constraint 1.** | −1 full SpMV, simplifying | moderate | 3 | pending |
| A8 | Symmetric upper-triangle-only storage. Halves `rowval`+`nzval` but reintroduces SpMV atomics. Classic memory-vs-speed trade — judge under the conflict rule. | −6.55 GiB, slower SpMV | high ↑ | — | **likely reject** — confirm in Phase 2 |
| A9 | Uniform-`D` specialization omitting `nzval` entirely (off-diagonals all −1, diagonal = degree). | −6.55 GiB | — | — | **defer** — this is matrix-free with extra steps; belongs in `MATRIX_FREE_PLAN.md` |

### B-series — speed-dominant

| id | change | est. gain | complexity | phase | status |
| --- | --- | --- | --- | --- | --- |
| B1 | **Hand CUSPARSE a CSR view instead of CSC.** The Laplacian is symmetric, so CSC and CSR are the same arrays. CUDA.jl's `CuSparseMatrixCSC` SpMV may transpose or convert internally on every call; wrapping as `CuSparseMatrixCSR` would skip that. Measure first — the cost may already be zero. | unknown, potentially large | trivial if it works | 4 | pending |
| B2 | **Atomic-free SpMV on the portable KA path.** `_spmv_kernel!` is column-parallel over CSC and does `Atomix.@atomic y[r] += v` per nonzero — 1.758 B atomics at 800³. Symmetry means the same arrays read as CSR give a row-parallel kernel with a register accumulator and zero atomics. Affects Metal, AMDGPU, and CPU (not CUDA, which uses CUSPARSE). | large on non-CUDA backends | neutral | 4 | pending |
| B3 | **Jacobi (diagonal) preconditioning for CG.** Currently unpreconditioned. The diagonal is already available; costs one vector (0.95 GiB at 800³) and typically cuts iteration count materially. A memory-for-speed trade — apply the conflict rule. | potentially large end-to-end | small | 4 | pending |
| B4 | **`CartesianIndices(im_gpu)[linear_idx]` inside hot kernels** costs an integer div/mod per thread. A 3-D `ndrange` gives `i, j, k` directly. Appears in both connectivity kernels. | unknown, likely real | simplifying | 4 | pending |
| B5 | **Launch configuration.** Workgroup sizes are hardcoded to 256 and `ndrange` covers the full grid, so ~50 % of threads idle on solid voxels at ε = 0.5. Consider launching over pore voxels, and tuning occupancy. | unknown | small | 4 | pending |
| B6 | **Multithread the CPU connectivity build.** `_build_connectivity_list_cpu` is a serial loop over every voxel. CPU-path speed matters for the PuMA comparison in the JOSS benchmarks. | large on CPU | moderate | 4 | pending |
| B7 | **CPU `build_adjacency_matrix` serial `colptr` loop** — same story, a serial scan over all edges. | moderate on CPU | small | 4 | pending |

### C-series — missing GPU paths and structural gaps

| id | gap | note | phase | status |
| --- | --- | --- | --- | --- |
| C1 | `find_boundary_nodes` explicitly copies the whole image device→host ("boundary detection is cheap, avoid GPU indexing issues"). At 800³ that is a 512 MB transfer plus CPU work — not cheap. | write the kernel | 5 | pending |
| C2 | Postprocessing (`reconstruct_field`, `tortuosity`, `effective_diffusivity`, `formation_factor`) runs on CPU because the struct deliberately keeps `img` host-side. At 800³ this is a large transfer plus a full CPU pass. Evaluate whether a device-side path is worth it. | measure first | 5 | pending |
| C3 | `Imaginator.trim_nonpercolating_paths` uses CPU-only `label_components`. This is in the normal user workflow and is likely a serious bottleneck at 800³. GPU connected-component labelling is nontrivial but well-studied. | judge cost/benefit | 5 | pending |
| C4 | `Imaginator.blobs` is CPU-only and takes ~60 s at 800³. Test-image generation rather than solver code — probably lower priority, but it does gate the benchmark harness. | agent's judgement | 5 | pending |

### Correction carried forward

An earlier draft claimed the atomic KA SpMV kernel (`_spmv_kernel!`) is the CUDA hot path. It is not — `ext/TortuosityCUDAExt.jl` overrides `mul!` to call CUSPARSE. The atomic kernel is what Metal, AMDGPU, and CPU use. Any SpMV optimization must be evaluated separately for the CUSPARSE path and the portable KA path (hence B1 and B2 being distinct items).

## Progress log

**This log is the campaign's state.** Read it before doing anything; append to it after every change, including rejections, blockers, and reverts. An empty log is the only thing that means Phase 0 has not run.

Format: `date — id(s) — status — memory delta — speed delta — commit sha — reviewer verdict`.

_(empty — Phase 0 not yet started)_
