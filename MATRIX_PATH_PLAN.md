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
2. **CSC row ordering — ~~as stated below~~ CORRECTED 2026-08-08 by audits (a) and (b), independently, with evidence.** The original claim was: *"Connectivity rows must stay grouped by column and ascending within a column… CUSPARSE also depends on it."* **That is false for the GPU path.** `write_connections_offset_kernel!` (`kernels/graph.jl:131-205`) buckets by `conns[:,1]`, i.e. by *row*, and `_scatter_coo_to_csc_kernel!` (`graph.jl:242-253`) then re-scatters by *column* with per-column atomics — so row order within a column is **nondeterministic**, and the committed suite already documents it: `test/test_gpu_parity.jl:266-278` measures ~90 % of columns unsorted on a 24³ open image, `test_impl_parity.jl:199-202` says the same, and the suite contains a passing test named *"CUSPARSE SpMV tolerates unsorted row indices within a column"*. The real invariant is only: **rows must be grouped by column** (`colptr` must bound each column correctly). Ascending order within a column holds only for the CPU `build_adjacency_matrix(::Array{Int,2})` method (`topotools.jl:161-175`).

   **Ruling on `test/test_gpu_parity.jl:291` (`@test unsorted > 0`).** A5/A12 done owner-parallel emit ascending rows *for free* (column-major monotonicity of the neighbour offsets `−nx·ny, −nx, −1, +1, +nx, +nx·ny`), which turns that assertion red. This is permitted and is **not** a weakened test: replacing "unsorted output exists" with "output is sorted" asserts a strictly stronger property. Conditions: (i) the change must be otherwise accepted on its own merits; (ii) the *"CUSPARSE tolerates unsorted"* test must be preserved by constructing an unsorted matrix explicitly rather than by relying on assembly output, so no coverage is lost; (iii) log it as an `OVERRIDE` and have the reviewer sign off. Do **not** add A5's proposed insertion sort — sortedness is free by construction, so the sort is dead code.
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
| A10 | Same as A1 for the transient path — `zero_rows!` calls `dropzeros!` internally. **The plan's stated implementation breaks a pinned test** (audit d, correction 1): `test/test_sparse_ops.jl:186` asserts `nnz(P) == 2` after `zero_rows!(P, [1])`, and `zero_rows!`'s docstring (`src/sparse_type.jl:36-45`) states "then drop the resulting structural zeros" as its *contract*. Removing `dropzeros!` from `zero_rows!` is a contract change, not an optimization. **Correct form: stop *calling* `zero_rows!` in `build_transient_operator` and inline the row-zeroing there** — same win, contract and tests intact. Measured cost of keeping the zeros: ~7.7 M extra nnz = 0.44 %. | ~−35 GiB transient (dropzeros' true footprint, see correction 7) | simplifying | 1 | pending |
| A5 | **Fused connectivity → Laplacian assembly.** The pipeline currently computes the CSC structure twice and materializes a COO list it never needs: the connectivity kernel's histogram *is* the column count, its exclusive scan *is* `colptr − 1`, and `conns[:,1]` *is* `rowval` — then `build_adjacency_matrix` throws that away and recomputes it. Emit the Laplacian directly: one thread per grid voxel, skip solids, own column `j = idx[i,j,k]` entirely, write its contiguous slot range. No COO, no adjacency matrix, no scatter, no atomics in assembly. **Must preserve ascending row order within each column (constraint 2) — a ≤ 7-element per-thread insertion sort is free.** | −42 GiB, large setup speedup | moderate ↑ | 3 | pending |
| A6 | Compute edge weights inline inside A5's kernel. For uniform `D` the current code materializes ~5.6 GiB of a repeated constant; for variable `D`, `interpolate_edge_values` materializes another ~5.6 GiB. Both are three flops from `D[p]`, `D[q]`. | −5.6…11.2 GiB, − a kernel launch | neutral within A5 | 3 | pending |
| A7 | Apply Dirichlet BCs during assembly rather than by mutating afterwards. Deletes `zero_rows_cols!`, `set_diag!`, `dropzeros!`, and the `b .-= A * x_bc` step — the last being a full 1.758 B-nonzero SpMV against a vector that is zero at 99.75 % of entries. **Must satisfy the six identities in constraint 1.** | −1 full SpMV, simplifying | moderate | 3 | pending |
| A11 | **`I = conns[:,1]` / `J = conns[:,2]` materialize full copies** (`src/topotools.jl:203-204`). `getindex`, not `@view` — two `nedges` Int32 vectors, both live when `rowval`/`nzval` are allocated, so it sets peak. One-line fix. *(audit a, rank 1)* | −11.2 GiB, −2 full device copies | simplifying | 1 | pending |
| A12 | **Swap the emitted pair in the connectivity kernel so `conns` is grouped by CSC column** (`src/kernels/graph.jl:148-149` + 5 identical pairs). The kernel buckets by `neighbor_val_idx` but stores it in column 1, so the list is grouped by `rowval` — the wrong key. Swapping makes `d_bucket_write_counters` *be* the inclusive column scan; `colptr = [1; counters.+1]`, `rowval = conns[:,1]`. Deletes steps 1–3 of `build_adjacency_matrix` (`topotools.jl:205-227`) entirely. *(audit a, rank 2)* | −19.6 GiB, −2 atomic kernels −1 scan | neutral→simplifying (net −25 lines) | 3 | pending |
| A13 | **No `unsafe_free!` anywhere on the setup path** (`src/simulations.jl:195-202`). `conns` (11.2 GiB), `gd` (5.6 GiB), `am` (11.2 GiB) stay reachable while the next stage allocates. Likely the actual cause of the baseline's "40.46 GiB pool reserved against a 23.89 GiB device". *(audit a, rank 5)* | up to −17 GiB peak reserved | +1 concept, ~8 lines | 1 | pending |
| A14 | **`laplacian` computes degrees with a full SpMV** (`src/sparse_type.jl:247-250`): `ones_v = fill(1, n)` (0.95 GiB) + `mul!(degrees, am, ones_v)` = a real CUSPARSE SpMV over 1.503 B nonzeros to get what is a per-column reduction of `nzval`. *(audit a, rank 7)* | −0.95 GiB, −1 full SpMV | simplifying (6-line kernel) | 1 | pending |
| A15 | **`b` is built on the host and uploaded** (`src/simulations.jl:191`): 0.95 GiB host alloc + PCIe upload, then overwritten with zeros. `fill!(similar(img_dev, T, nnodes), zero(T))` does it on device. *(audit a, rank 8)* | −0.95 GiB host, −0.1 s PCIe | simplifying | 1 | pending |
| A20 | **The Krylov workspace is allocated TWICE per solve** (`src/simulations.jl:213` via LinearSolve). `__init` builds a full `CgWorkspace` (the `zeroinit` path cannot shrink a `PortableSparseCSC`, so it hits `KS(A,b)`), then sets `isfresh=true`, so `solve!` builds a **second** one and drops the first. Fix: one `LinearSolve.init_cacheval(::KrylovJL, ::PortableSparseCSC, …)` method returning an empty-`S` workspace when `zeroinit=true`. **THIS IS THE ITEM THAT DECIDES FIT VS OOM** — see plan correction 5 below. *(audit d, rank 1)* | −3.8…4.75 GiB at the peak moment | ~10 lines, +0 concepts | **1 (promoted)** | pending |
| A21 | **τ needs 4 slices; `reconstruct_field` builds all 512 M voxels to serve 2.56 M of them** (`src/utils.jl:49`, `src/dnstools.jl:31`). `effective_diffusivity` reads only slices 1, N, ind, ind+1. Written-to-read ratio is 200:1. `SteadyDiffusionProblem` **already computes** the inlet/outlet pore-index lists (`simulations.jl:181-182`) and throws them away. Keep the `(c, img)` methods so golden tests are untouched; add `tortuosity(u, sim)`. **Needs its own parity test — the golden tests exercise the OLD API and will not catch a bug here.** *(audit d, rank 2)* | −3.1 GB host; ~2–3.5 s → ~10 ms | +40 lines, +1 field | 5 | pending |
| A22 | **`reconstruct_field` copies a `BitArray` it does not need to** (`src/utils.jl:53`). `img isa Array` is false for `BitArray`, triggering a full `Array(img)` — 512 MB at 800³ — although CPU logical indexing works on `BitArray` directly. The test suite and `TransientDiffusionProblem` both use `BitArray`; `Imaginator.blobs` returns `Array{Bool}`, which is why nobody noticed. Test `_on_gpu(img)`, which is what the comment says it means. *(audit d, rank 14)* | −512 MB host | simplifying | 1 | pending |
| A16 | **The CUSPARSE `_cache` pins stale device buffers** (`ext/TortuosityCUDAExt.jl:45`). `A._cache[]` holds a `CuSparseMatrixCSC` referencing `A.colptr/rowval/nzval`; `b .-= A * x_bc` populates it with the *pre-elimination* arrays, then `dropzeros!` swaps in new ones — old `rowval` + `nzval` (14.1 GiB) stay reachable. One line: `A._cache[] = nothing` before reassignment. **Also note the perverse coupling** — the pointer-equality invalidation at `ext:36-40` is only sound *because* the cache pins the memory; adding `unsafe_free!` (A13) to those arrays turns pointer equality into silent corruption. Replace with a generation counter. *(audit b, rank 2)* | −14.1 GiB | simplifying | 1 | pending |
| A17 | **`dropzeros!` allocates a fully redundant nnz array** (`src/kernels/sparse.jl:275-278`). `scan_output` holds `scan_inclusive` shifted by one, but the kernel branch is only taken when `flags[k]`, so `new_idx == scan_inclusive[k]` exactly. The array, the `fill!`, and the two-view `copyto!` are dead weight. *(audit b, rank 6)* | −7.03 GiB (28 % of dropzeros peak) | simplifying (−4 lines) | 1 | pending |
| A18 | **`laplacian` pays 2.04 GiB to discover something structurally known** (`src/sparse_type.jl:253-262`): `diag_missing` + `extra_scan` + 2 kernels + a scan, to handle self-loops that `build_connectivity_list` can never produce. When no column carries a self-loop, `L_colptr[j+1] == A_colptr[j+1] + j` in closed form. *(audit b, rank 8)* | −2.04 GiB, −2 launches −1 scan | +1 branch, generic path kept | 1 | pending |
| A19 | **Three small free wins** (*audit b, rank 13*): (a) `Base.:*` (`sparse_type.jl:134`) `fill!`s a 1.02 GB result both `mul!` paths immediately overwrite; (b) the non-`Int32` `_as_cusparse` fallback (`ext:52-60`) converts `colptr` **and** `rowval` on **every** SpMV — 7.03 GB allocated per call, uncached, silently, reachable from any user `inds::Array{Int,3}`; (c) no 5-arg `mul!` for `PortableSparseCSC` outside the CUDA ext, so a 5-arg call on Metal/AMDGPU/CPU falls into `generic_matvecmul!` and dies on scalar indexing (latent). | −1.02 GiB; avoids a silent ~100× slowdown | simplifying / +7 lines | 1 | pending |
| A23 | **CPU Dirichlet application is the largest single CPU setup cost, and the inventory had no item for it** (`src/pdetools.jl:60-83`). `findnz` allocates `I,J,V` (3 × nnz × 8 B = 5.2 GiB at 400³) and **discards `V`**; then two hashed `Set` scans + a `union` + a scatter; then a full-matrix `b .-= A*x_bc`. Replace with one pass over `colptr`, folding the RHS from BC columns only (~0.25 % of columns). Must reproduce all six identities at `test_assembly.jl:405-410`, must zero BC rows **and** BC columns including BC–BC couplings, and must iterate BC columns in **ascending** order for a bit-identical RHS (`vcat(inlet,outlet)` is NOT sorted). *(audit c, rank 2)* | −5.7 GiB @400³; 20–50× on this stage | simplifying | 3 | pending |
| A24 | **Sparse `setindex!` insertion cliff — silent, data-dependent** (`src/pdetools.jl:80`). `A[diag_inds] .= diag_vals` is scalar `SparseMatrixCSC` `setindex!`. Verified live: `spdiagm(d) - am` **prunes** a zero-degree node's diagonal, so every isolated BC voxel forces a structural **insert** = `resize!` + memmove of all of `rowval`/`nzval` — ~0.5 s each at 400³. Zero on a clean duct, **tens of seconds on an untrimmed blob**, and 33/36 blob fixtures have such a voxel. Looks like "assembly is just slow". `test_assembly.jl:319-337` pins this case; keep the unit-diagonal fallback. *(audit c, rank 3)* | removes an O(nnz) memmove per isolated BC voxel | simplifying (falls out of A23/A25) | 3 | pending |
| A25 | **Fused CPU assembly — A5/A6/A7 written once as a shared KA kernel** (`src/topotools.jl:45,161,13`). Today CPU allocates a `6·npore × 2` `Int` matrix, copies it down, copies column 1 again as `rowval`, fills a constant weight vector, builds `spdiagm`, then does a sparse–sparse merge for `D − A`; `conns[:,2]` is write-only for uniform `D`. **KA's CPU backend already multithreads** (`KernelAbstractions/src/cpu.jl:98-121` spawns over `Threads.nthreads()`), so writing A5 once as a KA kernel gives CPU threading free, **subsumes B6 and B7 at zero extra cost**, and deletes the CPU/GPU drift surface (the two paths already disagree — CPU sorts `conns` by column, GPU by row). Do **not** route CPU through the *existing* KA path: `_scatter_coo_to_csc_kernel!` uses atomics (per-nonzero lock cmpxchg on CPU) and destroys row order. Keep the old functions exported — `test_assembly.jl:387,451` and `bench/cpu_bench.jl` call them directly. *(audit c, ranks 4+5)* | 400³ peak ~15 GiB → ~4.5 GiB; assembly ~2–3× | moderate up-front, simplifying after | 3 | pending |
| A26 | **CPU CSC uses `Int64` indices where GPU uses `Int32`** (`src/topotools.jl:161-174`). SpMV is bandwidth-bound at 16 B/nnz → 12 B/nnz. Brushes the deferred `Int32` overflow: `nnz_L` = 1.758 B < `typemax(Int32)` fits, but with no margin at 800³ — **gate on `nedges` and keep `Int64` above the threshold**. *(audit c, rank 11)* | −0.75 GiB @400³ (−21 % of final matrix); +~15 % every SpMV | neutral (one type param) | 4 | pending |
| A8 | Symmetric upper-triangle-only storage. ~~Halves `rowval`+`nzval` but reintroduces SpMV atomics.~~ | −6.0 GiB exact (plan's −6.55 was ~9 % high) | high ↑ | — | **REJECTED** (audit b, rank 10). The plan's stated cost was too weak: it does not merely "reintroduce atomics", it **forfeits CUSPARSE entirely** — the generic cuSPARSE API has no symmetric SpMV, so `y = Ux + Uᵀx − diag(U)x` must be hand-written in KA, replacing a vendor-tuned kernel with 751 M atomics per iteration on the *only benchmarked backend*. The conflict rule also retires it: once A5+A6+A7 clear the 19 GiB target, speed wins and A8 loses on both axes. |
| A9 | Uniform-`D` specialization omitting `nzval` entirely (off-diagonals all −1, diagonal = degree). | −7.03 GiB (not −6.55) | — | — | **REJECTED** (audit b, rank 11) — and the plan's *stated reason was wrong*. It is **not** "matrix-free with extra steps": it keeps `rowval` and `colptr`, so it is squarely an assembled-path item. It fails for A8's reason — a matrix without `nzval` cannot be handed to CUSPARSE. Reject here; do **not** export it to `MATRIX_FREE_PLAN.md` as a live item. |

### B-series — speed-dominant

| id | change | est. gain | complexity | phase | status |
| --- | --- | --- | --- | --- | --- |
| B1 | **Hand CUSPARSE a CSR view instead of CSC — guarded by a symmetry flag.** ~~CUDA.jl's `CuSparseMatrixCSC` SpMV may transpose or convert internally on every call~~ — **mechanism misattributed**: CUDA.jl 5.11.3 does neither, it passes a native CSC descriptor (`cusparseCreateCsc`, `helpers.jl:220`). The cost is *inside* cuSPARSE plus an **uncached per-call `with_workspace` device allocation** (`lib/utils/call.jl:23,61-63`) — cuSPARSE documents extra workspace for SpMV on CSC and none for CSR/op=N, so any nonzero buffer is a fresh device malloc+free **every iteration**. 10-second zero-GPU check that settles it: print `cusparseSpMV_bufferSize` for the same arrays as CSC vs CSR. **DANGER (audit b):** the "Laplacian is symmetric" premise **fails on the transient path** — `build_transient_operator` → `zero_rows!` (`transient.jl:305`) zeroes rows and *not* columns, so A is asymmetric from then on and a blanket CSR reinterpretation computes `Aᵀc` and returns a *plausible wrong answer*. Needs a `symmetric::Bool` field cleared by `zero_rows!` and any future row-only mutator. Also correct the stale comment at `test_gpu_parity.jl:273`. | removes a per-`mul!` workspace alloc; potentially 1.5–3× on every SpMV | +1 field, ~10 lines | 4 | pending |
| B2 | **Atomic-free SpMV on the portable KA path.** CONFIRMED: `_spmv_kernel!` (`sparse_type.jl:106-117`) is column-parallel over CSC and does `Atomix.@atomic y[r] += v` per nonzero — 1.758 B atomics plus a preceding `fill!(y,0)` per SpMV. The symmetric row-parallel rewrite is *shorter* than the current kernel and drops the `fill!`. **Must share B1's symmetry flag** or it silently computes `Aᵀc` on the transient path. Plan detail corrected: **"and CPU" is overstated** — the production CPU path returns `SparseMatrixCSC` (`topotools.jl:161-175`) and never touches `_spmv_kernel!`; it is CPU-relevant only in tests. | large on Metal/AMDGPU | simplifying (net −1 line) | 4 | pending |
| B14 | **Dirichlet row-zeroing is O(nnz) when it can be O(\|bc\|·d²)** (`src/kernels/sparse.jl:102-112,138-173`). `zero_rows_kernel!` sweeps all 1.758 B nonzeros to zero ~1.28 M rows. For a structurally symmetric matrix row `j`'s entries live in the ≤6 columns listed in `rowval[colptr[j]:colptr[j+1]-1]` — one thread per BC node finds them all. Replaces a 7 GB read + 7 GB write with ~46 M scattered ops. **Overlaps A7, which deletes the call entirely — do this only if A7 slips.** *(audit b, rank 5)* | −0.25 GiB, ~38× less work on that step | +25 lines, +1 concept | 4 | pending |
| B15 | **`dropzeros!` does a binary search *and* an atomic per nonzero** (`src/kernels/sparse.jl:224-227`): `searchsortedlast(colptr_old, k)` = 1.758 B × ~28 random probes, plus 1.758 B `Atomix` increments, purely to recover which column a nonzero belongs to. A column-parallel kernel (thread per column, count kept entries in its own slot range) needs zero searches and zero atomics, and lets `new_col_counts` write `new_colptr` directly. *(audit b, rank 7)* | large on compaction, −1.0 GiB | neutral | 4 | pending |
| B16 | **BC diagonal handling touches every column instead of 0.25 % of them** (`src/pdetools.jl:95,117` → `kernels/sparse.jl:83-93,34-53`). `get_diag` and `set_diag!` launch n-wide kernels scanning ~7 rowvals each; only the ~1.28 M BC columns change. The zero-degree interior-node carve-out at `pdetools.jl:110-115` must survive verbatim. *(audit b, rank 9)* | ~50 ms at 800³, −1.02 GiB | neutral | 4 | pending |
| B3 | **Jacobi (diagonal) preconditioning for CG.** **REJECTED** *(audit d, rank 10)*. CG is confirmed unpreconditioned and adding one is genuinely cheap, but both of the plan's numbers are wrong. **Memory: 1.90 GiB, not 0.95** — Krylov allocates `z` lazily via `allocate_if(!MisI, …)` (`cg.jl:142`), so *any* preconditioner adds a workspace vector on top of the diagonal itself. **Speed: 5–10 %, not "large"** — after elimination the diagonal *is* the degree, 1…6 for uniform `D`, so Jacobi rescales rows by a factor bounded by 6 and leaves untouched the low-frequency mode that sets κ ~ O(N²); iterations ~ √κ improve by at most √(6/5.4) ≈ 5 %, and for a constant-diagonal Laplacian Jacobi is **exactly the identity** (CG is invariant to scalar scaling). Fails the heuristic on both sides: <15 % gain for 10 % of the memory budget, at a point where headroom is zero. **Revisit only for strongly variable `D`**, where the diagonal spans the `D` contrast. | <15 % | +2 lines but +1.90 GiB | — | **rejected** |
| B4 | **`CartesianIndices(im_gpu)[linear_idx]` inside hot kernels** costs an integer div/mod per thread. A 3-D `ndrange` gives `i, j, k` directly. Appears in both connectivity kernels. | unknown, likely real | simplifying | 4 | pending |
| B5 | **Launch configuration.** Workgroup sizes are hardcoded to 256 and `ndrange` covers the full grid, so ~50 % of threads idle on solid voxels at ε = 0.5. Consider launching over pore voxels, and tuning occupancy. | unknown | small | 4 | pending |
| B6 | **Multithread the CPU connectivity build.** Validated — the loop is serial over ~188 M edges at 400³. **But a naive `Threads.@threads` over voxels with a shared `row` counter silently corrupts the CSC**: `build_adjacency_matrix(::Array{Int,2})` requires `conns` grouped by *ascending* column with ascending rows inside, which the serial loop guarantees *structurally* (`CartesianIndices` order = pore-numbering order, six neighbour probes emitted in ascending linear-index order). Pinned at `test_assembly.jl:104-111`. *(audit c, rank 7)* | ~3× on connectivity = ~15–20 % of assembly, <1 % of e2e | free inside A25; +30 lines standalone | 4 | **accept only inside A25; reject as a standalone hand-threaded rewrite** |
| B7 | **CPU `build_adjacency_matrix` serial `colptr` loop.** **REFUTED as a standalone** *(audit c, rank 12)*: `conns[:,2]` is sorted, so this is sequential-access run counting (~0.5 s at 400³), not a random scatter — <1 % of e2e. Parallelising it needs per-thread n-sized histograms = 256 MB/thread at 400³, a **memory regression**. The degree count is free inside A25's first pass. | <1 % of e2e | — | — | **rejected standalone; subsumed by A25** |
| B17 | **Threaded symmetric CPU SpMV** (`src/simulations.jl:213`, `src/sparse_type.jl:119`). The CPU solve calls SparseArrays' **single-threaded** `SparseMatrixCSC` `mul!`; CG is ~10³ iterations × 3.5 GiB/SpMV, so it is **>90 % of CPU end-to-end** and uses 1 of 4 threads. A is symmetric post-elimination, so CSC-read-as-CSR gives a row-parallel atomic-free kernel. **This, not B2, is the real CPU SpMV work** — B2 delivers nothing on CPU (the CPU path returns `SparseMatrixCSC` and never touches `_spmv_kernel!`). Must be opt-in per matrix, never blanket — the transient operator is not symmetric. *(audit c, rank 1)* | solve 2–3×, e2e ~2–2.5× on CPU | +40 lines, +1 concept | 4 | pending |
| B18 | **Fold the `−1/voxel_size²` scaling into the edge weights** (`src/transient.jl:302`). `nonzeros(A) .= nonzeros(A) ./ (−voxel_size^2)` is a read-modify-write over all 1.758 B nonzeros to apply a constant. For scalar `D`, `gd` is a scalar (line 296) so the factor rides on it free; `L(αA) = αL(A)`, bit-identical up to one multiply-vs-divide rounding. *(audit d, rank 4)* | −14 GB device traffic (~10 ms) | simplifying (−1 line) | 4 | pending |
| B19 | **`reconstruct_slice` copies the whole solution vector per slice** (`src/geometry.jl:71`). `c[mask] .= Array(u)[ind_slice[mask]]` — full materialise then gather, called twice per `flux`, and `StopAtFluxBalance` (`transient.jl:450`) additionally does `Array(c)` on **every ODE step**. `StopAtPeriodicState` (`transient.jl:580`) already does the right thing and documents "~180× faster, ~380× less memory" — same fix. *(audits c+d)* | ~1000× per fire | simplifying (−2 lines) | 4 | pending |
| B20 | **CPU `zero_rows!` is a serial `Set`-hash over every nonzero** (`src/transient.jl:311-320`): `A.rowval[i] in target` for 1.758 B entries, single-threaded hash probes. `is_bc = falses(n); is_bc[rows] .= true` is two lines shorter and ~8–10× faster. *(audits c+d)* | ~8–50× on this stage | simplifying, −1 concept | 4 | pending |
| B21 | **No `@inbounds` and a broadcast write in the CPU connectivity loop** (`src/topotools.jl:59-87`). ~7 bounds checks per pore voxel, and `conns[row,:] .= a,b` builds a `SubArray` to write 2 elements a full column-length apart. `@inbounds` must stay inside the existing i/j/k guards — `idx` is `similar(img,Int)`, i.e. **uninitialised at solid voxels**, read-safe only because `img[]` is checked first. Free win **if A25 slips**. *(audit c, rank 8)* | ~1.3–1.8× on this loop | neutral (−1 line) | 4 | pending |
| B22 | **Reuse the pore index for boundary nodes** (`src/topotools.jl:239-276`, `src/simulations.jl:181`). `find_boundary_nodes` walks the entire image **twice** (once per face) purely to recover pore ordinals that `build_connectivity_list` is about to compute anyway; `geometry.jl:52` `slice_indices` already does the cheap version. The transient additionally builds `pore_index` (`transient.jl:141`) then throws it away — `build_connectivity_list` accepts `inds=` and is never given it. ~2 s of **serial host time in the middle of the GPU pipeline** at 800³. Ascending-order contract pinned at `test_geometry_ops.jl:197`. *(audits a+c+d)* | −2 full-image passes, ~2 s @800³ | neutral→simplifying | 4 | pending — supersedes C1 |
| B23 | **No iteration cap; tolerance policy is size- and precision-dependent** (`src/simulations.jl:213`). LinearSolve defaults `maxiters = length(b)` = **254.6 M** at 800³, and `abstol = reltol = sqrt(eps(eltype(b)))` — 1.49e-8 on CPU (Float64), 3.45e-4 on GPU (Float32). Krylov stops at `atol + rtol·‖r0‖`, so atol dominates at small sizes and rtol at large ones: **accuracy is not comparable across sizes or devices**. With κ ~ (2N/π)² ~ 2.6e5 at N=800, Float32 CG's attainable relative residual (~κ·eps ~ 3e-2) is **worse than the requested 3.45e-4** — the recursive residual reports convergence the true residual has not reached, or the run walks toward a 254.6 M-iteration cap. **Golden-safe**: every golden assertion passes an explicit `reltol` (1e-10/1e-12), so defaults are free to change; re-check the physics tests at `reltol=1e-6`. *(audit d, rank 6)* | removes an unbounded downside | neutral | 4 | pending |
| B8 | **Atomic-free histogram** (`src/kernels/graph.jl:33-85`). Pass 1 scatters 6 atomic RMWs per pore thread into a 0.95 GiB array. By adjacency symmetry the contributions to bucket `b` sum to `degree(b)`, so thread `b` can compute its own count and do one coalesced store. 1.503 B uncoalesced atomic RMWs → 0.95 GiB of coalesced stores. *(audit a, rank 3)* | large on assembly | simplifying | 4 | pending |
| B9 | **Owner-parallel deterministic pass 2** (`src/kernels/graph.jl:131-205`). Thread `v` owns column `j = idx[v]` and writes its whole contiguous slot run — no `Atomix.modify!` at all, and rows come out ascending for free. **Turns `test/test_gpu_parity.jl:291` (`@test unsorted > 0`) red** — that test asserts unsortedness as a guard and documents itself as "should be revisited" if assembly starts emitting sorted columns. *(audit a, rank 4)* | removes 1.503 B atomics | neutral | 4 | pending — treat test change as OVERRIDE, needs reviewer sign-off |
| B10 | **Both graph kernels stream `im_gpu` redundantly** (`kernels/graph.jl:40-82, 138-199`): `idx_gpu[n] > 0` already equals `im_gpu[n]`; the code tests both. ~20 % of these kernels' DRAM traffic. *(audit a, rank 10)* | ~20 % of 2 kernels | simplifying | 4 | pending |
| B11 | **7 redundant `KernelAbstractions.synchronize` calls** on the setup path (`topotools.jl:146,208,214,227`; `sparse_type.jl:257,263,281`). KA's CPU backend is already synchronous inside the launch; GPU backends are stream-ordered. Syncs immediately before a host read must stay. *(audit a, rank 11)* | small | simplifying | 4 | pending |
| B12 | **CPU `laplacian` builds two throwaway sparse matrices** (`src/topotools.jl:13-17`): `spdiagm(degrees)` then a generic sparse–sparse `-`, ~3× nnz of transient allocation on the path the PuMA comparison cares about. Must reproduce `D - A`'s pruning of a zero diagonal or the CPU Dirichlet path changes behaviour on zero-degree nodes. *(audit a, rank 15)* | >2× on that CPU stage | +20 lines | 4 | pending |
| B13 | **`find_boundary_nodes` does two full serial column-major walks** over 512 M BitArray elements with a carried `ordinal` dependency, one per face (`src/topotools.jl:239-276`). Closed forms exist: `:bottom` → `1:count(view(img,:,:,1))`, `:top` → an offset of `count(img)`, dim-1 faces → `cumsum(vec(sum(img;dims=1)))`. This is what C1 should have been. *(audit a, rank 9)* | ~2× by fusing, much more via closed forms | neutral / +15 lines | 4 | pending |

### C-series — missing GPU paths and structural gaps

| id | gap | note | phase | status |
| --- | --- | --- | --- | --- |
| C1 | ~~`find_boundary_nodes` explicitly copies the whole image device→host.~~ **PREMISE REFUTED — rejected as written**, independently by audits (a), (c) and (d). Both production call sites (`simulations.jl:181-182`, `transient.jl:276-279`) pass the **host** `img` *before* the GPU transfer, and both structs keep `img` host-side by design; the `Array(img)` branch at `topotools.jl:241` is **dead in production**. There is no 512 MB device→host copy and no kernel to write. The real cost — two serial full-image host passes, ~2 s at 800³, blocking the GPU pipeline — is now **B22**. | — | 5 | **rejected — premise false; re-scoped as B22** |
| C2 | ~~Postprocessing runs on CPU… evaluate whether a device-side path is worth it.~~ **PREMISE INVERTED — rejected as framed** *(audit d, rank 11)*. Postprocessing is not slow because it runs on CPU; it is slow because `reconstruct_field` materialises 512 M voxels so that four 640 k-element slices can be read. A device-side reconstruct would allocate 2.05 GB on a device with **zero headroom at solve peak** and fix nothing. Re-scoped as **A21** ("compute τ from the pore vector using slice indices only"). | — | 5 | **rejected — premise inverted; re-scoped as A21** |
| C3 | `Imaginator.trim_nonpercolating_paths` uses CPU-only `label_components`. This is in the normal user workflow and is likely a serious bottleneck at 800³. GPU connected-component labelling is nontrivial but well-studied. | judge cost/benefit | 5 | pending |
| C4 | `Imaginator.blobs` is CPU-only and takes ~60 s at 800³. Test-image generation rather than solver code — probably lower priority, but it does gate the benchmark harness. | agent's judgement | 5 | pending |
| C5 | **No rocSPARSE fast path.** `ext/TortuosityAMDGPUExt.jl` registers a backend and nothing else, so AMD SpMV runs the 1.758 B-atomic KA kernel. `AMDGPU.rocSPARSE` exists and mirrors the CUSPARSE API (~40 lines mirroring `ext/TortuosityCUDAExt.jl:32-75`). **Unverifiable on this machine (no AMD device), so it would land untested** — which argues for doing B2 instead, since that helps AMD with no new dependency and is testable. *(audit b, rank 12)* | 5 | **likely reject in favour of B2** |

### Correction carried forward

An earlier draft claimed the atomic KA SpMV kernel (`_spmv_kernel!`) is the CUDA hot path. It is not — `ext/TortuosityCUDAExt.jl` overrides `mul!` to call CUSPARSE. The atomic kernel is what Metal, AMDGPU, and CPU use. Any SpMV optimization must be evaluated separately for the CUSPARSE path and the portable KA path (hence B1 and B2 being distinct items).

## Progress log

**This log is the campaign's state.** Read it before doing anything; append to it after every change, including rejections, blockers, and reverts. An empty log is the only thing that means Phase 0 has not run.

Format: `date — id(s) — status — memory delta — speed delta — commit sha — reviewer verdict`.

**Turn-budget note.** The master launched the four Phase 2 auditors *concurrently with* Phase 0 rather than after it (plan says Phase 2 follows Phase 0). Justified under principle 3: auditors are strictly read-only and cannot conflict with write work, and the `/goal` run has a finite turn budget. No write-agent ran concurrently with another.

- 2026-08-08 — Phase 0 preconditions — **done** — branch `perf/matrix-path` confirmed; `test_assembly.jl` + `test_regression_golden.jl` confirmed committed; `git status --short -- src test ext bench` empty — commit `680f883` (`docs: add matrix path optimization campaign plan`)
- 2026-08-08 — Phase 0 baseline test — **done** — `Pkg.test()` **PASS, 11360/11360 assertions, 2m50.6s** (306.97 s wall incl. startup). **This is the floor the campaign must never drop below.**
- 2026-08-08 — Phase 2 audit (a) GPU setup path — **done** — 15 candidates; 5 new A-items (A11–A15), 6 new B-items (B8–B13), 10 plan corrections
- 2026-08-08 — Phase 2 audit (b) sparse/SpMV — **done** — 13 candidates; 4 new A-items (A16–A19), 3 new B-items (B14–B16), 1 new C-item (C5); **A8 and A9 rejected with evidence**
- 2026-08-08 — Phase 2 audit (c) CPU path — **done** — 12 candidates; 4 new A-items (A23–A26), 7 new B-items (B17–B23); **B7 rejected standalone**, B6 accepted only inside A25
- 2026-08-08 — Phase 2 audit (d) solve/postprocess — **done** — 14 candidates; 3 new A-items (A20–A22); **B3 rejected**, **C1 and C2 rejected on refuted premises**

### Phase 2 verdict — the four audits changed the campaign's shape

Five findings that alter the plan rather than extend it:

1. **Constraint 2 was factually wrong** (see the corrected constraint above). Three of four auditors independently found that the GPU path does *not* guarantee ascending row order, and that the suite contains a test *asserting* unsortedness. The plan's mandated insertion sort in A5 is dead code.
2. **A20 decides fit vs OOM, and nothing in the original inventory covered it.** The live solve set is `A` 14.05 + `b` 0.95 + `u` 0.95 + CG's `r,p,Ap` 2.85 = **18.8 GiB — the 19 GiB target with zero slack** — and LinearSolve allocates the Krylov workspace *twice*, adding 3.8–4.75 GiB. Even if every A-series item lands perfectly, 800³ does not fit until A20 is fixed. Promoted to Phase 1.
3. **Two of the four C-series items rest on false premises** (C1: the device→host copy does not happen; C2: the cost is materialising 512 M voxels, not the transfer). Both rejected and re-scoped rather than implemented.
4. **The single largest one-line win was missing entirely** — A11, `conns[:,1]`/`conns[:,2]` materialising 11.2 GiB of copies because they are `getindex` and not `@view`.
5. **CPU memory is not a headline metric but should be.** The CPU path peaks at ~15 GiB for a 400³ image whose final matrix is 3.5 GiB — a 4× overhead and the binding constraint for a reviewer reproducing the PuMA comparison on a 16 GiB machine. `bench/scaling_bench.jl` should report peak CPU RSS.

Inventory grew from 21 items to 40. A-series 11→26, B-series 7→23, C-series 4→5.
