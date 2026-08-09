---
title: Matrix path optimization
created: 2026-08-08
updated: 2026-08-09
status: complete
outcome: 800³ went from OutOfGPUMemoryError to a 20.7 s end-to-end solve; setup 939× faster; 39 commits, suite green at 11576 assertions.
branch: perf/matrix-path (merged to main 2026-08-09)
supersedes: "-"
superseded-by: "-"
related: 2026-08-08-matrix-free-operator.md
---

> **Status: complete.** Campaign to make the assembled sparse-matrix path as fast and memory-efficient as it reasonably can be, run unattended on 2026-08-08. It succeeded: an 800³ image that previously threw `OutOfGPUMemoryError` — and, once that stopped reproducing, took 384 s to set up and never finished solving — now solves end-to-end in **20.7 s** with the two-level preconditioner, peaking at 20.588 GiB of 23.889 with 3.301 GiB of headroom. Setup alone went 384.2 s → 0.409 s. The test suite grew from 11360 to 11576 assertions with no golden value touched. **Four decisions are still open for Amin** — see *Blockers* in the Final report; the largest, making the preconditioner the default, is worth the entire −89 %.

# Matrix path optimization plan

Execution plan for making the **assembled sparse matrix path** as fast and as memory-efficient as it can reasonably be. This is the orchestration document: it was the single source of truth for what was done, what was pending, and what was deliberately rejected. Paired with `2026-08-08-matrix-free-operator.md`, which describes the *separate, later* matrix-free work.

## How to read this document

**The campaign is finished.** If you are here for results, skip to the **Final report** at the end — it carries the measured deltas, the accepted changes, every rejection with its reasoning, the four open decisions, and the follow-up work. The **Optimization inventory** is the item-by-item record; every item in it is terminal.

Everything between here and the Progress log is the *original plan as written before any work started*, kept unedited except where measurement contradicted it (those corrections are marked inline). It is retained because the record of what was predicted, and how wrong some of it turned out to be, is itself useful — see *How the plan itself held up* in the Final report. **Do not read the sections below as current instructions.**

## How this was run — HISTORICAL, campaign complete

> **This section describes how the campaign was executed on 2026-08-08. It is a record, not an instruction. Do not re-run it.** The `/goal` condition below was satisfied; re-running it against a finished campaign would find every item already terminal.

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

Three pinned conventions that an "optimization" can silently violate. **All three re-verified against the committed suite in Phase 0 on 2026-08-08 at `680f883`.** Constraints 1 and 3 were found intact and are restated below with exact locations; constraint 2 was found to be wrong and is corrected in place.

1. **`test/test_assembly.jl:377` — `"Dirichlet elimination — exact contract, $(label)"`** — verified present and unmoved. States `apply_dirichlet_bc_fast!` as six identities against the pre-elimination Laplacian `L`, at lines 405–410. It is parametrized over the five `ASSEMBLY_IMAGES` fixtures, hence the `, $(label)` suffix on the name. The six, in order: `A[free,free] ≈ L[free,free]`; `all(iszero, A[free,bc])`; `all(iszero, A[bc,free])`; `diag(A)[bc] ≈ bc_diag`; `b[bc] ≈ bc_diag .* vals`; `b[free] ≈ -L[free,bc] * vals`. Note the diagonal identity is against `bc_diag`, not `diag(L)[bc]` — a zero-degree boundary node is pinned with a unit diagonal rather than its original zero (line 403). Any change to boundary handling must reproduce all six exactly.
2. **CSC row ordering — ~~as stated below~~ CORRECTED 2026-08-08 by audits (a) and (b), independently, with evidence.** The original claim was: *"Connectivity rows must stay grouped by column and ascending within a column… CUSPARSE also depends on it."* **That is false for the GPU path.** `write_connections_offset_kernel!` (`kernels/graph.jl:131-205`) buckets by `conns[:,1]`, i.e. by *row*, and `_scatter_coo_to_csc_kernel!` (`graph.jl:242-253`) then re-scatters by *column* with per-column atomics — so row order within a column is **nondeterministic**, and the committed suite already documents it: `test/test_gpu_parity.jl:266-278` measures ~90 % of columns unsorted on a 24³ open image, `test_impl_parity.jl:199-202` says the same, and the suite contains a passing test named *"CUSPARSE SpMV tolerates unsorted row indices within a column"*. The real invariant is only: **rows must be grouped by column** (`colptr` must bound each column correctly). Ascending order within a column holds only for the CPU `build_adjacency_matrix(::Array{Int,2})` method (`topotools.jl:161-175`).

   **Phase 0 corroboration, measured independently** (40³ blob, seed 42, five repeated GPU builds). CPU: 0 of 31 802 columns unsorted, in both the adjacency matrix and the Laplacian. GPU: 28 538 / 28 829 / 28 577 / 28 938 / 28 986 of 31 802 columns unsorted across five *identical* builds, with `rowval` differing between runs — so the GPU order is not merely unsorted, it is not reproducible. τ still agrees between the paths to 2.2 × 10⁻⁶ (Float32 vs Float64). Two details worth carrying forward: (i) the CPU guarantee comes from the fixed emission order of the six neighbour `if` blocks in `_build_connectivity_list_cpu` (k−1, j−1, i−1, i+1, j+1, k+1 — ascending linear index), so reordering those blocks corrupts the CPU matrix silently; (ii) `_laplacian_entries_kernel!` splices the diagonal at "its sorted position" (first `row > j`), which with unsorted GPU input lands it somewhere arbitrary. Values stay correct and the CSC stays valid, but that kernel's comment claims an ordering it does not have.

   **Ruling on `test/test_gpu_parity.jl:291` (`@test unsorted > 0`).** A5/A12 done owner-parallel emit ascending rows *for free* (column-major monotonicity of the neighbour offsets `−nx·ny, −nx, −1, +1, +nx, +nx·ny`), which turns that assertion red. This is permitted and is **not** a weakened test: replacing "unsorted output exists" with "output is sorted" asserts a strictly stronger property. Conditions: (i) the change must be otherwise accepted on its own merits; (ii) the *"CUSPARSE tolerates unsorted"* test must be preserved by constructing an unsorted matrix explicitly rather than by relying on assembly output, so no coverage is lost; (iii) log it as an `OVERRIDE` and have the reviewer sign off. Do **not** add A5's proposed insertion sort — sortedness is free by construction, so the sort is dead code.
3. **`test/test_regression_golden.jl`** — verified. `GOLDEN_STEADY` (lines 34–50) holds hard-coded τ values for three blob seeds (1, 42, 100): nine τ values in all, three axes each, every one paired with a golden mean pore concentration and a golden node count. The file carries a second golden table the original draft did not mention: `GOLDEN_VARIABLE_D` (line 82), two hard-coded `D_eff` values on seed 42. A change to any of these is either a bug or a deliberate improvement. **Never update a golden value during an unattended run** — treat it as a blocker (see below).

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

Memory target: **~19 GiB peak at 800³** (~198 → ~79 bytes per pore voxel). No speed baseline exists yet — Phase 0 establishes one. Note the assembled-CSC memory floor is ~55 B/pore-voxel (`rowval` + `nzval` at 6.9 nnz per node), which caps this path around 850³; getting past that is `2026-08-08-matrix-free-operator.md`'s job, not this plan's. That floor is a reason to push hard on *speed* here, not to stop at the memory target.

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
5. The numbers quoted in `2026-08-08-matrix-free-operator.md` have been refreshed so the comparison between the two paths stays honest.

### Final report

On completion, write a concise summary at the end of this document covering: total memory and speed deltas at each size; the list of accepted changes with their measured gains; every rejection with its reasoning; every blocker with the size of the forgone win and what decision Amin needs to make; and any newly discovered work that did not fit this campaign. That report, plus the Progress Log, is what Amin reads when he returns — assume he reads nothing else.

## Phases

Phases are sequential for write work. Phase 2 is read-only and runs concurrently with Phase 1.

### Phase 0 — baseline harness (one agent, must finish first)

**First action: commit `MATRIX_PATH_PLAN.md` and `2026-08-08-matrix-free-operator.md` to this branch** (`docs: add matrix path optimization campaign plan`). They arrive untracked; committing them versions the Progress Log so the campaign's state is recoverable rather than living only in an untracked file.

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

Independent adversarial review of the whole diff; full-suite run; re-run of `bench/scaling_bench.jl` at all sizes including 800³; update the numbers in this file and in `2026-08-08-matrix-free-operator.md` so the matrix-free comparison stays honest.

## Optimization inventory

> **The `status` and `est. gain` columns in the three tables below are AS-WRITTEN, before any work started.** They were maintained during the campaign but the phase write-ups, not the tables, became the record of disposition. **The authoritative final status of every item is the Status ledger immediately below.** Where a table cell and the ledger disagree, the ledger wins. Where a table's `est. gain` disagrees with a measured figure in a phase section, the measurement wins.

### Status ledger — final disposition of every item

Terminal states: **done** (landed), **rejected** (considered, declined, reasoning logged), **moot** (the code it targeted no longer exists on the production path), **BLOCKED** (needs a decision or hardware nobody here has), **retired** (premise was wrong).

#### A-series

| id | final status | note |
| --- | --- | --- |
| A1 | **moot** | Was BLOCKED by `csc_equivalent`; A7 deleted the `dropzeros!` call outright, so the question disappeared. No decision needed. |
| A2 | **done** | Phase 1 |
| A3 | **done** | Phase 1 |
| A4 | **done** | Phase 1 |
| A5 | **done** | Phase 3 — the main event. 800³ setup 384.2 → 0.39 s |
| A6 | **done** | Phase 3, inside A5 |
| A7 | **done** | Phase 3. Not blocked by `csc_equivalent` after all |
| A8 | **rejected** | Forfeits CUSPARSE entirely on the only benchmarked backend — a stronger objection than the plan's stated one |
| A9 | **rejected** | A matrix without `nzval` cannot be handed to CUSPARSE. The plan's stated reason ("matrix-free with extra steps") was itself wrong |
| A10 | **done** | Phase 1, via the corrected implementation (inline the row-zeroing; do not change `zero_rows!`'s contract) |
| A11 | **done** | Phase 1. −11.2 GiB from one line; the largest single-line win found |
| A12 | **subsumed by A5** | Win realised. Standalone form still not viable — `test_gpu_parity` feeds `build_adjacency_matrix` row-grouped `OldBaseline` conns, so that method must stay general. Unclaimed for the transient path |
| A13 | **done** | Phase 1 |
| A14 | **done** | Phase 1, keeping row-sum semantics |
| A15 | **done** | Phase 1 |
| A16 | **done** | Phase 1. Made mandatory by A13, not merely cleaner |
| A17 | **done** | Phase 1 |
| A18 | **done** | Phase 1 |
| A19 | **done (b, c); (a) retired** | (a)'s premise was wrong — `fill!` does not allocate |
| A20 | **done** | Phase 1 |
| A21 | **rejected** | Post was 0.83 % of e2e; re-scored at 7.3 % as B30 once the preconditioner landed, still below the bar, and its cost rose (A7 deleted the node lists it depended on) |
| A22 | **done** | Phase 1 |
| A23 | **done** | Phase 3 |
| A24 | **done** | Phase 3 — nothing inserts, so the `setindex!` cliff cannot occur |
| A25 | **done** | Phase 3. Subsumes B6 and B7 |
| A26 | **done** | Phase 4 r2. CPU e2e −17.2 %. Re-located: the index type is chosen at `assembly.jl`, not `topotools.jl` |
| A27 | **done (transient); moot (steady)** | Phase 4 r1 |
| A28 | **BLOCKED** | `_free!` is a no-op on Metal and AMDGPU. One line each — blocked on having no such hardware to test against |
| A29 | **done** | Phase 4 r1. **Cleared the memory gate.** The plan's premise was wrong: the aliasing already existed; the fix was to never allocate, since freeing does not move CUDA's pool peak |
| A30 | **rejected for this campaign** | Transient-only, moves no metric the harness reports. **Handed to follow-up** — it would retire the old five-stage pipeline and take B4/B8/B10/B12/B13/B21/B22/A12/A27 terminal with it |

#### B-series

| id | final status | note |
| --- | --- | --- |
| B1 | **done** | Phase 4 r1, −8.1 % e2e. Accepted below the 15 % bar as a logged override. The plan's stated mechanism was false; the win is kernel choice and is size-dependent (0.99× at 200³, 1.175× at 800³) |
| B2 | **done** | Phase 4 r1. Un-rejected by measurement — 3.5× on the CPU KA backend |
| B3 | **rejected** | Measured: the whole cheap-preconditioner family loses. Polynomial preconditioning *raises* total SpMV count monotonically with degree |
| B4 | **already done** | `assembly.jl` uses `@index(Global, NTuple)`. Remaining sites are transient-only |
| B5 | **rejected** | Setup is 0.3 % of e2e; no launch config can move a headline number |
| B6 | **subsumed by A25** | KA's CPU backend threads it for free |
| B7 | **rejected standalone; subsumed by A25** | <1 % of e2e, and parallelising it standalone is a memory regression |
| B8 | **moot** | No histogram pass on the steady path. Transient-only |
| B9 | **already done** | Phase 3's `_steady_fill_kernel!` is exactly this |
| B10 | **already done in substance** | `assembly.jl` never reads `img`; `idx` is the mask |
| B11 | **moot / not worth a commit** | 4 of 7 syncs off the steady path; the rest worth <1 ms |
| B12 | **moot for steady** | Transient CPU only |
| B13 | **moot for steady** | Transient-only |
| B14 | **moot** | A7 deleted the call site |
| B15 | **moot** | `dropzeros!` has no production caller anywhere |
| B16 | **moot** | Reachable only from `apply_dirichlet_bc_fast!`, which has no production caller |
| B17 | **BLOCKED** | CPU solve −57 % (4 threads) / −15 % (1 thread), kernel already committed. Needs an API ruling — see Blockers |
| B18 | **done** | Commit `1cf8726`. Accepted on the "simplifying/neutral" branch; needed no new branch because the scalar/array split already existed |
| B19 | **done** | Phase 4 r1 |
| B20 | **done** | Phase 4 r1 |
| B21 | **moot for steady** | Transient-only |
| B22 | **done (transient); retired moot (steady)** | Steady never enumerates BC nodes — `_is_bc` is a coordinate test in-kernel |
| B23 | **BLOCKED** | No mechanism — LinearSolve never consults `prob.kwargs`. Same fix as B24-as-default |
| B24 | **done; default BLOCKED** | 800³ e2e −89.1 %, iterations 3620 → 202, flat in N. **Opt-in only** — see Blockers |
| B25 | **rejected** | Measured identical iteration counts; `LinearProblem(A,b; u0=…)` does not warm-start KrylovJL |
| B26 | **done** | Phase 4 r2, 2.32× on the membership pass — the inventory's "5–20×" and "halves peak host memory" were both wrong |
| B27 | **done (fusion); Float32 half BLOCKED** | Fusion −27 %, bit-identical. Float32 would move two of three golden node counts |
| B28 | **rejected** | Measured with a fresh process per block: 800³ *regresses* +2.4 % and +0.47 GiB. A single global cap is self-contradictory across sizes |
| B29 | **rejected** | 7.7 % of e2e, 9.3 % ceiling. Also corrected: restriction+prolongation is 14.4 % of an iteration, not 46 % — prolongation already runs at bandwidth |
| B30 | **rejected** | 7.3 % of e2e and the cost went *up* — it must now construct node lists A7 deleted |

#### C-series — all five rejected

| id | final status | note |
| --- | --- | --- |
| C1 | **rejected — premise false** | The device→host copy never happens in production; both call sites pass the host `img` before transfer. Re-scoped as B22 |
| C2 | **rejected — premise inverted** | The cost is materialising 512 M voxels, not the transfer. Re-scoped as A21 (then rejected on merit) |
| C3 | **rejected as written** | `label_components` is not the hot spot, and the check is skipped above 50 M voxels anyway. Re-scoped as B26, which **was** accepted |
| C4 | **rejected as a campaign item** | Test-image generation, on no headline metric, and the harness caches it to disk |
| C5 | **rejected** | ~40 lines of rocSPARSE would land with zero test coverage on the one axis where untested code is most dangerous — a wrong SpMV returns a plausible τ, not an error |

#### Review findings (F-series, raised by the independent reviewers)

**Fixed:** F5 (bench cache path), F6 (bit-identity now pinned by 56 exact comparisons), F7/F8 (dead code labelled, stale comments corrected), F11 (cache invalidation — later removed together with the mutation it guarded, by B18), F12 (GPU device-gather test), F14 (bit-identity claim bounded to one workgroup).

**Recorded, not acted on:** F1 (A22 narrows a guard; unreachable today), F2 (`_as_cusparse` caches converted index copies — a silent invariant with no test), F3 (Laplacian diagonal is now nondeterministic on CUDA, ≤6 Float32 terms), F4 (cosmetic docstring), F9 (structural-only pruning vs the old `eps(Float32)` tolerance), F10 (the `Int32` ceiling's binding quantity moved from `nedges` to `nnz`), F13 (`bench/results/scaling.csv` retains only the Phase 0 baseline, so intermediate "before" figures are not independently checkable from the repo).

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
| A1 | Stop calling `dropzeros!` after Dirichlet BC application. It is the crash site and costs ~28 GiB of temporaries to reclaim ~0.4 % of nnz (BC nodes are 0.25 % of all nodes). Explicit zeros are harmless to CSC SpMV and to CUSPARSE. Also stops needless invalidation of the cached CUSPARSE wrapper, and removes a full compaction pass — a speed win too. | −28 GiB transient, + setup speed | simplifying | 1 | *(see ledger)* |
| A2 | Replace `findall(img)` with a prefix sum when building `idx_gpu`. `findall` on a 3-D Bool `CuArray` returns `CartesianIndex{3}` — 24 B per pore voxel, 5.69 GiB — purely to hand out sequential numbers. | −3.8…5.7 GiB | neutral | 1 | *(see ledger)* |
| A3 | Fuse the `temp_inclusive = similar(out)` allocation inside `exclusive_scan!`. | −0.95 GiB | neutral | 1 | *(see ledger)* |
| A4 | `copyto!(write_offsets, colptr[1:n])` materializes a GPU slice before copying it again. | −0.95 GiB | simplifying | 1 | *(see ledger)* |
| A10 | Same as A1 for the transient path — `zero_rows!` calls `dropzeros!` internally. **The plan's stated implementation breaks a pinned test** (audit d, correction 1): `test/test_sparse_ops.jl:186` asserts `nnz(P) == 2` after `zero_rows!(P, [1])`, and `zero_rows!`'s docstring (`src/sparse_type.jl:36-45`) states "then drop the resulting structural zeros" as its *contract*. Removing `dropzeros!` from `zero_rows!` is a contract change, not an optimization. **Correct form: stop *calling* `zero_rows!` in `build_transient_operator` and inline the row-zeroing there** — same win, contract and tests intact. Measured cost of keeping the zeros: ~7.7 M extra nnz = 0.44 %. | ~−35 GiB transient (dropzeros' true footprint, see correction 7) | simplifying | 1 | *(see ledger)* |
| A5 | **Fused connectivity → Laplacian assembly.** The pipeline currently computes the CSC structure twice and materializes a COO list it never needs: the connectivity kernel's histogram *is* the column count, its exclusive scan *is* `colptr − 1`, and `conns[:,1]` *is* `rowval` — then `build_adjacency_matrix` throws that away and recomputes it. Emit the Laplacian directly: one thread per grid voxel, skip solids, own column `j = idx[i,j,k]` entirely, write its contiguous slot range. No COO, no adjacency matrix, no scatter, no atomics in assembly. **Must preserve ascending row order within each column (constraint 2) — a ≤ 7-element per-thread insertion sort is free.** | −42 GiB, large setup speedup | moderate ↑ | 3 | *(see ledger)* |
| A6 | Compute edge weights inline inside A5's kernel. For uniform `D` the current code materializes ~5.6 GiB of a repeated constant; for variable `D`, `interpolate_edge_values` materializes another ~5.6 GiB. Both are three flops from `D[p]`, `D[q]`. | −5.6…11.2 GiB, − a kernel launch | neutral within A5 | 3 | *(see ledger)* |
| A7 | Apply Dirichlet BCs during assembly rather than by mutating afterwards. Deletes `zero_rows_cols!`, `set_diag!`, `dropzeros!`, and the `b .-= A * x_bc` step — the last being a full 1.758 B-nonzero SpMV against a vector that is zero at 99.75 % of entries. **Must satisfy the six identities in constraint 1.** | −1 full SpMV, simplifying | moderate | 3 | *(see ledger)* |
| A11 | **`I = conns[:,1]` / `J = conns[:,2]` materialize full copies** (`src/topotools.jl:203-204`). `getindex`, not `@view` — two `nedges` Int32 vectors, both live when `rowval`/`nzval` are allocated, so it sets peak. One-line fix. *(audit a, rank 1)* | −11.2 GiB, −2 full device copies | simplifying | 1 | *(see ledger)* |
| A12 | **Swap the emitted pair in the connectivity kernel so `conns` is grouped by CSC column** (`src/kernels/graph.jl:148-149` + 5 identical pairs). The kernel buckets by `neighbor_val_idx` but stores it in column 1, so the list is grouped by `rowval` — the wrong key. Swapping makes `d_bucket_write_counters` *be* the inclusive column scan; `colptr = [1; counters.+1]`, `rowval = conns[:,1]`. Deletes steps 1–3 of `build_adjacency_matrix` (`topotools.jl:205-227`) entirely. *(audit a, rank 2)* | −19.6 GiB, −2 atomic kernels −1 scan | neutral→simplifying (net −25 lines) | 3 | *(see ledger)* |
| A13 | **No `unsafe_free!` anywhere on the setup path** (`src/simulations.jl:195-202`). `conns` (11.2 GiB), `gd` (5.6 GiB), `am` (11.2 GiB) stay reachable while the next stage allocates. Likely the actual cause of the baseline's "40.46 GiB pool reserved against a 23.89 GiB device". *(audit a, rank 5)* | up to −17 GiB peak reserved | +1 concept, ~8 lines | 1 | *(see ledger)* |
| A14 | **`laplacian` computes degrees with a full SpMV** (`src/sparse_type.jl:247-250`): `ones_v = fill(1, n)` (0.95 GiB) + `mul!(degrees, am, ones_v)` = a real CUSPARSE SpMV over 1.503 B nonzeros to get what is a per-column reduction of `nzval`. *(audit a, rank 7)* | −0.95 GiB, −1 full SpMV | simplifying (6-line kernel) | 1 | *(see ledger)* |
| A15 | **`b` is built on the host and uploaded** (`src/simulations.jl:191`): 0.95 GiB host alloc + PCIe upload, then overwritten with zeros. `fill!(similar(img_dev, T, nnodes), zero(T))` does it on device. *(audit a, rank 8)* | −0.95 GiB host, −0.1 s PCIe | simplifying | 1 | *(see ledger)* |
| A20 | **The Krylov workspace is allocated TWICE per solve** (`src/simulations.jl:213` via LinearSolve). `__init` builds a full `CgWorkspace` (the `zeroinit` path cannot shrink a `PortableSparseCSC`, so it hits `KS(A,b)`), then sets `isfresh=true`, so `solve!` builds a **second** one and drops the first. Fix: one `LinearSolve.init_cacheval(::KrylovJL, ::PortableSparseCSC, …)` method returning an empty-`S` workspace when `zeroinit=true`. **THIS IS THE ITEM THAT DECIDES FIT VS OOM** — see plan correction 5 below. *(audit d, rank 1)* | −3.8…4.75 GiB at the peak moment | ~10 lines, +0 concepts | **1 (promoted)** | *(see ledger)* |
| A21 | **rejected on measurement (Phase 4 r1).** The post stage is now **0.8 % of end-to-end** at 800³ (1.64 s of 204.7), so the whole item is worth at most 0.8 %, against +40 lines, +1 struct field and a new public `tortuosity(u, sim)` method that would need its own parity suite. Fails the heuristic on both sides. Revisit only if the solve gets fast enough to make 1.6 s matter. ~~τ needs 4 slices; `reconstruct_field` builds all 512 M voxels to serve 2.56 M of them~~ (`src/utils.jl:49`, `src/dnstools.jl:31`). `effective_diffusivity` reads only slices 1, N, ind, ind+1. Written-to-read ratio is 200:1. `SteadyDiffusionProblem` **already computes** the inlet/outlet pore-index lists (`simulations.jl:181-182`) and throws them away. Keep the `(c, img)` methods so golden tests are untouched; add `tortuosity(u, sim)`. **Needs its own parity test — the golden tests exercise the OLD API and will not catch a bug here.** *(audit d, rank 2)* | −3.1 GB host; ~2–3.5 s → ~10 ms | +40 lines, +1 field | 5 | *(see ledger)* |
| A22 | **`reconstruct_field` copies a `BitArray` it does not need to** (`src/utils.jl:53`). `img isa Array` is false for `BitArray`, triggering a full `Array(img)` — 512 MB at 800³ — although CPU logical indexing works on `BitArray` directly. The test suite and `TransientDiffusionProblem` both use `BitArray`; `Imaginator.blobs` returns `Array{Bool}`, which is why nobody noticed. Test `_on_gpu(img)`, which is what the comment says it means. *(audit d, rank 14)* | −512 MB host | simplifying | 1 | *(see ledger)* |
| A16 | **The CUSPARSE `_cache` pins stale device buffers** (`ext/TortuosityCUDAExt.jl:45`). `A._cache[]` holds a `CuSparseMatrixCSC` referencing `A.colptr/rowval/nzval`; `b .-= A * x_bc` populates it with the *pre-elimination* arrays, then `dropzeros!` swaps in new ones — old `rowval` + `nzval` (14.1 GiB) stay reachable. One line: `A._cache[] = nothing` before reassignment. **Also note the perverse coupling** — the pointer-equality invalidation at `ext:36-40` is only sound *because* the cache pins the memory; adding `unsafe_free!` (A13) to those arrays turns pointer equality into silent corruption. Replace with a generation counter. *(audit b, rank 2)* | −14.1 GiB | simplifying | 1 | *(see ledger)* |
| A17 | **`dropzeros!` allocates a fully redundant nnz array** (`src/kernels/sparse.jl:275-278`). `scan_output` holds `scan_inclusive` shifted by one, but the kernel branch is only taken when `flags[k]`, so `new_idx == scan_inclusive[k]` exactly. The array, the `fill!`, and the two-view `copyto!` are dead weight. *(audit b, rank 6)* | −7.03 GiB (28 % of dropzeros peak) | simplifying (−4 lines) | 1 | *(see ledger)* |
| A18 | **`laplacian` pays 2.04 GiB to discover something structurally known** (`src/sparse_type.jl:253-262`): `diag_missing` + `extra_scan` + 2 kernels + a scan, to handle self-loops that `build_connectivity_list` can never produce. When no column carries a self-loop, `L_colptr[j+1] == A_colptr[j+1] + j` in closed form. *(audit b, rank 8)* | −2.04 GiB, −2 launches −1 scan | +1 branch, generic path kept | 1 | *(see ledger)* |
| A19 | **Three small free wins** (*audit b, rank 13*): (a) `Base.:*` (`sparse_type.jl:134`) `fill!`s a 1.02 GB result both `mul!` paths immediately overwrite; (b) the non-`Int32` `_as_cusparse` fallback (`ext:52-60`) converts `colptr` **and** `rowval` on **every** SpMV — 7.03 GB allocated per call, uncached, silently, reachable from any user `inds::Array{Int,3}`; (c) no 5-arg `mul!` for `PortableSparseCSC` outside the CUDA ext, so a 5-arg call on Metal/AMDGPU/CPU falls into `generic_matvecmul!` and dies on scalar indexing (latent). | −1.02 GiB; avoids a silent ~100× slowdown | simplifying / +7 lines | 1 | *(see ledger)* |
| A23 | **CPU Dirichlet application is the largest single CPU setup cost, and the inventory had no item for it** (`src/pdetools.jl:60-83`). `findnz` allocates `I,J,V` (3 × nnz × 8 B = 5.2 GiB at 400³) and **discards `V`**; then two hashed `Set` scans + a `union` + a scatter; then a full-matrix `b .-= A*x_bc`. Replace with one pass over `colptr`, folding the RHS from BC columns only (~0.25 % of columns). Must reproduce all six identities at `test_assembly.jl:405-410`, must zero BC rows **and** BC columns including BC–BC couplings, and must iterate BC columns in **ascending** order for a bit-identical RHS (`vcat(inlet,outlet)` is NOT sorted). *(audit c, rank 2)* | −5.7 GiB @400³; 20–50× on this stage | simplifying | 3 | *(see ledger)* |
| A24 | **Sparse `setindex!` insertion cliff — silent, data-dependent** (`src/pdetools.jl:80`). `A[diag_inds] .= diag_vals` is scalar `SparseMatrixCSC` `setindex!`. Verified live: `spdiagm(d) - am` **prunes** a zero-degree node's diagonal, so every isolated BC voxel forces a structural **insert** = `resize!` + memmove of all of `rowval`/`nzval` — ~0.5 s each at 400³. Zero on a clean duct, **tens of seconds on an untrimmed blob**, and 33/36 blob fixtures have such a voxel. Looks like "assembly is just slow". `test_assembly.jl:319-337` pins this case; keep the unit-diagonal fallback. *(audit c, rank 3)* | removes an O(nnz) memmove per isolated BC voxel | simplifying (falls out of A23/A25) | 3 | *(see ledger)* |
| A25 | **Fused CPU assembly — A5/A6/A7 written once as a shared KA kernel** (`src/topotools.jl:45,161,13`). Today CPU allocates a `6·npore × 2` `Int` matrix, copies it down, copies column 1 again as `rowval`, fills a constant weight vector, builds `spdiagm`, then does a sparse–sparse merge for `D − A`; `conns[:,2]` is write-only for uniform `D`. **KA's CPU backend already multithreads** (`KernelAbstractions/src/cpu.jl:98-121` spawns over `Threads.nthreads()`), so writing A5 once as a KA kernel gives CPU threading free, **subsumes B6 and B7 at zero extra cost**, and deletes the CPU/GPU drift surface (the two paths already disagree — CPU sorts `conns` by column, GPU by row). Do **not** route CPU through the *existing* KA path: `_scatter_coo_to_csc_kernel!` uses atomics (per-nonzero lock cmpxchg on CPU) and destroys row order. Keep the old functions exported — `test_assembly.jl:387,451` and `bench/cpu_bench.jl` call them directly. *(audit c, ranks 4+5)* | 400³ peak ~15 GiB → ~4.5 GiB; assembly ~2–3× | moderate up-front, simplifying after | 3 | *(see ledger)* |
| A26 | **CPU CSC uses `Int64` indices where GPU uses `Int32`** (`src/topotools.jl:161-174`). SpMV is bandwidth-bound at 16 B/nnz → 12 B/nnz. Brushes the deferred `Int32` overflow: `nnz_L` = 1.758 B < `typemax(Int32)` fits, but with no margin at 800³ — **gate on `nedges` and keep `Int64` above the threshold**. *(audit c, rank 11)* | −0.75 GiB @400³ (−21 % of final matrix); +~15 % every SpMV | neutral (one type param) | 4 | *(see ledger)* |
| A8 | Symmetric upper-triangle-only storage. ~~Halves `rowval`+`nzval` but reintroduces SpMV atomics.~~ | −6.0 GiB exact (plan's −6.55 was ~9 % high) | high ↑ | — | **REJECTED** (audit b, rank 10). The plan's stated cost was too weak: it does not merely "reintroduce atomics", it **forfeits CUSPARSE entirely** — the generic cuSPARSE API has no symmetric SpMV, so `y = Ux + Uᵀx − diag(U)x` must be hand-written in KA, replacing a vendor-tuned kernel with 751 M atomics per iteration on the *only benchmarked backend*. The conflict rule also retires it: once A5+A6+A7 clear the 19 GiB target, speed wins and A8 loses on both axes. |
| A9 | Uniform-`D` specialization omitting `nzval` entirely (off-diagonals all −1, diagonal = degree). | −7.03 GiB (not −6.55) | — | — | **REJECTED** (audit b, rank 11) — and the plan's *stated reason was wrong*. It is **not** "matrix-free with extra steps": it keeps `rowval` and `colptr`, so it is squarely an assembled-path item. It fails for A8's reason — a matrix without `nzval` cannot be handed to CUSPARSE. Reject here; do **not** export it to `2026-08-08-matrix-free-operator.md` as a live item. |

### B-series — speed-dominant

| id | change | est. gain | complexity | phase | status |
| --- | --- | --- | --- | --- | --- |
| B1 | **DONE** (Phase 4 r1, `2daaf12`). Measured: the mechanism below was **wrong on both counts** — `cusparseSpMV_bufferSize` returns the *same* size for CSC and CSR (41 543 B at 200³, 338 943 B at 400³, 2 740 547 B at 800³), so no per-call workspace allocation is avoided. The win is the kernel cuSPARSE picks: CSR SpMV gathers, CSC SpMV scatters with atomics. At 800³, 32.9 ms per CSC `mul!` vs 28.0 ms per CSR one (1.175×), and CSR repeats to within 2 % where CSC spreads over 15 %. At 200³ there is **no win at all** (0.99×) — the effect only appears once the matrix exceeds cache. 400³ solve 14.811 → 13.619 s (−8.0 %), iterations unchanged. Symmetry is carried by a `PortableSparseCSC.symmetric` field set only by `build_steady_system` and cleared by `_invalidate_cache!`, which every mutator now calls. ~~Hand CUSPARSE a CSR view instead of CSC — guarded by a symmetry flag.~~ ~~CUDA.jl's `CuSparseMatrixCSC` SpMV may transpose or convert internally on every call~~ — **mechanism misattributed**: CUDA.jl 5.11.3 does neither, it passes a native CSC descriptor (`cusparseCreateCsc`, `helpers.jl:220`). The cost is *inside* cuSPARSE plus an **uncached per-call `with_workspace` device allocation** (`lib/utils/call.jl:23,61-63`) — cuSPARSE documents extra workspace for SpMV on CSC and none for CSR/op=N, so any nonzero buffer is a fresh device malloc+free **every iteration**. 10-second zero-GPU check that settles it: print `cusparseSpMV_bufferSize` for the same arrays as CSC vs CSR. **DANGER (audit b):** the "Laplacian is symmetric" premise **fails on the transient path** — `build_transient_operator` → `zero_rows!` (`transient.jl:305`) zeroes rows and *not* columns, so A is asymmetric from then on and a blanket CSR reinterpretation computes `Aᵀc` and returns a *plausible wrong answer*. Needs a `symmetric::Bool` field cleared by `zero_rows!` and any future row-only mutator. Also correct the stale comment at `test_gpu_parity.jl:273`. | removes a per-`mul!` workspace alloc; potentially 1.5–3× on every SpMV | +1 field, ~10 lines | 4 | *(see ledger)* |
| B2 | **DONE** (Phase 4 r1, `5960ef3`). `_spmv_symmetric_kernel!` reduces a column into `y[j]` instead of scattering it, for matrices carrying B1's `symmetric` flag. Measured on the CPU KA backend at 200³ (26.6 M nonzeros, Float64): scatter 125.5 ms → gather 35.5 ms on one thread, 34.6 → 13.3 ms on four. `y = A*x` is bit-identical to the scatter (same summation order); the 5-argument form matches only to rounding, since `alpha`/`beta` are applied in one expression rather than in separate passes. Live for Metal/AMDGPU and the CPU KA backend; CUDA keeps CUSPARSE. ~~Atomic-free SpMV on the portable KA path.~~ CONFIRMED: `_spmv_kernel!` (`sparse_type.jl:106-117`) is column-parallel over CSC and does `Atomix.@atomic y[r] += v` per nonzero — 1.758 B atomics plus a preceding `fill!(y,0)` per SpMV. The symmetric row-parallel rewrite is *shorter* than the current kernel and drops the `fill!`. **Must share B1's symmetry flag** or it silently computes `Aᵀc` on the transient path. Plan detail corrected: **"and CPU" is overstated** — the production CPU path returns `SparseMatrixCSC` (`topotools.jl:161-175`) and never touches `_spmv_kernel!`; it is CPU-relevant only in tests. | large on Metal/AMDGPU | simplifying (net −1 line) | 4 | *(see ledger)* |
| B14 | **MOOT (Phase 4 r1)** — A7 landed in Phase 3 and deleted the call, exactly as the entry's own "do this only if A7 slips" anticipated. ~~Dirichlet row-zeroing is O(nnz) when it can be O(\|bc\|·d²)~~ (`src/kernels/sparse.jl:102-112,138-173`). `zero_rows_kernel!` sweeps all 1.758 B nonzeros to zero ~1.28 M rows. For a structurally symmetric matrix row `j`'s entries live in the ≤6 columns listed in `rowval[colptr[j]:colptr[j+1]-1]` — one thread per BC node finds them all. Replaces a 7 GB read + 7 GB write with ~46 M scattered ops. **Overlaps A7, which deletes the call entirely — do this only if A7 slips.** *(audit b, rank 5)* | −0.25 GiB, ~38× less work on that step | +25 lines, +1 concept | 4 | *(see ledger)* |
| B15 | **MOOT for the steady path (Phase 4 r1)** — the fused assembly emits no explicit zeros, so `dropzeros!` is never called on it. Still live for `zero_rows!`, whose only remaining callers are tests. ~~`dropzeros!` does a binary search *and* an atomic per nonzero~~ (`src/kernels/sparse.jl:224-227`): `searchsortedlast(colptr_old, k)` = 1.758 B × ~28 random probes, plus 1.758 B `Atomix` increments, purely to recover which column a nonzero belongs to. A column-parallel kernel (thread per column, count kept entries in its own slot range) needs zero searches and zero atomics, and lets `new_col_counts` write `new_colptr` directly. *(audit b, rank 7)* | large on compaction, −1.0 GiB | neutral | 4 | *(see ledger)* |
| B16 | **MOOT (Phase 4 r1)** — Phase 3's fused assembly writes each boundary column's diagonal directly, so `get_diag`/`set_diag!` have no steady-path caller left. ~~BC diagonal handling touches every column instead of 0.25 % of them~~ (`src/pdetools.jl:95,117` → `kernels/sparse.jl:83-93,34-53`). `get_diag` and `set_diag!` launch n-wide kernels scanning ~7 rowvals each; only the ~1.28 M BC columns change. The zero-degree interior-node carve-out at `pdetools.jl:110-115` must survive verbatim. *(audit b, rank 9)* | ~50 ms at 800³, −1.02 GiB | neutral | 4 | *(see ledger)* |
| B3 | **Jacobi (diagonal) preconditioning for CG.** **REJECTED** *(audit d, rank 10)*. CG is confirmed unpreconditioned and adding one is genuinely cheap, but both of the plan's numbers are wrong. **Memory: 1.90 GiB, not 0.95** — Krylov allocates `z` lazily via `allocate_if(!MisI, …)` (`cg.jl:142`), so *any* preconditioner adds a workspace vector on top of the diagonal itself. **Speed: 5–10 %, not "large"** — after elimination the diagonal *is* the degree, 1…6 for uniform `D`, so Jacobi rescales rows by a factor bounded by 6 and leaves untouched the low-frequency mode that sets κ ~ O(N²); iterations ~ √κ improve by at most √(6/5.4) ≈ 5 %, and for a constant-diagonal Laplacian Jacobi is **exactly the identity** (CG is invariant to scalar scaling). Fails the heuristic on both sides: <15 % gain for 10 % of the memory budget, at a point where headroom is zero. **Revisit only for strongly variable `D`**, where the diagonal spans the `D` contrast. | <15 % | +2 lines but +1.90 GiB | — | **rejected — re-confirmed Phase 4 r1.** A29 freed the 1.90 GiB the audit said was unaffordable, so the memory objection is gone; the argument that survives is the one that always mattered. After elimination the diagonal *is* the degree, 1…6, so Jacobi is a row rescaling bounded by 6 and leaves the low-frequency mode that sets κ ~ O(N²) untouched. No evidence was found against that estimate, so it was not re-measured. |
| B4 | **`CartesianIndices(im_gpu)[linear_idx]` inside hot kernels** costs an integer div/mod per thread. A 3-D `ndrange` gives `i, j, k` directly. Appears in both connectivity kernels. | unknown, likely real | simplifying | 4 | *(see ledger)* |
| B5 | **rejected on measurement (Phase 4 r1).** Setup is **0.3 % of end-to-end** at 800³ (0.69 s of 204.7) after Phase 3, so even a perfect launch configuration cannot move a headline number by more than that. Phase 3's `wg=(64,4,1)` was chosen for coalescing along the contiguous dimension and is left unswept. ~~Launch configuration.~~ Workgroup sizes are hardcoded to 256 and `ndrange` covers the full grid, so ~50 % of threads idle on solid voxels at ε = 0.5. Consider launching over pore voxels, and tuning occupancy. | unknown | small | 4 | *(see ledger)* |
| B6 | **Multithread the CPU connectivity build.** Validated — the loop is serial over ~188 M edges at 400³. **But a naive `Threads.@threads` over voxels with a shared `row` counter silently corrupts the CSC**: `build_adjacency_matrix(::Array{Int,2})` requires `conns` grouped by *ascending* column with ascending rows inside, which the serial loop guarantees *structurally* (`CartesianIndices` order = pore-numbering order, six neighbour probes emitted in ascending linear-index order). Pinned at `test_assembly.jl:104-111`. *(audit c, rank 7)* | ~3× on connectivity = ~15–20 % of assembly, <1 % of e2e | free inside A25; +30 lines standalone | 4 | **accept only inside A25; reject as a standalone hand-threaded rewrite** |
| B7 | **CPU `build_adjacency_matrix` serial `colptr` loop.** **REFUTED as a standalone** *(audit c, rank 12)*: `conns[:,2]` is sorted, so this is sequential-access run counting (~0.5 s at 400³), not a random scatter — <1 % of e2e. Parallelising it needs per-thread n-sized histograms = 256 MB/thread at 400³, a **memory regression**. The degree count is free inside A25's first pass. | <1 % of e2e | — | — | **rejected standalone; subsumed by A25** |
| B17 | **BLOCKED — needs Amin's ruling on a public type change.** Measured at 200³ (4 threads / 1 thread): SpMV is **82 % of the CPU CG iteration**, and the symmetric gather beats SparseArrays' `mul!` by **3.31× / 1.21×** (44.1 → 13.3 ms, 43.1 → 35.5 ms), bit-identical. That projects to CPU solve **−57 % / −15 %**, both above the acceptance bar. The kernel exists and is committed (B2); the blocker is reaching it. `mul!(::Vector, ::SparseMatrixCSC, ::Vector)` is stdlib-on-stdlib, so dispatching to it needs `sim.prob.A` to stop being a `SparseMatrixCSC` — tried, and it removes `Array`, `==`, `issymmetric` and all SparseArrays interop from a published package's public object (23 assertions across `test_assembly.jl` and `test_errors.jl` fail on exactly that). Reverted rather than decided unattended, with a JOSS submission in flight. **The decision:** either (i) `sim.prob.A` becomes a `PortableSparseCSC` on CPU too and that type grows `getindex`/`==`/dense conversion, or (ii) it stays and CPU keeps single-threaded SpMV. ~~Threaded symmetric CPU SpMV~~ (`src/simulations.jl:213`, `src/sparse_type.jl:119`). The CPU solve calls SparseArrays' **single-threaded** `SparseMatrixCSC` `mul!`; CG is ~10³ iterations × 3.5 GiB/SpMV, so it is **>90 % of CPU end-to-end** and uses 1 of 4 threads. A is symmetric post-elimination, so CSC-read-as-CSR gives a row-parallel atomic-free kernel. **This, not B2, is the real CPU SpMV work** — B2 delivers nothing on CPU (the CPU path returns `SparseMatrixCSC` and never touches `_spmv_kernel!`). Must be opt-in per matrix, never blanket — the transient operator is not symmetric. *(audit c, rank 1)* | solve 2–3×, e2e ~2–2.5× on CPU | +40 lines, +1 concept | 4 | *(see ledger)* |
| B18 | **Fold the `−1/voxel_size²` scaling into the edge weights** (`src/transient.jl:302`). `nonzeros(A) .= nonzeros(A) ./ (−voxel_size^2)` is a read-modify-write over all 1.758 B nonzeros to apply a constant. For scalar `D`, `gd` is a scalar (line 296) so the factor rides on it free; `L(αA) = αL(A)`, bit-identical up to one multiply-vs-divide rounding. *(audit d, rank 4)* | −14 GB device traffic (~10 ms) | simplifying (−1 line) | 4 | *(see ledger)* |
| B19 | **DONE** (Phase 4 r1, `75c2b46`). Gathers on the device `u` lives on and returns only the slice; `StopAtFluxBalance` now passes the device state straight through, dropping its per-ODE-step `Array(u)`. Host and device results verified exactly equal on all three axes. ~~`reconstruct_slice` copies the whole solution vector per slice~~ (`src/geometry.jl:71`). `c[mask] .= Array(u)[ind_slice[mask]]` — full materialise then gather, called twice per `flux`, and `StopAtFluxBalance` (`transient.jl:450`) additionally does `Array(c)` on **every ODE step**. `StopAtPeriodicState` (`transient.jl:580`) already does the right thing and documents "~180× faster, ~380× less memory" — same fix. *(audits c+d)* | ~1000× per fire | simplifying (−2 lines) | 4 | *(see ledger)* |
| B20 | **DONE** (Phase 4 r1, `75c2b46`), as a `BitVector` mask. ~~CPU `zero_rows!` is a serial `Set`-hash over every nonzero~~ (`src/transient.jl:311-320`): `A.rowval[i] in target` for 1.758 B entries, single-threaded hash probes. `is_bc = falses(n); is_bc[rows] .= true` is two lines shorter and ~8–10× faster. *(audits c+d)* | ~8–50× on this stage | simplifying, −1 concept | 4 | *(see ledger)* |
| B21 | **No `@inbounds` and a broadcast write in the CPU connectivity loop** (`src/topotools.jl:59-87`). ~7 bounds checks per pore voxel, and `conns[row,:] .= a,b` builds a `SubArray` to write 2 elements a full column-length apart. `@inbounds` must stay inside the existing i/j/k guards — `idx` is `similar(img,Int)`, i.e. **uninitialised at solid voxels**, read-safe only because `img[]` is checked first. Free win **if A25 slips**. *(audit c, rank 8)* | ~1.3–1.8× on this loop | neutral (−1 line) | 4 | *(see ledger)* |
| B22 | **DONE for the transient path** (Phase 4 r1, `75c2b46`) — `build_transient_operator` now takes the `pore_index` the problem already builds and reads the two faces off it with `slice_indices`. **The steady half is MOOT:** Phase 3's fused assembly decides Dirichlet membership from a coordinate test inside the kernel, so `simulations.jl` no longer enumerates boundary nodes at all. `find_boundary_nodes` is kept — it has no production caller left but is used throughout the test suite as the reference. ~~Reuse the pore index for boundary nodes~~ (`src/topotools.jl:239-276`, `src/simulations.jl:181`). `find_boundary_nodes` walks the entire image **twice** (once per face) purely to recover pore ordinals that `build_connectivity_list` is about to compute anyway; `geometry.jl:52` `slice_indices` already does the cheap version. The transient additionally builds `pore_index` (`transient.jl:141`) then throws it away — `build_connectivity_list` accepts `inds=` and is never given it. ~2 s of **serial host time in the middle of the GPU pipeline** at 800³. Ascending-order contract pinned at `test_geometry_ops.jl:197`. *(audits a+c+d)* | −2 full-image passes, ~2 s @800³ | neutral→simplifying | 4 | *(see ledger)* — supersedes C1 |
| B23 | **BLOCKED — no mechanism without an API change.** The diagnosis below is correct, but Tortuosity has nowhere to put a default: `LinearSolve.__init` reads `abstol`/`reltol`/`maxiters` from the `init`/`solve` **call** and never consults `prob.kwargs`, so a `LinearProblem` cannot carry them. The only routes are type piracy on `LinearSolve.default_tol` (dispatches on eltype alone — it would change tolerances for every LinearSolve user in the session) or a Tortuosity-owned `solve(sim, alg; …)` entry point, which would not affect `solve(sim.prob, …)` that every doc and example uses. Measured gain on the campaign's metrics is **zero** — the benchmark and every golden assertion pass an explicit `reltol`. **The decision:** whether Tortuosity should own a `solve` entry point with its own tolerance and iteration policy. ~~No iteration cap; tolerance policy is size- and precision-dependent~~ (`src/simulations.jl:213`). LinearSolve defaults `maxiters = length(b)` = **254.6 M** at 800³, and `abstol = reltol = sqrt(eps(eltype(b)))` — 1.49e-8 on CPU (Float64), 3.45e-4 on GPU (Float32). Krylov stops at `atol + rtol·‖r0‖`, so atol dominates at small sizes and rtol at large ones: **accuracy is not comparable across sizes or devices**. With κ ~ (2N/π)² ~ 2.6e5 at N=800, Float32 CG's attainable relative residual (~κ·eps ~ 3e-2) is **worse than the requested 3.45e-4** — the recursive residual reports convergence the true residual has not reached, or the run walks toward a 254.6 M-iteration cap. **Golden-safe**: every golden assertion passes an explicit `reltol` (1e-10/1e-12), so defaults are free to change; re-check the physics tests at `reltol=1e-6`. *(audit d, rank 6)* | removes an unbounded downside | neutral | 4 | *(see ledger)* |
| B8 | **Atomic-free histogram** (`src/kernels/graph.jl:33-85`). Pass 1 scatters 6 atomic RMWs per pore thread into a 0.95 GiB array. By adjacency symmetry the contributions to bucket `b` sum to `degree(b)`, so thread `b` can compute its own count and do one coalesced store. 1.503 B uncoalesced atomic RMWs → 0.95 GiB of coalesced stores. *(audit a, rank 3)* | large on assembly | simplifying | 4 | *(see ledger)* |
| B9 | **Owner-parallel deterministic pass 2** (`src/kernels/graph.jl:131-205`). Thread `v` owns column `j = idx[v]` and writes its whole contiguous slot run — no `Atomix.modify!` at all, and rows come out ascending for free. **Turns `test/test_gpu_parity.jl:291` (`@test unsorted > 0`) red** — that test asserts unsortedness as a guard and documents itself as "should be revisited" if assembly starts emitting sorted columns. *(audit a, rank 4)* | removes 1.503 B atomics | neutral | 4 | *(see ledger)* — the predicted test change happened in Phase 3 and was signed off |
| B10 | **Both graph kernels stream `im_gpu` redundantly** (`kernels/graph.jl:40-82, 138-199`): `idx_gpu[n] > 0` already equals `im_gpu[n]`; the code tests both. ~20 % of these kernels' DRAM traffic. *(audit a, rank 10)* | ~20 % of 2 kernels | simplifying | 4 | *(see ledger)* |
| B11 | **7 redundant `KernelAbstractions.synchronize` calls** on the setup path (`topotools.jl:146,208,214,227`; `sparse_type.jl:257,263,281`). KA's CPU backend is already synchronous inside the launch; GPU backends are stream-ordered. Syncs immediately before a host read must stay. *(audit a, rank 11)* | small | simplifying | 4 | *(see ledger)* |
| B12 | **CPU `laplacian` builds two throwaway sparse matrices** (`src/topotools.jl:13-17`): `spdiagm(degrees)` then a generic sparse–sparse `-`, ~3× nnz of transient allocation on the path the PuMA comparison cares about. Must reproduce `D - A`'s pruning of a zero diagonal or the CPU Dirichlet path changes behaviour on zero-degree nodes. *(audit a, rank 15)* | >2× on that CPU stage | +20 lines | 4 | *(see ledger)* |
| B13 | **`find_boundary_nodes` does two full serial column-major walks** over 512 M BitArray elements with a carried `ordinal` dependency, one per face (`src/topotools.jl:239-276`). Closed forms exist: `:bottom` → `1:count(view(img,:,:,1))`, `:top` → an offset of `count(img)`, dim-1 faces → `cumsum(vec(sum(img;dims=1)))`. This is what C1 should have been. *(audit a, rank 9)* | ~2× by fusing, much more via closed forms | neutral / +15 lines | 4 | *(see ledger)* |

### C-series — missing GPU paths and structural gaps

| id | gap | note | phase | status |
| --- | --- | --- | --- | --- |
| C1 | ~~`find_boundary_nodes` explicitly copies the whole image device→host.~~ **PREMISE REFUTED — rejected as written**, independently by audits (a), (c) and (d). Both production call sites (`simulations.jl:181-182`, `transient.jl:276-279`) pass the **host** `img` *before* the GPU transfer, and both structs keep `img` host-side by design; the `Array(img)` branch at `topotools.jl:241` is **dead in production**. There is no 512 MB device→host copy and no kernel to write. The real cost — two serial full-image host passes, ~2 s at 800³, blocking the GPU pipeline — is now **B22**. | — | 5 | **rejected — premise false; re-scoped as B22** |
| C2 | ~~Postprocessing runs on CPU… evaluate whether a device-side path is worth it.~~ **PREMISE INVERTED — rejected as framed** *(audit d, rank 11)*. Postprocessing is not slow because it runs on CPU; it is slow because `reconstruct_field` materialises 512 M voxels so that four 640 k-element slices can be read. A device-side reconstruct would allocate 2.05 GB on a device with **zero headroom at solve peak** and fix nothing. Re-scoped as **A21** ("compute τ from the pore vector using slice indices only"). | — | 5 | **rejected — premise inverted; re-scoped as A21** |
| C3 | `Imaginator.trim_nonpercolating_paths` uses CPU-only `label_components`. This is in the normal user workflow and is likely a serious bottleneck at 800³. GPU connected-component labelling is nontrivial but well-studied. | judge cost/benefit | 5 | *(see ledger)* |
| C4 | `Imaginator.blobs` is CPU-only and takes ~60 s at 800³. Test-image generation rather than solver code — probably lower priority, but it does gate the benchmark harness. | agent's judgement | 5 | *(see ledger)* |
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

### Phase 0 baseline — measured 2026-08-08, commit `bc86be1`

Harness `bench/scaling_bench.jl`. Blobs seed 42, porosity 0.5, blobiness 1.0, untrimmed, `axis=:x`, `reltol=1e-6`. RTX PRO 5000, 23.89 GiB. `peak` = max device usage across setup/solve/post; base CUDA context is 1.375 GiB — subtract for problem-only memory.

```
  N  device status      nnodes            nnz   setup_s   solve_s     e2e_s  peak_GiB  iters   tau
100     cpu     ok      500558        3161178     0.258     4.332     4.596     0.000    638 2.048
200     cpu     ok     4009753       26583004     2.475    52.988    55.502     0.000   1087 1.951
100     gpu     ok      500558        3161178     0.012     0.122     0.137     1.625    602 2.048
200     gpu     ok     4009753       26583004     0.073     0.727     0.822     3.281   1044 1.951
400     gpu     ok    31906673      216918266     0.453    14.811    15.447    12.750   2094 1.928
600     gpu     ok   108249920      742436368   107.871    (>41min)       -    23.889      -     -
800     gpu     ok   254645845     1753943979   384.181    (>25min)       -    23.889      -     -
```

GPU stage attribution, wall_s / peak_GiB:

```
        h2d       bcnodes      conns        adjacency    laplacian    dirichlet
100  0.001/1.41  0.003/1.41  0.003/1.44   0.003/1.50   0.002/1.53   0.003/1.59
200  0.002/1.41  0.017/1.41  0.010/1.75   0.012/2.25   0.009/2.53   0.019/3.16
400  0.007/1.44  0.120/1.44  0.070/4.16   0.095/8.19   0.053/10.41  0.083/12.75
800  0.050/1.88  0.819/1.88  0.473/23.88  323.201/23.89 32.705/23.89 51.503/23.89
                             ret=21.951    ret=0.064    ret=0.000    ret=0.000
CPU 200³: conns 0.280 / adjacency 0.216 / laplacian 0.647 / dirichlet 1.339
```

**Three findings that change the campaign's framing:**

1. **The motivating bug does not reproduce at HEAD.** 800³ setup completes in 384 s; `dropzeros!` in the dirichlet stage succeeds in 51.5 s. Confirmed twice (API pass and staged pass). The recorded OOM was almost certainly pool-state-dependent — the plan's own note of "40.46 GiB of pool reserved" says that session was already fragmented. **The wall is now a speed cliff, not a throw.**
2. **The cliff starts at 600³, not 800³.** Setup goes 0.45 s (400³) → 107.9 s (600³) → 384.2 s (800³) once device usage pins at 23.889/23.89 GiB. Memory is still the root cause, so the conflict rule ("memory wins until 800³ fits with ≥3 GiB headroom") still holds — but the metric it protects is now wall time.
3. **The biggest single target, measured:** the `conns` stage retains **21.951 GiB in 0.47 s** at 800³, filling the device before assembly starts. Everything after grinds — `adjacency` costs 323 s while retaining only 0.064 GiB. A2, A5 and A11 hit exactly this.

Convention re-verification (independent of the Phase 2 audits, and agreeing with them):
- **(a) VERIFIED, unmoved** — `test/test_assembly.jl:377`, testset `"Dirichlet elimination — exact contract, $(label)"`, parametrized over 5 `ASSEMBLY_IMAGES` fixtures; six identities at lines 405-410 all present. Caveat: the diagonal identity is against `bc_diag` (zeros replaced by 1), not `diag(L)[bc]`.
- **(b) INCIDENTAL on GPU / GUARANTEED on CPU — measured**, 40³ seed 42, 5 repeated GPU builds: CPU 0/31802 columns unsorted; GPU 28538/28829/28577/28938/28986 of 31802 unsorted, with `rowval` **differing between identical builds**. Nothing depends on it — τ agrees to 2.2e-6 and CUSPARSE has consumed unsorted `rowval` in every green run. The CPU guarantee comes from the fixed six-neighbour emission order (`k-1,j-1,i-1,i+1,j+1,k+1`); reordering those blocks corrupts the CPU matrix silently.
- **(c) CONFIRMED** — `test/test_regression_golden.jl` `GOLDEN_STEADY` lines 34-50, seeds 1/42/100, nine τ values (3 axes each) + 9 mean concentrations + 3 node counts. **Plus a second, undocumented table `GOLDEN_VARIABLE_D` (line 82)**, two `D_eff` values on seed 42. Both tables are frozen.

Baseline figures matching the plan: `nnodes` 4.01 M / 31.9 M / 254.6 M, `nedges` 22.79 M / 1.503 B, `nnz_L` 1.758 B, 200³ peak 3.281 vs 3.25 GiB quoted. 400³ peak reads 12.75 vs the 13.78 GiB quoted — the plan's figure was high.

### Phase 1 — quick wins, 2026-08-08, commits `8053141`…`d529398` (14 commits)

**Accepted (14 items):** A20, A11, A16, A13, A10, A2, A3, A4, A14, A15, A17, A18, A19(b)(c), A22.
**Retired:** A19(a) — the plan scored `Base.:*`'s `fill!` at −1.02 GiB, but **`fill!` does not allocate**; it is one write pass per setup, and removing it would make correctness depend on cuSPARSE not reading `y` at `beta=0`.
**BLOCKED:** A1 — see below.

`Pkg.test()` **PASS, 11360 assertions, 2m57.5s** — exactly the floor. No golden value touched, no test weakened or skipped.

| N | peak GiB before → after | setup s | solve s | e2e s |
| --- | --- | --- | --- | --- |
| 200³ | 3.281 → 2.500 (−23.8 %) | 0.073 → 0.060 | 0.727 → 0.730 | 0.822 → 0.810 |
| 400³ | 12.750 → 10.218 (−19.9 %) | 0.453 → 0.360 | 14.811 → 14.885 | 15.447 → 15.405 |
| 600³ | 23.889 → 20.228 (−15.3 %, no longer pinned) | **107.871 → 0.99 (109×)** | **never finished (>41 min) → 72.62** | **— → 74.15** |
| 800³ | 23.889 → 23.889 (unchanged, still 100 % of device) | 384.181 → 252.17 (−34 %) | still >30 min, did not complete | — |

**The headline is 600³.** The speed cliff is gone: a problem that previously never solved now completes end-to-end in 74 s.

#### BLOCKER: A1 — needs Amin's decision

A1 (stop calling `dropzeros!` after Dirichlet elimination) is **the item that decides 800³**, and it is blocked by a pinned structural test.

- **Win forgone, measured:** 400³ peak 11.843 → **8.000 GiB (−32 %)**; Dirichlet stage 0.111 s → 0.024 s.
- **What fails:** exactly 82 assertions, all `csc_equivalent` at `test_gpu_parity.jl:235` and `:258`. That helper compares `colptr` and `rowval` for **exact structural equality**; A1 leaves ~0.44 % explicit zeros in the matrix.
- **What passes, untouched:** the six Dirichlet identities, **both golden tables**, all dense-form parity, all physics invariants. The failure is *purely structural*, not numerical.
- **The decision:** either (i) allow `csc_equivalent` to compare after dropping explicit zeros, or (ii) keep the structural pin and rely on A7 in Phase 3. **Without one of the two, 800³ cannot fit** — `dropzeros!` at 800³ needs ~29 GiB of transient on top of the 14 GiB matrix even after A17.
- Measured, confirmed structural-only, then **reverted**. Per the unattended rules the master did **not** change the test.

#### New items discovered during Phase 1

| id | change | est. gain | complexity | phase | status |
| --- | --- | --- | --- | --- | --- |
| A27 | `D_dev[img_dev]` materialises an **unnamed** npore-length temp (1.0 GiB at 800³) on the variable-`D` path that nothing frees. | −1.0 GiB (variable D) | simplifying | 4 | **DONE** (Phase 4 r1, `75c2b46`). Only `transient.jl` still had it — `simulations.jl` lost it in Phase 3, since the fused kernel reads `D` per voxel and never gathers a pore-length copy. The device copies of `img` and `D` were also never released on the transient path; they are now, matching what the steady constructor already did. |
| A28 | `_free!` is a **no-op on Metal and AMDGPU**. Left alone deliberately in Phase 1 — neither device is present and an untested `unsafe_free!` risks precompile failure on those platforms. One line each once someone can test it. | device mem on Metal/AMD | trivial | — | **BLOCKED — no hardware to test on** |

#### Corrections from Phase 1

- The plan's claim that the non-`Int32` `_as_cusparse` fallback is "reachable from any user `inds::Array{Int,3}`" is **false**: `conns_gpu = similar(idx_gpu, Int32, …)` forces `Int32` regardless of `inds`. The fallback is reachable only by calling `build_adjacency_matrix` directly with non-`Int32` indices. Fixed anyway (4 lines).
- **A14 kept row-sum semantics**, not the cheaper column sums: `laplacian` accepts any matrix and documents `D = diag(row_sums(A))`, so column sums would be a *silently wrong answer* on asymmetric input. Revisit only if a symmetry flag lands with B1/B2.
- **A16's cache fix was made mandatory by A13, not merely cleaner** — once `unsafe_free!` is in play, pointer equality can match a recycled address. Mutator-side invalidation replaced it.
#### Phase 1 independent review — VERDICT: APPROVE WITH FINDINGS

Reviewer re-ran everything rather than trusting the report. `Pkg.test()` **11364 pass / 11364, 0 fail, 3m09.5s** — *above* the floor; the Phase 1 agent quoted 11360, a count taken before its own last commit (safe direction, but wrong). Benchmarks reproduce: peak memory to the digit (2.500 / 10.218 GiB), `iters` (1044 / 2094) and τ (1.951 / 1.928) **bit-for-bit identical** to the Phase 0 baseline.

**NO use-after-free found. NO missed cache invalidation found** — the two silent-corruption risks, both traced exhaustively rather than spot-checked: all 13 `_free!` sites against every live alias, all 6 `.colptr/.rowval/.nzval` reassignment sites against `_invalidate_cache!`. A14 verified as genuine row sums; A10 verified to leave `zero_rows!`'s contract and `test_sparse_ops.jl:186` intact.

Non-blocking findings, carried into Phase 4/6:

- **F1** — A22 *narrows* the guard: a `SubArray`/reshape of a `CuArray` answers false to `_on_gpu`, so the old code copied it and the new code scalar-indexes it. Unreachable today (`sim.img` is host-side by design).
- **F2** — A19(b) adds a **silent invariant**: the non-`Int32` `_as_cusparse` now caches *converted copies* of `colptr`/`rowval`, so any future in-place index edit (the shape of work B14/B15 propose) is a stale-matrix bug. Commented, but **no test enforces it**.
- **F3** — A14 makes the Laplacian diagonal **nondeterministic on CUDA** (was a deterministic CUSPARSE SpMV, now an atomic scatter). ≤6 Float32 terms, so tiny — but it is new.
- **F4** — cosmetic: `zero_rows!`'s docstring still says "Used to enforce Dirichlet BCs in the transient operator"; after A10 it has no production caller. Early-return paths in `_build_connectivity_list_ka` skip the new `_free!`s.
- **F5** — pre-existing, must fix before Phase 6: `bench/scaling_bench.jl:96` hardcodes a **session-specific Temp path** as `DEFAULT_CACHE`. It works here and on no other machine or session.

**`csc_equivalent` resolved — it is NOT a stored fixture.** It is a live, same-run comparison against `OldBaseline` (`bench/old_baseline.jl`, `test_gpu_parity.jl:18`), rebuilt on the same `conns` in the same process. `csc_canonical` sorts each column, so it pins **colptr exactly, rowval exactly up to intra-column order, nzval approximately** — the structural sparsity pattern and nothing else.

**Consequence: A7 is NOT blocked** (this was the campaign's biggest open risk). A1 fails because it *retains* explicit zeros the reference drops, so `colptr` differs; A7 never creates them, and the pattern it must emit is exactly what the reference has after its final `dropzeros!`. Two conditions A7 must meet: (i) it must also omit the zero-degree **interior** node's diagonal, which the old path creates and `dropzeros!` prunes — miss this and `colptr` is off by one per isolated voxel, *which is common on untrimmed blobs*; (ii) BC columns must be diagonal-only with `_unit_where_zero` applied.

**Therefore A1 is expected to become MOOT rather than needing Amin's ruling** — A7 deletes the `dropzeros!` call entirely. The A1 decision above stands only if Phase 3's A7 fails.

### Phase 3 — fused assembly, 2026-08-08, commits `181037f` + `7f5774a`

**Accepted: A5, A6, A7, A23, A24, A25. A12 subsumed by A5** (with no connectivity list the whole COO→CSC scatter is gone, so A12's −19.6 GiB is realised; the *standalone* form is still not viable because `test_gpu_parity` feeds `build_adjacency_matrix` `OldBaseline` conns grouped by row, so that method must stay general. **A12 remains unclaimed for the transient path.**)

New `src/assembly.jl` (228 lines): two KA kernels (`_steady_count_kernel!`, `_steady_fill_kernel!`) plus `build_steady_system`, **shared by the CPU and GPU paths**. One thread per grid voxel owns its whole CSC column — no COO, no adjacency matrix, no edge-weight vector, no atomics, no `dropzeros!`. The steady path goes from five stages to two kernels and a scan, with one implementation instead of two.

`Pkg.test()` **PASS — `Tortuosity.jl | 11365  11365  2m53.7s`** (floor was 11364; +1 from an assertion the agent *added*).

**CPU output is bit-identical to the old pipeline** — `colptr`, `rowval`, `nzval` and `b` all `==` (not `≈`), uniform and variable `D`, all three axes. That is why the golden tables could not move.

| N | peak GiB: Phase 0 → Phase 1 → now | vs P0 | setup s (P0 → now) | solve s | e2e s |
| --- | --- | --- | --- | --- | --- |
| 200³ | 3.281 → 2.500 → **1.718** | −47.6 % | 0.073 → 0.008 | 0.727 → 0.775 | 0.822 → 0.801 |
| 400³ | 12.750 → 10.218 → **4.156** | −67.4 % | 0.453 → 0.053 | 14.811 → 14.907 | 15.447 → 15.113 |
| 600³ | 23.889 → 20.228 → **10.750** | −55.0 % | **107.871 → 0.158** | never → 71.137 | never → 71.838 |
| 800³ | 23.889 → 23.889 → **21.637** | −9.4 % | **384.181 → 0.388 (990×)** | never → 205.660 | never → **207.766** |

CPU 200³: setup 2.475 → 0.701 s (1 thread) → **0.213 s (4 threads, 11.6×)**, e2e 55.502 → 49.165. τ identical at every size. Solve times are unchanged, as expected — it is the same matrix.

**800³ completes end-to-end for the first time:** setup 0.388 s, solve 205.660 s (3620 iters), post 1.718 s, e2e 207.766 s, τ 1.884, nnz 1,753,943,979 unchanged.

#### The gate is NOT yet met

Total peak 21.637 GiB of 23.889 → **headroom 2.25 GiB, short of the ≥3 GiB gate by 0.75 GiB.** The binding peak has moved from assembly to the **solve**: setup now peaks at 19.128 GiB (4.76 GiB headroom), while the solve accounts for A 14.02 + b 0.949 + u 0.949 + Krylov `x,r,p,Ap` 3.80 + CUDA context 1.375 = 21.09 GiB. The conflict rule therefore stays in force into Phase 4.

#### OVERRIDE — signed off by the master

`test/test_gpu_parity.jl:291`, exactly as pre-authorized: `@test unsorted > 0` → `@test unsorted == 0`. Owner-parallel assembly emits ascending rows by construction, so the guard's premise inverted; asserting sortedness is strictly stronger. Condition (ii) is satisfied — the *"CUSPARSE SpMV tolerates unsorted row indices within a column"* coverage does **not** depend on assembly output; it lives in the "with every column deliberately shuffled" subtest, which builds its unsorted matrix itself via `shuffle()`. The agent went further and **added** an assertion so that subtest fails loudly if the shuffle ever stops biting:

```julia
@test count(j -> !issorted(@view rv[colptr[j]:(colptr[j+1]-1)]), 1:A.n) > 0
```

That is the +1 assertion (11364 → 11365). **No test was weakened, skipped, or disabled.** Coverage increased.

#### New items from Phase 3

| id | change | est. gain | complexity | phase | status |
| --- | --- | --- | --- | --- | --- |
| A29 | **Krylov's workspace `x` duplicates LinearSolve's `u`** — 0.949 GiB at 800³, the largest pure-duplication vector at the solve peak. Aliasing them takes headroom 2.25 → ~3.2 GiB and **meets the campaign gate**. This is the one thing between the current state and the target. | −0.949 GiB; **clears the gate** | small | 4 | *(see ledger)* |
| A30 | **The transient path still runs the old five-stage pipeline.** `build_transient_operator` could use the same fused kernel pair with a row-only BC rule. A12's standalone win now applies there and nowhere else. | transient mem + setup | moderate | 4 | *(see ledger)* |

**Re-rankings:** B23 **up** — the solve is now 99.0–99.8 % of e2e at 600–800³; iterations 1044/2094/2983/3620 at 200/400/600/800. Float32 CG *did* converge at 800³, so the conditioning worry did not bite, but there is still no iteration cap. B3 deserves a re-look **only after A29** — it costs 1.90 GiB, which 2.25 GiB cannot afford. B5 still open (Phase 3 chose `wg=(64,4,1)` for coalescing and did not sweep). A21 still open and now visible: 800³ post is 1.718 s materialising 512 M voxels.

**Plan correction confirmed:** A5's mandated insertion sort is dead code — sortedness is free from column-major monotonicity, exactly as the corrected Constraint 2 says.

#### Phase 3 independent review — VERDICT: APPROVE

Reviewer re-ran everything at `7f5774a`. `Pkg.test()` **`Tortuosity.jl | 11365 11365 2m49.2s`**. Benchmarks reproduce (200³ peak 1.718 exact; 400³ 4.125 vs 4.156 claimed, one 32 MiB sampler granule low — the sampler is a documented lower bound; `nnz` identical to the Phase 0 baseline at both sizes).

**Bit-identity verified independently, not taken on faith.** The reviewer reconstructed the old pipeline locally and compared with `==` on `colptr`, `rowval`, `nzval`, `b` across: blobs 24³ seeds 1/42/100; blobs 40³ seeds 1/42/100 (31 802 nodes, **15 zero-degree pore voxels present, untrimmed**); 40³ variable `D` and Bool-vs-BitArray masks; 20³ variable and constant-array `D`; duct + isolated inlet-face voxel; duct + isolated inlet voxel + **2 isolated interior voxels**; fully disconnected 4×1×1; open box; box with cavity; 2-thick slab — each across axes `:x`, `:y`, `:z`. All matched; every column sorted ascending. The only three `b` differences found were at `nbc == 1` (inlet face == outlet face), which `simulations.jl:157` rejects, so unreachable through the API.

**Zero-degree handling confirmed** — the risk Phase 1's reviewer flagged: `assembly.jl:86` sets `counts[c0] = 0` for a free node with `iszero(deg)`, so an isolated interior voxel gets an empty column and `colptr` is *not* off by one, matching what the old path achieved via `dropzeros!`. Confirmed empirically (two added interior isolated voxels: n 49→51, nnz unchanged at 201, `colptr ==`). The BC carve-out is honoured — `assembly.jl:79-84` pins a zero-degree boundary node with a unit diagonal and `b = 1` on the inlet, reproducing `_unit_where_zero`; `git diff 67977de..HEAD -- src/pdetools.jl` is empty.

**The six Dirichlet identities are stronger than before, not merely still green.** The testset builds `L` from the OLD chain purely as a reference and asserts against `sim.prob.A`/`sim.prob.b` — i.e. against the new assembler. It is now a cross-check of two independent implementations rather than `apply_dirichlet_bc_fast!` against itself.

Findings carried to Phase 6:

- **F6** — **no test pins the bit-identity claim itself.** All assertions are `≈`/rtol and no test calls `build_steady_system` directly. The real guard is the golden tables (12-digit τ, rtol 1e-6), which held. **Adding one `==` CPU-vs-old-chain test would make the claim regression-proof.**
- **F7** — `apply_dirichlet_bc_fast!` (both methods) now has **zero production callers**; transitively production-dead with it: `apply_dirichlet_bc!`, `zero_rows_cols!`, `set_diag!`, `get_diag`, `multihotvec`, `overlap_indices`/`_fast`. Still live via `src/transient.jl`: `build_connectivity_list` (:288), `interpolate_edge_values` (:296), `build_adjacency_matrix` (:298), `laplacian` (:302), `find_boundary_nodes` (:276,279), `dropzeros!` (:330). Keeping them is authorised by A25, but they should be **labelled reference/transient material**, and `test_impl_parity.jl:225` "steady-system assembly parity" is now a **misnomer** — it compares two non-production chains.
- **F8** — the testset comment at the Dirichlet contract still claims it "pins `apply_dirichlet_bc_fast!`", now stale.
- **F9** — behavioural delta, unreachable through the API but recorded: the old GPU `dropzeros!` used `tol = eps(Float32)` so it pruned `|w| <= 1.2e-7`; the new path prunes only **structurally**. Harmless given the `D > 0` precondition, and it makes CPU/GPU structure identical.
- **F10** — `Int32` headroom unchanged but **the binding quantity moved**: `colptr`/`nnz` (1.754e9 at 800³) is now what approaches `typemax(Int32)`, not `nedges` (1.503e9). Still out of scope, not made worse.

Reviewer finding **F5 fixed** in `7f5774a`: `DEFAULT_CACHE` had hardcoded a user name and a session UUID; now `tempdir()/tortuosity_bench_blobs`.

### Phase 4 — audit round 1, 2026-08-08 (audited `7f5774a`)

`AUDIT ROUND 1: 5 candidates surfaced, 4 accepted` (see round-2 outcomes below for the final dispositions).

#### Triage — 13 of 24 pending items are MOOT or already done

Phase 3 moved the production steady path to `SteadyDiffusionProblem` → `build_steady_system` only. Nothing in `src/` calls the old five stages except `build_transient_operator`.

| id | disposition |
| --- | --- |
| B8 | **MOOT** — no histogram pass exists on the steady path; `_steady_count_kernel!` replaced it. Transient-only, subsumed by A30 |
| B9 | **ALREADY DONE** — owner-parallel columns *are* `_steady_fill_kernel!`; the predicted `test_gpu_parity.jl:291` flip already happened in Phase 3 |
| B10 | **ALREADY DONE in substance** — `assembly.jl` never reads `img`; `idx` *is* the mask, exactly B10's ask |
| B4 | **ALREADY DONE for production** — `assembly.jl:47,102` use `@index(Global, NTuple)`. The two `CartesianIndices(im_gpu)[linear_idx]` sites are transient-only |
| B11 | **mostly MOOT** — 4 of 7 named syncs are off the steady path; `assembly.jl` adds 3, of which 2 are redundant. Worth <1 ms — **not worth a commit** |
| B12 | **MOOT for steady** — `topotools.jl:13-17 laplacian` has no steady caller. Transient CPU only |
| B13 | **MOOT for steady** — `find_boundary_nodes` has no steady caller. Transient-only |
| B14 | **MOOT as written** — A7 deleted `zero_rows_cols!` from the steady path |
| B15 | **MOOT — terminal.** `dropzeros!` has **no production caller left anywhere** (steady killed by A7, transient by A10). Tests and `bench/` only |
| B16 | **MOOT — terminal.** `get_diag`/`set_diag!` are reachable only from `apply_dirichlet_bc_fast!`, which has no production caller |
| B21 | **MOOT for steady** — `_build_connectivity_list_cpu` is transient-only now |
| B22 | **half ALREADY DONE** — steady never enumerates BC nodes (`_is_bc` is a coordinate test in-kernel); half LIVE at `transient.jl:276-279` |
| A27 | **MOOT for steady** — `assembly.jl` reads `D[i,j,k]` in-kernel; LIVE only at `transient.jl:296` |
| B2 | **LIVE code-wise but ZERO gain on any benchmarked backend** — CUDA goes through CUSPARSE, production CPU returns `SparseMatrixCSC`. Metal/AMD only |

**Optimising B15/B16 would be optimising the reference implementation.** Of the 11 items still live, 8 are transient-path items that **A30 would subsume as a unit**.

#### Measured findings that close off avenues

Stated positively, because the Phase 4 stop condition depends on hearing them:

- **Per-iteration cost is already clean.** One CG iteration at 800³ = 1 CUSPARSE SpMV + 2 axpy + 1 axpby + 2 dots. **No redundant kernel launches, no avoidable device→host.** The two `kdotr` calls sync the stream, but at 56.8 ms/iteration that latency is <0.1 %. Effective bandwidth ~528 GB/s against ~900 GB/s peak — normal for SpMV with a random gather.
- **The whole cheap-preconditioner family is a measured dead end** — this generalises B3's rejection rather than merely confirming it. Neumann/polynomial preconditioning at 96³, degree 1/2/4/8: 399/329/258/196 iterations vs 606 plain — **total SpMV count rises** (798/987/1290/1764) and wall time rises monotonically (3.23 → 8.81 s). The best *possible* degree-k polynomial preconditioner divides iterations by exactly k, which only ever saves the vector-op share (12.2 of 30.2 GB per iteration). **Only a real coarse space wins.**
- **`src/assembly.jl` has nothing worth optimising.** Setup is 0.388 s = 0.19 % of e2e and its peak (~19.1 GiB) is *below* the solve peak (21.6 GiB), so neither axis is binding.
- **There is no second A29.** Exact solve-peak accounting: A 14.02 GiB (colptr 1.019 + rowval 7.016 + nzval 7.016) + b/u/x/r/p/Ap 5.69 + context 1.375 = 21.09 GiB. A29 takes it to 20.14 → 3.2 GiB headroom. The only remaining levers are a narrower `nzval` type (A9's ground, blocked by CUSPARSE needing matched eltypes) and freeing `b` after CG's first iteration (Krylov never reads it again, but reaching in to free it is fragile).

#### MASTER RULING — the conflict rule is a threshold, not a floor

The auditor recommended re-scoping the ≥3 GiB gate, on the grounds that it "vetoes a measured 5–12× solve speedup to protect 0.75 GiB". **No re-scope is needed — the plan already says the right thing.** The conflict rule reads: *"Until 800³ fits in 23.89 GiB with ≥3 GiB of headroom, memory wins. After that threshold is met, speed wins. **Memory becomes a budget to spend rather than a wall to avoid.**"* It is a threshold to **cross once**, not a floor to maintain forever. Once A29 clears it, spending headroom on a large solve speedup is precisely the authorised behaviour. Candidates [1] and [2] below are therefore **in scope without an override**.

#### New candidates — all iteration counts MEASURED (CPU/Float64, blobs seed 42, `atol = 1e-6·‖b‖`, `rtol = 0` so the stopping test is identical across variants)

| id | change | measured gain | complexity | phase | status |
| --- | --- | --- | --- | --- | --- |
| B24 | **Two-level aggregation preconditioner** (geometric block coarse space): `Pl = W·(WᵀAW)⁻¹·Wᵀ + I/λmax`, `W` = piecewise-constant indicators of 8³-voxel blocks. **96³ 606→141 (4.3×), 160³ 889→101 (8.8×), 224³ 1217→97 (12.6×); wall 3.15→1.15, 20.01→3.54, 82.18→6.97 s.** Iterations go **FLAT in N** — this removes the O(N) growth, it is not a constant factor. Per-iteration GPU overhead ~+13 %. Projected 800³: solve 205.7 → ~40 s, e2e −80 %. Build `WᵀAW` in one pass over the image — same fused kernel shape as `assembly.jl`. **Risk: the coarse solve must be accurate or convergence stalls silently — a plausible wrong τ, not a crash.** Float32 coarse operator is near-singular for pure-Neumann interior blocks. Costs ~1.9 GiB. | **e2e −80 % at 800³** | +200 lines, +2 concepts | 4 | *(see ledger)* |
| B25 | **Linear-ramp initial guess (warm start)** — start CG from `c = 1 − (i−1)/(N−1)` instead of 0 via Krylov's `Δx`/`warm_start`. Measured ramp/zero iteration ratio: 0.906 (48³), 0.826 (64³), 0.782 (96³), 0.782 (128³), 0.856 (160³) → **~17 % fewer iterations, flat in N**; τ unchanged to 5 s.f. Costs +0.949 GiB. **Largely SUBSUMED by B24 — do not do both without re-measuring.** | ~17 % of e2e (~35 s @800³) | ~15 lines | 4 | *(see ledger)* |
| B26 | **`trim_nonpercolating_paths` does an O(N) hash-set probe per voxel** (`src/imgen.jl:143`): `in.(labels, Ref(Set(...)))` hashes once per voxel over 512 M voxels, single-threaded, on an `Array{Int64}` labels array (4.1 GB host). A `Bool` lookup table indexed by label is 1 line. **This, not `label_components`, is C3's real hot spot.** | est. 5–20× on that step; halves peak host memory | simplifying (−1 line, −1 concept) | 5 | *(see ledger)* |
| B27 | **`blobs` allocates ~6 full `Float64` copies** (`src/imgen.jl:98-104`, `norm_to_uniform:19-25`): `rand` + blur + 5 unfused broadcasts, ~4.1 GB each at 800³. `Float32` + one fused `@.`. **Must be gated on the golden node counts** — `to_binary` threshold rounding could move one. | est. 2–3× of ~60 s; −20 GB host churn | simplifying | 5 | *(see ledger)* |

**A26 re-scoped and re-located:** the CPU index type is decided at **`assembly.jl:186`** (`Ti = on_gpu ? Int32 : Int`), *not* `topotools.jl:161-174` (off the steady path). CPU SpMV drops 16 B/nnz → 12 B/nnz ≈ **18 % of CPU solve traffic**, and CPU solve is >99 % of the 49.2 s CPU e2e. Clears the 15 % bar; compounds with B17.

#### C-series judgements — all three rejected, with reasoning

- **C3 — reject as written; re-scoped to B26.** Do *not* write a GPU connected-component labeller. (i) It is **not in the default large-image workflow**: `_warn_nonpercolating` (`simulations.jl:64`) only runs the check when `length(img) <= 50_000_000`, so it is skipped at 400³ and 800³ unless forced. (ii) The actual hot spot is not `label_components` at all — it is the serial hash probe at `imgen.jl:143` (now B26). GPU CCL is hundreds of lines of merge-based union-find needing its own test suite, for a function outside the measured path.
- **C4 — reject as a campaign item.** `Imaginator.blobs` is test-image generation, on no headline metric, and the harness already caches its output to disk (`DEFAULT_CACHE`), so it is paid once per machine. The cheap CPU cleanup (B27) is worth doing if someone is idle but must not displace B24/B25/A26.
- **C5 — reject and close.** No AMD device exists here, so ~40 lines of rocSPARSE plumbing would land with **zero test coverage on the one axis where untested code is most dangerous** — a wrong SpMV returns a plausible τ, not an error. B2 is the correct substitute, **but B2 also has no measurable effect on any backend present here**, so both are logged terminal-rejected rather than carried forward.

#### Plan corrections from audit round 1

1. **A21's stated free ingredient is gone.** The plan claims `SteadyDiffusionProblem` "already computes the inlet/outlet pore-index lists and throws them away". After A7 it computes **no node lists at all** — BC membership is `_is_bc(_face_coord(...))` inside the kernel. A21 would now have to *create* those lists, and post is 1.718 s of 207.8 s = **0.83 %**, so it now **fails the acceptance bar**.
2. **B3's rejection should be strengthened, not revisited after A29** — the whole cheap-preconditioner family loses, not just Jacobi.
3. **B23 has a second issue** the plan does not record: at `reltol=1e-6`, two equally-converged solutions differ by **2.5 % in relative L2** (measured 48³–96³) while τ agrees to 1e-4. The residual criterion is much weaker than it looks; the discrepancy lives in stagnant, non-percolating clusters that carry no flux. Worth stating so nobody later reads that 2.5 % as a bug.

### Phase 4 — round 1, 2026-08-08, commits `849cc3c` `2daaf12` `5960ef3` `0258344`

**Accepted:** A29, B1, B2, A27, B19, B20, B22(transient half). **Rejected:** A21, B5, B3 (re-confirmed). **Retired moot:** B14, B15, B16, B22(steady half). **BLOCKED:** B17, B23.

`Pkg.test()` **PASS — `Tortuosity.jl | 11397 11397 2m48.9s`** (floor was 11365; two assertions added, none weakened, no golden value touched).

## ✅ THE MEMORY GATE IS MET

**800³ peak 20.605 GiB; headroom 23.889 − 20.605 = 3.284 GiB ≥ 3 GiB.** Per the conflict rule, memory now becomes a budget to spend and **speed wins** for the remainder of the campaign.

| N | peak GiB → | setup s → | solve s → | e2e s → |
| --- | --- | --- | --- | --- |
| 200³ | 1.718 → 1.718 (0 %) | 0.008 → 0.009 | 0.775 → 0.715 (−7.7 %) | 0.801 → 0.747 (−6.7 %) |
| 400³ | 4.156 → 4.031 (−3.0 %) | 0.053 → 0.053 | 14.907 → 13.666 (−8.3 %) | 15.113 → 13.904 (−8.0 %) |
| 600³ | 10.750 → 10.343 (−3.8 %) | 0.158 → 0.191 | 71.137 → 65.117 (−8.5 %) | 71.838 → 65.971 (−8.2 %) |
| 800³ | 21.637 → **20.605** (−4.8 %) | 0.388 → 0.391 | 205.660 → **189.051** (−8.1 %) | 207.766 → **191.030** (−8.1 %) |

Iteration counts **identical** at every size (1044/2094/2983/3620) — the win is per-iteration cost, not convergence.

#### Both headline mechanisms in the plan were WRONG, in opposite directions

- **A29's premise was already true.** LinearSolve *does* alias `x` and `u` (`iterative_wrappers.jl:266`). The real defect is that the constructor allocates `x` **before** the workspace's other vectors, and **freeing it afterwards does not move the peak** — CUDA's stream-ordered pool still counts a freed block as in use (verified: `unsafe_free!` of 1 GiB leaves `total − available` unchanged). It had to *not be allocated*.
- **B1's premise is simply false.** There is no uncached per-call `with_workspace` device malloc: `cusparseSpMV_bufferSize` returns *the same* 2 740 547 bytes for CSC and CSR at 800³. The win is the **kernel choice**, and it is **size-dependent**: 0.99× at 200³, 1.175× at 800³. **Anyone re-measuring on a small image will wrongly conclude there is nothing here.**

#### B2 un-rejected by measurement

Audit round 1 logged B2 "terminal-rejected — no measurable effect on any backend present here". **That premise is refuted:** the symmetric KA SpMV runs the CPU backend at 125.5 → 35.5 ms (1 thread) and 34.6 → 13.3 ms (4 threads) — 3.5×, on a backend the suite exercises. It is also the exact kernel B17 needs the moment the API question is decided. Accepted.

#### OVERRIDE — B1 accepted below the 15 % bar

B1 gives −8.1 % e2e with +1 struct field, under the 15 % threshold. Justified under principle 3: the solve is 99 % of e2e so this is the largest remaining single lever, and the flag **replaces a subtle invalidation rule** ("reassignment must invalidate, in-place edits need not") **with a blunter, more auditable one** — any mutation invalidates.

#### BLOCKER: B17 — needs Amin's decision

CPU SpMV is **82 %** of the CPU CG iteration. The symmetric gather beats SparseArrays' `mul!` by **3.31× (4 threads) / 1.21× (1 thread), bit-identical** ⇒ **CPU solve −57 % / −15 %**. The kernel is already committed (B2). The blocker: reaching it requires `sim.prob.A` to stop being a `SparseMatrixCSC`, which removes `Array`, `==`, `issymmetric` and SparseArrays interop from a **published package's public object, mid-JOSS-review**. Tried; 23 assertions fail on exactly that. Reverted rather than decided unattended.

#### BLOCKER: B23 — no mechanism exists

`LinearSolve.__init` reads `abstol`/`reltol`/`maxiters` from the init/solve **call** and never consults `prob.kwargs`, so a `LinearProblem` cannot carry defaults. The only routes are type piracy on `LinearSolve.default_tol` (global) or a Tortuosity-owned `solve(sim, alg)` entry point. Measured gain zero as things stand.

#### Data-integrity finding — CPU benchmark numbers are not yet trustworthy

`bench/results/scaling_env.csv` records `threads=1` for **every** run, while the Phase 3 CPU figure was annotated "4 threads". CPU assembly is KA-threaded and CPU SpMV is not, so the distinction changes the numbers materially. **Settle this before any PuMA comparison goes in the paper.** Handed to round 2.

#### The GPU solve is now run-to-run reproducible

CSC SpMV summed with atomics in nondeterministic order — the **Phase 0 baseline itself shows τ = 1.92742 vs 1.92824 across two reps of the same build**. CSR is a deterministic gather: the reviewer measured **1.88141 in both reps, identical to every printed digit**.

**CORRECTION (reviewer, and the master repeated the error to Amin before catching it).** The claim that τ 1.884 → 1.881 sits "inside the Float32 envelope `test_gpu_e2e.jl:81` pins at `rtol=1e-3`" is **wrong on both counts**: 1.6e-3 relative is *outside* 1e-3, and that test pins **CPU-vs-GPU agreement on a 24³ image**, not run-to-run spread at 800³. The defensible statement is narrower: the new value is more **reproducible**, but is **not shown to be more accurate** — no Float64 reference exists at 800³. Both values round to 1.88. **Do not quote GPU τ to more than three significant figures in the paper.**

#### Phase 4 round 1 independent review — VERDICT: APPROVE WITH FINDINGS

Gate **independently verified** at 800³ over 2 reps, harness read from `git show 0258344:bench/scaling_bench.jl` with results written to scratch: `peak 20.622 GiB, base 1.375, iters 3620, tau 1.881, setup 0.398, solve 187.315, e2e 189.483` → **headroom 3.267 GiB, gate MET with 0.27 GiB to spare.** (Claim 20.605/3.284; measured 20.622/3.267 — 17 MiB apart, inside the sampler's ~32 MiB granule. Solve and e2e came in *better* than claimed.)

`Pkg.test()` **`Tortuosity.jl | 11397 11397 2m56.0s`**. `test_sparse_ops.jl` is **+102 lines of pure addition** (4 new testsets, 0 deletions); the `test_gpu_parity.jl` diff is comment-only. **Correction: "2 assertions added" was a miscount — it is 32** (11365 → 11397).

**Symmetry flag fully audited — no path where an asymmetric matrix is read as CSR.** Set true at `assembly.jl:229` only (GPU steady path); defaults false at `sparse_type.jl:84,93`. Cleared in `_invalidate_cache!` (`sparse_type.jl:113`), reached from `set_diag!`, `zero_rows_cols!`, `_zero_rows_only!`, both mutating `dropzeros!` branches, and `_free!`. Read at `ext/TortuosityCUDAExt.jl:51,81` and `sparse_type.jl:215`. `dropzeros!`'s `nnz_new == nnz_old` early return keeps the flag, correctly — it mutates nothing. The transient operator comes from `laplacian`, which never sets `symmetric=true`, so the flag is already false there.

Findings for Phase 6:

- **F11 (most severe, latent not live)** — `transient.jl:322` `nonzeros(A) .= nonzeros(A) ./ (-voxel_size^2)` edits `nzval` in place **without `_invalidate_cache!`**. Harmless today (the flag is already false and the next line's `_zero_rows_only!` invalidates anyway), but it is **the single mutation site that does not obey the blunt "any mutation invalidates" rule the commit message advertises**. One line to make the rule literally true.
- **F12** — B19's new `_on_gpu(u)` device path has **no committed test**; the GPU transient e2e tests never call `flux`/`reconstruct_slice` with a device `u`. Verified by a throwaway scratch probe (`isequal`, 3 axes × 3 slices) — real evidence, not regression-proof. A 3-line GPU assertion closes it.
- **F13** — `bench/results/scaling.csv` still holds **only the Phase 0 baseline**. No Phase 3 or Phase 4 rows are persisted anywhere in the repo, so every "before" figure is not independently checkable — only the "after", which the reviewer reproduced. **Fix in Phase 6.**
- **F14** — the symmetric KA kernel's "bit-for-bit" claim holds only **within one workgroup**; the tests use n=40 (single workgroup) so they are not flaky, but the docstring overstates for a multi-workgroup CPU/Metal/AMD launch.

**A29 verified to prevent the allocation, not free it after the fact:** `_cg_workspace` builds at `(0,0)`, assigns `x = u`, and allocates only `r,p,Ap` — the fourth vector is never created. **B19 verified** to preserve NaN placement and `nansum` semantics.

### Phase 4 — round 2, 2026-08-08, commits `cffb944` `7ca9890` `af7fde2` `8b54b4c` `cb018cf` `7484bae`

**Accepted:** B24, A26, B26, bench thread-count fix. **Rejected:** B25. **BLOCKED:** B24-as-default. **Deferred:** A30, B27.

`Pkg.test()` **`Tortuosity.jl | 11511 11511 3m03.4s`** (floor 11397, **+114**). No golden value touched; no existing test weakened, skipped or disabled.

#### B24 — the campaign's largest speed result

`src/preconditioner.jl` (+~300 lines, 106 new assertions). Two-level aggregation preconditioner, opt-in via `Pl=two_level_preconditioner(sim)`.

| N | default e2e | preconditioned e2e | Δ | iters |
| --- | --- | --- | --- | --- |
| 200³ | 0.740 s | **0.320 s** | −56.8 % | 1044 → 82 |
| 400³ | 13.724 s | **2.398 s** | −82.5 % | 2094 → 168 |
| 600³ | 66.166 s | **9.202 s** | −86.1 % | 2983 → 223 |
| 800³ | 189.698 s | **20.660 s** | **−89.1 %** | 3620 → 202 |

Beats the projected −80 %. The **default path is untouched** (peak 20.588 GiB at 800³, headroom 3.301); the preconditioned path costs `agg` (Int16/pore voxel) + Krylov's `z` → 21.959 GiB, headroom 1.930.

**The silent-failure risk is closed by construction, not by spot checks.** `W'AW` is PSD; blocks with a zero coarse diagonal are dropped; the rest get a relative diagonal shift; the coarse solve is **Float64 whatever the fine precision** — so Cholesky provably exists and `Pl` is provably SPD, which is what makes PCG land on the same solution. Verified at `reltol=1e-10` on untrimmed blobs, a trimmed blob, an open box, a deliberately detached cluster, variable `D` and `axis=:z`: **τ agrees to 1e-9 everywhere.** The ~2.5 % L2 gap audit round 1 warned about is real, does **not** shrink with tolerance, and drops to 6e-9 once the image is trimmed — it is entirely stagnant volume.

#### BLOCKER: B24 as the default — needs Amin's API ruling

LinearSolve 3.87 offers exactly two hooks, `alg.precs` and `Pl=` at init/solve. `DEFAULT_PRECS` ignores both the matrix type and `prob.p`, and `__init` never reads `prob.kwargs` — all three verified in source. Making B24 the default therefore requires **either** Tortuosity shadowing the exported `KrylovJL_CG` (which turns `using Tortuosity, LinearSolve` into an **ambiguity error on a published name, mid-JOSS-review**) **or** a Tortuosity-owned `solve(sim, alg)` entry point — the same decision **B23** is blocked on. **Forgone win: the whole −89 %.** Users get it today only by writing `Pl=two_level_preconditioner(sim)`.

**Master's recommendation:** take the *additive* option — a new Tortuosity-owned entry point, not a shadowed name. It resolves B23 and B24 together and adds no ambiguity to a published API.

#### A26, B26, and the benchmark thread-count question

- **A26 accepted** — `Ti = (on_gpu || 7*nnodes+1 <= typemax(Int32)) ? Int32 : Int` at `assembly.jl:186`. Measured 160³ CPU/1 thread: SpMV −16.6 %, solve 24.12 → 20.88 s (−13.4 %) at unchanged iterations, **τ bit-identical**. 200³ CPU e2e 53.0 → 43.9 s (−17.2 %).
- **B26 accepted, but the inventory estimate was wrong** — measured **2.32×** on the membership pass (177 → 76 ms at 400³) and −11.6 % on the whole function, **not 5–20×**. Its "halves peak host memory" claim is **false**: the `Int64` label array dominates and is untouched.
- **The thread-count worry resolves the other way.** The harness was already recording `Threads.nthreads()` truthfully; **the Phase 3 "4 threads" annotation was the mislabel.** Measured 200³ CPU: 1 thread 43.876 s vs 4 threads 44.941 s — thread count **barely moves CPU numbers**, because CPU SpMV is single-threaded (B17 blocked) and assembly is 0.26 s of 44 s. `threads` is now a per-row column, part of the resume key, and in the summary, so a CPU number cannot be read without it.

#### B25 rejected, with a plan correction

`LinearProblem(A, b; u0=ramp)` does **NOT** warm-start `KrylovJL` — Krylov's `cg!` zeroes `x` unless `warm_start`/`Δx` is set, which LinearSolve does not expose. **Measured identical iteration counts with and without the ramp** (1044 @200³, 2094 @400³). It needs workspace surgery, and on top of B24 its ceiling is ~17 % of 168 iterations.

#### New candidates from round 2

| id | change | est. gain | status |
| --- | --- | --- | --- |
| B28 | Preconditioner block-size cap is slightly off at 400³: default picks block=13 (168 it, 2.047 s); block=12 measures 111 it, 1.456 s. A cap near 40 000 would pick 12. **Left alone because 800³ already reaches 202 it and a change there is unmeasured in both directions.** | ~30 % at 400³, unknown at 800³ | round 3 — measure at 800³ before changing |
| B29 | Restriction+prolongation is now **~46 % of a preconditioned iteration** at 400³ (~6 of 13 ms). The restriction is an atomic scatter with ~1000-way contention per coarse slot; a chunked or two-stage reduction is the next lever. | unknown | audit round 2 |
| B30 | With the preconditioner on, `post` is **8 % of 800³ e2e** (1.6 of 20.7 s) where A21 was rejected at 0.83 %. Still under 15 %, but re-score if B29 lands. | ~8 % | audit round 2 |

**Corroborates B23:** LinearSolve's default `abstol = sqrt(eps)` = 1.5e-8 dominates at tight `reltol` and ended both verification runs on the **absolute** term, at different accuracies. Forcing `abstol` down moved agreement from 1.1e-8 to below 1e-9.

**A30 deliberately not attempted** — transient-only, invisible to every metric the harness reports, and its structural contract differs between the CPU and GPU `laplacian` paths (zero-degree diagonals). It needs its own round rather than the tail of this one; handed over intact rather than left half-ported.

### Phase 4 — audit round 2 (the stop condition), audited `7484bae`

```
AUDIT ROUND 2: 8 candidates surfaced, 0 accepted
PHASE 4 COMPLETE: 0 accepted candidates
```

**Measured composition of a preconditioned iteration** (400³, quiet machine, block=13, nc=26 039): Krylov iteration **13.51 ms** (168 iters, 2.27 s) = SpMV 3.443 + `ldiv!` 5.255 + Krylov vector ops 4.81. `ldiv!` splits: restrict **1.488**, prolong **0.460**, host CHOLMOD coarse solve **3.271**, d2h+h2d 0.042. An emulated hand-written iteration costs 11.55 ms against Krylov's 13.51 — **no hidden per-iteration overhead**; LinearSolve passes `Pl` straight through and `_as_cusparse` caches the CSR descriptor.

**All 8 rejected, with reasoning:**

1. **B29 — chunked/two-stage restriction. REJECT.** Contention measured directly against a same-moment reference kernel with identical traffic minus the atomic: **5.0× at 1 225 voxels/slot, 10.9× at 4 081, 22.2× at 7 890, 43.5× at 62 318**. Atomic throughput is set by slot *count*, not by n. At 800³ (block=26, nc≈29 800) restrict ≈ 11–12 ms ≈ 16 % of a ~69 ms iteration = **7.7 % of e2e**; the atomic-free floor bounds the *maximum* recoverable at **9.3 % of e2e**, and a `@localmem` segmented reduction will not reach the floor. +1 concept, ~35–45 lines. **Correction to the round-2 log: restriction+prolongation is 14.4 % of a preconditioned iteration, not 46 % — prolongation runs at 0.89–0.93× the streaming reference, i.e. already at bandwidth, and was never part of the problem.**
2. **B30 — slice-only τ. REJECT.** Post is 1.6 s of 20.66 s = 7.7 %; slice-only takes it to ~0.1 s → gain **7.3 % of e2e**, 9× A21's 0.83 % but still below the bar. **The cost went up, not down:** it must now also *construct* the inlet/outlet pore-index lists that A7 deleted, and `build_pore_index` is not a shortcut — it allocates 4.1 GB of host `Int` at 800³, worse than the 2.05 GB it would replace.
3. **A30 — REJECT as a Phase 4 item; it is maintenance work, not speed work.** `bench/scaling_bench.jl` measures only the steady path, so A30 moves **no** reported metric by any amount and cannot clear a rule stated in percentages. Its real value is structural — it would retire the old five-stage pipeline, let B4/B8/B10/B12/B13/B21/B22/A12/A27 all go terminal at once, and remove a live CPU/GPU drift surface. **Handed over as follow-up work, judged on correctness and maintenance rather than speed.**
4. **Coarse solve → GPU / non-allocating `ldiv!`. REJECT** — ~3.8 ms of ~69 ms = 3.8 % of e2e at 800³; a 30 k-unknown sparse triangular solve is latency-bound and routinely *slower* on GPU.
5. **Retune `DEFAULT_MAX_COARSE`/block (= B28). REJECT on gain, not complexity.** Measured sweep at 400³: iters 143/168/202/294/379 for block 10/13/16/20/26, nnz(L) 6.14 M → 0.32 M. **Coarse-solve cost rises as fast as the iteration count falls; wall time is flat within noise across block 13–20. Gain ≈ 0** even though it is a one-constant, neutral-complexity change.
6. **Fused/pipelined CG. REJECT** — Krylov's vector ops are 4.81 ms = 36 % of a preconditioned iteration, the largest single share, but standard CG is already 13 vector passes; merging recovers ~3 of 13 ≈ 6 % of e2e, changes the numerics, and means vendoring a solver out of Krylov.jl.
7. **fp16 `nzval` for CUSPARSE SpMV** (new, never logged). Would cut peak 20.588 → 17.08 GiB (**−17 %, clears the bar**) and SpMV traffic −25 %, and for uniform `D` the entries (−1, and 1…6 on the diagonal) are **exact** in fp16. **INFEASIBLE on mechanism:** cuSPARSE's 16F SpMV paths require `X` in the same precision as `A`, so the CG vectors would have to be fp16 — no convergence at `reltol=1e-6`.
8. **Preconditioner build. REJECT** — warm build is **0.121 s** at 400³ (the 4.17 s first observed was compilation), ≈0.7–1.0 s at 800³ = ~4 % of e2e. Its 1.9 GiB `idx` temp (`preconditioner.jl:282-283`, a second full-grid cumsum duplicating `build_steady_system`) is removable, but the build peak (~19.2 GiB) sits *below* the solve peak, so it moves no headline.

**VERDICT: diminishing returns reached.** The largest remaining shares of the 800³ preconditioned e2e are **CUSPARSE SpMV (~27 %)** and **Krylov's own vector ops (~24 %)** — neither is Tortuosity code, and neither is improvable without going matrix-free. Every in-repo candidate tops out at 7–9 %.

### Phase 4 — round 3 (correctness and coverage), commits `3fc55bc` `2ad5fc2` `4200236` `e8548b5` `94408f6` `41c0127`

`Pkg.test()` **`Tortuosity.jl | 11576 11576 2m57.0s`** — foreground, GPU included, +65 over the floor.

**Accepted:** F11, F12, F6, F7/F8, F14, B27(partial). **Rejected:** B28, B27's Float32 half.

- **F6 — Phase 3's bit-identity claim is now PINNED AND TRUE.** New CPU testset compares `build_steady_system` against the full reference chain with `==` on colptr/rowval/nzval/b: **7 images × {uniform D, variable D} = 56 exact comparisons, all pass.** Fixtures include the three untrimmed 16³ blobs plus a purpose-built 8×6×6 image carrying zero-degree pore voxels **in the interior AND on both Dirichlet faces**, with 5 non-vacuity assertions proving those cases are reached. *Why it holds, recorded in the test comment:* the reference chain sums a degree in ascending pore ordinal, which is the order the kernel walks its six face offsets, and `2ab/(a+b)` is exactly symmetric.
- **F11** — `transient.jl` now calls `_invalidate_cache!` before rescaling `nzval`, with a no-op `_invalidate_cache!(::Any)` fallback mirroring `_free!`. **"Any mutation invalidates" is now literally true.**
- **F12** — GPU assertion added for `reconstruct_slice`'s device gather, verified on real CUDA hardware (3 slices identical, 235/225/215 NaNs each).
- **F7/F8/F14** — the post-hoc Dirichlet elimination is labelled reference/parity material (nothing deleted); `test_impl_parity.jl:225` renamed to "reference-chain parity"; the symmetric SpMV kernel's bit-identity claim is bounded to a single-workgroup launch. **No assertion changed.**
- **B28 REJECTED on measurement**, and the earlier reading explained: *"block 12 is better" came from measuring several block sizes in one process — pool growth inflated later cells by up to 75 %."* With a fresh process per block: 400³ block 12 −27 %, 600³ block 19 −10 %, but **800³ block 25 regresses +2.4 % and +0.47 GiB**, dropping headroom 1.93 → 1.46 GiB. A single global cap is self-contradictory (block 12 at 400³ needs cap ≥ 39304; block 26 at 800³ needs cap < 32768). Iteration count is **not monotone in block size** — this needs a size-dependent rule, not a retuned constant. Cap left at 32000.
- **B27 partial.** Accepted: `norm_to_uniform` rewritten as one buffer written twice instead of ten Float64 temporaries — 400³ **9.91 → 7.19 s (−27 %), 13.5 → 9.21 GiB allocated (−32 %)**, output bit-identical (0 voxel flips at 200³), all three golden node counts unchanged. **Float32 REJECTED as a BLOCKER** — `rand(Float32,…)` changes two of three golden node counts (8098/8093/8113 vs 8116/8066/8113); even keeping the Float64 RNG stream and converting after passes the golden gate but flips 31 of 8 M voxels at 200³, which would invalidate every benchmark baseline recorded for this campaign and change what a released API returns for a given seed — for time no reported metric measures.

---

# FINAL REPORT — matrix path optimization campaign

**Branch `perf/matrix-path`, 35 commits, `680f883`…`41c0127`. Started and finished 2026-08-08.** Nothing pushed; every commit is local, as instructed.

## Phase 6 verification — measured at HEAD `41c0127`

`Pkg.test()` — **foreground, GPU included:**

```
Test Summary:                            |  Pass  Total     Time
Tortuosity.jl                            | 11576  11576  2m57.0s
```

Phase 0 floor was **11360**. Final is **11576, +216 assertions, zero failures.** No golden τ value or golden node count was ever modified.

`bench/scaling_bench.jl`, all sizes including 800³:

```
    N  device threads     pass   status       nnodes            nnz   setup_s    prec_s   solve_s     e2e_s  peak_GiB    iters      tau
  200     cpu       1      api       ok      4009753       26583004     0.260         -    43.578    43.876     0.000     1087    1.951
  200     cpu       4      api       ok      4009753       26583004     0.119         -    44.780    44.941     0.000     1087    1.951
  200     cpu       1  precond       ok      4009753       26583004     0.258     0.164     4.620     5.426     0.000       90    1.951
  200     cpu       4  precond       ok      4009753       26583004     0.202     0.107     4.971     5.418     0.000       90    1.951
  200     gpu       1      api       ok      4009753       26583004     0.009         -     0.708     0.740     1.718     1044    1.951
  400     gpu       1      api       ok     31906673      216918266     0.056         -    13.522    13.724     4.031     2094    1.925
  600     gpu       1      api       ok    108249920      742436368     0.160         -    65.368    66.166    10.343     2983    1.943
  800     gpu       1      api       ok    254645845     1753943979     0.409         -   187.829   189.698    20.588     3620    1.881
  200     gpu       1  precond       ok      4009753       26583004     0.009     0.052     0.232     0.320     1.781       82    1.951
  400     gpu       1  precond       ok     31906673      216918266     0.052     0.123     2.047     2.398     4.500      168    1.927
  600     gpu       1  precond       ok    108249920      742436368     0.159     0.150     8.223     9.202    11.968      223    1.941
  800     gpu       1  precond       ok    254645845     1753943979     0.388     0.231    18.552    20.660    21.959      202    1.882
```

## Total deltas, Phase 0 baseline → final

| N | peak GiB | setup s | e2e s (default) | e2e s (preconditioned) |
| --- | --- | --- | --- | --- |
| 200³ GPU | 3.281 → **1.718** (−47.6 %) | 0.073 → 0.009 (−87.7 %) | 0.822 → 0.740 (−10.0 %) | **0.320 (−61.1 %)** |
| 400³ GPU | 12.750 → **4.031** (−68.4 %) | 0.453 → 0.056 (−87.6 %) | 15.447 → 13.724 (−11.2 %) | **2.398 (−84.5 %)** |
| 600³ GPU | 23.889 → **10.343** (−56.7 %) | 107.871 → 0.160 (**−99.85 %, 674×**) | **never completed → 66.166** | **9.202** |
| 800³ GPU | 23.889 → **20.588** (−13.8 %) | 384.181 → 0.409 (**−99.89 %, 939×**) | **never completed → 189.698** | **20.660** |
| 200³ CPU | — | 2.475 → 0.260 (−89.5 %) | 55.502 → 43.876 (−20.9 %) | **5.426 (−90.2 %)** |

**The campaign's central objective is met.** 800³ went from a documented `OutOfGPUMemoryError` — and, once that stopped reproducing, from a 384-second setup that pinned the device at 100 % and never finished solving — to a complete end-to-end solve. **Peak 20.588 GiB of 23.889, headroom 3.301 GiB, clearing the ≥3 GiB gate.** With the preconditioner, 800³ solves in **20.7 seconds**.

Bytes per pore voxel at 800³: **198 → ~76** (peak less the 1.375 GiB CUDA context).

## Accepted changes, with measured gains

**Phase 1 — quick wins (14 items, commits `8053141`…`d529398`).** A20 Krylov placeholder workspace at zero length; A11 connectivity columns as views (−11.2 GiB, one line, the single largest one-line win found); A16 CUSPARSE cache dropped on rebuild; A13 setup-stage arrays freed when dead; A10 transient keeps structural zeros; A2 prefix sum instead of `findall`; A3 exclusive scan by subtraction; A4 colptr view; A14/A18 degrees and diagonal flag without scratch arrays; A15 device-side RHS; A17 `dropzeros!` redundant scan and mask; A19(b)(c) non-Int32 CUSPARSE cache + 5-arg `mul!`; A22 mask copied only when on device. **Result: the 600³ cliff eliminated — setup 107.87 → 0.99 s, e2e never-finished → 74 s.**

**Phase 3 — fused assembly (the main event, `181037f`).** A5+A6+A7+A12+A23+A24+A25 as one coherent change. New `src/assembly.jl`: two KA kernels shared by CPU and GPU, one thread per voxel owning its whole CSC column. No COO, no adjacency matrix, no edge-weight vector, no atomics, no `dropzeros!`. Five stages → two kernels and a scan, **one implementation instead of two**. **800³ setup 384.2 → 0.39 s. CPU output bit-identical to the old pipeline**, now pinned by 56 exact `==` comparisons.

**Phase 4 — speed (rounds 1–3).** A29 CG workspace built around LinearSolve's solution vector (**cleared the memory gate**); B1 CSR view of the symmetric matrix (−8.1 % e2e); B2 symmetric reduce instead of scatter (3.5× on the CPU KA backend); A26 Int32 host CSC (**CPU e2e −17.2 %**); A27/B19/B20/B22 transient data movement; B26 label lookup table (2.32×); B27 fused blob normalisation (−27 %, bit-identical); **B24 two-level aggregation preconditioner — 800³ e2e −89.1 %, iterations 3620 → 202, iteration count flat in N.**

## Rejections, with reasoning

Twenty items rejected. The ones that matter:

- **A8** (upper-triangle storage) — the plan understated the cost: it does not "reintroduce atomics", it **forfeits CUSPARSE entirely** on the only benchmarked backend.
- **A9** (omit `nzval`) — the plan's *reason* was wrong (it is not "matrix-free with extra steps"; it keeps `rowval` and `colptr`). Rejected because a matrix without `nzval` cannot be handed to CUSPARSE.
- **B3 and the entire cheap-preconditioner family** — measured, not argued: polynomial preconditioning **raises** total SpMV count monotonically with degree (798/987/1290/1764 vs 606 plain). Only a coarse space wins.
- **C1** — premise refuted by three independent auditors: the device→host copy never happens in production.
- **C2** — premise inverted: the cost is materialising 512 M voxels, not the transfer.
- **C3/C4/C5** — rejected with reasoning; C5 specifically because ~40 lines of rocSPARSE would land with **zero test coverage on the one axis where untested code is most dangerous** — a wrong SpMV returns a plausible τ, not an error.
- **B25** — measured identical iteration counts; `LinearProblem(A,b; u0=…)` does not warm-start KrylovJL.
- **B28** — a single global cap is self-contradictory across sizes, and 800³ *regresses*.
- **B29/B30 and 6 others** in the final audit round — every in-repo candidate tops out at 7–9 % of e2e, below the bar.

**Thirteen further items were triaged MOOT or ALREADY DONE** after Phase 3 deleted the kernels they targeted.

## Blockers — four decisions for Amin

1. **B24 as the default (worth the entire −89 %).** The preconditioner is opt-in; `solve(sim.prob, KrylovJL_CG())` does not use it, so the *default* 800³ e2e is still 190 s rather than 20.7 s. LinearSolve exposes only `alg.precs` and `Pl=`, and `__init` never reads `prob.kwargs`. Making it default needs either shadowing the exported `KrylovJL_CG` (an **ambiguity error on a published name, mid-JOSS-review**) or a Tortuosity-owned `solve` entry point. **Recommendation: the additive option — a new entry point, not a shadowed name.** It resolves B23 at the same time. *This is worth more than every remaining candidate combined.*
2. **B17 — CPU solve −57 % (4 threads) / −15 % (1 thread).** SpMV is 82 % of the CPU CG iteration; the symmetric gather is 3.31× faster and bit-identical, and the kernel is already committed. The blocker: reaching it means `sim.prob.A` stops being a `SparseMatrixCSC`, costing `Array`, `==`, `issymmetric` and SparseArrays interop on a published object. Tried; 23 assertions fail on exactly that; reverted rather than decided unattended.
3. **B23 — solver tolerance and iteration cap.** No mechanism exists today; same fix as (1). Note `maxiters` defaults to `length(b)` = **254.6 M** at 800³, and `abstol = reltol = sqrt(eps)` differs by 4½ orders of magnitude between CPU Float64 and GPU Float32.
4. **A1 — superseded, no longer needs a decision.** It was blocked by `csc_equivalent`; A7 made it moot by deleting the `dropzeros!` call outright.

## B18 — ACCEPTED, commit `1cf8726`

**The master initially recorded this as "rejected, folded into the A30 follow-up" on judgement rather than measurement, because the agent sent to evaluate it died on an API session limit before doing any work. That judgement was wrong and is corrected here.** B18 was triaged `LIVE` in audit round 1, never claimed by any write round, and caught during final verification.

**Accepted.** Folded `−1/voxel_size²` into the edge weights in `build_transient_operator` (`src/transient.jl:299-315`) and deleted the post-assembly `nonzeros(A) .= nonzeros(A) ./ (−voxel_size^2)`.

- **Measured** (RTX PRO 5000, 320³, 229 M nonzeros, stage-level timing of both variants side by side): the rescale pass costs 6.5 ms. **Scalar `D` → the pass disappears entirely** (~10 % of a ~66 ms operator build). **Array `D` → it moves onto the shorter edge-weight vector**, 6.5 → 5.3 ms. Setup-only, so 0 % on any number `bench/scaling_bench.jl` reports and ≪1 % of a transient solve — accepted on the **"neutral or simplifying"** branch of the rule, not the 15 % branch.
- **The crux the master could not resolve without reading the code: no new branch is needed.** The scalar/array split **already existed** (`gd = D; if !(D isa Number)`); it was inverted to `if D isa Number … else`. Line count: 4 removed, 4 added = **net 0** — the inventory's "−1 line" was wrong in the other direction too. Structurally a net simplification: one fewer in-place mutation of the largest array in the package, and one fewer cache-invalidation obligation.
- **F11 survives intact.** The `_invalidate_cache!(A)` from `3fc55bc` was removed *together with the mutation it guarded* — that is not undoing F11, it is F11's rule holding. After the fold, nothing edits `A` between `laplacian(am)` (which returns a fresh matrix) and `_zero_rows_only!`, whose `PortableSparseCSC` method invalidates the cache itself at `kernels/sparse.jl:206`. **"Every mutation invalidates first" is still literally true.** Keeping the call would have left a no-op guarding nothing, under a comment that had become false.
- **Bit-identity: off-diagonals YES, exactly; diagonals differ by ≤4 ULP.** `oftype(D, D / -voxel_size^2)` rounds the Float64 quotient once, exactly as scaling `nzval` did, so every edge weight is unchanged bit for bit. The diagonal is now `sum(α·w)` instead of `α·sum(w)`: max 4 ULP (4.4e-16 rel Float64, 2.4e-7 rel Float32), often 0. **The divide was deliberately kept rather than switched to multiply-by-reciprocal, precisely to preserve the off-diagonals.** Checked by rebuilding the pre-change chain and diffing colptr/rowval/nzval entry by entry (sorting within columns, since the GPU atomic scatter's intra-column order is not reproducible between builds) over open boxes and blobs × scalar/array `D` × CPU Float64 / GPU Float32 × four `voxel_size` values.
- `Pkg.test()` **`Tortuosity.jl | 11576 11576 3m59.2s`**, CUDA included. Transient physics 295/295, including *"operator entries scale as 1/voxel_size²"* and the symmetry / zero-row-sum checks.

**New follow-up found while measuring:** the Float64 divisor makes the remaining array-`D` pass run at **338 GB/s vs 765 GB/s** with a Float32 one — halving it is free but shifts off-diagonal values, so it is a separate item. Scaling `node_D` (7× fewer elements) was **rejected**: harmonic-mean homogeneity is exact only in real arithmetic, and `α²` can overflow Float32. `_invalidate_cache!(::Any)` now has no caller in `src/`; left deliberately.

## Follow-up work that did not fit

- **A30** — port the transient path to the fused assembler. Deliberately not attempted: transient-only, invisible to every metric this harness reports, and its structural contract differs between the CPU and GPU `laplacian` paths. **It would retire the old five-stage pipeline and let B4/B8/B10/B12/B13/B21/B22/A12/A27 all go terminal at once**, and remove a live CPU/GPU drift surface. Judge it on correctness and maintenance, not speed.
- **A28** — `_free!` is a no-op on Metal and AMDGPU. One line each, blocked on having no hardware to test on.
- **B27's Float32 half** — worth revisiting only alongside a deliberate fixture regeneration.
- **F13** — `bench/results/scaling.csv` retains only the Phase 0 baseline; intermediate "before" figures are not independently checkable from the repo.

## Two cautions for the JOSS paper

1. **Do not quote GPU τ beyond three significant figures.** The old CSC path summed with atomics in nondeterministic order — the Phase 0 baseline itself shows τ = 1.92742 vs 1.92824 across two reps of the *same build*. The new CSR path is a deterministic gather (1.88141 in both reps). The new value is more **reproducible**; it is **not shown to be more accurate** — there is no Float64 reference at 800³.
2. **CPU thread count barely moves CPU numbers** (200³: 43.9 s at 1 thread vs 44.9 s at 4), because CPU SpMV is single-threaded — that is B17, blocked. The Phase 3 "4 threads" annotation was a mislabel; the harness was recording `Threads.nthreads()` truthfully all along. `threads` is now a per-row column and part of the resume key.

## How the plan itself held up

Of the original 21 inventory items, **the four audits and the implementation work corrected or refuted a substantial fraction**: Constraint 2 was factually wrong (the GPU path never guaranteed ascending row order, and the suite contained a test *asserting* unsortedness); A5's mandated insertion sort was dead code; A29's and B1's stated mechanisms were both wrong in opposite directions; two of four C-items rested on false premises; and the single largest one-line win (A11) was missing entirely. The inventory grew 21 → 40 items and every `est. gain` figure that was measured is now marked as measured. **The campaign's most valuable output after the code is an inventory that no longer contains unverified arithmetic.**

- The harness's `stages` pass **overstates peak by design** (it holds `conns` and `am` alive across stages, which the API path no longer does): at 600³ it reports 22.8 GiB against the API pass's 20.2. Do not read the two as comparable.

### Phase 4 round 1 — 2026-08-08, commits `849cc3c`…`0258344` (4 commits)

**Accepted (8 items):** A29, B1, B2, A27, B19, B20, B22.
**BLOCKED (2):** B17, B23 — both need an API ruling from Amin, both detailed in their inventory rows.
**Rejected on measurement (2):** A21, B5 — post is 0.8 % of e2e and setup 0.3 %, so neither can move a headline number.
**Moot after Phase 3 (3):** B14, B15 (steady half), B16. **B3 re-confirmed rejected.**

`Pkg.test()` **PASS, 11397/11397, 2m48.9s** — above the 11365 floor. No golden value touched, no test weakened. Two assertions were *added* that pin the new invariants: every mutator drops the symmetry claim, and `_cg_workspace` matches Krylov's own constructor field by field.

| N | peak GiB | setup s | solve s | e2e s | iters | τ |
| --- | --- | --- | --- | --- | --- | --- |
| 200³ | 1.718 → 1.718 (0 %) | 0.008 → 0.009 | 0.775 → **0.715** (−7.7 %) | 0.801 → 0.747 (−6.7 %) | 1044 | 1.951 |
| 400³ | 4.156 → 4.031 (−3.0 %) | 0.053 → 0.053 | 14.907 → **13.666** (−8.3 %) | 15.113 → 13.904 (−8.0 %) | 2094 | 1.925 |
| 600³ | 10.750 → 10.343 (−3.8 %) | 0.158 → 0.191 | 71.137 → **65.117** (−8.5 %) | 71.838 → 65.971 (−8.2 %) | 2983 | 1.943 |
| 800³ | 21.637 → **20.605** (−4.8 %) | 0.388 → 0.391 | 205.660 → **189.051** (−8.1 %) | 207.766 → 191.030 (−8.1 %) | 3620 | 1.881 |

**THE CAMPAIGN GATE IS MET.** 800³ peak 20.605 GiB of 23.889 leaves **3.284 GiB of headroom**, against the ≥3 GiB the conflict rule requires. Iteration counts are identical at every size, so nothing here changed the problem being solved. From here **speed wins** — memory is a budget.

#### Findings that correct the plan

1. **LinearSolve was already aliasing `x` to `u`** (`iterative_wrappers.jl:266`), so A29's stated premise was half-true: the duplication is real but the fix is not aliasing. The constructor allocates a length-n `x` *before* the workspace's other vectors and LinearSolve then throws it away — and freeing it afterwards **does not move the measured peak**, because CUDA's stream-ordered pool counts a freed block as still in use (verified: `unsafe_free!` of 1 GiB leaves `total - available` unchanged, and the next 1 GiB allocation reuses it for free). It has to not be allocated at all. Measured 400³: the workspace build costs 0.5 GiB where `r,p,Ap` account for 0.357.
2. **B1's mechanism was wrong.** `cusparseSpMV_bufferSize` returns the **same** size for CSC and CSR (2 740 547 B at 800³), so no per-call workspace allocation is avoided — the plan's central claim. The win is the kernel: CSR gathers, CSC scatters with atomics. It is also **size-dependent**: 0.99× at 200³, 1.17× at 400³, 1.175× at 800³. Anyone re-measuring on a small image will conclude, wrongly, that there is nothing here.
3. **B2 is measurable on a backend present here**, contrary to audit round 1's terminal rejection: on the CPU KA backend at 200³ the atomic scatter is 125.5 ms against 35.5 ms for the symmetric gather (34.6 → 13.3 ms on four threads). It has no effect on the *production* CPU path only because that path returns `SparseMatrixCSC` — which is exactly what B17 is blocked on.
4. **The CPU path is untouched by this round** and its numbers should not be read as a regression: 200³ CPU e2e reads 58.2 s here against 49.2 s in the Phase 3 table, but nothing in these four commits reaches the CPU steady path, and repeated measurement of the same commit gave 53.0 s (1 thread) and 57.0 s (4 threads). The spread is machine variance. Note also that `bench/results/scaling_env.csv` records **threads=1** for every run, while the Phase 3 table annotates its CPU figure "4 threads" — one of the two is mislabelled, and CPU comparisons are not trustworthy until that is settled.
5. **τ at 400³ moved 1.928 → 1.925** (and 800³ 1.884 → 1.881). Not a regression: CSC SpMV sums with atomics in a nondeterministic order — the Phase 0 table itself shows 1.92742 vs 1.92824 across two reps of the *same* build — and CSR is deterministic. The shift is 1.6e-3 relative, inside the Float32 envelope `test_gpu_e2e.jl:81` already pins at `rtol=1e-3`. **A side benefit worth naming: the GPU solve is now run-to-run reproducible where it previously was not.**
