# JOSS submission — handoff

Context transfer for continuing the Tortuosity.jl JOSS paper on a different machine. Written 2026-08-05.

## TL;DR

The paper draft is real and structurally complete, but its **Performance comparison section reports numbers from a benchmark run that was later thrown away and redone**. The figures are unusable. PuMA was never benchmarked despite being a hard requirement. Everything else (benchmark harness, taufactor fork, JOSS PDF workflow) exists but is parked in a git stash.

Read "Blocking gaps" before touching the prose.

## Where everything lives

| Thing | Location | State |
|---|---|---|
| Paper draft + figures | branch `joss`, commit `bc05072`, pushed to `origin` | committed 2026-08-05 |
| Everything else | `stash@{0}` ("benchmark-wip"), created 2026-04-09 14:34 | **local only, never pushed** |
| Base branch for the stash | local branch `docs/benchmark` @ `ea46a98` | local only, not on origin |
| Modified taufactor | `github.com/ma-sadeghi/taufactor`, branch `node-centered-bc` @ `d05aa2e` | fork exists, verified live |

The stash holds two commits worth of content: `13eb441` (tracked edits) and `6fb04e4` (untracked files). Extract without popping:

```bash
git checkout 6fb04e4 -- benchmarks/ .github/ CONTRIBUTING.md NODE_VS_CELL.md REVIEW.md
git checkout 13eb441 -- .gitmodules docs/src/benchmark.md .gitignore
```

**The stash is the only copy of the benchmark harness and results.** Those CSVs took hours of GPU time. Commit or back them up early.

### Stash contents

Untracked (`6fb04e4`):
- `paper/` — already recovered onto `joss`
- `benchmarks/` — `bench_tortuosity.jl`, `bench_taufactor.py`, `bench_puma.py`, `generate_images.jl`, `plot_results.py`, `run_benchmarks.sh`, `pixi.lock`, `pyproject.toml`, `README.md`, `data/images.h5`, `results/results_{tortuosity,taufactor}.csv`
- `.github/workflows/draft-pdf.yml` — the JOSS draft-PDF action (**not on `main`**)
- `docs/src/assets/benchmark_*.png` — byte-identical to the `paper/` copies
- `CONTRIBUTING.md`, `NODE_VS_CELL.md`, `REVIEW.md`

Tracked edits (`13eb441`): `.gitmodules` (taufactor submodule), `docs/src/benchmark.md` (+94 lines), `.gitignore`, `docs/make.jl`, `README.md`, `benchmarks/taufactor` submodule pointer.

## What the April 8–9 session actually did

A 74-prompt session (`b5227a43`, 2026-04-08 02:45 → 2026-04-09 14:33). Its transcript has been garbage-collected; only the prompts survive in `~/.claude/history.jsonl`. Reconstructed arc:

1. **Drafted the paper** from the JOSS template — authors, sections, `paper.bib`, and `draft-pdf.yml`.
2. **Built a three-way benchmark harness** (Tortuosity.jl / taufactor / PuMA) under `benchmarks/`, with pixi managing the conda+PyPI Python environment.
3. **Established apples-to-apples comparison** — this was the bulk of the work. On small (10³) cases, forced all three solvers onto the same discrete problem before trusting any timing. Required forking taufactor to:
   - use node-centered Dirichlet BCs (upstream uses ghost cells / cell-centered)
   - respect the user's `conv_crit` (upstream has a hard-coded `2e-3` tau-stability check that silently overrides it)
4. **Discovered the node-vs-cell-centered convention bug** in our own transient fitting code as a side effect → wrote `NODE_VS_CELL.md`. **This is now resolved** — see "Already fixed, don't redo".
5. **Ran the final sweep** (100/200/300/400³ × 5 porosities × adaptive tolerance ladder), with GPU memory freeing, 150 s per-case timeout, skip-existing, and triplicate timing.
6. **Opened issues #48 and #49** (both still open) from problems the benchmark exposed.
7. **Vendored the taufactor fork as a submodule**, then stashed everything to clear the tree.

The session ended mid-cleanup. The paper prose was never revisited after step 5 produced new data.

## Blocking gaps

### 1. The paper's benchmark numbers are from the abandoned run

`paper.md` describes the April-8 benchmark. The April-9 rerun changed the sweep entirely. Every one of these is wrong:

| `paper.md` claims | Final data actually says |
|---|---|
| sizes 50³–200³ | 100³, 200³, 300³, 400³ |
| porosities 0.3, 0.5, 0.7, 0.9 | ≈0.18, 0.40, 0.60, 0.80, 0.95 |
| both at fixed `1e-5` convergence | adaptive tolerance ladder, 0.5 → 1e-7 |
| "Tortuosity.jl errors below 1e-4 across all cases" | many cases 1e-4…1.2e-3 |
| "taufactor errors 1e-3 (high ε) to 1e-1 (low ε)" | taufactor reaches 2e-5…9e-3; **beats us on error at high ε** |
| "at N=200, ε=0.3, roughly 5× faster" | at N=200, ε≈0.2: 1.44 s vs 16.58 s ≈ **11.5×** |

Real numbers from `results/*.csv` (best-accuracy row per case):

| N | ε | Tortuosity.jl err / time | taufactor err / time |
|---|---|---|---|
| 100 | 0.2 | 5.4e-4 / 0.18 s | 9.7e-4 / 1.46 s |
| 100 | 0.8 | 1.5e-4 / 0.11 s | 5.5e-5 / **0.07 s** |
| 200 | 0.2 | 3.6e-5 / 1.44 s | 5.1e-4 / 16.58 s |
| 200 | 0.8 | 4.7e-4 / 1.41 s | 9.8e-4 / **0.51 s** |
| 300 | 0.2 | 1.2e-3 / 11.04 s | 5.3e-4 / 163.12 s |
| 300 | 0.6+ | ~7–9 s | timed out |
| 400 | all | 21–27 s | timed out |

The honest story is **not** "we win everywhere". It is: we win decisively at low porosity and large domains (10–15×, and taufactor stops finishing at all past 300³), while taufactor is competitive-to-faster on small high-porosity images where SOR converges quickly. Writing that nuance will read as more credible to JOSS reviewers than the current overclaim, and it still lands the argument.

`docs/src/benchmark.md` (in the stash, +94 lines) has the **same stale numbers** and needs the same rewrite.

### 2. PuMA was never benchmarked

You pushed back hard on this once already — *"You absolutely must include it. I don't know why you excluded it."* `bench_puma.py` was written and the pixi env includes PuMA, but **there is no `results_puma.csv`** and the paper's comparison section covers taufactor only. The State-of-the-field section discusses PuMA but presents no data for it.

Either run it, or make the omission explicit and justified in the prose. Silently dropping it is the one thing likely to draw a reviewer's fire, since PuMA is cited.

### 3. The figures are unusable

`paper/benchmark_time.png` is a **~19-panel horizontal strip** (13800×600 px) of solve-time-vs-domain-size, one panel per case. At page width it is illegible.

The caption in `paper.md` line 144 describes something else entirely — *"Solve time (left), tortuosity agreement (center), and relative error (right)"* — i.e. a 3-panel composite that does not exist. `benchmark_tau.png` and `benchmark_error.png` are never referenced in the paper at all.

You asked for *"some clever condensed representations (maybe think of 3 or 4), that would convey the message much faster by looking at it"* and *"publication quality in terms of style and aesthetics"*. Neither was delivered. This is the largest single piece of remaining work.

### 4. Housekeeping not on `main`

- `.github/workflows/draft-pdf.yml` — needed to get a compiled PDF from GitHub Actions. Currently stash-only, so `joss` has no PDF build.
- `CONTRIBUTING.md` — JOSS checks for contribution guidelines. Stash-only.
- Sawyer Hossfeld's ORCID is blank in the front-matter. JOSS wants ORCIDs where available.
- Archival DOI (Zenodo) — you noted this would be done "right before submitting". Still outstanding.

## Already fixed — don't redo

- **`NODE_VS_CELL.md` is resolved.** Issue #71 closed 2026-04-14; `src/transient_fitting.jl` now uses `(depth_idx - 1) * voxel_size` at lines 95 and 221, and the flux path correctly keeps `(depth_idx - 0.5)` at line 119. The doc also predates the `dx` → `voxel_size` rename (#61), so its line numbers and code snippets are stale. Recover it for the reasoning, not the instructions.
- Issues #48 (peak GPU memory ≥400³) and #49 (matrix-free CG) were filed from this session and remain open. #48 is why taufactor has no 400³ rows to compare against — worth a sentence in the paper's limitations if you mention domain ceilings.

## Reproducing the benchmark

```bash
git submodule update --init --recursive   # pulls ma-sadeghi/taufactor @ node-centered-bc
cd benchmarks/
pixi install
./run_benchmarks.sh                        # --overwrite to force re-run of existing cases
```

`benchmarks/data/images.h5` is stashed deliberately so runs stay deterministic. `benchmarks/.gitignore` had entries commented out to let results be stashed — check it before assuming a file is tracked.

Note `benchmarks/README.md` says the tolerance sweep is "15 log-spaced, 1 to 1e-5"; the code was later changed to start at 0.5 and end at 1e-7. The README is stale.

## Your stated preferences from that session

Worth preserving, since they shaped the harness:

- Timing must include a **warm-up call with identical kwargs** (`gpu=true`) on a small domain — Julia's first-call compilation must not pollute the measurement.
- Comparison fairness is measured as **actual relative error against ground truth**, not each package's own tolerance knob. Tolerance is just the dial being swept.
- Reference solution = Tortuosity.jl at the tightest tolerance.
- Triplicate + median timing, but only worth it below ~60 s per run.
- Minimal modification to third-party source; vendor as a submodule when modification is unavoidable.
- Skip cases with no percolating path after trimming.
- Logging via `loguru` (Python) / native (Julia) so long runs are followable.

## Suggested order of work

1. Recover the stash onto `joss` and commit the benchmark harness + results (protect the data first).
2. Add `draft-pdf.yml` so the PDF builds on push.
3. Decide the PuMA question — run it, or write the justified omission.
4. Design the condensed figures (3–4 candidates, publication quality).
5. Rewrite the Performance comparison in `paper.md` and `docs/src/benchmark.md` against the real CSVs.
6. Fill ORCID, add `CONTRIBUTING.md`, mint the Zenodo DOI.
