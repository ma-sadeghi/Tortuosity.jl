# Benchmarks

A reproducible comparison of Tortuosity.jl against [taufactor](https://github.com/tldr-group/taufactor) (Python/PyTorch) and [PuMA](https://github.com/nasa/puma) (Python/C++), over domain size, porosity and microstructure, on both the CPU and the GPU, measuring both time-to-accuracy and peak memory.

Every tool solves the same images, against the same ground truth, at the same accuracy targets, with the same thread budget. Anything that differs between them is meant to be a property of the tool.

## Layout

```
config.toml            the campaign: grids, porosities, blobiness, ladders, targets, threads
src/                   Julia harness — case grid, image store, result files, memory probes
benchkit/              Python harness — the same, for the tools written in Python

generate_images.jl     build the shared image store
compute_references.jl  ground truth: Float64 CPU solves
bench_tortuosity.jl    Tortuosity.jl, either device, either operator, time or memory
bench_taufactor.py     taufactor, either device, time or memory
bench_puma.py          PuMA (CPU only), time or memory
make_figures.py        post-processing: figures from the CSVs, nothing else

run/                   orchestration; see run/README.md for the rented-machine flow
data/images/           the image store and its manifest (regenerable; not in git)
results/               the measured dataset: timings/, memory/, references.csv
results/legacy/        the superseded pre-2026-08 dataset, kept unmodified
results/archive/       datasets a later change made incomparable; see its README
figures/               output of make_figures.py
```

`config.toml` is the single source of truth. Both harnesses read it — Julia through the `TOML` stdlib, Python through `tomllib` — so no grid, ladder or target is defined twice and the two languages cannot drift apart.

## Setup

```bash
cd benchmarks/
./run/setup.sh
```

That resolves the Julia project, installs the Python environment with [pixi](https://pixi.sh), checks out the vendored taufactor fork at its pinned commit, and checks that the GPU and all three tools actually work before anything is measured.

Two things about the environments are deliberate and easy to undo by accident:

- **The Julia project here is separate from the package above it.** `bench_tortuosity.jl` needs CUDA, which is a *weak* dependency of Tortuosity.jl. Adding it to the package's own `Project.toml` would turn the CUDA extension into a hard dependency of the released package. Always run with `--project=.` from this directory.
- **taufactor is checked out under `vendor/` and installed editable.** Editable because the campaign measures patches to that fork, and a non-editable install fails silently: `pixi install` resolves a path dependency by version rather than by source content, so it keeps the copy it built earlier and the benchmark reports numbers for code nobody edited. Under `vendor/` because `benchmarks/` is `sys.path[0]` when the scripts run — a checkout at `benchmarks/taufactor/` is found there first and, despite having no `__init__.py`, is returned by `PathFinder` as a PEP 420 namespace package. That counts as a successful import, so `import taufactor` yields a module with no `Solver` and the editable install's finder, which sits behind `PathFinder` on `sys.meta_path`, never runs. `run/setup.sh` asserts the import resolves into the working tree and that the patched signature is present, and exits non-zero if not.

## Running

```bash
./run/campaign.sh --grid=smoke      # validate the machinery, minutes
./run/campaign.sh --grid=full       # the real grid
./run/figures.sh                    # post-processing; no GPU needed
```

`--grid=smoke` is the full grid with every dimension divided by ten: the same 75 cases, the same stages, the same code paths, finishing in minutes. Run it before committing a large machine to anything.

Every stage resumes from its own results file and runs cases cheapest first, so the campaign can be interrupted and re-run with the same command. `run/README.md` covers the rented-machine flow, what has to be copied where, and what each stage costs. `run/ORCHESTRATION.md` is the step-by-step runbook for actually driving one, with a check after every step.

A stage can run for hours, so every one of them logs as it goes — through `loguru` in Python and Julia's own logger — and `run/campaign.sh` appends each stage's output to `logs/<stage>.log`, one line per case. A long run nobody can follow is a long run nobody can diagnose afterwards.

### Stages, individually

```bash
julia --project=. generate_images.jl --grid=full
julia --project=. -t auto compute_references.jl --grid=full
julia --project=. -t 1 bench_tortuosity.jl --device=gpu --operator=matrixfree --measure=time
julia --project=. -t 1,1 bench_tortuosity.jl --device=cpu --operator=assembled --measure=memory
pixi run python bench_taufactor.py --device=cpu --measure=time
pixi run python bench_puma.py --measure=memory
pixi run python make_figures.py --only=memory,summary
```

Every stage takes the same selection flags: `--grid=`, `--sizes=`, `--porosities=`, `--blobiness=`, `--cases=`, `--overwrite`, `--dry-run`. They compose, `--cases=` overrides the rest, and an unrecognised flag is an error rather than something quietly ignored.

Run the stages **serially**. They contend for the same GPU and the same cores, and concurrency contaminates the very timings the benchmark exists to measure. Run the CPU and GPU passes as **separate processes**: a process that has already run large `Float64` CPU sweeps carries a multi-gigabyte heap, and GPU sweeps that follow it in the same process stop being monotonic in the iteration count. A longer solve comes back faster, which is not something a solver can do.

## What is measured, and how

### The grid

75 images: 5 domain sizes × 5 target porosities × 3 blobiness values.

| axis | values |
|---|---|
| domain size | 200, 400, 600, 800, 1000 (smoke grid: 20 … 100) |
| target porosity | 0.20, 0.40, 0.60, 0.80, 0.95 |
| blobiness | 0.5, 1.0, 2.0 |

Blobiness is the feature-size knob: `Imaginator.blobs` blurs with `sigma = mean(shape)/40/blobiness`, so a higher value gives finer features and longer transport paths. Sigma scales with the domain, which keeps structures self-similar across the size sweep — a 200³ and a 1000³ image at the same blobiness hold the same number of blobs, so the size sweep measures scaling rather than a changing geometry. Porosity alone does not describe a porous medium. At one pore fraction a coarse structure and a fine one differ in tortuosity substantially, and the campaign covers three so that a ranking between solvers can be shown to survive that.

Every image is trimmed to the pore space that percolates along the transport axis. This is not cosmetic: an isolated pore cluster contributes nodes that no boundary condition reaches, leaving the operator singular on that subspace, and solvers differ in how gracefully they absorb that. An untrimmed image would therefore measure error handling rather than transport. A case whose percolating pore space is empty (possible at low porosity with coarse structures on small domains) is recorded with zero nodes and skipped by every later stage, so the gap in the grid is explained rather than merely absent.

### The image store

Images are generated once and cached as one HDF5 file per case, indexed by `data/images/manifest.csv`, which records a SHA-256 of each. Generation is deterministic in the seed, so a machine that rebuilds the store gets byte-identical images — and the hash is what turns that from an assumption into something checked. Every stage that loads an image verifies it and refuses to proceed on a mismatch. That is what lets the store be regenerated on a rented machine instead of copied to it, which matters because it reaches tens of gigabytes at the full grid.

Case identifiers look like `n400_b100_p020`: 400³, blobiness 1.00, target porosity 0.20. Every result row carries the identifier and the target porosity as well as the realised one, so joining across tools never depends on two languages agreeing about a float.

### Ground truth

A Float64 CPU solve at `reltol = 1e-10`, computed once per image and reused by every tool. Deliberately not on the GPU: GPU solves run in Float32, whose epsilon (~1.2e-7) falls inside the error range being measured, so a GPU reference could not resolve the errors it certifies.

Do not loosen the tolerance. `reltol` bounds the residual and the error in the solution is bounded by κ(A)·reltol with κ ~ N², so at N=400 a 1e-8 reference would admit ~1.6e-3 error — larger than the 0.1% target it exists to certify.

References are the most expensive thing the campaign computes, so each is appended to `results/references.csv` the moment it is solved rather than at the end of the stage. This stage is the one place the whole machine is used (`-t auto`): a reference is a value, not a timing, so its thread count cannot change the answer.

### Time

Each tool is swept over the knob that best traces *its own* accuracy–time frontier. What is compared is the frontier, not the knob value. A row is written for every rung, so the time to reach any looser target is answerable from the same data without re-measuring.

| tool | knob | range |
|---|---|---|
| Tortuosity.jl | CG iteration count | 18 log-spaced, 1 … 20000 |
| taufactor | SOR iteration count | 18 log-spaced, 1 … 20000 |
| PuMA | CG iteration count | 18 log-spaced, 1 … 20000 |

Iteration count rather than tolerance, for all three. Tolerance samples the frontier badly at both ends — the loosest settings return τ ≈ 0 and the tightest can step straight past the target in one rung — and taufactor evaluates its own `conv_crit` only every 100 iterations, which puts the entire coarse-accuracy regime out of reach through that knob. PuMA needs two workarounds to reach the same axis, both in `bench_puma.py`: its `PropertySolver.solve` raises rather than returning a partial result when SciPy stops on `maxiter`, and it passes only `atol` to SciPy and never `tol`, leaving that at its 1e-5 default so that every tolerance rung below `1e-5·‖b‖` was a duplicate of the one above it. Driving SciPy's conjugate gradient directly, over PuMA's own operator and preconditioner, fixes both. Nothing about PuMA's algorithm changes, only the stopping rule, which is the thing being swept.

### Warm-up

Every stage solves a throwaway image before it measures anything, on a case that is not in the grid. No reported number includes a first-call cost: Julia compiles on first execution, PyTorch pays CUDA context creation on its first kernel launch, and SciPy and PuMA's compiled `compute_flux` each pay one of their own. Warming on a measured case instead would double that case's cost for nothing.

The warm-up has to exercise the same code as the measurement, not merely the same package. Julia specialises on types rather than array sizes, so a 64³ image compiles what a 1000³ one runs — but a timing run traces its ladder through a different path than a plain solve, and warming only the latter leaves the first measured case carrying the trace path's compilation. Both paths are warmed in all three harnesses. taufactor shows why most plainly: a run shorter than 100 iterations never reaches a convergence check and so never calls `compute_metrics`, which is the one thing every checkpoint calls. 64³ is also the smallest size that clears the pore count below which `precond=:auto` declines to build a coarse space, so the preconditioner is warmed too.

### One solve per case, not one per rung

Each tool's whole ladder is traced from a **single** solve. A Krylov or SOR iterate is deterministic — iterate *k* is the same vector whether the run stopped there or carried on — so reading tortuosity off at each rung reports exactly what one solve per rung reported, for a fraction of the cost. The time recorded against a rung has the cost of the readings taken so far subtracted back out, so it stays comparable with a plain run that stopped at that iteration.

Verified rather than assumed, on both devices and all three tools: τ comes back **bit-identical** at every rung, and the traced time lands within a few percent of one-solve-per-rung, converging on it as the rung grows. This is what makes the campaign affordable — the CPU stages were 6.5 of the 7 hours of the previous 200³ run.

Two things this required. `abstol = 0` for Tortuosity.jl, because LinearSolve otherwise defaults it to `sqrt(eps(T))` — 3.4e-4 in `Float32` — which ends the solve long before the iteration cap and leaves most of the ladder unreachable. And a `checkpoints` argument on the vendored taufactor fork, listed with its other patches below.

Each rung is run three times and the **median** reported, with the spread recorded alongside. The median rather than one sample because wall time varies between launches, and because the preconditioner's restriction once scattered with atomic float adds whose order is not fixed. Those adds moved τ between runs by roughly the size of the accuracy target, and made whether a case "reached" the target partly luck. That scatter is now a gather over a fixed coarse-to-fine adjacency, and every three-repeat GPU row in the current results reports a τ spread of exactly zero. The coarse operator's own assembly still uses atomics, so bit-for-bit equality across runs is not guaranteed even though it is what we now measure. A first repeat slower than `repeat_threshold_s` abandons the remaining repeats, and those rows carry `repeats = 1` and a NaN spread — a spread of zero is the claim that three runs agreed exactly, which is not the same thing.

**Every tool is clocked from the moment it receives the image to the moment tortuosity can be read.** One rule, applied identically: problem construction, matrix assembly and preconditioner build are all inside the timed region, for all three. Only the image itself, and the tortuosity read-off at the end, sit outside — the first because it is the input, the second because it is instrumentation rather than work a user does.

This replaces an earlier convention that excluded setup for taufactor and PuMA on the grounds that their users "pay it before solving", and described the result as conservative. It was not conservative by a little. taufactor's `Solver` constructor builds the SOR checkerboard from an N³ float64 array and a three-way N³ meshgrid. Measured at 200³ on a GPU it costs **0.415 s against a 0.48 s solve — 45% of the total**, and it grows as N³. Charging Tortuosity.jl for its assembly and coarse space while charging taufactor nothing skewed the GPU comparison by about the whole margin being measured, and it inverted the ranking at the loose end of the accuracy ladder.

### Memory

Measured by its own stage, never read off a timing sweep. The two questions want opposite things from a run: a timing must not be perturbed and so cannot afford a sampler, while a peak needs one and does not care what it costs. Separating them is what lets each be measured properly instead of both being measured badly. The memory stage runs one short fixed-length solve per case, needs no ground truth, and is cheap.

| | host | device |
|---|---|---|
| Julia | resident set, sampled on an interactive thread | `CUDA.memory_stats().live`, sampled |
| Python | resident set, sampled by `psutil` | `torch.cuda.max_memory_allocated()` |

Host memory is the process resident set in **both** languages, sampled at the same interval. That is what makes a Julia figure and a Python figure comparable at all: Julia's own `gc_live_bytes` would count only what its collector manages, and PuMA's solver allocates in C where a Python-level tracer such as `tracemalloc` sees nothing. Figures report the *increase* over a baseline taken with the image already loaded, because raw resident totals would rank runtimes rather than solvers — a Julia process starts near a gigabyte and a Python one near a tenth of that.

**The memory stage runs one process per case, and that is not a tuning choice.** A resident-set increase only means anything in a process that has not already faulted in comparable pages. Measured within one process the readings are worthless: torch's CPU allocator reuses pages it already holds, so an 80³ taufactor solve holding several full-grid tensors reported 0.6 MB, and the series was not even monotonic in the domain size. It is the host-side twin of the pool problem below — a caching allocator defeats a delta measurement — and isolation is the only honest fix. `run/campaign.sh` enumerates the cases with `--list-cases` and spawns a process for each. Resume still applies, so an interrupted stage picks up where it stopped.

By default the memory stage measures only the reference blobiness (`memory.blobinesses` in `config.toml`). Memory tracks pore count, which at a fixed porosity barely moves with the structure, and every memory figure is drawn at one structure for the same reason — so measuring all three would spend three times the processes redrawing one curve.

On the device the figure is what the solve holds, not what the allocator took. `CUDA.total_memory() - CUDA.available_memory()` and `torch.cuda.max_memory_reserved()` measure a caching allocator's footprint, which grows opportunistically and saturates at the card's capacity under pressure — reporting the same number for every configuration, precisely where the comparison matters most. Both are recorded for context in `pool_device_bytes` and must not be compared between configurations.

### Threads

Nothing is pinned. Every tool takes the machine the way a user running it would — `-t auto` for Julia, torch's own pool for taufactor, whatever NumPy and SciPy size their BLAS to for PuMA — and every row records the count that run actually got.

An earlier policy pinned all three to a single thread. It was wrong in both directions: it forced taufactor and PuMA down to one thread while Julia's OpenBLAS quietly kept eight, and then recorded `cpu_threads = 1` for runs that were nothing of the sort. What this campaign claims is a speedup, not an algorithmic advantage, so a tool that parallelises better is entitled to the result that follows from it — and one that does not is not protected from it. The three parallelise very differently (Tortuosity.jl's matrix-free apply is a KernelAbstractions kernel that scales with Julia threads, taufactor's CPU path is threaded through torch, PuMA's conjugate gradient is effectively serial), and that difference is a property of the tools worth reporting rather than one worth suppressing.

## Result files

```
results/references.csv          ground truth, one row per case
results/timings/<tool>-<device>[-<variant>].csv
results/memory/<tool>-<device>[-<variant>].csv
results/environment.csv         what produced each batch of rows
results/archive/<change>/       datasets superseded by that change
```

The identifying columns — tool, device, variant, case, thread count, host, timestamp — are repeated on every row rather than encoded only in the filename, so post-processing concatenates the whole directory without parsing a name to know what it is reading. `environment.csv` exists because timings are only comparable within one machine and one software stack, and this campaign spans a laptop and a rented host by design.

A sweep is complete only once one of its rows carries a `stop_reason` (`target_reached`, `timeout`, `ladder_exhausted`, `error` or `oom`). Resume keys on that rather than on the mere presence of a row, because a case interrupted halfway up its ladder would otherwise be silently accepted as converged — and a partial ladder is indistinguishable from a converged one once it is in the file.

`results/legacy/` holds the superseded dataset measured before this harness existed. It is on a different schema, was measured over a different size grid at a single blobiness, and cannot be merged with the current one. It is kept only so the earlier figures can still be rebuilt. The image store it was measured against, `data/images.h5`, is likewise superseded — a single 3.9 GB HDF5 file rather than the per-case store — and can be deleted once those figures are no longer needed.

`results/archive/` holds datasets a later change made incomparable with the current ones, one directory per change, each named for the change it precedes. They are kept because a before-and-after is the only way to show what a change cost, and one of them is cited by `docs/src/benchmark.md`. The `results/archive/` README says what each is. Their rows share the current schema and must still never be concatenated with it, because the solver underneath them is different.

## The taufactor fork

taufactor is vendored as a pinned checkout of [`ma-sadeghi/taufactor@a4bc5f9`](https://github.com/ma-sadeghi/taufactor/commit/a4bc5f9), which carries the node-centered boundary patch of [`d05aa2e`](https://github.com/ma-sadeghi/taufactor/commit/d05aa2e) plus the checkpoint patch on top. The fork is a small change to `taufactor.py` that makes all three tools solve the *same discrete problem* and be measurable in the same way. A difference in the reported τ then reflects the solver rather than the discretisation or the harness. No solver logic is changed.

Third-party source is modified as little as the comparison allows, and where modification is unavoidable the result is vendored at a pinned commit rather than patched in place, so the exact source behind every number is recoverable. `run/setup.sh` clones it and checks that commit out. `vendor/` is ignored rather than tracked.

It is deliberately not a git submodule. A gitlink in the tree makes this repository's git tree hash — the one the Julia registry records — disagree with the hash Pkg recomputes by walking an installed package's files, where the empty submodule directory is skipped. Every Linux and macOS user of Tortuosity.jl would then pay for the benchmark harness on `Pkg.add`, with a "tarball content does not match git-tree-sha1" warning and a fall back to a full git clone.

- **Node-centered Dirichlet BCs.** Upstream places the boundary half a voxel outside the domain (`top_bc, bot_bc = -0.5, 0.5` with a `1/(2·Nx)` shift in `init_field`), a cell-centered ghost-cell convention. Tortuosity.jl and PuMA both pin concentrations at the boundary voxels themselves. The fork switches to `(0.0, 1.0)` pinned at voxels 1 and `Nx`, and zeroes the SOR checkerboard on those two slices so they are never relaxed.
- **Domain length spans `Nx-1` intervals, not `Nx`.** Follows directly: with the boundaries *at* voxels 1 and `Nx`, the distance between them is `(Nx-1)·dx`. Upstream's `D_rel = mean_fl * Nx / ΔC` becomes `mean_fl * (Nx - 1) / ΔC`. This is an O(1/N) bias — 1% at N=100 — that would otherwise be attributed to the solver.
- **The convergence criterion honours `conv_crit`.** Upstream gates convergence on a hard-coded `tau_error < 2e-3` that silently overrides what the user passed, which makes a tolerance sweep impossible.
- **`tau` is finalised when a run stops between convergence checks.** Upstream assigns `self.tau` only inside `check_convergence`, which fires every 100 iterations, so a run capped below 100 iterations reports a stale or unset value. This is what makes `iter_limit` usable as a sweep knob. Converged runs are untouched.
- **`solve` takes `checkpoints` and `checkpoint_hook`.** Reads τ off at a list of iteration counts without stopping, recording `(iter, tau, time_s)` with the cost of the readings subtracted back out of the clock, and lets the caller end the run from a checkpoint. This is what lets one solve trace the whole ladder. It touches no state the SOR sweep reads: `compute_metrics` only reads the field, which is why iterate *k* comes back bit-identical either way.

Upstream taufactor seeds SOR with an exact linear concentration profile while Tortuosity.jl's CG starts from a zero vector, and the fork preserves that. It is worth knowing when reading the results: for a high-porosity medium the linear profile is already close to the solution, so taufactor begins nearly converged and Tortuosity.jl does not.

## Known constraints

- **Never import taufactor and PuMA in one process.** Both link an OpenMP runtime and abort on the duplicate. Each tool gets its own process, which is also required for the timing reasons above.
- **`torch` from PyPI is CPU-only on Windows.** It must come from the CUDA index, as `pyproject.toml` specifies, or taufactor gets benchmarked on the CPU against a GPU competitor.
- **The Julia memory stage needs an interactive thread** (`-t N,1`) so the sampler has somewhere to run while the solver saturates the default pool. Without one it warns and falls back to readings taken either side of the solve.
- **PuMA's checkpoint allocates a full flux field.** Reading tortuosity off an iterate goes through PuMA's own `compute_flux`, whose fourth return value is a per-voxel flux vector — `(N, N, N, 3)` float64, so 192 MB at 200³ and about 1.5 GB at 400³ — allocated and discarded at every rung. Its wall time is excluded from the reported figure, but the allocator churn is not, and at the larger sizes it is a plausible source of a host `MemoryError` that would be recorded against PuMA rather than against the harness. There is no way around it without reimplementing a compiled routine. If the grid is ever pushed past 200³, watch for it.
