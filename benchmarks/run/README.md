# Running the campaign on a rented machine

**Driving a real campaign? Follow [`ORCHESTRATION.md`](ORCHESTRATION.md)** — an
ordered procedure with a check after every step. This file explains why the
stages are shaped the way they are. That one is what to type, in order.

The stages are split so that a rented machine only ever pays for measurement.
Data generation (`campaign.sh`) writes CSVs and draws nothing. Post-processing
(`figures.sh`) reads those CSVs and needs no GPU, no images and no solver. The
same scripts run locally — the only difference is how long they take.

## What has to travel

Very little, in either direction.

| | Direction | Size | Why |
|---|---|---|---|
| the repository | up | small | `git clone`, or `git push` a branch and pull it there |
| `results/*.csv` | down | kilobytes | the entire measured dataset |
| `data/images/manifest.csv` | down | kilobytes | the record of what geometry was measured |
| `logs/` | down, optional | megabytes | for diagnosing a stage that failed |

The image store itself never moves. It reaches tens of gigabytes at the full
grid, and it does not need to. Images are generated deterministically from the
seed in `config.toml`, and every one carries a SHA-256 in the manifest, so a
store rebuilt on another machine is *checked* to be identical rather than assumed
to be. Any stage that loads an image verifies it and refuses to proceed on a
mismatch.

## The sequence

```bash
# on the rented machine
git clone <this repo> && cd Tortuosity/benchmarks
./run/setup.sh                        # resolve both environments, check the GPU

./run/campaign.sh --grid=smoke        # ~minutes: same shapes, one tenth the size
./run/campaign.sh --grid=full         # the real grid

# back on a laptop
rsync -av pod:Tortuosity/benchmarks/results/ results/
rsync -av pod:Tortuosity/benchmarks/data/images/manifest.csv data/images/
./run/figures.sh
```

**Run the smoke grid first, every time.** It is the full grid divided by ten in
each dimension — the same 75 cases, the same stages, the same code paths — and it
finishes in minutes. It exists so that a broken flag, a missing dependency or an
unwritable directory is found before an hourly meter is running.

## Stopping early, and picking up again

Every stage resumes from its own results file, and cases run cheapest first. So a
campaign can be interrupted at any point and re-run with the same command. It
continues rather than restarting, and what it has already measured is the small
end of the grid rather than a scatter of half-finished large cases.

A sweep is only treated as complete once one of its rows carries a `stop_reason`,
so a case interrupted halfway up its ladder is redone rather than silently
accepted as converged.

To measure part of the grid, pass the selection through:

```bash
./run/campaign.sh --grid=full --stages=timings --tools=tortuosity --sizes=800,1000
./run/campaign.sh --grid=full --stages=timings --tools=puma,porespy --sizes=200
./run/campaign.sh --grid=full --stages=memory --blobiness=1.0
```

Add `--dry-run` to any of it to see the case list without measuring anything.

## Choosing the machine

- **A GPU with as much memory as possible.** Device memory, not compute, is what
  ends the size sweep: the assembled operator and taufactor both hold quantities
  proportional to the whole voxel grid, so 1000³ is where they stop.
- **Plenty of host RAM.** The ground-truth stage solves in `Float64` on the CPU,
  which at 1000³ and high porosity is several tens of gigabytes of Krylov vectors
  alone. PuMA also runs entirely in host memory.
- **Cores set what the CPU numbers mean.** Nothing is pinned: every tool takes
  the whole machine, the way a user running it would, and every row records the
  thread count it actually got. The three parallelise very differently, and that
  is part of what the CPU comparison reports.

## What each stage costs

Roughly, and worth knowing before renting anything:

| stage | dominated by | notes |
|---|---|---|
| `images` | the largest sizes | one pass, deterministic, resumable |
| `references` | `Float64` CPU solves | by far the most expensive; each value is written the moment it is solved |
| `timings` | the accuracy ladders | six configurations by default, the GPU pair first since the paper needs those most; PuMA and PoreSpy are opt-in and are run at 200³ alone |
| `memory` | nothing much | one short fixed-length solve per case |

If the budget is uncertain, run `--stages=references` on its own first: every
later stage depends on those values, and they are the part that cannot be
cheaply redone.
