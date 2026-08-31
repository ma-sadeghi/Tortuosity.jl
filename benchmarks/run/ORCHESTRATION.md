# Orchestration runbook

An ordered procedure for driving a full campaign on a rented machine. Written for
whoever — or whatever — is holding the SSH key. Every step has a check that must
pass before the next one starts, because the failures that matter here are silent.
A contaminated timing looks exactly like a real one once it is in the CSV.

`README.md` in this directory explains *why* the stages are split. This file is
the *what to type*, in order.

## Before anything

Collect and record these. They belong in the campaign notes, not in anyone's head.

| | why it matters |
|---|---|
| host, user, SSH key | — |
| GPU model and memory | device memory, not compute, is what ends the size sweep |
| host RAM | the ground-truth stage is `Float64` on the CPU; at 1000³ and ε=0.95 that is roughly 8 GB per Krylov vector |
| free disk | the image store reaches about 27 GB at the full grid |
| CPU core count | every CPU *measurement* takes the whole machine, so it sets what the CPU numbers mean |

**The whole timing dataset must come from one machine.** Timings measured on two
hosts cannot be compared, and nothing downstream will stop you mixing them —
`results/environment.csv` records the host per batch so the mistake is at least
auditable afterwards. If a pod dies mid-campaign, the timings already collected on
it are only usable if the replacement is the same instance type. Even then it
is safer to re-run the timing stages from scratch.

## 1. Get the code there

```bash
ssh POD
git clone https://github.com/ma-sadeghi/Tortuosity.jl.git   # or push a branch and clone that
cd Tortuosity.jl/benchmarks
```

**Run everything under `tmux` or `screen`.** A campaign runs for hours and an SSH
drop kills a foreground job with it — leaving a half-written results file that
resume handles correctly, but also leaving the meter running with nothing on it.

```bash
tmux new -s campaign
```

## 2. Set up, and prove the machine works

```bash
./run/setup.sh
```

**Gate.** `setup.sh` prints the Julia and CUDA versions, the device name, the torch
CUDA availability, and confirms taufactor exposes `Solver` and pumapy imports. Do
not continue past a warning here. In particular:

- `CUDA is not functional` — every GPU stage will refuse to run. Fix the driver
  before spending anything.
- `torch.cuda.is_available(): False` — taufactor would be benchmarked on the CPU
  against a GPU competitor. On Linux this usually means the wrong torch wheel.
  `pyproject.toml` pins the CUDA index for a reason.

## 3. Smoke grid, every time

```bash
./run/campaign.sh --grid=smoke
```

Minutes, not hours: the same 75 cases and the same code paths at one tenth the
size in each dimension. It exists so that a broken flag, a missing dependency or
an unwritable directory is found before an hourly meter is running.

**Gate.** All of these must hold before going further:

```bash
grep -c . data/images/manifest.csv          # 76 = header + 75 cases
wc -l results/references.csv                # 73 = header + 72 (three smoke cases do not percolate)
ls results/timings results/memory           # seven files each
grep -h . logs/*.log | grep -i "error\|refus\|failed" | head
```

Then draw the figures once, locally or there, and look at them. A figure that
renders is not the same as a figure that is right.

## 4. Ground truth, on its own, first

```bash
./run/campaign.sh --grid=full --stages=images,references
```

This is the expensive, irreplaceable half. Every later stage depends on these
values and nothing else does. A campaign that runs out of budget here has still
produced the thing that cannot be cheaply redone. Each reference is appended the
moment it is solved, so an interruption costs at most one case.

**Gate.**

```bash
wc -l results/references.csv                # header + one row per percolating case
awk -F, 'NR>1 && $7+0 <= 0' results/references.csv   # must print nothing: tau <= 0 is not a solution
```

If a size is unaffordable, stop here and continue with the sizes you have:
everything downstream takes `--sizes=`, and a missing size is reported as missing
rather than guessed at.

## 5. Measurement

```bash
./run/campaign.sh --grid=full --stages=timings,memory
```

Fourteen stages, serially, ordered so that an interrupted night leaves the results
the paper most needs. Do not add parallelism. The stages contend for the same GPU
and the same cores, and concurrency corrupts the very timings this exists to
measure. `campaign.sh` takes a PID lock and refuses to start beside a live
campaign, but that only catches the obvious case.

PuMA and PoreSpy are not in the default tool list. Both are CPU only, both are an
order of magnitude or more off the pace at the smallest size, and sweeping either
over the size grid costs days to settle a question already settled at 200³.
They are run on purpose, at one size:

```bash
./run/campaign.sh --grid=full --stages=timings --tools=puma,porespy --sizes=200
```

To split the rest of the work over several sittings, pass the stage and tool
through:

```bash
./run/campaign.sh --grid=full --stages=timings --tools=tortuosity
./run/campaign.sh --grid=full --stages=timings --tools=taufactor
./run/campaign.sh --grid=full --stages=memory
```

**Watch for.** `logs/<stage>.log` is written per stage and the exit code is printed
after each. A non-zero exit is not fatal by design — a configuration that dies at
the largest size still leaves every smaller size measured — but read the log before
assuming that is what happened.

Expect and do not treat as failures:

- `oom` rows from the assembled operator and from taufactor at the largest sizes.
  Holding a quantity proportional to the whole voxel grid is what ends those paths,
  and where it ends is a result.
- `timeout` rows from PuMA at the largest sizes. Its conjugate gradient is
  effectively serial even with the whole machine available, and the grid is
  large. That too is a result.
- Blank cells in a figure. Every one has a `stop_reason` in the CSV explaining it.

## 6. Stopping, killing and resuming

To stop cleanly, `Ctrl-C` the campaign inside its tmux window and confirm nothing
survived:

```bash
pgrep -af 'campaign.sh|bench_tortuosity|bench_taufactor|bench_puma'
```

**Killing the solver process is not enough.** The campaign script waits on its
child. If you kill only the child, the script advances to the next stage, so a
replacement campaign started afterwards runs concurrently with it. Match on the
command line and kill the shell too:

```bash
pkill -f 'campaign.sh|bench_tortuosity|bench_taufactor|bench_puma'
rm -f logs/.campaign.lock
```

Resuming is just re-running the same command. Every stage resumes from its own
results file, cases run cheapest first, and a sweep counts as complete only once
one of its rows carries a `stop_reason` — so a case interrupted halfway up its
ladder is redone rather than silently accepted as converged.

**After any interruption, check coverage rather than values.** A timing taken while
something else was running is indistinguishable from a clean one by inspection. The
tell is a results file containing more cases than the run had time to complete.

## 7. Bring the results home

Only the CSVs and the manifest travel. The image store does not: it is regenerable
from the seed in `config.toml`, and every image carries a SHA-256 in the manifest,
so a rebuild elsewhere is checked identical rather than assumed to be.

```bash
rsync -av POD:Tortuosity.jl/benchmarks/results/ results/
rsync -av POD:Tortuosity.jl/benchmarks/data/images/manifest.csv data/images/
rsync -av POD:Tortuosity.jl/benchmarks/logs/ logs/      # optional, for diagnosing a stage
```

Then, locally:

```bash
./run/figures.sh
```

No GPU, no images, no solver. Redraw as often as the paper needs.

**Before releasing the pod**, confirm the results are actually on the local disk
and readable — `wc -l results/timings/*.csv results/memory/*.csv` and one
`./run/figures.sh` that succeeds. A pod terminated with the only copy of a
campaign on it is the one failure nothing here can recover from.

## Things not to do

- **Do not edit `config.toml` mid-campaign.** The grid, the ladders and the thread
  budget are recorded in the rows already written. Changing one silently produces a
  file whose rows were measured under two different definitions.
- **Do not run the stages in parallel**, on any machine, for any reason.
- **Do not run anything else on the machine** while a timing stage is running.
  Post-processing included — it is cheap, but it is not free, and it can wait.
- **Do not pin a timing or memory stage to fewer threads than the machine has.**
  Every tool takes the whole machine, which is what a user running it gets, and
  every row records what it actually got. Pinning one tool and not another is how
  the earlier campaign came to record `cpu_threads = 1` for runs where OpenBLAS
  was quietly using eight.
- **Do not merge results measured on different machines** into one timing figure.
