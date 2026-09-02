"""Sample how many CPU cores each tool actually occupies while solving.

    pixi run python measure_occupancy.py
    pixi run python measure_occupancy.py --tools=porespy-cpu --out=results/x.json

The paper argues that the CPU margin is won on less of the machine rather than
more, which is only worth saying if the occupancy behind it was sampled on the
same host as everything else. This samples it there, under the campaign's own
settings -- `threads = "auto"`, so each tool takes the machine the way its own
defaults let it.

Occupancy is `psutil.cpu_percent` over the process *and its children*, divided by
100, sampled twice a second. Startup and teardown are trimmed before the summary:
a Julia process spends its first seconds compiling on one core and a PuMA process
spends its first seconds building a workspace, and neither is the solve. What is
reported is the median over the trimmed middle, which is the quantity the
sentence in the paper is about.

No harness here takes an output path, so each writes into the published results
files. This backs those files up, runs, restores them, and verifies the restore
by SHA-256 -- an occupancy sample must not perturb a single published number.
"""

import argparse
import contextlib
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import psutil

from benchkit.occupancy import published_results_lock

# Trim this fraction off each end before summarising: the head is compilation and
# workspace construction, the tail is teardown and writing results.
TRIM = 0.20
INTERVAL_S = 0.5

TOUCHED = [
    Path("results/timings/tortuosity-cpu-matrixfree-hostcg.csv"),
    Path("results/timings/puma-cpu.csv"),
    Path("results/timings/porespy-cpu.csv"),
    Path("results/environment.csv"),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def pixi_binary():
    """Where `pixi` is, for a child that needs a different environment than ours.

    A `pixi run` environment does not carry pixi itself on `PATH`, so a tool
    launched from inside one cannot simply name it.
    """
    found = shutil.which("pixi") or Path.home() / ".pixi" / "bin" / "pixi"
    if not Path(found).is_file():
        raise SystemExit(f"cannot find the pixi executable (looked at {found})")
    return str(found)


def sample(cmd, logpath):
    """Run `cmd`, sampling total CPU occupancy of the process tree until it exits.

    The child's output goes to a file rather than to /dev/null: a tool that dies
    on a signal reports only its exit code, and without its stderr there is
    nothing to diagnose from.

    Every process is kept and reused between samples. `cpu_percent` reports the
    share used since that same object's previous call, so its first call always
    returns 0.0; building the child objects afresh each iteration would make
    every call a first call and report a solve that used no CPU at all. This
    matters only when the work happens below the process we launched — a tool run
    through `pixi run` — which is why it went unnoticed while every tool here was
    launched directly.
    """
    log = open(logpath, "wb")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    parent = psutil.Process(proc.pid)
    tracked = {parent.pid: parent}
    parent.cpu_percent(None)  # prime; the first call always returns 0.0
    samples = []
    started = time.perf_counter()
    while proc.poll() is None:
        time.sleep(INTERVAL_S)
        try:
            for child in parent.children(recursive=True):
                if child.pid not in tracked:
                    tracked[child.pid] = child
                    with contextlib.suppress(psutil.Error):
                        child.cpu_percent(None)  # prime it too
        except psutil.Error:
            pass
        total = 0.0
        for pid, process in list(tracked.items()):
            try:
                total += process.cpu_percent(None)
            except psutil.Error:
                del tracked[pid]  # exited between one sample and the next
        samples.append(total / 100.0)
    log.close()
    return proc.returncode, time.perf_counter() - started, samples


def summarise(samples):
    if len(samples) < 5:
        return {"note": f"only {len(samples)} samples; solve too short to summarise"}
    lo = int(len(samples) * TRIM)
    hi = max(lo + 1, int(len(samples) * (1 - TRIM)))
    mid = samples[lo:hi]
    return {
        "n_samples": len(samples),
        "n_used": len(mid),
        "median_cores": round(statistics.median(mid), 2),
        "mean_cores": round(statistics.fmean(mid), 2),
        "peak_cores": round(max(samples), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="n200_b100_p040")
    ap.add_argument("--out", default="results/core-occupancy.json")
    # Sampling one tool must not cost the others their numbers. `julia` is not on
    # the benchmark host's path under `pixi run`, so a full run replaces a good
    # Julia measurement with a crashed one; ours is sampled by
    # `measure_occupancy_ours.py` instead. Pair this with `--out`.
    ap.add_argument("--tools", help="comma-separated subset to sample (default: all)")
    args = ap.parse_args()

    print(f"host: {psutil.cpu_count(logical=False)} physical / "
          f"{psutil.cpu_count(logical=True)} logical cores")
    print(f"case: {args.case}\n")

    runs = {
        "tortuosity-cpu-matrixfree": [
            "julia", "-t", "auto", "--project=.", "bench_tortuosity.jl",
            "--device=cpu", "--operator=matrixfree", "--measure=time",
            f"--cases={args.case}", "--overwrite",
        ],
        "puma-cpu": [
            sys.executable, "bench_puma.py", "--measure=time",
            f"--cases={args.case}", "--overwrite",
        ],
        # PoreSpy resolves in an environment of its own, so it cannot be launched
        # with this process's interpreter the way the other two are. `pixi run`
        # adds a parent process, which costs nothing here: the sampler already
        # walks the tree recursively, and pixi itself is idle while the child
        # solves. `pixi` is resolved by absolute path because a `pixi run`
        # environment does not put pixi itself on `PATH`.
        "porespy-cpu": [
            pixi_binary(), "run", "-e", "porespy", "python", "bench_porespy.py",
            "--measure=time", f"--cases={args.case}", "--overwrite",
        ],
    }

    if args.tools:
        wanted = [name.strip() for name in args.tools.split(",")]
        unknown = [name for name in wanted if name not in runs]
        if unknown:
            raise SystemExit(f"unknown tool(s) {', '.join(unknown)}; "
                             f"this script runs {', '.join(runs)}")
        runs = {name: cmd for name, cmd in runs.items() if name in wanted}

    out = {
        "case": args.case,
        "host_physical_cores": psutil.cpu_count(logical=False),
        "host_logical_cores": psutil.cpu_count(logical=True),
        "tools": {},
    }
    with published_results_lock():
        before = {p: digest(p) for p in TOUCHED}
        backups = {}
        for p in TOUCHED:
            backup = p.with_suffix(p.suffix + ".occupancy-backup")
            if backup.is_file():
                raise RuntimeError(f"stale occupancy backup exists: {backup}")
        try:
            for p in TOUCHED:
                if p.is_file():
                    backup = p.with_suffix(p.suffix + ".occupancy-backup")
                    shutil.copy2(p, backup)
                    backups[p] = backup
            print(f"backed up {len(backups)} results file(s)")
            for name, cmd in runs.items():
                print(f"running {name} ...", flush=True)
                child_log = f"results/occupancy-{name}.log"
                code, elapsed, samples = sample(cmd, child_log)
                out["tools"][name] = {
                    "exit": code, "elapsed_s": round(elapsed, 1),
                    "child_log": child_log, **summarise(samples)
                }
                print(f"  exit={code}  {elapsed:.1f}s  "
                      f"{out['tools'][name]}\n", flush=True)
        finally:
            for p in TOUCHED:
                if p in backups:
                    shutil.move(str(backups[p]), str(p))
                elif before[p] is None and p.is_file():
                    p.unlink()
                else:
                    partial = p.with_suffix(p.suffix + ".occupancy-backup")
                    if partial.is_file():
                        partial.unlink()
            restored = all(digest(p) == before[p] for p in TOUCHED)
            out["published_results_unchanged"] = restored
            print(f"results files restored and verified by SHA-256: {restored}")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
