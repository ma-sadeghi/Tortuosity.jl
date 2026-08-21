"""Sample how many CPU cores each tool actually occupies while solving.

    pixi run python measure_occupancy.py

The paper claims our CPU path occupies about two cores and PuMA about one. That
number was sampled on a different machine from every other number in the
campaign, which is exactly the kind of detail a reviewer asks about. This
re-samples it on the benchmark host under the campaign's own settings --
`threads = "auto"`, so each tool takes the machine the way its own defaults let
it.

Occupancy is `psutil.cpu_percent` over the process *and its children*, divided by
100, sampled twice a second. Startup and teardown are trimmed before the summary:
a Julia process spends its first seconds compiling on one core and a PuMA process
spends its first seconds building a workspace, and neither is the solve. What is
reported is the median over the trimmed middle, which is the quantity the
sentence in the paper is about.

Neither harness takes an output path, so both write into the published results
files. This backs those files up, runs, restores them, and verifies the restore
by SHA-256 -- an occupancy sample must not perturb a single published number.
"""

import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import psutil

# Trim this fraction off each end before summarising: the head is compilation and
# workspace construction, the tail is teardown and writing results.
TRIM = 0.20
INTERVAL_S = 0.5

TOUCHED = [
    Path("results/timings/tortuosity-cpu-matrixfree.csv"),
    Path("results/timings/puma-cpu.csv"),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def sample(cmd, logpath):
    """Run `cmd`, sampling total CPU occupancy of the process tree until it exits.

    The child's output goes to a file rather than to /dev/null: a tool that dies
    on a signal reports only its exit code, and without its stderr there is
    nothing to diagnose from.
    """
    log = open(logpath, "wb")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    parent = psutil.Process(proc.pid)
    parent.cpu_percent(None)  # prime; the first call always returns 0.0
    samples = []
    started = time.perf_counter()
    while proc.poll() is None:
        time.sleep(INTERVAL_S)
        total = 0.0
        try:
            total += parent.cpu_percent(None)
            for child in parent.children(recursive=True):
                try:
                    total += child.cpu_percent(None)
                except psutil.Error:
                    pass
        except psutil.Error:
            break
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
    args = ap.parse_args()

    before = {p: digest(p) for p in TOUCHED}
    backups = {}
    for p in TOUCHED:
        if p.is_file():
            b = p.with_suffix(p.suffix + ".occupancy-backup")
            shutil.copy2(p, b)
            backups[p] = b
    print(f"backed up {len(backups)} results file(s)")

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
    }

    out = {
        "case": args.case,
        "host_physical_cores": psutil.cpu_count(logical=False),
        "host_logical_cores": psutil.cpu_count(logical=True),
        "tools": {},
    }
    try:
        for name, cmd in runs.items():
            print(f"running {name} ...", flush=True)
            child_log = f"results/occupancy-{name}.log"
            code, elapsed, samples = sample(cmd, child_log)
            out["tools"][name] = {
                "exit": code, "elapsed_s": round(elapsed, 1),
                "child_log": child_log, **summarise(samples)
            }
            print(f"  exit={code}  {elapsed:.1f}s  {out['tools'][name]}\n", flush=True)
    finally:
        for p, b in backups.items():
            shutil.move(str(b), str(p))
        restored = all(digest(p) == before[p] for p in backups)
        out["published_results_unchanged"] = restored
        print(f"results files restored and verified by SHA-256: {restored}")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
