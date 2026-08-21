"""Sample our own CPU path's core occupancy, at two sizes.

Companion to `measure_occupancy.py`, which covers PuMA. Split out because the two
tools need different sampling: a PuMA solve at $200^3$ runs for six minutes and
gives hundreds of samples, while ours finishes the same case in three seconds and
gives six. Six samples is not a measurement, so this samples ten times a second at
the small size and adds a larger case where the solve dominates outright.

Julia has been seen to die on SIGSEGV during exit cleanup under this harness,
after the solve has completed and the results row has been written. The exit code
is therefore reported but not treated as a failure: what matters is whether the
samples cover a real solve, which the sample count and the child log show.
"""

import argparse
import hashlib
import os
import json
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import psutil

TRIM = 0.20
TOUCHED = Path("results/timings/tortuosity-cpu-matrixfree.csv")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


# pixi exports LD_LIBRARY_PATH and CONDA_PREFIX pointing at its own lib directory.
# A child Julia inherits them, and once it loads the full package stack those
# libraries shadow the ones Julia ships with, which ends the process on SIGSEGV
# before it prints a line. Hand the child a clean environment instead.
POISON = ("LD_LIBRARY_PATH", "LD_PRELOAD", "CONDA_PREFIX", "PYTHONHOME", "PYTHONPATH")


def clean_env():
    return {k: v for k, v in os.environ.items() if k not in POISON}


def sample(cmd, interval, logpath):
    log = open(logpath, "wb")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=clean_env())
    parent = psutil.Process(proc.pid)
    parent.cpu_percent(None)
    samples = []
    started = time.perf_counter()
    while proc.poll() is None:
        time.sleep(interval)
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
    if len(samples) < 8:
        return {"note": f"only {len(samples)} samples -- not a measurement"}
    lo = int(len(samples) * TRIM)
    hi = max(lo + 1, int(len(samples) * (1 - TRIM)))
    mid = samples[lo:hi]
    return {
        "n_samples": len(samples), "n_used": len(mid),
        "median_cores": round(statistics.median(mid), 2),
        "mean_cores": round(statistics.fmean(mid), 2),
        "peak_cores": round(max(samples), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/core-occupancy-ours.json")
    args = ap.parse_args()

    before = digest(TOUCHED)
    backup = TOUCHED.with_suffix(TOUCHED.suffix + ".occupancy-backup")
    shutil.copy2(TOUCHED, backup)

    plan = [
        # (label, case, sampling interval): the small case needs a fast interval
        # to yield enough samples, the large one does not.
        ("n200_b100_p040", "n200_b100_p040", 0.1),
        ("n600_b100_p040", "n600_b100_p040", 0.5),
    ]
    out = {"tool": "tortuosity-cpu-matrixfree",
           "host_physical_cores": psutil.cpu_count(logical=False),
           "host_logical_cores": psutil.cpu_count(logical=True),
           "cases": {}}
    try:
        for label, case, interval in plan:
            cmd = ["julia", "-t", "auto", "--project=.", "bench_tortuosity.jl",
                   "--device=cpu", "--operator=matrixfree", "--measure=time",
                   f"--cases={case}", "--overwrite"]
            child_log = f"results/occupancy-ours-{label}.log"
            print(f"running {label} (interval {interval}s) ...", flush=True)
            code, elapsed, samples = sample(cmd, interval, child_log)
            out["cases"][label] = {"exit": code, "elapsed_s": round(elapsed, 1),
                                   "interval_s": interval, "child_log": child_log,
                                   **summarise(samples)}
            print(f"  exit={code} {elapsed:.1f}s {out['cases'][label]}\n", flush=True)
    finally:
        shutil.move(str(backup), str(TOUCHED))
        out["published_results_unchanged"] = digest(TOUCHED) == before
        print(f"results restored and verified: {out['published_results_unchanged']}")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
