"""Sample our own CPU path's core occupancy, at two sizes.

Companion to `measure_occupancy.py`, which covers PuMA. Split out because the two
tools need different sampling: a PuMA solve at $200^3$ runs for six minutes,
while ours finishes in seconds. The Julia child marks the benchmark-case phase
after warm-up, samples outside it are discarded, and the small case is sampled
ten times a second.

Julia has been seen to die on SIGSEGV during exit cleanup under this harness,
after the solve has completed and the results row has been written. The exit code
is therefore reported but not treated as a failure: what matters is whether the
samples cover a real solve, which the sample count and the child log show.
"""

import argparse
import csv
import json
import os
import socket
import statistics
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import psutil

from benchkit.occupancy import (
    benchmark_measurement_lock,
    install_termination_handler,
    measurement_process,
)

# pixi exports LD_LIBRARY_PATH and CONDA_PREFIX pointing at its own lib directory.
# A child Julia inherits them, and once it loads the full package stack those
# libraries shadow the ones Julia ships with, which ends the process on SIGSEGV
# before it prints a line. Hand the child a clean environment instead.
POISON = ("LD_LIBRARY_PATH", "LD_PRELOAD", "CONDA_PREFIX", "PYTHONHOME", "PYTHONPATH")


def clean_env(output_dir, marker):
    env = {k: v for k, v in os.environ.items() if k not in POISON}
    env["TORTUOSITY_BENCHMARK_OUTPUT_DIR"] = output_dir
    env["TORTUOSITY_BENCHMARK_PHASE_MARKER"] = str(marker)
    return env


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def latest_environment(output_dir):
    """Runtime provenance written by the child that just completed."""
    path = Path(output_dir) / "environment.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("benchmark child wrote no environment record")
    keys = (
        "measured_at", "host", "stage", "tool", "device", "variant",
        "runtime", "runtime_version", "cpu_threads", "accelerator", "notes",
    )
    return {key: rows[-1][key] for key in keys}


def require_target_result(output_dir, case):
    """Require the isolated case to have reached the benchmark target."""
    path = (
        Path(output_dir)
        / "timings"
        / "tortuosity-cpu-matrixfree-hostcg.csv"
    )
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["case_id"] == case and row["stop_reason"]
        ]
    if len(rows) != 1 or rows[0]["stop_reason"] != "target_reached":
        raise RuntimeError(f"{case} did not reach the target in isolated output")


def sample(cmd, interval, logpath, output_dir, marker, case):
    start_marker = Path(f"{marker}.start")
    active_marker = Path(f"{marker}.active")
    transition_marker = Path(f"{marker}.transitions")
    success_marker = Path(f"{marker}.success")
    end_marker = Path(f"{marker}.end")
    with open(logpath, "wb") as log:
        with measurement_process(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=clean_env(output_dir, marker),
        ) as (proc, tracked):
            parent = psutil.Process(proc.pid)
            tracked[parent.pid] = parent
            parent.cpu_percent(None)
            samples = []
            started = time.perf_counter()
            while proc.poll() is None:
                active_before = active_marker.is_file()
                transitions_before = (
                    transition_marker.stat().st_size
                    if transition_marker.is_file()
                    else 0
                )
                time.sleep(interval)
                total = 0.0
                try:
                    total += parent.cpu_percent(None)
                    for child in parent.children(recursive=True):
                        if child.pid not in tracked:
                            tracked[child.pid] = child
                            try:
                                child.cpu_percent(None)
                            except psutil.Error:
                                pass
                            continue
                        try:
                            total += child.cpu_percent(None)
                        except psutil.Error:
                            pass
                except psutil.Error:
                    break
                active_after = active_marker.is_file()
                transitions_after = (
                    transition_marker.stat().st_size
                    if transition_marker.is_file()
                    else 0
                )
                if (
                    active_before
                    and active_after
                    and transitions_before == transitions_after
                ):
                    samples.append(total / 100.0)
            code = proc.wait()
    if (
        not start_marker.is_file()
        or not success_marker.is_file()
        or not end_marker.is_file()
        or not transition_marker.is_file()
    ):
        raise RuntimeError("benchmark child did not mark its measured phase")
    require_target_result(output_dir, case)
    provenance = latest_environment(output_dir)
    return code, time.perf_counter() - started, samples, provenance


def summarise(samples):
    if len(samples) < 8:
        return {"note": f"only {len(samples)} samples -- not a measurement"}
    return {
        "n_samples": len(samples), "n_used": len(samples),
        "median_cores": round(statistics.median(samples), 2),
        "mean_cores": round(statistics.fmean(samples), 2),
        "peak_cores": round(max(samples), 2),
    }


def main():
    install_termination_handler()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/core-occupancy-ours.json")
    args = ap.parse_args()

    plan = [
        # (label, case, sampling interval): the small case needs a fast interval
        # to yield enough samples, the large one does not.
        ("n200_b100_p040", "n200_b100_p040", 0.1),
        ("n600_b100_p040", "n600_b100_p040", 0.5),
    ]
    out = {
        "tool": "tortuosity-cpu-matrixfree-hostcg",
        "host": socket.gethostname(),
        "started_at": timestamp(),
        "host_physical_cores": psutil.cpu_count(logical=False),
        "host_logical_cores": psutil.cpu_count(logical=True),
        "cases": {},
    }
    with benchmark_measurement_lock():
        with tempfile.TemporaryDirectory(
            prefix="tortuosity-occupancy-"
        ) as output_dir:
            for label, case, interval in plan:
                cmd = ["julia", "-t", "auto", "--project=.",
                       "bench_tortuosity.jl", "--device=cpu",
                       "--operator=matrixfree", "--measure=time",
                       f"--cases={case}", "--overwrite"]
                child_log = f"results/occupancy-ours-{label}.log"
                marker = Path(output_dir) / f"phase-{label}"
                print(f"running {label} (interval {interval}s) ...", flush=True)
                code, elapsed, samples, provenance = sample(
                    cmd, interval, child_log, output_dir, marker, case
                )
                out["cases"][label] = {
                    "exit": code, "elapsed_s": round(elapsed, 1),
                    "interval_s": interval,
                    "child_log": child_log,
                    "benchmark_environment": provenance,
                    **summarise(samples),
                }
                print(f"  exit={code} {elapsed:.1f}s "
                      f"{out['cases'][label]}\n", flush=True)
        out["results_mode"] = "isolated"
        out["published_results_unchanged"] = True
        out["completed_at"] = timestamp()
        print("published results were not opened by occupancy children")
        Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
