"""Sample how many CPU cores each external CPU tool occupies while solving.

    pixi run python measure_occupancy.py
    pixi run python measure_occupancy.py --tools=porespy-cpu --out=results/x.json

The CPU margin combines algorithmic work reduction with each tool's own
parallelism, which is only interpretable if occupancy is sampled on the same
host as the timings. This samples it there under `threads = "auto"`, so each
tool takes the machine the way its defaults allow.

Occupancy is `psutil.cpu_percent` over the process *and its children*, divided by
100, sampled twice a second. Startup and teardown are trimmed before the summary. What is reported is the
median over the middle of each external tool's process.

Each child receives a temporary output root through the shared benchmark
configuration. Published results are never opened, so a concurrent campaign
cannot be overwritten when occupancy sampling finishes.
"""

import argparse
import json
import os
import shutil
import socket
import statistics
import sys
import tempfile
from pathlib import Path

import psutil

from benchkit.occupancy import (
    benchmark_measurement_lock,
    install_termination_handler,
    latest_environment,
    require_target_result,
    sample_cpu_occupancy,
    timestamp,
)

# Trim this fraction off each end before summarising: the head is compilation and
# workspace construction, the tail is teardown and writing results.
TRIM = 0.20
INTERVAL_S = 0.5


def pixi_binary():
    """Where `pixi` is, for a child that needs a different environment than ours.

    A `pixi run` environment does not carry pixi itself on `PATH`, so a tool
    launched from inside one cannot simply name it.
    """
    found = shutil.which("pixi") or Path.home() / ".pixi" / "bin" / "pixi"
    if not Path(found).is_file():
        raise SystemExit(f"cannot find the pixi executable (looked at {found})")
    return str(found)


def sample(cmd, logpath, output_dir):
    """Run one external tool under the shared sampler, in an isolated output root."""
    env = os.environ.copy()
    env["TORTUOSITY_BENCHMARK_OUTPUT_DIR"] = output_dir
    code, elapsed, samples = sample_cpu_occupancy(cmd, INTERVAL_S, logpath, env)
    provenance = latest_environment(output_dir)
    return code, elapsed, samples, provenance


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
    install_termination_handler()
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
        "host": socket.gethostname(),
        "started_at": timestamp(),
        "host_physical_cores": psutil.cpu_count(logical=False),
        "host_logical_cores": psutil.cpu_count(logical=True),
        "tools": {},
    }
    with benchmark_measurement_lock():
        with tempfile.TemporaryDirectory(
            prefix="tortuosity-occupancy-"
        ) as output_dir:
            for name, cmd in runs.items():
                print(f"running {name} ...", flush=True)
                child_log = f"results/occupancy-{name}.log"
                code, elapsed, samples, provenance = sample(
                    cmd, child_log, output_dir
                )
                if code != 0:
                    raise RuntimeError(f"{name} exited with status {code}")
                require_target_result(output_dir, name, args.case)
                out["tools"][name] = {
                    "exit": code, "elapsed_s": round(elapsed, 1),
                    "child_log": child_log,
                    "benchmark_environment": provenance,
                    **summarise(samples),
                }
                print(f"  exit={code}  {elapsed:.1f}s  "
                      f"{out['tools'][name]}\n", flush=True)
        out["results_mode"] = "isolated"
        out["published_results_unchanged"] = True
        out["completed_at"] = timestamp()
        print("published results were not opened by occupancy children")
        Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
