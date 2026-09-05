"""Individual GPU cells and the one projected GPU value cited in the docs prose."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from benchkit import config as bkconfig
from benchkit import figures as fig

TARGET = 0.001

cfg = bkconfig.load_config()
timings = fig.load_results(cfg.resultsdir, "timings")
memory = fig.load_results(cfg.resultsdir, "memory")

porosities = sorted(timings["porosity_target"].unique())
sizes = sorted(timings[timings.device == "gpu"]["size"].unique())
ours = fig.series_label(*fig.REFERENCE_SERIES)
tau = fig.series_label("taufactor", "sor")


def detail(case_id, series):
    """The row at which a tool first reaches the target, with its iteration count."""
    sub = timings[(timings.case_id == case_id) & (timings.device == "gpu")
                  & (timings["series"] == series) & (timings.rel_error <= TARGET)]
    if sub.empty:
        return None
    row = sub.loc[sub["time_s"].idxmin()]
    return row


for case in ("n600_b100_p020", "n200_b200_p095"):
    print(f"### {case}")
    for label, series in (("ours", ours), ("taufactor", tau)):
        row = detail(case, series)
        if row is None:
            print(f"   {label:10s}  never reached the target")
            continue
        print(f"   {label:10s}  time {row['time_s']:9.3f} s   knob {row['knob']:>6}   "
              f"rel_err {row['rel_error']:.2e}")
    print()

print("### Projected GPU cell: taufactor, blobiness 1.0, eps ~ 0.2, N = 1000")
theirs = fig.target_times(timings, TARGET, device="gpu", series=tau, sizes=sizes,
                          porosities=porosities, blobiness=1.0)


def node_count(size, por):
    sub = timings[(timings["size"] == size)
                  & np.isclose(timings["porosity_target"], por)
                  & np.isclose(timings["blobiness"], 1.0)]
    return sub["nnodes"].iloc[0] if len(sub) else np.nan


por = 0.2
measured = [n for n in sizes if np.isfinite(theirs[(n, por)])]
counts = [node_count(n, por) for n in measured]
a, p, _ = fig.power_law(counts, [theirs[(n, por)] for n in measured])
est = a * node_count(1000, por) ** p
print(f"   fitted on sizes {measured}, exponent {p:.3f}")
print(f"   projected taufactor time at N=1000: {est:.0f} s")
print(f"   measured floor (timed out, did not converge): see the timeout row")

print()
print("### Peak device memory, matrix-free, blobiness 1.0, N = 1000")
mem = memory[(memory.device == "gpu") & (memory["variant"] == "matrixfree")
             & np.isclose(memory["blobiness"], 1.0)
             & (memory["size"] == 1000)]
for _, r in mem.sort_values("porosity_target").iterrows():
    gib = r["peak_device_bytes"] / 1024 ** 3
    gb = r["peak_device_bytes"] / 1000 ** 3
    print(f"   eps~{r['porosity_target']:.2f}: {gib:.3f} GiB   ({gb:.3f} GB)")
