"""Estimate machine drift between the archived and re-measured GPU sweeps.

    pixi run python drift_check.py results/archive/pre-gpu-remeasure-<date>

Compares only rows that did identical work — same case, same variant, same
ladder rung — so the ratio is machine drift plus the code change, not a
different amount of solving. Rows whose rung is unchanged across the two runs
isolate drift; the speedups the paper quotes are only sound to the extent this
is small, because taufactor's GPU timings were recorded in the older session.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from benchkit import figures as fig

archive = Path(sys.argv[1] if len(sys.argv) > 1 else "results/archive")
KEY = ["case_id", "variant", "knob"]


def load(path):
    frame = pd.read_csv(path)
    return frame[frame.time_s > 0]


for variant in ("matrixfree", "assembled"):
    old_path = archive / "timings" / f"tortuosity-gpu-{variant}.csv"
    new_path = Path("results/timings") / f"tortuosity-gpu-{variant}.csv"
    if not (old_path.exists() and new_path.exists()):
        print(f"skipping {variant}: missing {old_path if not old_path.exists() else new_path}")
        continue

    old, new = load(old_path), load(new_path)
    merged = old.merge(new, on=KEY, suffixes=("_old", "_new"))
    if merged.empty:
        print(f"{variant}: no rows share a case and rung")
        continue

    ratio = merged["time_s_old"] / merged["time_s_new"]
    print(f"\n== {variant}: {len(merged)} rows at an identical case and rung ==")
    print(f"   speedup old/new  geomean {fig.geomean(ratio.tolist()):.3f}   "
          f"min {ratio.min():.3f}   max {ratio.max():.3f}")

    # The cheapest rungs are dominated by fixed setup, which is exactly what the
    # readout and percolation-check commits changed, so split them out. `knob` is
    # the iteration cap on a geometric ladder, not its index, so rank it.
    ladder = sorted(merged["knob"].unique())
    rung = merged["knob"].map({k: i for i, k in enumerate(ladder)})
    for lo, hi, label in ((0, 1, "two cheapest rungs, setup dominated"),
                          (2, len(ladder), "rung 3 and up, solve dominated")):
        part = merged[(rung >= lo) & (rung <= hi)]
        if len(part):
            r = (part["time_s_old"] / part["time_s_new"]).tolist()
            print(f"   {label}: n={len(part):4d}  geomean {fig.geomean(r):.3f}")
    assert len(merged) == sum(len(merged[(rung >= lo) & (rung <= hi)])
                              for lo, hi in ((0, 1), (2, len(ladder)))), "rungs do not partition"

    by_size = merged.groupby("size_old").apply(
        lambda g: fig.geomean((g["time_s_old"] / g["time_s_new"]).tolist()),
        include_groups=False,
    )
    print("   by size:", {int(k): round(float(v), 3) for k, v in by_size.items()})

    same_tau = np.isclose(merged["tau_old"], merged["tau_new"], rtol=1e-3)
    print(f"   tau agrees within 0.1% on {same_tau.sum()}/{len(merged)} shared rows")
