"""The GPU device-memory table in docs/src/benchmark.md, in the page's own shape.

The "as shipped" rows come from the current campaign; the "solve only" rows come
from the pre-refinement archive, which this campaign did not re-measure. Printing
both together is what makes the refinement delta checkable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from benchkit import config as bkconfig

cfg = bkconfig.load_config()
RESULTS = Path(cfg.resultsdir)
PRE_REFINE = RESULTS / "archive" / "pre-refine-2026-08-20" / "memory"

POROSITIES = (0.2, 0.4, 0.6, 0.8, 0.95)
SIZES = (200, 400, 600, 800, 1000)
BLOB = 1.0


def load(path):
    df = pd.read_csv(path)
    return df[np.isclose(df["blobiness"], BLOB)]


def lookup(df, size, por, column="peak_device_bytes"):
    row = df[(df["size"] == size) & np.isclose(df["porosity_target"], por)]
    if row.empty:
        return None
    r = row.iloc[0]
    if str(r["status"]) != "ok":
        return "oom"
    return r[column]


shipped_mf = load(RESULTS / "memory" / "tortuosity-gpu-matrixfree.csv")
shipped_as = load(RESULTS / "memory" / "tortuosity-gpu-assembled.csv")
solve_mf = load(PRE_REFINE / "tortuosity-gpu-matrixfree.csv") if PRE_REFINE.exists() else None


def cells(df, size, column="peak_device_bytes"):
    out = []
    for por in POROSITIES:
        v = lookup(df, size, por, column)
        out.append("*oom*" if v == "oom" else ("—" if v is None else f"{v / 1000 ** 3:.3f}"))
    return out


print("### Peak device memory, blobiness 1.0, GB (as the docs table prints it)")
for size in SIZES:
    if solve_mf is not None:
        print(f"| {size} | matrix-free, solve only | " + " | ".join(cells(solve_mf, size)) + " |")
    print(f"| | matrix-free, as shipped | " + " | ".join(cells(shipped_mf, size)) + " |")
    print(f"| | assembled, as shipped | " + " | ".join(cells(shipped_as, size)) + " |")

if solve_mf is None:
    print()
    print(f"NOTE: pre-refine archive not found at {PRE_REFINE}")
    raise SystemExit(0)

print()
print("### Refinement delta, bytes per pore node (as shipped minus solve only)")
deltas = []
for size in SIZES:
    for por in POROSITIES:
        a = lookup(shipped_mf, size, por)
        b = lookup(solve_mf, size, por)
        if a in (None, "oom") or b in (None, "oom"):
            continue
        row = shipped_mf[(shipped_mf["size"] == size)
                         & np.isclose(shipped_mf["porosity_target"], por)].iloc[0]
        d = (a - b) / row["nnodes"]
        deltas.append(d)
        print(f"   N={size:5d} eps~{por:.2f}: {d:8.4f} B/node")
print(f"   median {np.median(deltas):.4f}   n={len(deltas)}")

print()
print("### Operator ratios on the GPU (assembled / matrix-free), cases where both complete")
ratios_shipped = []
for size in SIZES:
    for por in POROSITIES:
        a = lookup(shipped_as, size, por)
        m = lookup(shipped_mf, size, por)
        if a in (None, "oom") or m in (None, "oom"):
            continue
        ratios_shipped.append(a / m)
gm = float(np.exp(np.mean(np.log(ratios_shipped))))
print(f"   as shipped: geomean {gm:.3g}x  range {min(ratios_shipped):.3g}x - "
      f"{max(ratios_shipped):.3g}x  over {len(ratios_shipped)} cases")
