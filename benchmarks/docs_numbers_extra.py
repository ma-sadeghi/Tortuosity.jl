"""The GPU numbers in `docs/src/benchmark.md` that `docs_numbers.py` does not cover.

    pixi run python docs_numbers_extra.py

Prose on that page cites individual cells — the closest cell to taufactor, the
per-blobiness pooled means, the one projected GPU cell — that no table in
`docs_numbers.py` prints. Re-measuring the GPU sweeps invalidates those too.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from benchkit import config as bkconfig
from benchkit import figures as fig

TARGET = 0.001

cfg = bkconfig.load_config()
timings = fig.load_results(cfg.resultsdir, "timings")

porosities = sorted(timings["porosity_target"].unique())
blobinesses = sorted(timings["blobiness"].unique())
sizes = sorted(timings[timings.device == "gpu"]["size"].unique())
ours = fig.series_label(*fig.REFERENCE_SERIES)
tau = fig.series_label("taufactor", "sor")


def rows(target=TARGET):
    out = []
    for blob in blobinesses:
        mine = fig.target_times(timings, target, device="gpu", series=ours,
                                sizes=sizes, porosities=porosities, blobiness=blob)
        theirs = fig.target_times(timings, target, device="gpu", series=tau,
                                  sizes=sizes, porosities=porosities, blobiness=blob)
        for size in sizes:
            for por in porosities:
                a, b = mine[(size, por)], theirs[(size, por)]
                if np.isfinite(a) and np.isfinite(b) and a > 0:
                    out.append(dict(blob=blob, size=size, por=por,
                                    ours=a, theirs=b, ratio=b / a))
    return out


all_rows = rows()

print("### Closest cells to taufactor on the GPU (0.1% target)")
for r in sorted(all_rows, key=lambda r: r["ratio"])[:5]:
    case = f"n{r['size']}_b{int(r['blob'] * 100):03d}_p{int(round(r['por'] * 100)):03d}"
    print(f"   {case:20s} ratio {r['ratio']:6.3f}x   ours {r['ours']:8.3f} s   "
          f"taufactor {r['theirs']:8.3f} s")

print()
print("### Pooled geometric mean by blobiness (0.1% target)")
for blob in blobinesses:
    vals = [r["ratio"] for r in all_rows if r["blob"] == blob]
    print(f"   blobiness {blob}: {fig.geomean(vals):.3g}x over {len(vals)} cases")

print()
print("### The n600_b100_p020 cell cited in prose")
sub = timings[(timings.device == "gpu") & (timings.case_id == "n600_b100_p020")]
for series in (ours, tau):
    s = sub[sub["series"] == series] if "series" in sub.columns else None
print("   (see per-tool detail below)")

print()
print("### Determinism: three-repeat rows with tau_spread exactly 0")
for name in ("tortuosity-gpu-matrixfree", "tortuosity-gpu-assembled"):
    path = Path(cfg.resultsdir) / "timings" / f"{name}.csv"
    import pandas as pd
    df = pd.read_csv(path)
    three = df[df["repeats"] == 3]
    zero = three[three["tau_spread"] == 0]
    print(f"   {name}: {len(zero)} of {len(three)}")
