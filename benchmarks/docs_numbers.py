"""Recompute the GPU-dependent tables in `docs/src/benchmark.md` from results/.

    pixi run python docs_numbers.py

The benchmark page states its tables as prose rather than generating them, so
re-measuring the GPU sweeps means re-deriving several dozen cells. This prints
them in the page's own shape, making the update a comparison rather than
arithmetic done by hand.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from benchkit import config as bkconfig
from benchkit import figures as fig

TARGETS = (0.10, 0.01, 0.001)
TARGET = 0.001

cfg = bkconfig.load_config()
timings = fig.load_results(cfg.resultsdir, "timings")
memory = fig.load_results(cfg.resultsdir, "memory")

porosities = sorted(timings["porosity_target"].unique())
blobinesses = sorted(timings["blobiness"].unique())
sizes = sorted(timings[timings.device == "gpu"]["size"].unique())
ours = fig.series_label(*fig.REFERENCE_SERIES)
tau = fig.series_label("taufactor", "sor")


def pairs(target, blobs):
    """Every (blob, size, porosity, ratio) both tools solved on the GPU."""
    out = []
    for blob in blobs:
        mine = fig.target_times(timings, target, device="gpu", series=ours,
                                sizes=sizes, porosities=porosities, blobiness=blob)
        theirs = fig.target_times(timings, target, device="gpu", series=tau,
                                  sizes=sizes, porosities=porosities, blobiness=blob)
        for size in sizes:
            for por in porosities:
                a, b = mine[(size, por)], theirs[(size, por)]
                if np.isfinite(a) and np.isfinite(b) and a > 0:
                    out.append((blob, size, por, b / a))
    return out


def cell(rows, size=None, porosity=None):
    vals = [r[3] for r in rows
            if (size is None or r[1] == size)
            and (porosity is None or np.isclose(r[2], porosity))]
    return fig.geomean(vals) if vals else np.nan


def fmt(x):
    return "-" if not np.isfinite(x) else f"{x:.3g}x"


hdr = " | ".join(f"N={s}" for s in sizes)
rule = "|---" * (len(sizes) + 1) + "|"
print(f"sizes {sizes}   porosities {porosities}   blobinesses {blobinesses}")

# --- the blobiness-1.0 slice the figure draws ---
one = pairs(TARGET, [1.0])
print(f"\n### Against taufactor on the GPU - blobiness 1.0, {TARGET:.1%} target\n")
print(f"| Porosity | {hdr} |")
print(rule)
for por in porosities:
    cells = " | ".join(fmt(cell(one, size=s, porosity=por)) for s in sizes)
    print(f"| eps ~ {por:.2f} | {cells} |")
gm = " | ".join(fmt(cell(one, size=s)) for s in sizes)
print(f"| geometric mean | {gm} |")

# --- pooled over all three microstructures ---
pool = pairs(TARGET, blobinesses)
allr = [r[3] for r in pool]
print(f"\n### Pooled over all microstructures - {len(pool)} paired cases")
print(f"pooled geometric mean {fig.geomean(allr):.3g}x, range {min(allr):.2f}x to {max(allr):.3g}x\n")
print(f"| | {hdr} | row |")
print("|---" * (len(sizes) + 2) + "|")
for por in porosities:
    cells = " | ".join(fmt(cell(pool, size=s, porosity=por)) for s in sizes)
    print(f"| eps ~ {por:.2f} | {cells} | **{fmt(cell(pool, porosity=por))}** |")
cols = " | ".join(f"**{fmt(cell(pool, size=s))}**" for s in sizes)
print(f"| column | {cols} | **{fmt(fig.geomean(allr))}** |")

# --- the fixed family set that supports a five-size scaling claim ---
families = []
for blob in blobinesses:
    for por in porosities:
        got = [r for r in pool if np.isclose(r[0], blob) and np.isclose(r[2], por)]
        if len(got) == len(sizes):
            families.append((blob, por))
print(f"\n### Fixed set of {len(families)} families solved by both tools at every size")
print(f"{families}\n")
print(f"| | {hdr} |")
print(rule)
for label, fn in (("geometric mean", fig.geomean), ("worst case", min), ("best case", max)):
    row = []
    for s in sizes:
        vals = [r[3] for r in pool if (r[0], r[2]) in families and r[1] == s]
        row.append(fmt(fn(vals)) if vals else "-")
    print(f"| {label} | {' | '.join(row)} |")

# --- does the ranking survive a different accuracy demand? ---
print("\n### Sensitivity to the accuracy target\n")
print("| target | paired cases | pooled geometric mean | cases taufactor wins | where |")
print("|---|---|---|---|---|")
for t in TARGETS:
    rows = pairs(t, blobinesses)
    vals = [r[3] for r in rows]
    wins = [r for r in rows if r[3] < 1]
    where = sorted({int(r[1]) for r in wins})
    span = f"all at N = {where[0]}" if len(where) == 1 else str(where)
    print(f"| {t:.3g} | {len(rows)} | {fig.geomean(vals):.3g}x | {len(wins)} | {span} |")

# --- cases taufactor never finished, which appear in no ratio ---
gpu = timings[timings.device == "gpu"]
theirs_all = gpu[gpu.series == tau]
solved = set(theirs_all[theirs_all.stop_reason == "target_reached"]["case_id"])
failed = sorted(set(theirs_all["case_id"]) - solved)
print(f"\n### taufactor never reached {TARGET:.1%} on {len(failed)} GPU cases (lower bounds)\n")
for case in failed:
    t = theirs_all[theirs_all.case_id == case]
    if not t["rel_error"].notna().any():
        continue
    best = t.loc[t["rel_error"].idxmin()]
    mine = gpu[(gpu.series == ours) & (gpu.case_id == case)
               & (gpu.stop_reason == "target_reached")]
    ours_t = mine["time_s"].min() if len(mine) else np.nan
    bound = best["time_s"] / ours_t if np.isfinite(ours_t) and ours_t > 0 else np.nan
    print(f"   {case}: taufactor {best['time_s']:.1f} s at {best['rel_error']:.2e}; "
          f"ours {ours_t:.2f} s  -> lower bound {bound:.0f}x")

mine_all = gpu[gpu.series == ours]
reached = mine_all[mine_all.stop_reason == "target_reached"]["case_id"].nunique()
print(f"   Tortuosity.jl reached the target on {reached} of {mine_all['case_id'].nunique()} GPU cases")

# --- the GPU against our own CPU path ---
print("\n### GPU over the HostCG CPU path")
cpu_sizes = sorted(timings[timings.device == "cpu"]["size"].unique())
ratios, per_size = [], {}
for blob in blobinesses:
    g = fig.target_times(timings, TARGET, device="gpu", series=ours,
                         sizes=sizes, porosities=porosities, blobiness=blob)
    c = fig.target_times(timings, TARGET, device="cpu", series=ours,
                         sizes=cpu_sizes, porosities=porosities, blobiness=blob)
    for size in sizes:
        for por in porosities:
            a, b = g.get((size, por), np.nan), c.get((size, por), np.nan)
            if np.isfinite(a) and np.isfinite(b) and a > 0:
                ratios.append(b / a)
                per_size.setdefault(size, []).append(b / a)
if ratios:
    print(f"   geometric mean {fig.geomean(ratios):.3g}x over {len(ratios)} cases, "
          f"range {min(ratios):.2f}x to {max(ratios):.2f}x")
    print("   by size:", {int(k): f"{fig.geomean(v):.2f}x" for k, v in sorted(per_size.items())})

# --- matrix-free against assembled, same device ---
print("\n### Operator comparison on the GPU")
asm = fig.series_label("tortuosity", "assembled")
op_ratios = []
for blob in blobinesses:
    m = fig.target_times(timings, TARGET, device="gpu", series=ours,
                         sizes=sizes, porosities=porosities, blobiness=blob)
    a = fig.target_times(timings, TARGET, device="gpu", series=asm,
                         sizes=sizes, porosities=porosities, blobiness=blob)
    for size in sizes:
        for por in porosities:
            x, y = m[(size, por)], a[(size, por)]
            if np.isfinite(x) and np.isfinite(y) and x > 0:
                op_ratios.append(y / x)
if op_ratios:
    print(f"   matrix-free is {fig.geomean(op_ratios):.3g}x faster end to end "
          f"over {len(op_ratios)} paired cases")

# --- determinism: the tau_spread claim ---
print("\n### Run-to-run determinism on the GPU")
for variant in ("matrixfree", "assembled"):
    sub = gpu[(gpu.variant == variant) & (gpu.tool == "tortuosity") & (gpu.repeats == 3)]
    zero = int((sub["tau_spread"] == 0).sum())
    print(f"   {variant}: {zero} of {len(sub)} three-repeat rows have tau_spread exactly 0")

# --- peak device memory ---
print("\n### Peak device memory, matrix-free, blobiness 1.0 (GiB)\n")
print(f"| | {hdr} |")
print(rule)
for por in porosities:
    row = []
    for s in sizes:
        gib = fig.memory_gib(memory, device="gpu", series=ours, size=s, porosity=por,
                             blobiness=1.0, column="peak_device_bytes")
        row.append(f"{gib:.3f}" if np.isfinite(gib) else "*oom*")
    print(f"| eps ~ {por:.2f} | {' | '.join(row)} |")
