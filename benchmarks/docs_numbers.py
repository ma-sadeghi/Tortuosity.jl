"""Recompute the GPU-dependent tables in `docs/src/benchmark.md` from results/.

    pixi run python docs_numbers.py

The benchmark page states its tables as prose rather than generating them, so
re-measuring the GPU sweeps means re-deriving several dozen cells. This prints
them in the page's own shape, making the update a comparison rather than
arithmetic done by hand. Sections run in the order their claims appear on the
page, and each heading names the part of the page it supports.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from benchkit import campaign as bkcampaign
from benchkit import figures as fig

TARGETS = (0.10, 0.01, 0.001)
TARGET = 0.001
BLOB = 1.0

run = bkcampaign.load()
cfg, timings, memory = run.cfg, run.timings, run.memory
porosities, blobinesses, sizes = run.porosities, run.blobinesses, run.sizes
ours, tau = run.ours, run.tau

RESULTS = Path(cfg.resultsdir)
PRE_REFINE = RESULTS / "archive" / "pre-refine-2026-08-20" / "memory"


def pairs(target, blobs):
    """Every (blob, size, porosity) both tools solved on the GPU, with both times."""
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
                    out.append(dict(blob=blob, size=size, por=por,
                                    ours=a, theirs=b, ratio=b / a))
    return out


def cell(rows, size=None, porosity=None):
    vals = [r["ratio"] for r in rows
            if (size is None or r["size"] == size)
            and (porosity is None or np.isclose(r["por"], porosity))]
    return fig.geomean(vals) if vals else np.nan


def fmt(x):
    return "-" if not np.isfinite(x) else f"{x:.3g}x"


hdr = " | ".join(f"N={s}" for s in sizes)
rule = "|---" * (len(sizes) + 1) + "|"
print(f"sizes {sizes}   porosities {porosities}   blobinesses {blobinesses}")

# --- the blobiness-1.0 slice the figure draws ---
one = pairs(TARGET, [BLOB])
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
allr = [r["ratio"] for r in pool]
print(f"\n### Pooled over all microstructures - {len(pool)} paired cases")
print(f"pooled geometric mean {fig.geomean(allr):.3g}x, range {min(allr):.2f}x to {max(allr):.3g}x\n")
print(f"| | {hdr} | row |")
print("|---" * (len(sizes) + 2) + "|")
for por in porosities:
    cells = " | ".join(fmt(cell(pool, size=s, porosity=por)) for s in sizes)
    print(f"| eps ~ {por:.2f} | {cells} | **{fmt(cell(pool, porosity=por))}** |")
cols = " | ".join(f"**{fmt(cell(pool, size=s))}**" for s in sizes)
print(f"| column | {cols} | **{fmt(fig.geomean(allr))}** |")


# --- the two cells the prose names, with the iteration counts it quotes ---
def detail(case_id, series):
    """The row at which a tool first reaches the target, with its iteration count."""
    sub = timings[(timings.case_id == case_id) & (timings.device == "gpu")
                  & (timings["series"] == series) & (timings.rel_error <= TARGET)]
    if sub.empty:
        return None
    row = sub.loc[sub["time_s"].idxmin()]
    return row


print()
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

# --- the fixed family set that supports a five-size scaling claim ---
families = []
for blob in blobinesses:
    for por in porosities:
        got = [r for r in pool if np.isclose(r["blob"], blob) and np.isclose(r["por"], por)]
        if len(got) == len(sizes):
            families.append((blob, por))
print(f"### Fixed set of {len(families)} families solved by both tools at every size")
print(f"{families}\n")
print(f"| | {hdr} |")
print(rule)
for label, fn in (("geometric mean", fig.geomean), ("worst case", min), ("best case", max)):
    row = []
    for s in sizes:
        vals = [r["ratio"] for r in pool if (r["blob"], r["por"]) in families and r["size"] == s]
        row.append(fmt(fn(vals)) if vals else "-")
    print(f"| {label} | {' | '.join(row)} |")

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

# --- does the ranking survive a different accuracy demand? ---
print("\n### Sensitivity to the accuracy target\n")
print("| target | paired cases | pooled geometric mean | cases taufactor wins | where |")
print("|---|---|---|---|---|")
for t in TARGETS:
    rows = pairs(t, blobinesses)
    vals = [r["ratio"] for r in rows]
    wins = [r for r in rows if r["ratio"] < 1]
    where = sorted({int(r["size"]) for r in wins})
    span = f"all at N = {where[0]}" if len(where) == 1 else str(where)
    print(f"| {t:.3g} | {len(rows)} | {fig.geomean(vals):.3g}x | {len(wins)} | {span} |")

# --- the narrowest margins, one of which the prose quotes as the closest cell ---
print(f"\n### Closest cells to taufactor on the GPU ({TARGET:.1%} target)")
for r in sorted(pool, key=lambda r: r["ratio"])[:5]:
    case = f"n{r['size']}_b{int(r['blob'] * 100):03d}_p{int(round(r['por'] * 100)):03d}"
    print(f"   {case:20s} ratio {r['ratio']:6.3f}x   ours {r['ours']:8.3f} s   "
          f"taufactor {r['theirs']:8.3f} s")

# --- the one projected GPU cell, quoted in the projection warning ---
print()
print("### Projected GPU cell: taufactor, blobiness 1.0, eps ~ 0.2, N = 1000")
theirs = fig.target_times(timings, TARGET, device="gpu", series=tau, sizes=sizes,
                          porosities=porosities, blobiness=BLOB)


def node_count(size, por):
    sub = timings[(timings["size"] == size)
                  & np.isclose(timings["porosity_target"], por)
                  & np.isclose(timings["blobiness"], BLOB)]
    return sub["nnodes"].iloc[0] if len(sub) else np.nan


por = 0.2
measured = [n for n in sizes if np.isfinite(theirs[(n, por)])]
counts = [node_count(n, por) for n in measured]
a, p, _ = fig.power_law(counts, [theirs[(n, por)] for n in measured])
est = a * node_count(1000, por) ** p
print(f"   fitted on sizes {measured}, exponent {p:.3f}")
print(f"   projected taufactor time at N=1000: {est:.0f} s")
print("   measured floor (timed out, did not converge): see the timeout row")

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


# --- the device-memory table, in the page's own shape ---
#
# The "as shipped" rows come from the current campaign; the "solve only" rows come
# from the pre-refinement archive, which this campaign did not re-measure. Printing
# both together is what makes the refinement delta checkable.
def load_memory_csv(path):
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


def gb_cells(df, size, column="peak_device_bytes"):
    out = []
    for por in porosities:
        v = lookup(df, size, por, column)
        out.append("*oom*" if v == "oom" else ("—" if v is None else f"{v / 1000 ** 3:.3f}"))
    return out


shipped_mf = load_memory_csv(RESULTS / "memory" / "tortuosity-gpu-matrixfree.csv")
shipped_as = load_memory_csv(RESULTS / "memory" / "tortuosity-gpu-assembled.csv")
solve_mf = (load_memory_csv(PRE_REFINE / "tortuosity-gpu-matrixfree.csv")
            if PRE_REFINE.exists() else None)

print()
print("### Peak device memory, blobiness 1.0, GB (as the docs table prints it)")
for size in sizes:
    if solve_mf is not None:
        print(f"| {size} | matrix-free, solve only | " + " | ".join(gb_cells(solve_mf, size)) + " |")
    print("| | matrix-free, as shipped | " + " | ".join(gb_cells(shipped_mf, size)) + " |")
    print("| | assembled, as shipped | " + " | ".join(gb_cells(shipped_as, size)) + " |")

print()
print("### Refinement delta, bytes per pore node (as shipped minus solve only)")
if solve_mf is None:
    print(f"   skipped: pre-refine archive not found at {PRE_REFINE}")
else:
    deltas = []
    for size in sizes:
        for por in porosities:
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
for size in sizes:
    for por in porosities:
        a = lookup(shipped_as, size, por)
        m = lookup(shipped_mf, size, por)
        if a in (None, "oom") or m in (None, "oom"):
            continue
        ratios_shipped.append(a / m)
print(f"   as shipped: geomean {fig.geomean(ratios_shipped):.3g}x  range {min(ratios_shipped):.3g}x - "
      f"{max(ratios_shipped):.3g}x  over {len(ratios_shipped)} cases")

# --- the same footprint in the units the ceiling paragraph quotes ---
print("\n### Peak device memory, matrix-free, blobiness 1.0 (GiB)\n")
print(f"| | {hdr} |")
print(rule)
for por in porosities:
    row = []
    for s in sizes:
        gib = fig.memory_gib(memory, device="gpu", series=ours, size=s, porosity=por,
                             blobiness=BLOB, column="peak_device_bytes")
        row.append(f"{gib:.3f}" if np.isfinite(gib) else "*oom*")
    print(f"| eps ~ {por:.2f} | {' | '.join(row)} |")

print()
print("### Peak device memory, matrix-free, blobiness 1.0, N = 1000")
mem = memory[(memory.device == "gpu") & (memory["variant"] == "matrixfree")
             & np.isclose(memory["blobiness"], BLOB)
             & (memory["size"] == 1000)]
for _, r in mem.sort_values("porosity_target").iterrows():
    gib = r["peak_device_bytes"] / 1024 ** 3
    gb = r["peak_device_bytes"] / 1000 ** 3
    print(f"   eps~{r['porosity_target']:.2f}: {gib:.3f} GiB   ({gb:.3f} GB)")

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

# --- does the margin survive a change of microstructure? ---
print()
print(f"### Pooled geometric mean by blobiness ({TARGET:.1%} target)")
for blob in blobinesses:
    vals = [r["ratio"] for r in pool if r["blob"] == blob]
    print(f"   blobiness {blob}: {fig.geomean(vals):.3g}x over {len(vals)} cases")

# --- determinism: the tau_spread claim ---
print("\n### Run-to-run determinism on the GPU")
for variant in ("matrixfree", "assembled"):
    sub = gpu[(gpu.variant == variant) & (gpu.tool == "tortuosity") & (gpu.repeats == 3)]
    zero = int((sub["tau_spread"] == 0).sum())
    print(f"   {variant}: {zero} of {len(sub)} three-repeat rows have tau_spread exactly 0")
