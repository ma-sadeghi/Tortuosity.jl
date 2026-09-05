"""Recompute the headline numbers `paper.md` quotes, straight from results/.

    pixi run python paper_numbers.py

Exists so the manuscript can be re-checked against a regenerated dataset
mechanically rather than by reading values off a figure.
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
memory = fig.load_results(cfg.resultsdir, "memory")

porosities = sorted(timings["porosity_target"].unique())
blobinesses = sorted(timings["blobiness"].unique())
configured = [float(b) for b in cfg["image"]["blobinesses"]]
ref_blob = 1.0 if 1.0 in configured else configured[len(configured) // 2]
ours = fig.series_label(*fig.REFERENCE_SERIES)
tau = fig.series_label("taufactor", "sor")


def times(device, series, sizes, blobiness):
    return fig.target_times(timings, TARGET, device=device, series=series,
                            sizes=sizes, porosities=porosities, blobiness=blobiness)


gpu_sizes = sorted(timings[timings.device == "gpu"]["size"].unique())
print(f"sizes on gpu: {gpu_sizes}")
print(f"porosities: {porosities}   blobinesses: {blobinesses}   reference blobiness: {ref_blob}")

# --- the one cell where taufactor wins, at the reference feature size ---
mine = times("gpu", ours, gpu_sizes, ref_blob)
theirs = times("gpu", tau, gpu_sizes, ref_blob)
print("\n== GPU, reference blobiness: time to <=0.1% error (s) ==")
print(f"{'size':>6} {'eps':>6} {'ours':>10} {'taufactor':>10} {'ratio':>8}")
for size in gpu_sizes:
    for por in porosities:
        a, b = mine[(size, por)], theirs[(size, por)]
        ratio = b / a if np.isfinite(a) and np.isfinite(b) and a > 0 else float("nan")
        flag = "  <-- taufactor faster" if np.isfinite(ratio) and ratio < 1 else ""
        print(f"{size:>6} {por:>6} {a:>10.3f} {b:>10.3f} {ratio:>8.2f}{flag}")

# --- geometric mean over case families where both tools reached the target at every size ---
largest = gpu_sizes[-1]
ratios, families = [], []
for blob in blobinesses:
    m = times("gpu", ours, gpu_sizes, blob)
    t = times("gpu", tau, gpu_sizes, blob)
    for por in porosities:
        cells = [(m[(s, por)], t[(s, por)]) for s in gpu_sizes]
        if all(np.isfinite(a) and np.isfinite(b) and a > 0 for a, b in cells):
            families.append((blob, por))
            a, b = m[(largest, por)], t[(largest, por)]
            ratios.append(b / a)
print(f"\n== families reaching the target at all {len(gpu_sizes)} sizes: {len(families)} ==")
print(f"   {families}")
if ratios:
    print(f"   geometric mean of the ratio at {largest}^3: {fig.geomean(ratios):.2f}x")

# --- peak device memory at the largest size ---
print(f"\n== peak device memory at {largest}^3 (GiB), matrix-free, reference blobiness ==")
for por in porosities:
    gib = fig.memory_gib(memory, device="gpu", series=ours, size=largest,
                         porosity=por, blobiness=ref_blob, column="peak_device_bytes")
    print(f"   eps={por}: {gib:.2f}")

# --- operator agreement in single precision on the GPU ---
gpu = timings[(timings.device == "gpu") & (timings.stop_reason == "target_reached")]
mf = gpu[gpu.variant == "matrixfree"].set_index(["case_id"])["tau"]
asm = gpu[gpu.variant == "assembled"].set_index(["case_id"])["tau"]
shared = mf.index.intersection(asm.index)
if len(shared):
    rel = (mf.loc[shared] - asm.loc[shared]).abs() / asm.loc[shared].abs()
    rel = rel.groupby(level=0).max()
    print(f"\n== operator agreement on the GPU over {len(rel)} paired cases ==")
    print(f"   median {rel.median():.2e}   mean {rel.mean():.2e}   max {rel.max():.2e}")
