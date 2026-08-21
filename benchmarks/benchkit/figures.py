"""Loading and drawing for the post-processing stage.

Nothing here touches an image, a GPU or a solver: the figures are built from the
result CSVs alone. That separation is what lets data be generated on a rented
machine and every figure be redrawn afterwards on a laptop, as many times as the
paper needs, without paying for the machine again.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Publication style ────────────────────────────────────────────────

STYLE = {
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
}

# How each (tool, variant) is named and drawn. The device is a facet rather than
# part of a series, because CPU and GPU never share an axis: mixing them would
# conflate the algorithm with the hardware.
SERIES = {
    ("tortuosity", "matrixfree"): ("Tortuosity.jl (matrix-free)", "o", "#2166ac"),
    ("tortuosity", "assembled"): ("Tortuosity.jl (assembled)", "^", "#67a9cf"),
    ("tortuosity", "matrixfree-nopc"): ("Tortuosity.jl (matrix-free, unpreconditioned)", "v", "#7b3294"),
    ("tortuosity", "assembled-nopc"): ("Tortuosity.jl (assembled, unpreconditioned)", "P", "#c2a5cf"),
    ("taufactor", "sor"): ("taufactor", "s", "#b2182b"),
    ("puma", "fv-cg"): ("PuMA", "D", "#4dac26"),
}

# The series every speedup on a device is measured against.
REFERENCE_SERIES = ("tortuosity", "matrixfree")

# The external tool the headline comparison targets on each device. taufactor
# runs on both, but on the CPU the established point of comparison is PuMA,
# whose finite-volume conjugate gradient is the closest counterpart to what
# Tortuosity.jl does there.
HEADLINE_COMPETITOR = {"gpu": ("taufactor", "sor"), "cpu": ("puma", "fv-cg")}

# Accuracy targets every figure is resolved over. No extra measurement is needed:
# each sweep holds a full ladder per case, so the time to reach any target is
# already in the data. Reporting one target would hide that the ranking depends
# on how much accuracy is demanded — at 10% a solver that starts near the answer
# wins on its initial guess; at 0.1% it has to solve.
THRESHOLDS = [0.10, 0.01, 0.001]
THRESHOLD_LABELS = ["10\\%", "1\\%", "0.1\\%"]
PAPER_TARGET = 0.001


UNKNOWN_STYLE = ("x", "0.4")


def series_label(tool, variant):
    """Display name for a (tool, variant) pair, including ones not in the table.

    A configuration nobody has styled still gets a label and still gets drawn.
    Dropping it would be the worst outcome: a measured series silently missing
    from a figure looks exactly like a series that was never measured.
    """
    entry = SERIES.get((tool, variant))
    return entry[0] if entry else f"{tool} ({variant})"


def style_for(label):
    """`(marker, colour)` for a display name, falling back for unstyled series."""
    for key, (name, marker, color) in SERIES.items():  # noqa: B007 - key unused by design
        if name == label:
            return marker, color
    return UNKNOWN_STYLE


def load_results(resultsdir, kind):
    """Concatenate every CSV under ``results/<kind>/`` into one frame.

    The identifying columns are repeated on every row, so which file a row came
    from carries no information and the whole directory can simply be stacked.
    """
    directory = Path(resultsdir) / kind
    frames = []
    for path in sorted(directory.glob("*.csv")):
        if path.stat().st_size == 0:
            continue
        frame = pd.read_csv(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data["series"] = [series_label(t, v) for t, v in zip(data["tool"], data["variant"])]
    return data


def geomean(values):
    """Geometric mean of the positive, finite entries; NaN when there are none.

    Solve times span orders of magnitude and speedups are ratios, where the
    arithmetic mean is the wrong statistic: it lets one slow porosity dominate an
    aggregate, and it averages "2x faster" and "2x slower" to 1.25x rather than
    to parity.
    """
    v = np.asarray([x for x in values if np.isfinite(x) and x > 0], dtype=float)
    return float(np.exp(np.log(v).mean())) if v.size else np.nan


def time_to_target(frame, target):
    """Wall time of the fastest measured run in `frame` that reaches `target`.

    Solve time rises monotonically as the knob tightens, so the cheapest run
    meeting the target is the loosest one that does. NaN when none did, which is
    a real outcome — the solver never got there — rather than missing data.
    """
    ok = frame[frame["rel_error"].notna() & (frame["rel_error"] <= target) & (frame["time_s"] > 0)]
    return float(ok["time_s"].min()) if len(ok) else np.nan


def target_times(data, target, *, device, series, sizes, porosities, blobiness):
    """`{(size, porosity): seconds}` for one series on one device, at one blobiness."""
    sub = data[(data["device"] == device) & (data["series"] == series)
               & np.isclose(data["blobiness"], blobiness)]
    out = {}
    for size in sizes:
        for por in porosities:
            cell = sub[(sub["size"] == size) & np.isclose(sub["porosity_target"], por)]
            out[(size, por)] = time_to_target(cell, target) if len(cell) else np.nan
    return out


def memory_gib(memory, *, device, series, size, porosity, blobiness, column):
    """Peak memory one series held solving one case, in GiB."""
    sub = memory[(memory["device"] == device) & (memory["series"] == series)
                 & (memory["size"] == size) & np.isclose(memory["porosity_target"], porosity)
                 & np.isclose(memory["blobiness"], blobiness) & (memory["status"] == "ok")]
    if sub.empty:
        return np.nan
    values = pd.to_numeric(sub[column], errors="coerce").dropna()
    values = values[values > 0]
    return float(values.max()) / 2**30 if len(values) else np.nan


def host_delta(memory):
    """Resident-set increase over the state with the image already loaded.

    The raw resident set is not comparable across languages: a Julia process
    carrying a compiled runtime starts near a gigabyte and a Python one near a
    tenth of that, so comparing totals would rank runtimes rather than solvers.
    Both harnesses take their baseline at the same point — image loaded, nothing
    else built — which makes the increase the like-for-like quantity.
    """
    return memory["peak_rss_bytes"] - memory["baseline_rss_bytes"]


# ── Drawing helpers ──────────────────────────────────────────────────

def format_speedup(value):
    """Label a speedup with enough digits to stay informative on both sides of 1.

    A fixed one-decimal format collapses every loss worse than about 20x into
    "0.0x", which reads as missing data rather than as the largest deficit shown.
    """
    if not np.isfinite(value):
        return "—"
    if value >= 10:
        return f"{value:.0f}×"
    if value >= 1:
        return f"{value:.1f}×"
    if value >= 0.1:
        return f"{value:.2f}×"
    return f"{value:.3f}×"


def speedup_span(matrices):
    """Half-width of a log colour scale covering every matrix given.

    Panels sharing a scale can be compared cell to cell; panels each normalised
    to their own extreme cannot, and three of those side by side invite exactly
    that mistake.
    """
    finite = [m[np.isfinite(m)] for m in matrices]
    finite = np.concatenate(finite) if finite else np.array([])
    if finite.size == 0:
        return None
    return max(float(np.abs(np.log10(finite)).max()), np.log10(2))


def blank_aware_cmap():
    """RdBu with an unmistakable colour for cells that were never measured.

    Parity sits at the pale centre of RdBu, so a blank left white is
    indistinguishable from a 1x cell. Grey says "no measurement" instead.
    """
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("0.88")
    return cmap


def draw_speedup_heatmap(ax, matrix, sizes, porosities, span=None):
    """Render a speedup matrix on a log colour scale diverging about parity.

    Speedup is multiplicative, so 0.5x and 2x are equal and opposite departures
    from parity. A linear scale floored at 0.5 flattens every loss into one
    colour, hiding the cases where Tortuosity.jl is the slower tool.
    """
    span = span if span is not None else speedup_span([matrix])
    image = None
    if span is not None:
        image = ax.imshow(np.log10(matrix), cmap=blank_aware_cmap(), aspect="auto",
                          vmin=-span, vmax=span, origin="lower")
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes)
    ax.set_yticks(range(len(porosities)))
    ax.set_yticklabels([f"{p:.2f}" for p in porosities])
    ax.set_xlabel("Domain size $N$ (voxels per side)")
    ax.set_ylabel("Porosity $\\varepsilon$")
    for i in range(len(porosities)):
        for j in range(len(sizes)):
            value = matrix[i, j]
            strong = np.isfinite(value) and span and abs(np.log10(value)) > span * 0.6
            color = ("white" if strong else "black") if np.isfinite(value) else "0.45"
            ax.text(j, i, format_speedup(value), ha="center", va="center",
                    fontsize=8, fontweight="bold", color=color)
    return image


def speedup_colorbar(fig, image, axes, span, label):
    ticks = [t for t in (-3, -2, -1, -np.log10(5), -np.log10(2), 0,
                         np.log10(2), np.log10(5), 1, 2, 3) if abs(t) <= span]
    bar = fig.colorbar(image, ax=axes, shrink=0.85, ticks=ticks)
    bar.ax.set_yticklabels([f"{10 ** t:g}×" for t in ticks])
    bar.set_label(label, fontsize=8)
    return bar


def log_size_axis(ax, sizes):
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
