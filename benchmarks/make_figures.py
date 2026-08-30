"""Draw every benchmark figure from the result CSVs.

    pixi run python make_figures.py
    pixi run python make_figures.py --only=memory,summary --no-publish

Post-processing only. This stage reads `results/` and nothing else — no images,
no solvers, no GPU — so the expensive half of the campaign can run on a rented
machine while the figures are redrawn as often as the paper needs on a laptop.

By default the figures the paper and the documentation embed are copied into
`paper/` and `docs/src/assets/`. Copying rather than leaving that to hand is what
keeps a regenerated dataset from leaving a stale figure behind: the failure is
silent, because the document still builds and still shows a plausible plot, just
of superseded numbers.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from benchkit import config as bkconfig  # noqa: E402
from benchkit import figures as fig  # noqa: E402

logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}")

# Which memory reading is meaningful on each device. On the GPU the host heap
# holds the image rather than the solve; on the CPU there is no device at all.
MEMORY_COLUMN = {"gpu": "peak_device_bytes", "cpu": "host_delta_bytes"}
MEMORY_LABEL = {
    "gpu": "Peak device memory (GiB)",
    "cpu": "Peak host memory above baseline (GiB)",
}

# Figures the paper and the documentation embed: `{drawn name: (published name,
# destinations)}`. The published names keep the `benchmark_` prefix the documents
# already reference, so renaming a figure here never silently breaks a document —
# a broken image is a build that still succeeds and a page that quietly loses a
# result. `speedup_taufactor_gpu` publishes under the unsuffixed name for the
# same reason: it is the comparison the docs linked before taufactor was also run
# on the CPU.
# Only the two figures `paper.md` embeds are published into `paper/`. JOSS builds
# the manuscript from that directory, so every other figure goes to the docs alone
# rather than leaving an unreferenced image beside the paper.
PUBLISH = {
    "summary.png": ("benchmark_summary.png", ["paper", "docs/src/assets"]),
    "memory_gpu.png": ("benchmark_memory_gpu.png", ["paper", "docs/src/assets"]),
    "memory_cpu.png": ("benchmark_memory_cpu.png", ["docs/src/assets"]),
    "blobiness.png": ("benchmark_blobiness.png", ["docs/src/assets"]),
    "scaling_gpu.png": ("benchmark_scaling_gpu.png", ["docs/src/assets"]),
    "scaling_cpu.png": ("benchmark_scaling_cpu.png", ["docs/src/assets"]),
    "time_bars_gpu.png": ("benchmark_time_bars_gpu.png", ["docs/src/assets"]),
    "time_bars_cpu.png": ("benchmark_time_bars_cpu.png", ["docs/src/assets"]),
    "speedup_taufactor_gpu.png": ("benchmark_speedup_taufactor.png", ["docs/src/assets"]),
    "speedup_taufactor_cpu.png": ("benchmark_speedup_taufactor_cpu.png", ["docs/src/assets"]),
    "speedup_puma_cpu.png": ("benchmark_speedup_puma_cpu.png", ["docs/src/assets"]),
    "pareto_gpu.png": ("benchmark_pareto.png", ["docs/src/assets"]),
    "pareto_cpu.png": ("benchmark_pareto_cpu.png", ["docs/src/assets"]),
}

MISSING_MARKER = "x"


class Campaign:
    """Everything the figures are drawn from, loaded once."""

    def __init__(self, cfg, outdir):
        self.cfg = cfg
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.timings = fig.load_results(cfg.resultsdir, "timings")
        self.memory = fig.load_results(cfg.resultsdir, "memory")
        if self.timings.empty and self.memory.empty:
            raise SystemExit(f"no results under {cfg.resultsdir} — run the benchmarks first")
        if not self.memory.empty:
            self.memory["host_delta_bytes"] = fig.host_delta(self.memory)
        # Ground truth is read from its own file rather than off a timing row, so
        # the structure panel can be drawn before any tool has been benchmarked.
        refpath = cfg.resultsdir / "references.csv"
        self.references = pd.read_csv(refpath) if refpath.is_file() else pd.DataFrame()

        configured = [float(b) for b in cfg["image"]["blobinesses"]]
        # Everything except the blobiness figure is drawn at one structure, so
        # that a size or porosity trend is not silently averaged over three
        # different microstructures. 1.0 is the value the earlier campaigns used.
        self.reference_blobiness = 1.0 if 1.0 in configured else configured[len(configured) // 2]
        self.porosities = [float(p) for p in cfg["image"]["porosities"]]

    def devices(self, frame):
        return [d for d in ("gpu", "cpu") if not frame.empty and (frame["device"] == d).any()]

    def series_on(self, frame, device):
        """Series measured on `device`, in the order the style table declares.

        Anything measured but not in that table is appended rather than dropped:
        a series silently missing from a figure looks exactly like one that was
        never measured.
        """
        if frame.empty:
            return []
        present = set(frame.loc[frame["device"] == device, "series"])
        ordered = [fig.series_label(t, v) for t, v in fig.SERIES]
        known = [s for s in ordered if s in present]
        return known + sorted(present - set(known))

    def sizes(self, frame, device=None, blobiness=None):
        sub = frame
        if device is not None:
            sub = sub[sub["device"] == device]
        if blobiness is not None:
            sub = sub[np.isclose(sub["blobiness"], blobiness)]
        return sorted(int(n) for n in sub["size"].unique())

    def save(self, figure, name):
        figure.savefig(self.outdir / name)
        plt.close(figure)
        logger.success(f"saved {name}")


# ── Scaling at matched accuracy ──────────────────────────────────────

# Fewest sizes a porosity must have been measured at before its power law is
# fitted and used to stand in for a size it was not measured at. Two points
# define a line with nothing left over to judge it by.
PROJECTION_MIN_SIZES = 3


def out_of_memory(timings, *, device, series, size, porosity, blobiness):
    """Whether this case failed on the hardware rather than going unmeasured.

    Every `error` row in the campaign is an out-of-memory failure, so a cell
    holding one is not a measurement someone chose to skip: the tool cannot solve
    that image on this machine at all. Projecting a time for it would put a
    number on the page for a run that can never happen, which is a different
    claim from projecting one that was merely never afforded.
    """
    cell = timings[(timings["device"] == device) & (timings["series"] == series)
                   & (timings["size"] == size)
                   & np.isclose(timings["porosity_target"], porosity)
                   & np.isclose(timings["blobiness"], blobiness)]
    return bool((cell["stop_reason"] == "error").any())


def scaling_points(campaign, device, target, sizes, series):
    """Time against size for one series: `(points, projected_sizes, exponent)`.

    Averaging over only the porosities a tool happened to reach makes the plotted
    quantity a different one at every size, and biases it in a known direction:
    the porosity a tool drops first is the one it is slowest on. taufactor's
    hardest image at each size is its densest, so losing that image pulls the
    mean down far enough to draw a solver that was still slowing as though it had
    levelled off, and on the CPU as though it had started to speed up. Neither
    happens. Both are the average changing underneath the curve.

    So every size here averages over the same five porosities. A porosity the
    tool was never run at, or ran at without reaching the target inside its
    budget, is filled in from its own fitted power law over the sizes it did
    reach. A porosity that ran out of memory is not filled in, and takes its
    whole size off the curve: there is no honest mean over a set of images one of
    which the tool cannot solve at all.
    """
    times = fig.target_times(campaign.timings, target, device=device, series=series,
                             sizes=sizes, porosities=campaign.porosities,
                             blobiness=campaign.reference_blobiness)
    # One power law per porosity rather than one for the series. The exponent
    # rises steeply as the pore space closes -- taufactor spans 3.1 to 3.9 on the
    # GPU and 3.2 to 5.0 on the CPU -- so a single exponent would project the
    # dense images, the ones that are missing, at the open images' rate.
    fits = {}
    for por in campaign.porosities:
        measured = [n for n in sizes if np.isfinite(times[(n, por)])]
        if len(measured) >= PROJECTION_MIN_SIZES:
            fits[por] = fig.power_law(measured, [times[(n, por)] for n in measured])

    points, projected = [], []
    for size in sizes:
        filled, borrowed = [], False
        for por in campaign.porosities:
            value = times[(size, por)]
            if np.isfinite(value):
                filled.append(value)
                continue
            fit = fits.get(por)
            if fit is None or out_of_memory(campaign.timings, device=device, series=series,
                                            size=size, porosity=por,
                                            blobiness=campaign.reference_blobiness):
                filled = None
                break
            a, exponent, _ = fit
            filled.append(a * size ** exponent)
            borrowed = True
        if not filled:
            continue
        points.append((size, fig.geomean(filled)))
        if borrowed:
            projected.append(size)

    # The line is fitted to the measured means alone. Refitting it through the
    # projected ones would draw a model back through its own output and hide
    # whether the two agree, which is the one thing a reader should be able to
    # check by eye.
    solid = [(n, t) for n, t in points if n not in projected]
    fit = fig.power_law([n for n, _ in solid], [t for _, t in solid]) if len(solid) >= 3 else None
    # Say what was projected and what was dropped. A curve that quietly stops
    # short, or quietly rests on a fit, reads exactly like one that does neither.
    if projected:
        logger.info(f"{series} on the {device} at {target:g}: projected {projected} "
                    f"from a per-porosity power law")
    dropped = [n for n in sizes if n not in [p for p, _ in points]]
    if dropped:
        logger.info(f"{series} on the {device} at {target:g}: left {dropped} off, where a "
                    "porosity was neither measured nor projectable")
    return points, projected, fit


def draw_scaling_panel(ax, campaign, device, target, sizes):
    """One accuracy-matched scaling panel: time against size, one line per tool.

    The solid line joins measurements and the dashed one runs on to sizes reached
    by projection, drawn hollow. The fitted exponent rides in the legend rather
    than as a line through the points, because it is a summary and not the
    measurement: `Tortuosity.jl` is deliberately not a single power law, since a
    setup cost that dominates a small image and vanishes on a large one bends the
    curve, and a fitted line drawn through those points would look like a bad fit
    where the data is fine.
    """
    for series in campaign.series_on(campaign.timings, device):
        points, projected, fit = scaling_points(campaign, device, target, sizes, series)
        if not points:
            continue
        marker, color = fig.style_for(series)
        label = series if fit is None else f"{series} ($N^{{{fit[1]:.1f}}}$)"
        measured = [p for p in points if p[0] not in projected]
        # Where the solid line hands over to the dashed one. With nothing measured
        # the smallest size takes that role, which leaves the solid span a single
        # point, too short to draw, and dashes the whole curve.
        edge = max((n for n, _ in measured), default=min(n for n, _ in points))
        for span, style in (([p for p in points if p[0] <= edge], "-"),
                            ([p for p in points if p[0] >= edge], "--")):
            if len(span) >= 2:
                ax.plot(*zip(*span), style, color=color, zorder=2)
        if measured:
            ax.plot(*zip(*measured), marker, color=color, linestyle="none", label=label,
                    markeredgewidth=0.5, markeredgecolor="white", zorder=3)
        if projected:
            ax.plot(*zip(*[p for p in points if p[0] in projected]), marker,
                    color="white", markeredgewidth=1.0, markeredgecolor=color,
                    linestyle="none", zorder=4, label=None if measured else label)
    ax.set_xlabel("Domain size $N$ (voxels per side)")
    ax.set_ylabel("Solve time (s)")
    fig.log_size_axis(ax, sizes)
    ax.set_yscale("log")
    ax.grid(True)


def figure_scaling(campaign):
    for device in campaign.devices(campaign.timings):
        sizes = campaign.sizes(campaign.timings, device, campaign.reference_blobiness)
        if not sizes:
            continue
        figure, axes = plt.subplots(1, len(fig.THRESHOLDS), sharey=True,
                                    figsize=(3.2 * len(fig.THRESHOLDS), 2.9))
        for ax, target, label in zip(np.atleast_1d(axes), fig.THRESHOLDS, fig.THRESHOLD_LABELS):
            draw_scaling_panel(ax, campaign, device, target, sizes)
            ax.set_title(f"target $\\leq {label}$")
            # A legend per panel rather than one for the figure. The exponent is
            # part of each label and it is a property of the target as much as of
            # the tool, so a shared legend would print the leftmost panel's
            # exponents over all three. Every curve rises with N, which leaves the
            # large-N corner free for it.
            ax.legend(loc="lower right", framealpha=0.9, fontsize=6)
        for ax in np.atleast_1d(axes)[1:]:
            ax.set_ylabel("")
        np.atleast_1d(axes)[0].set_ylabel("Solve time (s), geometric mean")
        figure.suptitle(f"Scaling at matched accuracy on the {device.upper()} "
                        f"(blobiness {campaign.reference_blobiness:g}; solid: measured; "
                        "dashed and hollow: projected; legend: fitted exponent)", y=1.02)
        figure.tight_layout()
        campaign.save(figure, f"scaling_{device}.png")


# ── Per-porosity solve time ──────────────────────────────────────────

def figure_time_bars(campaign):
    """Grouped bars rather than a curve through an average.

    Collapsing five porosities into one number hides that the ranking flips
    between a dense medium, where the solve dominates, and a nearly-open one,
    where startup does.
    """
    for device in campaign.devices(campaign.timings):
        sizes = campaign.sizes(campaign.timings, device, campaign.reference_blobiness)
        series = campaign.series_on(campaign.timings, device)
        if not sizes or not series:
            continue
        rows, cols = len(fig.THRESHOLDS), len(sizes)
        figure, axes = plt.subplots(rows, cols, squeeze=False, sharey="col",
                                    figsize=(3.0 * cols, 2.4 * rows))
        x = np.arange(len(campaign.porosities))
        width = 0.80 / len(series)
        blanks = []

        for r, (target, label) in enumerate(zip(fig.THRESHOLDS, fig.THRESHOLD_LABELS)):
            for k, name in enumerate(series):
                times = fig.target_times(campaign.timings, target, device=device, series=name,
                                         sizes=sizes, porosities=campaign.porosities,
                                         blobiness=campaign.reference_blobiness)
                _, color = fig.style_for(name)
                offset = (k - (len(series) - 1) / 2) * width
                for c, size in enumerate(sizes):
                    ax = axes[r][c]
                    for i, por in enumerate(campaign.porosities):
                        value = times[(size, por)]
                        if np.isfinite(value):
                            ax.bar(x[i] + offset, value, width, color=color,
                                   edgecolor="white", linewidth=0.4, zorder=3)
                        else:
                            blanks.append((ax, x[i] + offset))
            for c, size in enumerate(sizes):
                ax = axes[r][c]
                ax.set_yscale("log")
                ax.set_xticks(x)
                ax.set_xticklabels([f"{p:.2f}" for p in campaign.porosities])
                ax.set_title(f"$N = {size}$,  target $\\leq {label}$")
                ax.grid(True, axis="y", zorder=0)
                if r == rows - 1:
                    ax.set_xlabel("Porosity $\\varepsilon$")
                if c == 0:
                    ax.set_ylabel("Solve time (s)")

        # The limits have to exist before the "no result" marks can be placed on
        # each panel's floor, and sharey='col' means one call per column.
        for c in range(cols):
            heights = [p.get_height() for r in range(rows) for p in axes[r][c].patches
                       if p.get_height() > 0]
            if heights:
                axes[0][c].set_ylim(10 ** np.floor(np.log10(min(heights))),
                                    10 ** np.ceil(np.log10(max(heights))))
        for ax, xpos in blanks:
            ax.plot(xpos, ax.get_ylim()[0] * 1.25, marker=MISSING_MARKER, color="0.55",
                    markersize=3.5, markeredgewidth=0.8, linestyle="none", zorder=4)

        handles = [Patch(facecolor=fig.style_for(name)[1], label=name) for name in series]
        if blanks:
            handles.append(Line2D([], [], marker=MISSING_MARKER, color="0.55", linestyle="none",
                                  markeredgewidth=0.8, label="target never reached"))
        figure.legend(handles=handles, loc="lower center", ncol=len(handles),
                      frameon=False, bbox_to_anchor=(0.5, -0.02))
        figure.suptitle(f"Solve time per porosity on the {device.upper()}", y=1.0)
        figure.tight_layout()
        campaign.save(figure, f"time_bars_{device}.png")


# ── Speedup regime maps ──────────────────────────────────────────────

def speedup_matrix(campaign, device, competitor, target, sizes):
    """Speedup of the reference series over `competitor`, porosity by size.

    A cell is NaN when either tool never reached the target on that image, which
    is a real outcome and is drawn blank rather than filled with a guess.
    """
    reference = fig.series_label(*fig.REFERENCE_SERIES)
    ours = fig.target_times(campaign.timings, target, device=device, series=reference,
                            sizes=sizes, porosities=campaign.porosities,
                            blobiness=campaign.reference_blobiness)
    theirs = fig.target_times(campaign.timings, target, device=device, series=competitor,
                              sizes=sizes, porosities=campaign.porosities,
                              blobiness=campaign.reference_blobiness)
    matrix = np.full((len(campaign.porosities), len(sizes)), np.nan)
    for i, por in enumerate(campaign.porosities):
        for j, size in enumerate(sizes):
            mine, other = ours[(size, por)], theirs[(size, por)]
            if np.isfinite(mine) and np.isfinite(other) and mine > 0:
                matrix[i, j] = other / mine
    return matrix


def figure_speedups(campaign):
    reference = fig.series_label(*fig.REFERENCE_SERIES)
    for device in campaign.devices(campaign.timings):
        if reference not in campaign.series_on(campaign.timings, device):
            logger.warning(f"{reference} missing on the {device} — no speedups there")
            continue
        sizes = campaign.sizes(campaign.timings, device, campaign.reference_blobiness)
        for competitor in campaign.series_on(campaign.timings, device):
            if competitor == reference:
                continue
            matrices = [speedup_matrix(campaign, device, competitor, t, sizes)
                        for t in fig.THRESHOLDS]
            span = fig.speedup_span(matrices)
            if span is None:
                logger.warning(f"no overlapping runs for {reference} vs {competitor} on the {device}")
                continue
            figure, axes = plt.subplots(1, len(fig.THRESHOLDS), sharey=True, layout="constrained",
                                        figsize=(3.3 * len(fig.THRESHOLDS) + 0.8, 2.9))
            image = None
            for ax, matrix, label in zip(axes, matrices, fig.THRESHOLD_LABELS):
                image = fig.draw_speedup_heatmap(ax, matrix, sizes, campaign.porosities, span=span)
                ax.set_title(f"target $\\leq {label}$")
            for ax in axes[1:]:
                ax.set_ylabel("")
            fig.speedup_colorbar(figure, image, list(axes), span,
                                 "Tortuosity.jl slower ← → faster")
            figure.suptitle(f"{reference} vs {competitor}, both on the {device.upper()}")
            # Slugged from the whole label, not just the tool: two Tortuosity.jl
            # variants compared on one device would otherwise write to the same
            # file and the second would silently replace the first.
            campaign.save(figure, f"speedup_{fig.slugify(competitor)}_{device}.png")


# ── Memory ───────────────────────────────────────────────────────────

# A series whose per-voxel footprint is this consistent across the sizes it was
# measured at is holding dense arrays over the whole grid, and projecting it to a
# size it never reached is arithmetic rather than speculation. Looser than this
# and the projection is a guess, so it is not drawn at all.
PROJECTION_TOLERANCE = 0.05


def project_per_voxel(values, target):
    """Project `values` -- {size: GiB} -- to `target`, if the footprint is flat.

    Returns `(gib, per_voxel_spread)` or `(None, spread)` when the series does not
    scale cleanly with voxel count. Extrapolating anything else would put a number
    on the page that no measurement supports.
    """
    if len(values) < 3 or target in values:
        return None, None
    per_voxel = {n: gib / n ** 3 for n, gib in values.items()}
    lo, hi = min(per_voxel.values()), max(per_voxel.values())
    spread = (hi - lo) / lo if lo else float("inf")
    if spread > PROJECTION_TOLERANCE:
        return None, spread
    # Weight the largest measured sizes: the small ones carry a fixed overhead
    # that is a real part of their footprint and a vanishing part of the target.
    largest = sorted(per_voxel)[-2:]
    rate = sum(per_voxel[n] for n in largest) / len(largest)
    return rate * target ** 3, spread


def exhausted_memory(memory, *, device, series, size, porosity, blobiness):
    """Whether this case was run and ran out of memory, rather than not run."""
    sub = memory[(memory["device"] == device) & (memory["series"] == series)
                 & (memory["size"] == size)
                 & np.isclose(memory["porosity_target"], porosity)
                 & np.isclose(memory["blobiness"], blobiness)]
    return not sub.empty and (sub["status"] == "oom").any()


def figure_memory(campaign):
    """Peak memory by domain size, one panel per porosity, every tool.

    Grouped bars rather than lines. The comparison the paper rests on is between
    heights at one size -- the two operator forms against each other, and against
    the other packages -- and a bar chart puts those heights side by side instead
    of asking the reader to follow five markers across a log-log plot.

    A tool that was never run at a size gets a hatched bar carrying the projection
    of its own per-voxel footprint, and only when that footprint is flat enough
    across the sizes it *was* measured at for the projection to be arithmetic. The
    hatch and the legend say plainly that the bar is not a measurement.
    """
    for device in campaign.devices(campaign.memory):
        column = MEMORY_COLUMN[device]
        series = campaign.series_on(campaign.memory, device)
        sizes = campaign.sizes(campaign.memory, device, campaign.reference_blobiness)
        if not series or not sizes:
            continue
        # Every series is drawn over the union of sizes, so a projected bar has a
        # slot to stand in rather than leaving a gap the eye reads as zero.
        all_sizes = sorted(set(sizes) | set(campaign.sizes(campaign.memory, device, None) or sizes))
        # Gather every panel before drawing any: the bars sit on a log axis, and a
        # log axis has no zero to start them from, so the base has to be chosen
        # from the smallest value in the whole figure.
        panels = {}
        for por in campaign.porosities:
            measured = {}
            for name in series:
                values = {}
                for size in all_sizes:
                    gib = fig.memory_gib(campaign.memory, device=device, series=name, size=size,
                                         porosity=por, blobiness=campaign.reference_blobiness,
                                         column=column)
                    if np.isfinite(gib):
                        values[size] = gib
                if values:
                    measured[name] = values
            panels[por] = measured
        every = [v for m in panels.values() for vals in m.values() for v in vals.values()]
        if not every:
            continue
        base = min(every) / 4

        figure, axes = plt.subplots(1, len(campaign.porosities), sharey=True,
                                    figsize=(3.0 * len(campaign.porosities), 3.6))
        axes = np.atleast_1d(axes)
        projected_any = False
        exhausted_any = False
        for ax, por in zip(axes, campaign.porosities):
            measured = panels[por]

            width = 0.8 / max(len(measured), 1)
            index = np.arange(len(all_sizes), dtype=float)
            oom_slots = []
            for i, (name, values) in enumerate(sorted(measured.items())):
                _, color = fig.style_for(name)
                offset = (i - (len(measured) - 1) / 2) * width
                solid_x, solid_h, proj_x, proj_h = [], [], [], []
                for j, size in enumerate(all_sizes):
                    if size in values:
                        solid_x.append(index[j] + offset)
                        solid_h.append(values[size])
                        continue
                    if exhausted_memory(campaign.memory, device=device, series=name,
                                        size=size, porosity=por,
                                        blobiness=campaign.reference_blobiness):
                        # Ran, and did not fit. That is the measurement.
                        oom_slots.append((index[j] + offset, color))
                        continue
                    gib, _ = project_per_voxel(values, size)
                    if gib is not None:
                        proj_x.append(index[j] + offset)
                        proj_h.append(gib)
                if solid_h:
                    ax.bar(solid_x, [h - base for h in solid_h], width, bottom=base,
                           color=color, label=name, edgecolor="white", linewidth=0.4)
                if proj_h:
                    projected_any = True
                    ax.bar(proj_x, [h - base for h in proj_h], width, bottom=base,
                           color=color, alpha=0.55, hatch="////", edgecolor="white",
                           linewidth=0.4)

            ax.set_title(f"$\\varepsilon = {por:.2f}$")
            ax.set_xticks(index)
            ax.set_xticklabels([str(n) for n in all_sizes], fontsize=7)
            ax.set_yscale("log")
            ax.set_ylim(bottom=base)
            # Drawn after the limits settle so the bar reaches the top of the
            # panel: the point is that the case has no height on this scale.
            if oom_slots:
                exhausted_any = True
                top = ax.get_ylim()[1]
                for x, color in oom_slots:
                    ax.bar([x], [top - base], width, bottom=base, color=color,
                           alpha=0.30, hatch="xxx", edgecolor="white", linewidth=0.4)
            ax.set_xlabel("Domain size $N$ (voxels per side)")
            ax.grid(True, axis="y", alpha=0.4)
            ax.set_axisbelow(True)

        axes[0].set_ylabel(MEMORY_LABEL[device])
        handles, labels = axes[0].get_legend_handles_labels()
        if exhausted_any:
            handles = list(handles) + [Patch(facecolor="0.6", alpha=0.30, hatch="xxx",
                                             edgecolor="white")]
            labels = list(labels) + ["exhausted the device"]
        if projected_any:
            handles = list(handles) + [Patch(facecolor="0.6", alpha=0.55, hatch="////",
                                             edgecolor="white")]
            labels = list(labels) + ["projected, not measured"]
        if handles:
            axes[0].legend(handles, labels, loc="upper left", framealpha=0.9, fontsize=7)
        figure.suptitle(f"Memory held during the solve on the {device.upper()} "
                        f"(blobiness {campaign.reference_blobiness:g})", y=1.02)
        figure.tight_layout()
        campaign.save(figure, f"memory_{device}.png")


# ── Structure sensitivity ────────────────────────────────────────────

def figure_blobiness(campaign):
    """How the microstructure itself changes the answer, and the cost of getting it.

    Porosity alone does not describe a porous medium: at one pore fraction a
    coarse structure and a fine one differ in tortuosity by a factor that grows
    as the medium closes up. The left panel establishes that the three structures
    really are different problems; the rest ask whether the ranking between
    solvers survives that difference.
    """
    blobinesses = sorted(float(b) for b in campaign.cfg["image"]["blobinesses"])
    if len(blobinesses) < 2 or campaign.timings.empty:
        logger.warning("fewer than two blobiness values measured — skipping the structure figure")
        return
    devices = campaign.devices(campaign.timings)
    size = max(campaign.sizes(campaign.timings))

    figure, axes = plt.subplots(1, 1 + len(devices), squeeze=False,
                                figsize=(3.4 * (1 + len(devices)), 3.0))
    axes = axes[0]

    truth = (campaign.references[campaign.references["size"] == size]
             if not campaign.references.empty else campaign.references)
    for por in campaign.porosities:
        row = truth[np.isclose(truth["porosity_target"], por)].sort_values("blobiness")
        if row.empty:
            continue
        axes[0].plot(row["blobiness"], row["tau_ref"], marker="o",
                     label=f"$\\varepsilon = {por:.2f}$")
    axes[0].set_xlabel("Blobiness (higher = finer features)")
    axes[0].set_ylabel("Ground-truth tortuosity $\\tau$")
    axes[0].set_yscale("log")
    axes[0].set_title(f"(a) structure sets $\\tau$, $N = {size}$", loc="left")
    axes[0].grid(True)
    # Five curves fanning out from the lower right leave no corner reliably free,
    # so the placement is left to matplotlib rather than guessed at.
    axes[0].legend(fontsize=7, framealpha=0.9, loc="best")

    for ax, device in zip(axes[1:], devices):
        for name in campaign.series_on(campaign.timings, device):
            xs, ys = [], []
            for blob in blobinesses:
                times = fig.target_times(campaign.timings, fig.PAPER_TARGET, device=device,
                                         series=name, sizes=[size],
                                         porosities=campaign.porosities, blobiness=blob)
                hits = [t for t in times.values() if np.isfinite(t)]
                if hits:
                    xs.append(blob)
                    ys.append(fig.geomean(hits))
            if not xs:
                continue
            marker, color = fig.style_for(name)
            ax.plot(xs, ys, marker=marker, color=color, label=name,
                    markeredgewidth=0.5, markeredgecolor="white")
        ax.set_xlabel("Blobiness (higher = finer features)")
        ax.set_ylabel("Solve time to $\\leq 0.1\\%$ (s)")
        ax.set_yscale("log")
        ax.set_title(f"({'bcd'[devices.index(device)]}) {device.upper()}, $N = {size}$", loc="left")
        ax.grid(True)
        ax.legend(fontsize=7, framealpha=0.9)

    figure.suptitle("Sensitivity to microstructure at fixed porosity", y=1.03)
    figure.tight_layout()
    campaign.save(figure, "blobiness.png")


# ── Each tool's own accuracy/time frontier ───────────────────────────

def figure_pareto(campaign):
    """Each tool's own frontier, spanning every size measured.

    A property of one tool's own knob rather than a like-for-like ranking, so
    these are not restricted to the sizes the headline figures use.
    """
    for device in campaign.devices(campaign.timings):
        sizes = campaign.sizes(campaign.timings, device, campaign.reference_blobiness)
        if not sizes:
            continue
        figure, axes = plt.subplots(1, len(sizes), squeeze=False, sharey=True,
                                    figsize=(3.0 * len(sizes), 2.9))
        for ax, size in zip(axes[0], sizes):
            for name in campaign.series_on(campaign.timings, device):
                sub = campaign.timings[
                    (campaign.timings["device"] == device)
                    & (campaign.timings["series"] == name)
                    & (campaign.timings["size"] == size)
                    & np.isclose(campaign.timings["blobiness"], campaign.reference_blobiness)
                    & (campaign.timings["rel_error"] > 0) & (campaign.timings["time_s"] > 0)]
                if sub.empty:
                    continue
                grouped = sub.groupby("knob")[["rel_error", "time_s"]].agg(fig.geomean)
                grouped = grouped.sort_values("time_s").dropna()
                if grouped.empty:
                    continue
                marker, color = fig.style_for(name)
                ax.plot(grouped["time_s"], grouped["rel_error"], marker=marker, color=color,
                        label=name, markeredgewidth=0.5, markeredgecolor="white")
            ax.set_xlabel("Solve time (s)")
            ax.set_title(f"{device.upper()}, $N = {size}$")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-6)
            ax.grid(True)
        axes[0][0].set_ylabel("Relative error in $\\tau$")
        # Frontiers run top-left to bottom-right, leaving the lower-left corner free.
        axes[0][-1].legend(loc="lower left", framealpha=0.9, fontsize=7)
        figure.suptitle(f"Accuracy vs. solve time on the {device.upper()} "
                        "(geometric mean over porosities)", y=1.01)
        figure.tight_layout()
        campaign.save(figure, f"pareto_{device}.png")


# ── Paper summary ────────────────────────────────────────────────────

def figure_summary(campaign):
    """One row per device: how each tool scales, and where we win or lose."""
    reference = fig.series_label(*fig.REFERENCE_SERIES)
    devices = [d for d in campaign.devices(campaign.timings)
               if reference in campaign.series_on(campaign.timings, d)]
    if not devices:
        logger.warning("reference series missing everywhere — skipping the summary")
        return
    figure, axes = plt.subplots(len(devices), 2, squeeze=False, layout="constrained",
                                figsize=(7.6, 3.1 * len(devices)))
    tag = iter("abcdefgh")
    for row, device in enumerate(devices):
        ax_scale, ax_map = axes[row]
        sizes = campaign.sizes(campaign.timings, device, campaign.reference_blobiness)
        draw_scaling_panel(ax_scale, campaign, device, fig.PAPER_TARGET, sizes)
        ax_scale.set_ylabel("Solve time to $\\leq 0.1\\%$ error (s),\ngeometric mean over porosities")
        # Every curve rises with N, so the large-N/short-time corner holds no data.
        ax_scale.legend(loc="lower right", framealpha=0.9, fontsize=7)
        ax_scale.set_title(f"({next(tag)}) {device.upper()}: scaling at matched accuracy", loc="left")

        available = campaign.series_on(campaign.timings, device)
        preferred = fig.series_label(*fig.HEADLINE_COMPETITOR[device])
        rivals = [s for s in available if s != reference and not s.startswith("Tortuosity.jl")]
        if not rivals:
            ax_map.axis("off")
            continue
        competitor = preferred if preferred in rivals else rivals[0]
        matrix = speedup_matrix(campaign, device, competitor, fig.PAPER_TARGET, sizes)
        fig.draw_speedup_heatmap(ax_map, matrix, sizes, campaign.porosities)
        ax_map.set_title(f"({next(tag)}) vs {competitor}, both on the {device.upper()}", loc="left")

    figure.suptitle("Benchmark summary at $\\leq 0.1\\%$ relative error in $\\tau$ "
                    "(solid: measured; dashed and hollow: projected; legend: fitted exponent)")
    campaign.save(figure, "summary.png")


FIGURES = {
    "scaling": figure_scaling,
    "time_bars": figure_time_bars,
    "speedup": figure_speedups,
    "memory": figure_memory,
    "blobiness": figure_blobiness,
    "pareto": figure_pareto,
    "summary": figure_summary,
}


def publishable(campaign):
    """Whether the drawn data is the campaign grid rather than a validation run.

    Publishing copies figures into `paper/` and `docs/src/assets/`, where they are
    the numbers a reader sees. A smoke-grid run draws perfectly valid figures of
    20-100³ toy data, and nothing about the image says so — so without this guard
    a validation run silently replaces the paper's figures with toy plots, and the
    document still builds and still looks right.
    """
    configured = set(bkconfig.sizes(campaign.cfg, campaign.cfg["campaign"]["grid"]))
    drawn = set(campaign.sizes(campaign.timings)) if not campaign.timings.empty else set()
    return bool(drawn & configured)


def publish(campaign, repo_root):
    for name, (published, targets) in PUBLISH.items():
        source = campaign.outdir / name
        if not source.exists():
            logger.warning(f"{name} was not drawn — not publishing it")
            continue
        for target in targets:
            destination = repo_root / target
            if not destination.is_dir():
                logger.warning(f"{destination} does not exist — not publishing {name}")
                continue
            shutil.copyfile(source, destination / published)
            logger.info(f"published {name} → {target}/{published}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help=f"comma-separated subset of {', '.join(FIGURES)}")
    parser.add_argument("--outdir", help="where to write figures (default: benchmarks/figures)")
    parser.add_argument("--no-publish", action="store_true",
                        help="do not copy figures into paper/ and docs/src/assets/")
    parser.add_argument("--force-publish", action="store_true",
                        help="publish even when the results are not from the campaign grid")
    args = parser.parse_args()

    cfg = bkconfig.load_config()
    plt.rcParams.update(fig.STYLE)
    campaign = Campaign(cfg, args.outdir or cfg.root / "figures")

    wanted = [s.strip() for s in args.only.split(",")] if args.only else list(FIGURES)
    unknown = [w for w in wanted if w not in FIGURES]
    if unknown:
        raise SystemExit(f"unknown figure(s) {', '.join(unknown)}; choose from {', '.join(FIGURES)}")

    logger.info(f"timings: {len(campaign.timings)} rows, memory: {len(campaign.memory)} rows, "
                f"reference blobiness {campaign.reference_blobiness:g}")
    for name in wanted:
        logger.info(f"drawing {name}")
        FIGURES[name](campaign)

    if args.no_publish:
        pass
    elif publishable(campaign) or args.force_publish:
        publish(campaign, cfg.root.parent)
    else:
        logger.warning(f"not publishing: the results hold none of the "
                       f"{cfg['campaign']['grid']} grid's sizes, so this looks like a validation "
                       "run. Pass --force-publish to override.")
    logger.success(f"figures in {campaign.outdir}")


if __name__ == "__main__":
    main()
