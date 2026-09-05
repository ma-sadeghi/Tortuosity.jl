"""What the campaign is willing to estimate for a case nobody ran, and on what.

Kept apart from the drawing so the rules a projected number rests on — the fits,
the exponents, the footprint rates and the refusals — read as one model rather
than as asides between two matplotlib calls. Nothing here draws anything; the
figures decide how an estimate is marked, and this decides whether there is one.

`campaign` is passed through opaquely: these functions read the result frames and
the grid off it, and use it to hold the two caches every projected bar asks for.
"""

import numpy as np
from loguru import logger

from benchkit import figures as fig

# Fewest sizes a porosity must have been measured at before its power law is
# fitted and used to stand in for a size it was not measured at. Two points
# define a line with nothing left over to judge it by.
PROJECTION_MIN_SIZES = 3

# A series whose per-voxel footprint is this consistent across the sizes it was
# measured at is holding dense arrays over the whole grid, and projecting it to a
# size it never reached is arithmetic rather than speculation. Looser than this
# and the projection is a guess, so it is not drawn at all.
PROJECTION_TOLERANCE = 0.05


def pore_counts(campaign):
    """`{(size, porosity, blobiness): pore voxels}` from the reference table.

    Built once per campaign and cached: every projected memory bar asks for it,
    and the table is walked row by row.
    """
    if campaign.nodes is None:
        campaign.nodes = {} if campaign.references.empty else {
            (int(r["size"]), float(r.porosity_target), float(r.blobiness)): int(r.nnodes)
            for _, r in campaign.references.iterrows()}
    return campaign.nodes


def out_of_memory(timings, *, device, series, size, porosity, blobiness):
    """Whether this case failed on the hardware rather than going unmeasured.

    Every `error` row in the campaign is an out-of-memory failure, so a cell
    holding one is not a measurement someone chose to skip: the tool cannot solve
    that image on this machine at all. Projecting a time for it would put a
    number on the page for a run that can never happen, which is a different
    claim from projecting one that was merely never afforded.
    """
    cell = fig.select(timings, device=device, series=series, size=size,
                      porosity_target=porosity, blobiness=blobiness)
    return bool((cell["stop_reason"] == "error").any())


def unconverged_time(timings, device, series, size, porosity, blobiness):
    """Longest run this series spent on the image without reaching the target.

    A tool that ran and stopped short still proves its matched-accuracy time
    exceeds what it already spent. That makes the number a floor a projection has
    to clear: an estimate below it is contradicted by a measurement.
    """
    cell = fig.select(timings, device=device, series=series, size=size,
                      porosity_target=porosity, blobiness=blobiness)
    cell = cell[cell["time_s"] > 0]
    return float(cell["time_s"].max()) if len(cell) else None


def exhausted_memory(memory, *, device, series, size, porosity, blobiness):
    """Whether this case was run and ran out of memory, rather than not run."""
    sub = fig.select(memory, device=device, series=series, size=size,
                     porosity_target=porosity, blobiness=blobiness)
    return not sub.empty and (sub["status"] == "oom").any()


def porosity_power_laws(times, sizes, porosities, node_count, probe):
    """One power law per porosity, for filling cells a tool never reached.

    Returned as `{porosity: (a, exponent, r2)}`, so a missing cell is
    `a * n ** exponent` in that porosity's own transporting-voxel count. A
    porosity with too few measured sizes and no probe to borrow an exponent from
    is absent rather than guessed at.

    Both the scaling panels and the speedup maps fill their gaps from this, so
    the two agree by construction. Two projection methods in one figure would
    disagree at some cell, and the reader has no way to tell which they are
    looking at.
    """
    fits = {}
    for por in porosities:
        measured = [n for n in sizes if np.isfinite(times[(n, por)])]
        if len(measured) >= PROJECTION_MIN_SIZES:
            counts = [node_count(n, por) for n in measured]
            fits[por] = fig.power_law(counts, [times[(n, por)] for n in measured])
        elif probe is not None and measured:
            anchor = max(measured)
            anchor_voxels = node_count(anchor, por)
            fits[por] = (times[(anchor, por)] / anchor_voxels ** probe,
                         probe, np.nan)
    return fits


def scaling_points(campaign, device, target, sizes, series, *, project_oom=False):
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
    reach. A porosity that ran out of memory is not filled in by default, and
    takes its whole size off the curve. ``project_oom`` permits an explicitly
    hypothetical timing projection when the caller labels it as such.
    """
    times = fig.target_times(campaign.timings, target, device=device, series=series,
                             sizes=sizes, porosities=campaign.porosities,
                             blobiness=campaign.reference_blobiness)
    nodes = pore_counts(campaign)

    def node_count(size, porosity):
        return nodes[(size, porosity, campaign.reference_blobiness)]

    # One power law per porosity rather than one for the series. The transporting-
    # voxel exponent rises as the pore space closes -- taufactor spans 1.0 to 1.3
    # on the GPU and 1.1 to 1.7 on the CPU -- so one exponent would project the
    # dense images, the ones that are missing, at the open images' rate.
    # A tool swept at one size has no fit of its own, but the campaign measures
    # its size dependence directly in `results/scaling-probes.csv` — a matched
    # pair of solves on one image at two sizes. Given that exponent, the one
    # measured size fixes the constant and the whole curve follows. The result is
    # an estimate and is drawn as one, dashed and hollow at every projected size,
    # exactly as a projected porosity is.
    probe = fig.probe_exponent(campaign.probes, series, device)
    fits = porosity_power_laws(times, sizes, campaign.porosities, node_count, probe)

    # `probed` records that a filled-in value came from the probe rather than
    # from this series' own per-porosity fit, which is what the log below has to
    # distinguish. Named apart from `fit` so the series-level fit computed after
    # the loop is not read back through a name the loop also binds.
    points, projected, probed = [], [], False
    for size in sizes:
        filled, borrowed = [], False
        for por in campaign.porosities:
            value = times[(size, por)]
            if np.isfinite(value):
                filled.append(value)
                continue
            porosity_fit = fits.get(por)
            exhausted = out_of_memory(
                campaign.timings, device=device, series=series, size=size,
                porosity=por, blobiness=campaign.reference_blobiness
            )
            if porosity_fit is None or (exhausted and not project_oom):
                filled = None
                break
            a, exponent, _ = porosity_fit
            filled.append(a * node_count(size, por) ** exponent)
            borrowed = True
            probed = probed or (probe is not None and exponent == probe)
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
    solid_counts = [fig.geomean([node_count(n, por)
                                 for por in campaign.porosities])
                    for n, _ in solid]
    fit = fig.power_law(solid_counts, [t for _, t in solid]) if len(solid) >= 3 else None
    # Say what was projected and what was dropped. A curve that quietly stops
    # short, or quietly rests on a fit, reads exactly like one that does neither.
    if projected:
        source = (f"a probed exponent of {probe:.2f}" if probed
                  else "a per-porosity power law")
        logger.info(f"{series} on the {device} at {target:g}: projected {projected} from {source}")
    dropped = [n for n in sizes if n not in [p for p, _ in points]]
    if dropped:
        logger.info(f"{series} on the {device} at {target:g}: left {dropped} off, where a "
                    "porosity was neither measured nor projectable")
    # `estimated` says where the exponent in the legend came from: a fit through
    # this series' own measured means, or a probe run to calibrate a projection.
    # They deserve different labels, and only the caller knows how to draw them.
    estimated = fit is None and probe is not None
    if estimated:
        fit = (np.nan, probe, np.nan)
    return points, projected, fit, estimated


def projected_time(campaign, device, series, times, size, porosity, sizes):
    """Time for a cell nobody ran, from this series' probed exponent, or None.

    Only for the tools swept at a single size: everything else either has the
    cell or has enough of its own sizes to be fitted, and a projection laid over
    either would be drawing a model where a measurement already sits.
    """
    probe = fig.probe_exponent(campaign.probes, series, device)
    if probe is None:
        return None
    measured = [n for n in sizes if np.isfinite(times[(n, porosity)])]
    if not measured or len(measured) >= PROJECTION_MIN_SIZES:
        return None
    anchor = max(measured)
    nodes = pore_counts(campaign)
    anchor_key = (anchor, porosity, campaign.reference_blobiness)
    target_key = (size, porosity, campaign.reference_blobiness)
    if nodes.get(anchor_key, 0) <= 0 or nodes.get(target_key, 0) <= 0:
        return None
    voxel_ratio = nodes[target_key] / nodes[anchor_key]
    return times[(anchor, porosity)] * voxel_ratio ** probe


def speedup_matrix(campaign, device, competitor, target, sizes, *, project=False):
    """Speedup of the reference series over `competitor`, porosity by size.

    Returns `(matrix, projected)`. A cell is NaN when either tool never reached
    the target on that image. Without `project` that stays blank, which is the
    honest reading of a cell nobody measured.

    With `project`, a cell the competitor never reached is filled from its own
    per-porosity power law — the same fit the scaling panels use — and flagged in
    `projected` so the caller can draw it as an estimate. A cell we ourselves
    never reached is never filled: the ratio would then rest on two estimates,
    and the panel exists to report our margin, not to model both sides of it.
    """
    reference = fig.series_label(*fig.REFERENCE_SERIES)
    ours = fig.target_times(campaign.timings, target, device=device, series=reference,
                            sizes=sizes, porosities=campaign.porosities,
                            blobiness=campaign.reference_blobiness)
    theirs = fig.target_times(campaign.timings, target, device=device, series=competitor,
                              sizes=sizes, porosities=campaign.porosities,
                              blobiness=campaign.reference_blobiness)
    matrix = np.full((len(campaign.porosities), len(sizes)), np.nan)
    projected = np.zeros(matrix.shape, dtype=bool)

    fits, node_count = {}, None
    if project:
        nodes = pore_counts(campaign)

        def node_count(size, porosity):
            return nodes.get((size, porosity, campaign.reference_blobiness), 0)

        probe = fig.probe_exponent(campaign.probes, competitor, device)
        fits = porosity_power_laws(theirs, sizes, campaign.porosities, node_count, probe)

    for i, por in enumerate(campaign.porosities):
        for j, size in enumerate(sizes):
            mine, other = ours[(size, por)], theirs[(size, por)]
            if not np.isfinite(mine) or mine <= 0:
                continue
            if np.isfinite(other):
                matrix[i, j] = other / mine
                continue
            fit = fits.get(por)
            if fit is None or node_count(size, por) <= 0:
                continue
            a, exponent, _ = fit
            estimate = a * node_count(size, por) ** exponent
            floor = unconverged_time(campaign.timings, device, competitor, size, por,
                                     campaign.reference_blobiness)
            if floor is not None and estimate < floor:
                logger.warning(
                    f"{competitor} on the {device} at N={size}, eps={por:g}: projected "
                    f"{estimate:.0f} s is below the {floor:.0f} s it already spent "
                    "without converging — reporting the measured floor instead"
                )
                estimate = floor
            matrix[i, j] = estimate / mine
            projected[i, j] = True
    return matrix, projected


def footprint_rate(campaign, device, series, blobiness, column):
    """What one series' memory scales with: `("voxel" | "pore", rate)` or `(None, None)`.

    Decided from the measurements rather than declared. A footprint that is the
    same per grid voxel wherever it was measured is holding the grid; one that is
    the same per pore voxel is holding the pore space. Testing across porosity as
    well as across size is what lets a tool measured at a single size be projected
    at all: five porosities at one size separate the two shapes as sharply as
    three sizes at one porosity do, because the pore count moves by a factor of
    five across that range while the grid does not move at all.

    Rate and spread are both taken over the largest half of the measurements. The
    small ones carry a fixed interpreter and image overhead that is a real part of
    their own footprint and a vanishing part of the target's, and including them
    would fail a series that scales perfectly well.

    "Largest" is by the denominator being tested, and every measurement tied with
    the one at the cut is kept. Ordering the ties by their rate instead would pick
    the half whose rates already agree, which is the very thing the spread below
    is asking about: the gate would then pass on a series it should reject, and
    average a rate biased to one end of the spread it just ignored.
    """
    cached = campaign.rates.get((device, series, blobiness))
    if cached is not None:
        return cached
    nodes = pore_counts(campaign)
    rows = []
    for _, r in fig.select(campaign.memory, device=device, series=series,
                           blobiness=blobiness, status="ok").iterrows():
        gib = r[column] / 2 ** 30
        n = nodes.get((int(r["size"]), float(r.porosity_target), float(r.blobiness)))
        if np.isfinite(gib) and gib > 0 and n:
            rows.append((gib, int(r["size"]) ** 3, n))

    answer = (None, None)
    if len(rows) >= 3:
        for kind, index in (("voxel", 1), ("pore", 2)):
            pairs = sorted(((row[index], row[0] / row[index]) for row in rows),
                           key=lambda pair: pair[0])
            cut = pairs[len(pairs) // 2][0]
            largest = [rate for key, rate in pairs if key >= cut]
            lo, hi = min(largest), max(largest)
            if lo > 0 and (hi - lo) / lo <= PROJECTION_TOLERANCE:
                answer = (kind, sum(largest) / len(largest))
                break
    campaign.rates[(device, series, blobiness)] = answer
    return answer


def project_footprint(campaign, device, series, size, porosity, blobiness, column):
    """Projected GiB for a cell that was never run, or None when nothing supports one."""
    kind, rate = footprint_rate(campaign, device, series, blobiness, column)
    if kind is None:
        return None
    if kind == "voxel":
        return rate * size ** 3
    n = pore_counts(campaign).get((int(size), float(porosity), float(blobiness)))
    return rate * n if n else None
