"""Benchmark PoreSpy's ``tortuosity_fd`` on the shared image store. CPU only.

    pixi run -e porespy python bench_porespy.py
    pixi run -e porespy python bench_porespy.py --measure=memory

PoreSpy has no GPU path, so like PuMA it appears only in the CPU comparison. Its
method is the third distinct one in the campaign: the image becomes an OpenPNM
cubic network over the pore voxels alone, Fickian diffusion is assembled on it,
and the system is solved by Ruge-Stuben algebraic multigrid from PyAMG. That
makes it the closest external counterpart to what Tortuosity.jl does — a
pore-only system with a multilevel preconditioner — and the comparison worth
having for exactly that reason.

The sweep knob is the solver tolerance rather than an iteration count. PyAMG is
driven through ``PyamgRugeStubenSolver``, which takes a tolerance and returns
only when it has met it, so there is no iteration cap to trace a frontier with.
Tolerance is a usable knob here where it was a poor one for the Krylov and SOR
tools: multigrid converges at a rate that barely changes from one rung to the
next, so the rungs land evenly rather than piling up at one end.

Each rung is a fresh set of multigrid cycles over one shared setup. Trimming,
network construction, assembly and the multigrid hierarchy do not depend on the
tolerance, so they are built once and their measured cost is added to every
rung, which is what the tool charges a user who asks for that tolerance
directly. Rebuilding them per rung would charge one fixed cost eighteen times and
measure the ladder rather than the solver.

Runs in its own pixi environment. PoreSpy pulls OpenPNM, numba and scikit-image,
and the campaign's default environment holds torch and PuMA; keeping them apart
avoids both the import-order libstdc++ trap that torch sets and any question of
one tool's threading pool affecting another's timing.
"""

import logging
import os
import statistics
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchkit import config as bkconfig  # noqa: E402
from benchkit import images as bkimages  # noqa: E402
from benchkit import memory as bkmemory  # noqa: E402
from benchkit import results as bkresults  # noqa: E402
from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}")

parser = bkconfig.build_parser(__doc__.splitlines()[0])
parser.add_argument("--measure", default="time", choices=["time", "memory"])
parser.add_argument("--timeout", type=float, help="seconds; overrides sweep.timeout_s")
args = parser.parse_args()

cfg = bkconfig.load_config()

# Nothing is pinned, matching the rest of the CPU stages: PyAMG and SciPy size
# their pools to the machine, which is what a PoreSpy user gets.
cpu_threads = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()

import numpy as np  # noqa: E402
import openpnm as op  # noqa: E402
import porespy as ps  # noqa: E402
import pyamg  # noqa: E402
from porespy.filters import trim_nonpercolating_paths  # noqa: E402
from porespy.generators import faces  # noqa: E402

# Read off PoreSpy's own solver rather than written down here, so that a change
# in its default iteration cap follows through instead of being contradicted.
solver_maxiter = op.solvers.PyamgRugeStubenSolver().maxiter

warnings.filterwarnings("ignore", category=RuntimeWarning)

# PoreSpy logs a warning whenever trimming removes anything and an error whenever
# the inlet and outlet rates differ by more than 1e-4 relative. Both fire on
# images this campaign expects them to fire on, and neither says the answer is
# wrong: the accuracy of every rung is checked against the reference here, which
# is a stronger test than either. Silenced so a 15-case stage stays readable.
logging.getLogger("porespy").setLevel(logging.CRITICAL)

target_error = float(cfg["sweep"]["target_error"])
n_repeats = int(cfg["sweep"]["repeats"])
repeat_threshold = float(cfg["sweep"]["repeat_threshold_s"])
timeout_s = args.timeout if args.timeout else float(cfg["sweep"]["timeout_s"])
ladder = bkconfig.tolerance_ladder(cfg, "porespy_tolerance")
# What `tortuosity_fd` passes when it is given no solver of its own. The memory
# probe solves to this rather than to the tightest rung on the ladder: the
# footprint is set when the hierarchy is built and does not move with the
# tolerance, so the probe may as well measure the configuration a user gets.
probe_tolerance = 1e-8
sample_interval = float(cfg["memory"]["sample_interval_ms"])
axis = cfg.axis

# `_solve_setup` reads the flux along one axis and computes tortuosity from it,
# following `tortuosity_fd`, which is written for `axis` as an index into the
# image shape. Refusing rather than guessing keeps a solve along one axis from
# being reported along another.
AXES = {"x": 0, "y": 1, "z": 2}
if axis not in AXES:
    raise SystemExit(f"campaign.axis = {axis!r} is not one of {sorted(AXES)}")
axis_index = AXES[axis]

subdir = "timings" if args.measure == "time" else "memory"
outpath = cfg.outputdir / subdir / "porespy-cpu.csv"
columns = bkresults.TIMING_COLUMNS if args.measure == "time" else bkresults.MEMORY_COLUMNS
PREFIX = dict(tool="porespy", device="cpu", variant="fd-amg", cpu_threads=cpu_threads)

manifest = bkimages.read_manifest(cfg)
refs = bkresults.read_references(cfg)
cases = bkconfig.select_cases(cfg, args)
if args.measure == "memory":
    cases = bkconfig.restrict_memory_blobiness(cfg, args, cases)

if args.list_cases:
    print("\n".join(c.id for c in cases))
    raise SystemExit(0)

gaps = [c.id for c in cases if c.id not in manifest]
if gaps:
    raise SystemExit(f"no image for {', '.join(gaps)} — run generate_images.jl first")

done = set() if args.overwrite else (
    bkresults.completed_cases(outpath, knob_name="tolerance") if args.measure == "time"
    else bkresults.measured_cases(outpath)
)
solvable = [c for c in cases if manifest[c.id].nnodes > 0]
runnable = [c for c in solvable if c.id in refs] if args.measure == "time" else solvable
no_reference = [c.id for c in solvable if c.id not in refs]
pending = [c for c in runnable if c.id not in done]

if args.dry_run:
    bkconfig.report_plan(pending, f"bench_porespy --measure={args.measure}", done)
    if no_reference:
        print(f"no reference yet, skipped: {', '.join(no_reference)}")
    print(f"writing to {outpath}")
    raise SystemExit(0)

if no_reference:
    logger.warning(f"skipping {len(no_reference)} case(s) with no ground truth — run compute_references.jl")


def _solve_setup(img):
    """Everything `tortuosity_fd` does apart from the multigrid cycles themselves.

    Returns `(solve, setup_seconds)`, where `solve(tolerance)` runs one multigrid
    solve on the assembled system and returns tortuosity. Mirrors
    `porespy.simulations.tortuosity_fd` step for step, including the `L - 1` in
    the effective diffusivity, which is PoreSpy's correction for boundary
    conditions applied inside the domain rather than on its faces.

    The multigrid hierarchy is built here rather than per rung. OpenPNM's
    `PyamgRugeStubenSolver.solve` calls `pyamg.ruge_stuben_solver(A)` on every
    invocation, and that construction is the larger half of a default 200 cubed
    solve. It depends on the matrix and not on the tolerance, so rebuilding it
    per rung would charge one fixed cost eighteen times and measure the ladder
    rather than the solver. Everything else is PoreSpy's own: the same solver
    class, the same hierarchy, the same `maxiter`.
    """
    started = time.perf_counter()
    im = np.asarray(img, dtype=bool)
    # Computed and then unused, exactly as `tortuosity_fd` does: it keeps this
    # only to warn when trimming changed it. Kept here so the timed region covers
    # the same work the call does. Trimming can remove pore voxels and PoreSpy
    # divides by the porosity that survives, so `eps` below is the post-trim
    # value. The row still records the manifest porosity, which is what keeps all
    # four tools keyed on one number.
    eps0 = im.sum(dtype=np.int64) / im.size  # noqa: F841
    im = trim_nonpercolating_paths(
        im, inlets=faces(im.shape, inlet=axis_index), outlets=faces(im.shape, outlet=axis_index),
    )
    eps = im.sum(dtype=np.int64) / im.size
    if not eps:
        raise RuntimeError("no pores remain after trimming floating pores")

    net = op.network.CubicTemplate(template=im, spacing=1.0)
    phase = op.phase.Phase(network=net)
    phase["throat.diffusive_conductance"] = 1.0
    fd = op.algorithms.FickianDiffusion(network=net, phase=phase)
    inlets = net.coords[:, axis_index] <= 1
    outlets = net.coords[:, axis_index] >= im.shape[axis_index] - 1
    fd.set_value_BC(pores=inlets, values=1.0)
    fd.set_value_BC(pores=outlets, values=0.0)
    fd._update_A_and_b()
    hierarchy = pyamg.ruge_stuben_solver(fd.A.tocsr())
    length = im.shape[axis_index]
    area = np.prod(im.shape) / length
    setup_s = time.perf_counter() - started

    def solve(tolerance):
        fd.x, info = hierarchy.solve(fd.b, tol=tolerance, maxiter=solver_maxiter, return_info=True)
        if info:
            raise RuntimeError(f"pyamg failed to converge, exit code {info}")
        deff = fd.rate(pores=inlets)[0] * (length - 1) / area
        return eps / deff

    return solve, setup_s


def trace_once(img, tau_ref, rungs=None):
    """Walk the tolerance ladder over one shared setup, loosest rung first.

    Each row's time is the setup plus that rung's own solve, because those two
    together are what a user asking `tortuosity_fd` for that tolerance pays. The
    ladder stops at the first rung meeting the accuracy target, so a case that
    converges early never pays for the tighter rungs.
    """
    rungs = ladder if rungs is None else rungs
    solve, setup_s = _solve_setup(img)
    rows = []
    for tolerance in rungs:
        started = time.perf_counter()
        tau_val = solve(tolerance)
        elapsed = setup_s + (time.perf_counter() - started)
        rows.append((float(tolerance), float(tau_val), elapsed))
        if abs(tau_val - tau_ref) / tau_ref <= target_error or elapsed > timeout_s:
            break
    return rows


def sweep_case(writer, case, entry, img, tau_ref):
    """Trace one image once per repeat, writing a row per rung reached."""
    traces = []
    for rep in range(n_repeats):
        rows = trace_once(img, tau_ref)
        if not rows:
            raise RuntimeError("solve produced no checkpoints")
        traces.append(rows)
        if rep == 0 and rows[-1][2] > repeat_threshold:
            break

    n_rungs = min(len(t) for t in traces)
    for rung in range(n_rungs):
        tolerance = traces[0][rung][0]
        assert all(t[rung][0] == tolerance for t in traces), "repeats disagree about the ladder"
        taus = [t[rung][1] for t in traces]
        times = [t[rung][2] for t in traces]

        t_median = statistics.median(times)
        tau_val = statistics.median(taus)
        spread = (max(taus) - min(taus)) / tau_val if len(taus) > 1 else float("nan")
        rel_error = abs(tau_val - tau_ref) / tau_ref

        stop_reason = ""
        if rel_error <= target_error:
            stop_reason = "target_reached"
        elif t_median > timeout_s:
            stop_reason = "timeout"
        elif tolerance == ladder[-1]:
            stop_reason = "ladder_exhausted"
        elif rung == n_rungs - 1:
            # Repeats that disagree about whether the target was met stop at
            # different rungs, and only their common prefix can be aggregated.
            # Without this the case would end with nothing written, which reads
            # as "never measured" rather than "tau straddled the target".
            stop_reason = "repeats_diverged"

        writer.write_row({
            **bkresults.row_prefix(cfg, case, entry, **PREFIX),
            "knob_name": "tolerance", "knob": tolerance, "tau": tau_val, "tau_ref": tau_ref,
            "rel_error": rel_error, "time_s": t_median, "tau_spread": spread,
            "repeats": len(times), "stop_reason": stop_reason, "note": "",
        })
        logger.info(f"  [{rung + 1:2d}/{len(ladder)}] tol={tolerance:<9.3g} tau={tau_val:.4f} "
                    f"err={rel_error:.2e} t={t_median:.3f}s {stop_reason}")
        if stop_reason:
            return stop_reason
    return "ladder_exhausted"


def probe_case(writer, case, entry, img):
    """Measure one case's peak resident set at PoreSpy's own default tolerance.

    Multigrid allocates its whole hierarchy before the first cycle and reuses it,
    so the footprint is set by the problem rather than by how far it iterates.
    That leaves the tolerance free, and `probe_tolerance` — what `tortuosity_fd`
    passes when it is given no solver of its own — measures the configuration a
    user gets rather than a rung only this ladder visits.
    """
    def run():
        solve, _ = _solve_setup(img)
        return solve(probe_tolerance)

    usage, status, note = None, "ok", ""
    try:
        usage = bkmemory.with_peak_sampling(run, interval_ms=sample_interval, device="cpu")
    except MemoryError as exc:
        status, note = "oom", str(exc)[:200]
    except Exception as exc:  # noqa: BLE001 - a failed case must not end the run
        status, note = "error", f"{type(exc).__name__}: {exc}"[:200]

    ok = usage is not None
    writer.write_row({
        **bkresults.row_prefix(cfg, case, entry, **PREFIX),
        "iters": 0,
        "time_s": usage.elapsed if ok else float("nan"),
        "peak_rss_bytes": usage.peak_rss if ok else 0,
        "baseline_rss_bytes": usage.baseline_rss if ok else 0,
        "peak_device_bytes": 0, "pool_device_bytes": 0,
        "status": status, "note": note,
    })
    if ok:
        logger.info(f"  {status:<6} t={usage.elapsed:.2f}s rss={usage.peak_rss / 2**30:.2f}GiB "
                    f"(base {usage.baseline_rss / 2**30:.2f}) samples={usage.samples}")
    else:
        logger.warning(f"  {status}: {note}")
    del usage
    return status


logger.info(f"porespy {ps.__version__} / openpnm {op.__version__} on cpu, "
            f"threads={cpu_threads}, measure={args.measure}")
# Warmed on an image of its own: numba compiles inside PoreSpy's filters on first
# call, and without this the first case of the stage would carry that cost.
_warm = np.ones((16, 16, 16), dtype=np.uint8)
trace_once(_warm, float("inf"), rungs=ladder[:1])
bkresults.record_environment(
    cfg, stage=args.measure, tool="porespy", device="cpu", variant="fd-amg",
    cpu_threads=cpu_threads,
    notes=(f"porespy {ps.__version__}, openpnm {op.__version__}, "
           f"{len(ladder)} tolerance rungs, target={target_error}, timeout={timeout_s}s"),
)

logger.info(f"Starting: {len(pending)} pending, {len(done)} already done → {outpath}")
with bkresults.ResultsWriter(outpath, columns, overwrite=args.overwrite,
                            replace_cases={c.id for c in pending}) as writer:
    for i, case in enumerate(pending, start=1):
        entry = manifest[case.id]
        logger.info(f"[{i}/{len(pending)}] {case.id}  N={case.size}  blobiness={case.blobiness:.2f}  "
                    f"porosity={entry.porosity:.4f}  nodes={entry.nnodes}")
        img = bkimages.load_image(cfg, case)
        try:
            if args.measure == "time":
                sweep_case(writer, case, entry, img, refs[case.id])
            else:
                probe_case(writer, case, entry, img)
        except Exception as exc:  # noqa: BLE001 - one case must not end the run
            # A network this size can exhaust host memory outright, and PyAMG
            # raises rather than returning a partial result when it fails to
            # converge. Both are results about the tool; record them and go on.
            note = f"{type(exc).__name__}: {exc}"[:200]
            logger.warning(f"  {case.id} failed — {note}")
            if args.measure == "time":
                writer.write_row({
                    **bkresults.row_prefix(cfg, case, entry, **PREFIX),
                    "knob_name": "tolerance", "knob": 0, "tau": float("nan"),
                    "tau_ref": refs[case.id], "rel_error": float("nan"), "time_s": float("nan"),
                    "tau_spread": float("nan"), "repeats": 0, "stop_reason": "error", "note": note,
                })
        finally:
            del img

logger.success(f"Done → {outpath}")
