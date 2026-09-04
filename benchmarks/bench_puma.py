"""Benchmark PuMA (pumapy) on the shared image store. CPU only.

    pixi run python bench_puma.py
    pixi run python bench_puma.py --measure=memory

PuMA has no GPU path, so it appears only in the CPU comparison. Its solver is
SciPy's conjugate gradient over a matrix-free finite-volume operator, which
makes it the closest external counterpart to Tortuosity.jl's own CPU
configuration.

The sweep knob is the iteration count, the same one the other two tools use, and
the whole ladder is traced from a *single* solve. Reaching that took driving
SciPy's conjugate gradient directly over PuMA's own assembled operator and
preconditioner, rather than through ``PropertySolver.solve``, for two reasons:

- ``PropertySolver.solve`` raises rather than returning a partial result when
  SciPy stops on ``maxiter``, so an iteration cap produces an exception instead
  of an answer.
- It passes only ``atol`` to SciPy and never ``tol``, which therefore keeps its
  1e-5 default. The real stopping rule is ``‖r‖ ≤ max(1e-5·‖b‖, tolerance)``, so
  every tolerance rung below that point is a duplicate of the one above it — the
  tolerance ladder saturates and cannot trace the accurate end of the frontier
  at all.

Nothing about PuMA's algorithm changes: the operator, the preconditioner and the
solver are all still PuMA's. Only the stopping rule is ours, and the stopping
rule is the thing being swept. The time recorded against each rung has the cost
of computing tortuosity subtracted back out, matching the other two harnesses.

Never import this alongside taufactor. Both pull in an OpenMP runtime and the
two abort on a duplicate-runtime check in one process, which is one of several
reasons every tool gets its own.
"""

import contextlib
import io
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

# Nothing is pinned: NumPy and SciPy size their BLAS pools to the machine, which
# is what a user running PuMA gets. Recorded per row is the pool PuMA had
# available, not a measured degree of parallelism — its conjugate gradient is
# effectively serial regardless, which is itself part of the result.
cpu_threads = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()

# pumapy prints a TexGen availability notice to stdout when it is imported.
# Redirected to stderr so it stays visible without polluting stdout, which
# `--list-cases` uses to hand a case list to the orchestrator.
with contextlib.redirect_stdout(sys.stderr):
    import pumapy as puma  # noqa: E402
    from pumapy.material_properties.volume_fraction import compute_volume_fraction  # noqa: E402
    from pumapy.physics_models.finite_volume.isotropic_conductivity import (  # noqa: E402
        IsotropicConductivity,
    )
    from pumapy.physics_models.finite_volume.isotropic_conductivity_utils import (  # noqa: E402
        compute_flux,
    )
    from pumapy.physics_models.utils.property_maps import IsotropicConductivityMap  # noqa: E402

import numpy as np  # noqa: E402
from scipy.sparse.linalg import cg  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

target_error = float(cfg["sweep"]["target_error"])
n_repeats = int(cfg["sweep"]["repeats"])
repeat_threshold = float(cfg["sweep"]["repeat_threshold_s"])
timeout_s = args.timeout if args.timeout else float(cfg["sweep"]["timeout_s"])
ladder = bkconfig.iteration_ladder(cfg)
# Still the tolerance the memory probe solves to; the timing sweep no longer uses
# a tolerance at all.
probe_tolerance = float(bkconfig.tolerance_ladder(cfg)[0])
maxiter = int(cfg["sweep"]["ladder"]["puma_maxiter"])
memory_iters = int(cfg["memory"]["iters"])
sample_interval = float(cfg["memory"]["sample_interval_ms"])
axis = cfg.axis

# Both the tortuosity read off an intermediate iterate (`_tau_of`, which indexes
# `len_x`/`flux_x`) and the one read off the finished solve (`result[0][0]`) are
# written for the x direction, and PuMA back-transposes internally, so neither
# was ever validated against another axis. Passing `direction=axis` alone would
# therefore give a solve along one axis reported along another, silently and only
# in the numbers. Refuse instead of guessing PuMA's convention.
if axis != "x":
    raise SystemExit(
        f"config sets campaign.axis = {axis!r}, but this harness only implements x: "
        "its tortuosity readings index the x flux directly. Solve along x, or extend "
        "`_tau_of` and the `solve_once` return to the requested axis first."
    )

subdir = "timings" if args.measure == "time" else "memory"
outpath = cfg.outputdir / subdir / "puma-cpu.csv"
columns = bkresults.TIMING_COLUMNS if args.measure == "time" else bkresults.MEMORY_COLUMNS
PREFIX = dict(tool="puma", device="cpu", variant="fv-cg", cpu_threads=cpu_threads)

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
    bkresults.completed_cases(outpath, knob_name="iters") if args.measure == "time" else bkresults.measured_cases(outpath)
)
solvable = [c for c in cases if manifest[c.id].nnodes > 0]
runnable = [c for c in solvable if c.id in refs] if args.measure == "time" else solvable
no_reference = [c.id for c in solvable if c.id not in refs]
pending = [c for c in runnable if c.id not in done]

if args.dry_run:
    bkconfig.report_plan(pending, f"bench_puma --measure={args.measure}", done)
    if no_reference:
        print(f"no reference yet, skipped: {', '.join(no_reference)}")
    print(f"writing to {outpath}")
    raise SystemExit(0)

if no_reference:
    logger.warning(f"skipping {len(no_reference)} case(s) with no ground truth — run compute_references.jl")


def solve_once(img, tolerance):
    """Solve one image to ``tolerance`` and return tau with the wall time.

    Building the workspace sits outside the timed region, matching how the other
    two harnesses charge only the work their own entry points do. ``matrix_free``
    is PuMA's default and is left on: it is the configuration a PuMA user gets,
    and it is also the fair counterpart to Tortuosity.jl's matrix-free operator.
    """
    ws = puma.Workspace.from_array(img)
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        result = puma.compute_continuum_tortuosity(
            ws, cutoff=(1, 1), direction=axis, side_bc="s",
            tolerance=tolerance, maxiter=maxiter, solver_type="cg",
            display_iter=False, matrix_free=True,
        )
    elapsed = time.perf_counter() - started
    return result, float(result[0][0]), elapsed


class _StopSolve(Exception):
    """Raised from the solver callback to end a solve at a chosen iteration.

    SciPy's conjugate gradient ignores whatever its callback returns, so
    unwinding is the only way out of the loop that keeps the readings taken so
    far.
    """


def _cond_map():
    """The conductivity map ``compute_continuum_tortuosity`` builds for a binary
    pore mask under ``cutoff=(1, 1)``: the void phase conducts, nothing else does."""
    cond_map = IsotropicConductivityMap()
    cond_map.add_material((1, 1), 1)
    cond_map.add_material((0, 0), 0)
    cond_map.add_material((2, 32000), 0)
    return cond_map


def _tau_of(solver, xk, porosity):
    """Tortuosity of an intermediate iterate, without disturbing the solve.

    Mirrors ``IsotropicConductivity.compute_effective_coefficient`` for the x
    direction, the only one the campaign solves. That method cannot be called
    here: it consumes ``solver.x`` and transposes the conductivity array in
    place, so using it mid-solve would corrupt the solve it is reading from.
    """
    field = xk.reshape([solver.len_x, solver.len_y, solver.len_z], order="F")
    flux_x, _, _, _ = compute_flux(field, solver.cond, solver.len_x, solver.len_y, solver.len_z)
    keff_x = flux_x * (solver.len_x - 1)
    return porosity / keff_x if keff_x else float("inf")


def trace_once(img, tau_ref, rungs=None):
    """Trace the whole accuracy/time frontier from a single PuMA solve.

    The timed region starts at the solver's construction and covers assembly, so
    each rung's time is everything `compute_continuum_tortuosity` would have
    charged up to that iteration. That boundary is not arbitrary: the constructor
    copies the whole workspace, which is real work a PuMA user pays and which the
    tolerance-ladder version timed. Outside it are the workspace itself — the
    counterpart of an image already in memory — and the porosity, which is part
    of computing tortuosity rather than of solving.

    The solve stops at the first rung meeting the accuracy target, so a case that
    converges early never pays for the rungs above it.
    """
    rungs = ladder if rungs is None else rungs
    ws = puma.Workspace.from_array(img)
    porosity = compute_volume_fraction(ws, (1, 1))
    rows, pending, state = [], list(rungs), {"iter": 0, "excluded": 0.0}

    with contextlib.redirect_stdout(io.StringIO()):
        started = time.perf_counter()
        solver = IsotropicConductivity(ws, _cond_map(), axis, "s", None, 0.0,
                                       max(rungs), "cg", False, True)
        solver.error_check()
        solver.initialize()
        solver.assemble_bvector()
        solver.assemble_Amatrix()

        # Defined here, after the solver exists, so it closes over a name that is
        # already bound rather than one assigned further down.
        def callback(xk):
            state["iter"] += 1
            if not pending or state["iter"] != pending[0]:
                return
            pending.pop(0)
            mark = time.perf_counter()
            tau_val = _tau_of(solver, xk, porosity)
            elapsed = mark - started - state["excluded"]
            rows.append((state["iter"], tau_val, elapsed))
            state["excluded"] += time.perf_counter() - mark
            done = abs(tau_val - tau_ref) / tau_ref <= target_error
            if done or elapsed > timeout_s or not pending:
                raise _StopSolve

        try:
            # Driven directly rather than through `PropertySolver.solve`, which
            # cannot be given a stopping rule of our choosing — see the module
            # docstring. Everything passed in is PuMA's own.
            cg(solver.Amat, solver.bvec, x0=solver.initial_guess, M=solver.M,
               tol=0.0, atol=0.0, maxiter=max(rungs), callback=callback)
        except _StopSolve:
            pass
    return rows


def sweep_case(writer, case, entry, img, tau_ref):
    """Trace one image once per repeat, writing a row per rung reached."""
    traces = []
    for rep in range(n_repeats):
        rows = trace_once(img, tau_ref)
        if not rows:
            # Raised rather than returned: the caller's handler writes an error
            # row. Returning would leave the case with no row at all, and since
            # resume keys on a `stop_reason`, it would be retried on every resume
            # with nothing anywhere to say why.
            raise RuntimeError("solve produced no checkpoints")
        traces.append(rows)
        if rep == 0 and rows[-1][2] > repeat_threshold:
            break

    # Repeats can stop one rung apart when a tau near the target lands either
    # side of it, so only rungs every repeat reached can be aggregated.
    n_rungs = min(len(t) for t in traces)
    for rung in range(n_rungs):
        iters = traces[0][rung][0]
        assert all(t[rung][0] == iters for t in traces), "repeats disagree about the ladder"
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
        elif iters == ladder[-1]:
            stop_reason = "ladder_exhausted"
        elif rung == n_rungs - 1:
            # The case that would otherwise leave no verdict at all: repeats that
            # disagree about whether the target was met stop at different rungs,
            # only their common prefix can be aggregated, and if the target is not
            # met inside it the loop ends with nothing written. Silence then reads
            # as "not measured" when what happened is that tau straddled the target.
            stop_reason = "repeats_diverged"

        writer.write_row({
            **bkresults.row_prefix(cfg, case, entry, **PREFIX),
            "knob_name": "iters", "knob": iters, "tau": tau_val, "tau_ref": tau_ref,
            "rel_error": rel_error, "time_s": t_median, "tau_spread": spread,
            "repeats": len(times), "stop_reason": stop_reason, "note": "",
        })
        logger.info(f"  [{rung + 1:2d}/{len(ladder)}] iters={iters:<6d} tau={tau_val:.4f} "
                    f"err={rel_error:.2e} t={t_median:.3f}s {stop_reason}")
        if stop_reason:
            return stop_reason
    return "ladder_exhausted"


def probe_case(writer, case, entry, img):
    """Measure one case's peak resident set on the loosest rung of the ladder.

    PuMA cannot be capped at an iteration count, so the memory probe solves to a
    fixed tolerance instead. Which rung it solves to does not matter, and the
    cheapest one is therefore the right choice: SciPy's conjugate gradient
    allocates its working vectors up front and reuses them, and PuMA assembles the
    operator and the preconditioner before the first of them, so the footprint is
    set by the problem rather than by how far it iterates.
    """
    usage, status, note = None, "ok", ""
    try:
        usage = bkmemory.with_peak_sampling(
            lambda: solve_once(img, probe_tolerance),
            interval_ms=sample_interval, device="cpu",
        )
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


logger.info(f"pumapy on cpu, threads={cpu_threads}, measure={args.measure}")
# Warmed on an image of its own: no reported number may include a first-call
# cost. SciPy's sparse machinery and PuMA's compiled `compute_flux` both pay one,
# and without this the first case of the stage would carry it.
_warm = np.ones((16, 16, 16), dtype=np.int32)
with contextlib.redirect_stdout(io.StringIO()):
    solve_once(_warm, probe_tolerance)
    if args.measure == "time":
        trace_once(_warm, float("inf"), rungs=[1, 2])
bkresults.record_environment(
    cfg, stage=args.measure, tool="puma", device="cpu", variant="fv-cg",
    cpu_threads=cpu_threads,
    notes=f"{len(ladder)} iteration rungs, maxiter={maxiter}, target={target_error}, timeout={timeout_s}s",
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
            # PuMA raises rather than returning a partial result when SciPy stops
            # short, and at the largest sizes it can exhaust host memory outright.
            # Both are results about the tool; record them and keep going.
            note = f"{type(exc).__name__}: {exc}"[:200]
            logger.warning(f"  {case.id} failed — {note}")
            if args.measure == "time":
                writer.write_row({
                    **bkresults.row_prefix(cfg, case, entry, **PREFIX),
                    "knob_name": "iters", "knob": 0, "tau": float("nan"),
                    "tau_ref": refs[case.id], "rel_error": float("nan"), "time_s": float("nan"),
                    "tau_spread": float("nan"), "repeats": 0, "stop_reason": "error", "note": note,
                })
        finally:
            del img

logger.success(f"Done → {outpath}")
