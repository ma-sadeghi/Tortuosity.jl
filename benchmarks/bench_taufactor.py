"""Benchmark taufactor on the shared image store, on either device.

    pixi run python bench_taufactor.py --device=cuda
    pixi run python bench_taufactor.py --device=cpu --measure=memory

taufactor solves the same steady diffusion problem by successive
over-relaxation on the full voxel grid, in PyTorch, which is why it can be run
on the CPU as readily as on the GPU: the device is a tensor placement, not a
different code path. Running it on both is what makes a CPU comparison possible
at all, since PuMA has no GPU path.

The iteration count is the sweep knob rather than taufactor's own ``conv_crit``.
That criterion is evaluated only every 100 iterations, so its loosest reachable
answer is already 100 SOR sweeps deep and lands near 1e-4 relative error — the
entire coarse-accuracy half of the frontier is unreachable through it. Sweeping
iterations changes only where the result is read off, not how taufactor
iterates, and it lets taufactor stop earlier than ``conv_crit`` would ever have
allowed.

taufactor is clocked from the moment it receives the image, the same as the other
two, which means its ``Solver`` constructor is charged. Earlier versions started
the clock inside ``solve`` and so charged it nothing: measured at 200³ on a GPU,
that hid **0.415 s against a 0.48 s solve**, because the constructor builds the
SOR chequerboard from an N³ float64 array and a three-way N³ meshgrid. Since
Tortuosity.jl was charged for assembling its system and PuMA for assembling
hers, the comparison was skewed by roughly the whole margin it was measuring.

The whole ladder is traced from a *single* solve rather than one solve per rung.
An SOR sweep is deterministic, so iterate k is the same field whether the run
stopped there or carried on, which makes the two equivalent in what they report
and very far from equivalent in what they cost. The vendored fork grew a
``checkpoints`` argument for this; the time recorded against each rung has the
cost of reading tau subtracted back out, so it stays comparable with a plain run
that stopped at that iteration.
"""

import contextlib
import io
import statistics
import sys
import time
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
parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
parser.add_argument("--measure", default="time", choices=["time", "memory"])
parser.add_argument("--timeout", type=float, help="seconds; overrides sweep.timeout_s")
args = parser.parse_args()

cfg = bkconfig.load_config()

import numpy as np  # noqa: E402
import taufactor as tau  # noqa: E402
import torch  # noqa: E402

# Nothing is pinned: torch sizes its own pool, which is what a user running
# taufactor gets. What it chose is recorded in every row, rather than what the
# configuration asked for — the two used to disagree, and the row won.
cpu_threads = torch.get_num_threads()

device = args.device
if device == "cuda" and not torch.cuda.is_available():
    raise SystemExit("--device=cuda but torch reports no CUDA device")

# Disables taufactor's internal early exit so the iteration count is the only
# stopping rule: no relative error is ever below zero.
NO_EARLY_EXIT = 0.0

target_error = float(cfg["sweep"]["target_error"])
n_repeats = int(cfg["sweep"]["repeats"])
repeat_threshold = float(cfg["sweep"]["repeat_threshold_s"])
timeout_s = args.timeout if args.timeout else float(cfg["sweep"]["timeout_s"])
ladder = bkconfig.iteration_ladder(cfg)
memory_iters = int(cfg["memory"]["iters"])
sample_interval = float(cfg["memory"]["sample_interval_ms"])

device_label = "gpu" if device.startswith("cuda") else "cpu"
subdir = "timings" if args.measure == "time" else "memory"
outpath = cfg.outputdir / subdir / f"taufactor-{device_label}.csv"
columns = bkresults.TIMING_COLUMNS if args.measure == "time" else bkresults.MEMORY_COLUMNS

# taufactor applies its Dirichlet faces across the first array axis and reports
# tortuosity for that direction; this harness never transposes the image, so the
# axis it measures is x whatever the config says. Reading `cfg.axis` and ignoring
# it would be worse than not reading it — the reference tortuosity is computed
# along `campaign.axis`, so a mismatch compares two different quantities and the
# rel_error column would quietly stop meaning anything.
if cfg.axis != "x":
    raise SystemExit(
        f"config sets campaign.axis = {cfg.axis!r}, but this harness only implements x: "
        "it feeds the image to taufactor untransposed, so it always measures x. "
        "Solve along x, or transpose the image to the requested axis first."
    )

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
# A timing row is meaningless without ground truth to state its error against; a
# memory row needs no reference at all.
runnable = [c for c in solvable if c.id in refs] if args.measure == "time" else solvable
no_reference = [c.id for c in solvable if c.id not in refs]
pending = [c for c in runnable if c.id not in done]

if args.dry_run:
    bkconfig.report_plan(pending, f"bench_taufactor --device={device} --measure={args.measure}", done)
    if no_reference:
        print(f"no reference yet, skipped: {', '.join(no_reference)}")
    print(f"writing to {outpath}")
    raise SystemExit(0)

if no_reference:
    logger.warning(f"skipping {len(no_reference)} case(s) with no ground truth — run compute_references.jl")


def _settle():
    """Wait for queued device work, so a clock reading means what it says.

    A no-op on CPU, where torch is synchronous already.
    """
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def solve_once(img, iters):
    """Build a solver and run it for exactly ``iters`` SOR sweeps.

    Construction is inside the timed region, because every tool in this campaign
    is clocked from the moment it receives the image to the moment tortuosity can
    be read. That is not a detail here: `Solver.__init__` builds the SOR
    chequerboard from an N³ float64 array and a three-way N³ meshgrid, which at
    200³ measures 0.415 s against a 0.48 s solve — 45% of the total, and growing
    as N³. Below 100 iterations taufactor never reaches a convergence check and
    so never synchronises on its own, which is why the barriers are explicit.
    """
    started = time.perf_counter()
    solver = tau.Solver(img, device=device)
    _settle()
    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve(conv_crit=NO_EARLY_EXIT, iter_limit=iters, verbose=False)
    _settle()
    elapsed = time.perf_counter() - started
    return solver, float(solver.tau[0]), elapsed


def trace_once(img, tau_ref, rungs=None):
    """Run one solve, reading tau off at every rung of the ladder.

    Construction is inside the timed region — see `solve_once` for why it has to
    be. taufactor starts its own clock inside `solve`, after construction is
    already done, so the construction time is measured here and added to every
    rung. The barrier before reading it is what keeps the queued device work in
    construction from being charged to the solve that follows instead.

    The solve stops at the first rung that meets the accuracy target or blows the
    timeout, so a case that converges early never pays for the rungs above it.
    """
    rungs = ladder if rungs is None else rungs
    started = time.perf_counter()
    solver = tau.Solver(img, device=device)
    _settle()
    t_setup = time.perf_counter() - started

    def hook(iters, tau_val, elapsed):
        rel_error = abs(float(tau_val[0]) - tau_ref) / tau_ref
        return rel_error <= target_error or (t_setup + elapsed) > timeout_s

    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve(conv_crit=NO_EARLY_EXIT, iter_limit=max(rungs), verbose=False,
                     checkpoints=rungs, checkpoint_hook=hook)
    rows = [(iters, float(tau_val[0]), t_setup + elapsed)
            for iters, tau_val, elapsed in solver.checkpoints]
    del solver
    bkmemory.release_device(device)
    return rows


def sweep_case(writer, case, entry, img, tau_ref):
    """Trace one image once per repeat, writing a row per rung reached."""
    prefix = dict(tool="taufactor", device=device_label, variant="sor", cpu_threads=cpu_threads)
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
            **bkresults.row_prefix(cfg, case, entry, **prefix),
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
    """Measure one case's peak memory at a fixed iteration count."""
    prefix = dict(tool="taufactor", device=device_label, variant="sor", cpu_threads=cpu_threads)
    usage, status, note = None, "ok", ""
    try:
        usage = bkmemory.with_peak_sampling(
            # The solver is returned so its tensors are still reachable when the
            # closing sample is taken.
            lambda: solve_once(img, memory_iters),
            interval_ms=sample_interval, device=device,
        )
    except torch.cuda.OutOfMemoryError as exc:
        status, note = "oom", str(exc)[:200]
    except Exception as exc:  # noqa: BLE001 - a failed case must not end the run
        status, note = "error", f"{type(exc).__name__}: {exc}"[:200]

    ok = usage is not None
    writer.write_row({
        **bkresults.row_prefix(cfg, case, entry, **prefix),
        "iters": memory_iters,
        "time_s": usage.elapsed if ok else float("nan"),
        "peak_rss_bytes": usage.peak_rss if ok else 0,
        "baseline_rss_bytes": usage.baseline_rss if ok else 0,
        "peak_device_bytes": usage.peak_device if ok else 0,
        "pool_device_bytes": usage.pool_device if ok else 0,
        "status": status, "note": note,
    })
    if ok:
        logger.info(f"  {status:<6} t={usage.elapsed:.2f}s rss={usage.peak_rss / 2**30:.2f}GiB "
                    f"(base {usage.baseline_rss / 2**30:.2f}) device={usage.peak_device / 2**30:.2f}GiB "
                    f"pool={usage.pool_device / 2**30:.2f}GiB samples={usage.samples}")
    else:
        logger.warning(f"  {status}: {note}")
    del usage
    bkmemory.release_device(device)
    return status


# Warmed on an image of its own: no reported number may include a first-call
# cost, and PyTorch's first CUDA kernel launch pays context creation. Both paths
# are warmed, because they are not the same code: below 100 iterations a plain
# solve never reaches a convergence check, so it never calls `compute_metrics` —
# the very thing every checkpoint calls.
logger.info(f"torch {torch.__version__} device={device} threads={torch.get_num_threads()} measure={args.measure}")
_warm = np.ones((16, 16, 16), dtype=np.int32)
solve_once(_warm, 5)
if args.measure == "time":
    trace_once(_warm, float("inf"), rungs=[1, 2])
bkmemory.release_device(device)

bkresults.record_environment(
    cfg, stage=args.measure, tool="taufactor", device=device_label, variant="sor",
    accelerator=torch.cuda.get_device_name() if device.startswith("cuda") else "",
    cpu_threads=cpu_threads,
    notes=f"torch {torch.__version__}, {len(ladder)} rungs, target={target_error}, timeout={timeout_s}s",
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
            # taufactor holds several full-grid tensors, so the largest images can
            # exhaust the card. That is a result about the full-grid approach, not
            # a harness failure — record it and keep the images queued behind it.
            # Not only device memory: the chequerboard is built host-side from an
            # N³ float64 array and a three-way N³ meshgrid, so a host MemoryError
            # is the more likely of the two at the largest sizes.
            oom = isinstance(exc, (torch.cuda.OutOfMemoryError, MemoryError))
            logger.warning(f"  {'out of memory' if oom else type(exc).__name__} on {case.id}")
            if args.measure == "time":
                writer.write_row({
                    **bkresults.row_prefix(cfg, case, entry, tool="taufactor",
                                           device=device_label, variant="sor",
                                           cpu_threads=cpu_threads),
                    "knob_name": "iters", "knob": 0, "tau": float("nan"),
                    "tau_ref": refs[case.id], "rel_error": float("nan"), "time_s": float("nan"),
                    "tau_spread": float("nan"), "repeats": 0,
                    "stop_reason": "oom" if oom else "error",
                    "note": str(exc)[:200],
                })
        finally:
            del img
            bkmemory.release_device(device)

logger.success(f"Done → {outpath}")
