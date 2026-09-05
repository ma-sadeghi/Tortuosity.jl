"""The run skeleton the external-tool benchmark drivers share.

``bench_taufactor.py``, ``bench_puma.py`` and ``bench_porespy.py`` each drive a
different solver, and everything about *how* a tool is built, timed, warmed and
read off stays in its own file beside the docstring that justifies it. What is
here is everything around that: which cases an invocation runs, how a resumed
stage decides what is already done, how repeats are aggregated, and when a
ladder has been walked far enough.

The ladder verdict is the reason this is one module rather than three copies.
The campaign exists to compare tools, and a comparison only means anything if
``target_reached`` means the same thing for every one of them — three copies of
that rule is three chances for one to drift, and a drift here would be invisible
in the results rather than obvious in them.
"""

import statistics
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from benchkit import config as bkconfig
from benchkit import images as bkimages
from benchkit import results as bkresults


@dataclass(frozen=True)
class RunPlan:
    """What one invocation will measure, and where it will write it."""

    manifest: dict
    refs: dict
    pending: list
    done: set
    outpath: Path
    columns: list


@dataclass(frozen=True)
class Rung:
    """One aggregated rung: the numbers to record, and whether to stop here."""

    knob: object
    tau: float
    rel_error: float
    time_s: float
    tau_spread: float
    repeats: int
    stop_reason: str


def plan_run(cfg, args, *, filename, stage, knob_name):
    """Resolve the cases this invocation runs, or exit having said why not.

    Three of the questions a stage is asked have an answer but no measurement,
    and each ends the process here: ``--list-cases`` hands a case list to the
    orchestrator, ``--dry-run`` prints the plan so a large machine can be
    committed knowingly, and a case with no image is a mistake worth stopping on
    rather than a gap to step around silently.
    """
    subdir = "timings" if args.measure == "time" else "memory"
    outpath = cfg.outputdir / subdir / filename
    columns = bkresults.TIMING_COLUMNS if args.measure == "time" else bkresults.MEMORY_COLUMNS

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
        bkresults.completed_cases(outpath, knob_name=knob_name) if args.measure == "time"
        else bkresults.measured_cases(outpath)
    )
    solvable = [c for c in cases if manifest[c.id].nnodes > 0]
    # A timing row is meaningless without ground truth to state its error against; a
    # memory row needs no reference at all.
    runnable = [c for c in solvable if c.id in refs] if args.measure == "time" else solvable
    no_reference = [c.id for c in solvable if c.id not in refs]
    pending = [c for c in runnable if c.id not in done]

    if args.dry_run:
        bkconfig.report_plan(pending, stage, done)
        if no_reference:
            print(f"no reference yet, skipped: {', '.join(no_reference)}")
        print(f"writing to {outpath}")
        raise SystemExit(0)

    if no_reference:
        logger.warning(f"skipping {len(no_reference)} case(s) with no ground truth — run compute_references.jl")

    return RunPlan(manifest=manifest, refs=refs, pending=pending, done=done,
                   outpath=outpath, columns=columns)


def collect_traces(trace, n_repeats, repeat_threshold):
    """Trace one image once per repeat, and return the traces.

    ``trace()`` returns that repeat's ladder as ``(knob, tau, time_s)`` rungs.
    A first repeat slower than ``repeat_threshold`` seconds abandons the ones
    that would have followed it, which is what leaves a row carrying
    ``repeats = 1`` and a NaN spread rather than a spread of zero.
    """
    traces = []
    for rep in range(n_repeats):
        rows = trace()
        if not rows:
            # Raised rather than returned: the caller's handler writes an error
            # row. Returning would leave the case with no row at all, and since
            # resume keys on a `stop_reason`, it would be retried on every resume
            # with nothing anywhere to say why.
            raise RuntimeError("solve produced no checkpoints")
        traces.append(rows)
        if rep == 0 and rows[-1][2] > repeat_threshold:
            break
    return traces


def ladder_verdict(traces, ladder, tau_ref, target_error, timeout_s):
    """Aggregate the repeats and decide where the ladder stops.

    ``traces`` is one list of ``(knob, tau, time_s)`` rungs per repeat. Returns
    the rows to record — up to and including the one that ends the ladder — and
    that row's ``stop_reason``. Pure: it neither prints nor writes, so the one
    rule by which every tool is judged can be exercised without a solver.
    """
    # Repeats can stop one rung apart when a tau near the target lands either
    # side of it, so only rungs every repeat reached can be aggregated.
    n_rungs = min(len(t) for t in traces)
    rows = []
    for rung in range(n_rungs):
        knob = traces[0][rung][0]
        assert all(t[rung][0] == knob for t in traces), "repeats disagree about the ladder"
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
        elif knob == ladder[-1]:
            stop_reason = "ladder_exhausted"
        elif rung == n_rungs - 1:
            # The case that would otherwise leave no verdict at all: repeats that
            # disagree about whether the target was met stop at different rungs,
            # only their common prefix can be aggregated, and if the target is not
            # met inside it the loop ends with nothing written. Silence then reads
            # as "not measured" when what happened is that tau straddled the target.
            stop_reason = "repeats_diverged"

        rows.append(Rung(knob=knob, tau=tau_val, rel_error=rel_error, time_s=t_median,
                         tau_spread=spread, repeats=len(times), stop_reason=stop_reason))
        if stop_reason:
            return rows, stop_reason
    return rows, "ladder_exhausted"


def run_cases(cfg, args, plan, *, prefix, knob_name, sweep, probe, diagnose, cleanup=None):
    """Measure every pending case, recording a failure rather than ending the run.

    ``sweep(writer, case, entry, img, tau_ref)`` and ``probe(writer, case, entry,
    img)`` are the tool's own measurement entry points. ``diagnose(case, exc)``
    logs a failed case and returns the ``(stop_reason, note)`` its row carries —
    which of the two it is, and how it reads, is the tool's to say. ``cleanup``
    runs after every case whether or not it succeeded.
    """
    logger.info(f"Starting: {len(plan.pending)} pending, {len(plan.done)} already done → {plan.outpath}")
    with bkresults.ResultsWriter(plan.outpath, plan.columns, overwrite=args.overwrite,
                                 replace_cases={c.id for c in plan.pending}) as writer:
        for i, case in enumerate(plan.pending, start=1):
            entry = plan.manifest[case.id]
            logger.info(f"[{i}/{len(plan.pending)}] {case.id}  N={case.size}  blobiness={case.blobiness:.2f}  "
                        f"porosity={entry.porosity:.4f}  nodes={entry.nnodes}")
            img = bkimages.load_image(cfg, case)
            try:
                if args.measure == "time":
                    sweep(writer, case, entry, img, plan.refs[case.id])
                else:
                    probe(writer, case, entry, img)
            except Exception as exc:  # noqa: BLE001 - one case must not end the run
                stop_reason, note = diagnose(case, exc)
                if args.measure == "time":
                    writer.write_row({
                        **bkresults.row_prefix(cfg, case, entry, **prefix),
                        "knob_name": knob_name, "knob": 0, "tau": float("nan"),
                        "tau_ref": plan.refs[case.id], "rel_error": float("nan"),
                        "time_s": float("nan"), "tau_spread": float("nan"), "repeats": 0,
                        "stop_reason": stop_reason, "note": note,
                    })
            finally:
                del img
                if cleanup is not None:
                    cleanup()
    logger.success(f"Done → {plan.outpath}")
