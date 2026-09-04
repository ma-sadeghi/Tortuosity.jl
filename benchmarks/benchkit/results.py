"""Result files: fixed schemas, append-only writes, and the rule for resuming.

The column lists here are the same ones in ``src/results.jl`` and must stay
that way — the post-processing stage concatenates every file in a directory
without caring which language wrote it.

``note`` is last in every schema deliberately. It is the only column that can
contain a comma, and the Julia reader splits on commas without honouring quotes.
"""

import csv
import math
import platform
import socket
from datetime import datetime
from pathlib import Path

TIMING_COLUMNS = [
    "tool", "device", "variant", "cpu_threads",
    "case_id", "size", "blobiness", "porosity_target", "porosity", "nnodes",
    "knob_name", "knob", "tau", "tau_ref", "rel_error", "time_s", "tau_spread", "repeats",
    "stop_reason", "host", "measured_at", "note",
]

MEMORY_COLUMNS = [
    "tool", "device", "variant", "cpu_threads",
    "case_id", "size", "blobiness", "porosity_target", "porosity", "nnodes",
    "iters", "time_s", "peak_rss_bytes", "baseline_rss_bytes",
    "peak_device_bytes", "pool_device_bytes", "status", "host", "measured_at", "note",
]

ENVIRONMENT_COLUMNS = [
    "measured_at", "host", "stage", "tool", "device", "variant",
    "runtime", "runtime_version", "cpu_threads", "accelerator", "notes",
]


def timestamp():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _field(value):
    """Render one value the way the Julia writer does.

    Floats get ten significant digits, and NaN is written literally: a spread a
    single repeat could not measure is not the same thing as a spread of zero,
    and a blank field would read as the latter.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        return f"{value:.10g}"
    return str(value)


def drop_cases(path, columns, cases):
    """Rewrite `path` without the rows belonging to `cases`. Returns the count dropped.

    Writes a sibling file and moves it into place, so an interrupted rewrite
    leaves the original results intact rather than a half-copied file.
    """
    if "case_id" not in columns:
        return 0
    wanted = set(cases or ())
    if not wanted:
        return 0
    ci = columns.index("case_id")
    path = Path(path)
    kept, dropped = [], 0
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            if len(row) > ci and row[ci] in wanted:
                dropped += 1
                continue
            kept.append(row)
    tmp = path.with_suffix(path.suffix + ".rewriting")
    with open(tmp, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header if header else columns)
        writer.writerows(kept)
    tmp.replace(path)
    return dropped


class ResultsWriter:
    """An open results file with a fixed schema.

    Refuses to append to a file whose header is not the current schema. Silently
    appending rows in one shape under a header of another is the failure worth
    guarding against: the file still parses, the columns simply mean something
    else.
    """

    def __init__(self, path, columns, overwrite=False, replace_cases=None):
        self.path = Path(path)
        self.columns = list(columns)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = ",".join(self.columns)
        has_rows = self.path.is_file() and self.path.stat().st_size > 0
        # `--overwrite` means "do not resume", not "throw the file away". Given
        # the cases this run will measure, drop exactly those rows and keep the
        # rest, so re-measuring one stubborn case after a multi-hour sweep costs
        # that case and not the sweep. With no case set there is nothing to key
        # on, so it truncates — the right reading of `--overwrite` on a full grid.
        keeping = has_rows and (not overwrite or replace_cases is not None)
        if keeping:
            with open(self.path, newline="") as f:
                existing = f.readline().strip()
            if existing != header:
                raise ValueError(
                    f"{self.path.name} has header\n  {existing}\nbut this harness writes\n  {header}\n"
                    "Move the file aside or pass --overwrite; appending would produce a file whose "
                    "rows and header disagree."
                )
            if overwrite:
                drop_cases(self.path, self.columns, replace_cases)
        self._file = open(self.path, "a" if keeping else "w", newline="")
        self._writer = csv.writer(self._file, lineterminator="\n")
        if not keeping:
            self._writer.writerow(self.columns)
            self._file.flush()

    def write_row(self, row):
        """Append one row given as a dict covering the schema."""
        gaps = [c for c in self.columns if c not in row]
        if gaps:
            raise KeyError(f"row is missing {', '.join(gaps)} for {self.path.name}")
        self._writer.writerow([_field(row[c]) for c in self.columns])
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _read_column(path, key, value):
    path = Path(path)
    if not path.is_file():
        return {}
    out = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or key not in reader.fieldnames or value not in reader.fieldnames:
            return {}
        for row in reader:
            out[row[key]] = row[value]
    return out


def completed_cases(path, knob_name=None):
    """Case ids whose sweep ran to a conclusion.

    A case is finished when one of its rows carries a ``stop_reason`` — the
    ladder reached the accuracy target, timed out, or was exhausted. Keying
    resume on the mere presence of a row would silently treat a case interrupted
    halfway up its ladder as complete, and a partial ladder is indistinguishable
    from a converged one once it is in the file.

    ``knob_name`` is the axis this run sweeps. Pass it and a file swept on a
    different axis is refused rather than resumed. Every sweep shares one header,
    so nothing else distinguishes them: PuMA moved from a tolerance ladder to an
    iteration ladder, and without this check the new run would find every case
    already carrying a ``stop_reason``, measure nothing, and leave the published
    curve on an axis the other tools no longer use.
    """
    rows = _read_column(path, "case_id", "stop_reason")
    if knob_name is not None:
        seen = {k.strip() for k in _read_column(path, "case_id", "knob_name").values() if k.strip()}
        if seen - {knob_name}:
            raise SystemExit(
                f"{path} was swept on {'/'.join(sorted(seen))}, this run sweeps {knob_name}. "
                f"Rerun with --overwrite, or move the file aside; resuming would mix two axes "
                f"under one header."
            )
    return {case for case, reason in rows.items() if reason.strip()}


def measured_cases(path):
    """Case ids present at all — the resume rule for stages writing one row each."""
    return set(_read_column(path, "case_id", "case_id"))


def references_path(cfg):
    return cfg.resultsdir / "references.csv"


def read_references(cfg):
    """Cached ground-truth tortuosity by case id.

    Ground truth is produced once, by Tortuosity.jl on the CPU in ``Float64``,
    and every tool measures its error against the same values. A GPU reference
    could not serve: those solves run in ``Float32``, whose epsilon falls inside
    the error range being measured.
    """
    refs = {}
    for case, tau in _read_column(references_path(cfg), "case_id", "tau_ref").items():
        try:
            refs[case] = float(tau)
        except ValueError:
            continue
    return refs


def record_environment(cfg, *, stage, tool, device, variant="", accelerator="", notes="",
                       cpu_threads=None):
    """Record what produced a batch of rows, so results from two machines stay apart.

    Timings are only comparable within one machine and one software stack, and
    this campaign spans a laptop and a rented host by design.
    """
    path = cfg.outputdir / "environment.csv"
    with ResultsWriter(path, ENVIRONMENT_COLUMNS) as w:
        w.write_row({
            "measured_at": timestamp(),
            "host": socket.gethostname(),
            "stage": stage,
            "tool": tool,
            "device": device,
            "variant": variant,
            "runtime": "python",
            "runtime_version": platform.python_version(),
            "cpu_threads": cfg.cpu_threads if cpu_threads is None else cpu_threads,
            "accelerator": accelerator,
            "notes": notes,
        })
    return path


def row_prefix(cfg, case, entry, *, tool, device, variant, cpu_threads=None):
    """The identifying columns every row of every schema begins with."""
    return {
        "tool": tool,
        "device": device,
        "variant": variant,
        "cpu_threads": cfg.cpu_threads if cpu_threads is None else cpu_threads,
        "case_id": case.id,
        "size": case.size,
        "blobiness": case.blobiness,
        "porosity_target": case.porosity,
        "porosity": entry.porosity,
        "nnodes": entry.nnodes,
        "host": socket.gethostname(),
        "measured_at": timestamp(),
    }
