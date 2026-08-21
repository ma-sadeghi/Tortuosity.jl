"""Campaign configuration, the case grid, and the shared command-line surface.

The counterpart of ``src/config.jl`` and ``src/cli.jl``. Both halves read the
same ``config.toml``, so a grid is defined once and neither language can drift
from the other.
"""

import argparse
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BENCHDIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Case:
    """One benchmark image: a point on the (size, blobiness, porosity) grid.

    Attributes
    ----------
    id : str
        Grid identifier, e.g. ``n200_b100_p020``. Blobiness and porosity are
        written as hundredths so the identifier holds no decimal point.
    size : int
        Voxels per side.
    blobiness : float
        Feature-size knob the image was generated with.
    porosity : float
        The *target* pore fraction, not the realised one. Trimming
        non-percolating clusters removes pore voxels, so the realised value is
        always a little lower; every stage keys on the target, which is exact
        and identical in both languages.
    """

    id: str
    size: int
    blobiness: float
    porosity: float


def case_id(size, blobiness, porosity):
    """Identifier for a grid point, matching ``BenchHarness.case_id``."""
    return f"n{int(size)}_b{round(100 * blobiness):03d}_p{round(100 * porosity):03d}"


def make_case(size, blobiness, porosity):
    return Case(case_id(size, blobiness, porosity), int(size), float(blobiness), float(porosity))


class Config:
    """The parsed ``config.toml`` plus the paths derived from its location."""

    def __init__(self, raw, root):
        self.raw = raw
        self.root = Path(root)

    def __getitem__(self, key):
        return self.raw[key]

    @property
    def axis(self):
        return self.raw["campaign"]["axis"]

    @property
    def cpu_threads(self):
        """The configured CPU budget: an integer, or "auto" for each tool's own default.

        This is what the campaign asked for, not what a run got. Every result row
        records the latter instead, because that is the number a reader needs and
        the two used to disagree silently.
        """
        return self.raw["cpu"]["threads"]

    @property
    def imagedir(self):
        return self.root / "data" / "images"

    @property
    def resultsdir(self):
        return self.root / "results"


def load_config(path=None):
    """Read ``config.toml``, defaulting to the one beside the harness."""
    path = Path(path) if path else BENCHDIR / "config.toml"
    if not path.is_file():
        raise FileNotFoundError(f"no config at {path}")
    with open(path, "rb") as f:
        return Config(tomllib.load(f), path.parent)


def sizes(cfg, grid):
    """Sizes of the named grid."""
    grids = cfg["grid"]
    if grid not in grids:
        known = ", ".join(sorted(grids))
        raise KeyError(f'unknown grid "{grid}"; config defines {known}')
    return [int(n) for n in grids[grid]]


def case_grid(cfg, grid=None):
    """Every case of a named grid, ordered cheapest first.

    The ordering is what makes an interrupted campaign useful: every stage
    resumes from its own results file, so stopping early leaves the small cases
    complete rather than a scatter of half-finished large ones.
    """
    grid = grid or cfg["campaign"]["grid"]
    image = cfg["image"]
    cases = [
        make_case(n, b, p)
        for n in sizes(cfg, grid)
        for b in image["blobinesses"]
        for p in image["porosities"]
    ]
    return sorted(cases, key=lambda c: (c.size, c.blobiness, c.porosity))


def iteration_ladder(cfg):
    """Log-spaced iteration ladder, deduplicated and ascending."""
    spec = cfg["sweep"]["ladder"]["iters"]
    rungs = np.logspace(np.log10(spec["min"]), np.log10(spec["max"]), int(spec["count"]))
    return sorted({int(round(n)) for n in rungs})


def tolerance_ladder(cfg):
    """Log-spaced tolerance ladder, loosest first."""
    spec = cfg["sweep"]["ladder"]["tolerance"]
    return list(np.logspace(np.log10(spec["min"]), np.log10(spec["max"]), int(spec["count"])))


def add_selection_arguments(parser):
    """Attach the selection flags every stage shares, in both languages."""
    parser.add_argument("--grid", help="named size grid from config.toml (default: campaign.grid)")
    parser.add_argument("--sizes", help="restrict to these domain sizes, e.g. 200,400")
    parser.add_argument("--porosities", help="restrict to these target porosities")
    parser.add_argument("--blobiness", help="restrict to these blobiness values")
    parser.add_argument("--cases", help="run these case ids outright, overriding the other filters")
    parser.add_argument("--overwrite", action="store_true", help="re-measure instead of resuming")
    parser.add_argument("--dry-run", action="store_true", help="list the cases that would run, then stop")
    # Exists so a shell can drive one process per case. The memory stage needs
    # that: peak resident set is only that case's peak in a process that has not
    # already faulted in comparable pages, and an allocator that reuses them —
    # torch's on the CPU especially — makes a within-process reading report page
    # faults rather than footprint.
    parser.add_argument("--list-cases", action="store_true",
                        help="print the selected case ids, one per line, and nothing else")
    return parser


def _numbers(raw, cast):
    return None if not raw else [cast(s) for s in raw.split(",") if s.strip()]


def select_cases(cfg, args):
    """The cases this invocation should run, cheapest first.

    Filters compose: ``--sizes=200,400 --blobiness=1.0`` selects the
    intersection. ``--cases`` names grid points outright and overrides the rest.
    """
    grid = args.grid or cfg["campaign"]["grid"]
    cases = case_grid(cfg, grid)

    if args.cases:
        by_id = {c.id: c for c in cases}
        wanted = [s.strip() for s in args.cases.split(",") if s.strip()]
        unknown = [i for i in wanted if i not in by_id]
        if unknown:
            raise KeyError(f'no such case in grid "{grid}": {", ".join(unknown)}')
        return [by_id[i] for i in wanted]

    keep_sizes = _numbers(args.sizes, int)
    keep_blob = _numbers(args.blobiness, float)
    keep_por = _numbers(args.porosities, float)
    if keep_sizes is not None:
        cases = [c for c in cases if c.size in keep_sizes]
    if keep_blob is not None:
        cases = [c for c in cases if any(abs(c.blobiness - b) < 1e-9 for b in keep_blob)]
    if keep_por is not None:
        cases = [c for c in cases if any(abs(c.porosity - p) < 1e-9 for p in keep_por)]
    if not cases:
        raise ValueError(f'no cases match the selection in grid "{grid}"')
    return cases


def restrict_memory_blobiness(cfg, args, cases):
    """Narrow to the structures the memory stage measures, unless asked otherwise.

    Memory tracks pore count, which at a fixed porosity barely moves with
    blobiness, so measuring every structure would spend processes re-measuring
    one curve. An explicit ``--blobiness`` always wins.
    """
    if args.blobiness:
        return cases
    wanted = [float(b) for b in cfg["memory"].get("blobinesses", cfg["image"]["blobinesses"])]
    kept = [c for c in cases if any(abs(c.blobiness - b) < 1e-9 for b in wanted)]
    return kept or cases


def report_plan(cases, stage, skipped=()):
    """Print the selected cases and return, for checking a plan before paying."""
    print(f"{stage} would run {len(cases)} case(s):")
    for c in cases:
        print(f"  {c.id:<18} N={c.size:<5} blobiness={c.blobiness:.2f} porosity={c.porosity:.2f}")
    if skipped:
        print(f"skipping {len(skipped)} already complete: {', '.join(sorted(skipped))}")


def build_parser(description):
    """An argument parser carrying the shared selection flags."""
    parser = argparse.ArgumentParser(description=description)
    return add_selection_arguments(parser)
