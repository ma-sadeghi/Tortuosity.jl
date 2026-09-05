"""The campaign load every script that recomputes a published number starts from.

`paper_numbers.py` and `docs_numbers.py` re-derive the figures quoted in
`paper/paper.md` and `docs/src/benchmark.md` from `results/`, and both begin the
same way: parse `config.toml`, stack the timing and memory frames, and read the
grid's axes back off the data. Reading the axes rather than restating them is
the point — a script that hardcodes its own grid keeps printing after the
campaign has moved, and prints numbers for cells nobody measured.
"""

from dataclasses import dataclass

import pandas as pd

from . import config as bkconfig
from . import figures as fig


@dataclass(frozen=True)
class Campaign:
    """One loaded result set, with the axes and series labels it spans.

    Attributes
    ----------
    cfg : benchkit.config.Config
        The parsed ``config.toml`` and the paths derived from it.
    timings, memory : pandas.DataFrame
        Every CSV under ``results/timings/`` and ``results/memory/``, stacked.
    porosities, blobinesses : list
        The image axes, as they appear in the timings.
    sizes : list
        The domain sizes measured on the GPU.
    ours, tau : str
        Display names for the reference series and for taufactor's SOR solver.
    """

    cfg: bkconfig.Config
    timings: pd.DataFrame
    memory: pd.DataFrame
    porosities: list
    blobinesses: list
    sizes: list
    ours: str
    tau: str


def load():
    """Load the current campaign from the results directory ``config.toml`` names."""
    cfg = bkconfig.load_config()
    timings = fig.load_results(cfg.resultsdir, "timings")
    return Campaign(
        cfg=cfg,
        timings=timings,
        memory=fig.load_results(cfg.resultsdir, "memory"),
        porosities=sorted(timings["porosity_target"].unique()),
        blobinesses=sorted(timings["blobiness"].unique()),
        sizes=sorted(timings[timings.device == "gpu"]["size"].unique()),
        ours=fig.series_label(*fig.REFERENCE_SERIES),
        tau=fig.series_label("taufactor", "sor"),
    )
