"""Python half of the Tortuosity.jl benchmark harness.

Mirrors ``src/BenchHarness.jl``: the same ``config.toml``, the same case
identifiers, the same image store and the same result schemas. The two halves
are separate only because the tools they drive are written in different
languages; everything they disagree about would be a bug, so anything shared
lives in the configuration file rather than in either of them.
"""

from benchkit.config import (
    Case,
    Config,
    case_grid,
    iteration_ladder,
    load_config,
    select_cases,
    tolerance_ladder,
)
from benchkit.images import load_image, read_manifest
from benchkit.memory import PeakUsage, device_peaks, reset_device_peaks, with_peak_sampling
from benchkit.results import (
    MEMORY_COLUMNS,
    TIMING_COLUMNS,
    ResultsWriter,
    completed_cases,
    measured_cases,
    read_references,
    record_environment,
)

__all__ = [
    "MEMORY_COLUMNS",
    "TIMING_COLUMNS",
    "Case",
    "Config",
    "PeakUsage",
    "ResultsWriter",
    "case_grid",
    "completed_cases",
    "device_peaks",
    "iteration_ladder",
    "load_config",
    "load_image",
    "measured_cases",
    "read_manifest",
    "read_references",
    "record_environment",
    "reset_device_peaks",
    "select_cases",
    "tolerance_ladder",
    "with_peak_sampling",
]
