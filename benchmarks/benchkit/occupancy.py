"""Coordinate occupancy measurements that temporarily replace result files."""

from contextlib import contextmanager
from pathlib import Path

LOCK_PATH = Path("results/.occupancy.lock")


@contextmanager
def published_results_lock():
    """Hold the cross-script lock for published benchmark result files."""
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        LOCK_PATH.mkdir()
    except FileExistsError as error:
        raise RuntimeError(
            f"another occupancy measurement holds {LOCK_PATH}"
        ) from error
    try:
        yield
    finally:
        LOCK_PATH.rmdir()
