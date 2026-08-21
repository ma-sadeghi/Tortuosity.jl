"""Memory instrumentation: what a solve actually holds, on the host and the device.

The counterpart of ``src/memory.jl``, and deliberately built the same way.

Host memory is the process resident set, sampled by a background thread. That is
the quantity ``src/memory.jl`` samples too, which is what makes a Julia figure
and a Python figure comparable at all — and it is the only choice that covers
PuMA, whose solver allocates in C where a Python-level tracer such as
``tracemalloc`` sees nothing.

Device memory does not need sampling on this side: PyTorch's caching allocator
keeps an exact high-water mark, so the peak is read rather than estimated.
``max_memory_allocated`` is what a solve holds and ``max_memory_reserved`` is
what the allocator took from the driver; the second is recorded for context and
must not be compared between configurations, because a caching allocator's
footprint is dominated by its history.

``torch`` is imported lazily on purpose. Importing it alongside ``pumapy`` in one
process aborts on a duplicate OpenMP runtime, which is one of several reasons
every tool gets its own process.
"""

import threading
import time
from dataclasses import dataclass
from typing import Any

import psutil

_PROCESS = psutil.Process()


def current_rss():
    """Resident set of this process, in bytes.

    The right quantity for "how much memory did this need": it counts what is
    actually resident, includes allocations made by C libraries, and — unlike a
    peak counter — can be sampled repeatedly.
    """
    return int(_PROCESS.memory_info().rss)


def _torch():
    import torch

    return torch


def reset_device_peaks(device):
    """Clear the allocator's high-water marks so the next reading is this solve's."""
    if str(device).startswith("cuda"):
        torch = _torch()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def device_peaks(device):
    """``(allocated, reserved)`` peak device bytes since the last reset."""
    if not str(device).startswith("cuda"):
        return 0, 0
    torch = _torch()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())


def release_device(device):
    """Return cached blocks to the driver between cases."""
    if str(device).startswith("cuda"):
        torch = _torch()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


@dataclass
class PeakUsage:
    """What a sampled run held, alongside whatever the measured code returned."""

    value: Any
    elapsed: float
    baseline_rss: int
    peak_rss: int
    peak_device: int
    pool_device: int
    samples: int


class _RssSampler(threading.Thread):
    """Track the largest resident set seen while the measured code runs."""

    def __init__(self, interval_s):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.peak = current_rss()
        self.samples = 0
        # Not `_stop`: `threading.Thread` already uses that name for an internal
        # method that `join` calls, and shadowing it makes every join raise.
        self._finished = threading.Event()

    def run(self):
        while not self._finished.is_set():
            self.sample()
            self._finished.wait(self.interval_s)

    def sample(self):
        self.peak = max(self.peak, current_rss())
        self.samples += 1

    def stop(self):
        self._finished.set()
        self.join(timeout=5.0)


def with_peak_sampling(fn, *, interval_ms=10, device="cpu"):
    """Run ``fn`` while memory is monitored, and report the peaks.

    The sampler thread keeps running through the measured call because the
    solvers spend their time in C or CUDA code that releases the interpreter
    lock. A final sample is taken with everything ``fn`` allocated still
    reachable, so the end state is represented even if the thread was never
    scheduled.
    """
    import gc

    gc.collect()
    reset_device_peaks(device)

    sampler = _RssSampler(interval_ms / 1000.0)
    baseline = current_rss()
    sampler.start()
    try:
        started = time.perf_counter()
        value = fn()
        if str(device).startswith("cuda"):
            _torch().cuda.synchronize()
        elapsed = time.perf_counter() - started
        sampler.sample()
    finally:
        sampler.stop()

    peak_device, pool_device = device_peaks(device)
    return PeakUsage(value, elapsed, baseline, sampler.peak, peak_device, pool_device, sampler.samples)
