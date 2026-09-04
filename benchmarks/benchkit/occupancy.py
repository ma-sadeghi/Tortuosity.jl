"""Coordinate benchmark measurements that require exclusive host access."""

import contextlib
import csv
import os
import signal
import subprocess
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import psutil

from benchkit.results import ENVIRONMENT_COLUMNS

if os.name == "nt":
    DEFAULT_LOCK_PATH = (
        Path(tempfile.gettempdir())
        / "tortuosity-benchmark.measurement.lock"
    )
else:
    DEFAULT_LOCK_PATH = Path(
        "/tmp/tortuosity-benchmark.measurement.lock"
    )
LOCK_PATH = Path(
    os.environ.get("TORTUOSITY_BENCHMARK_LOCK_DIR", DEFAULT_LOCK_PATH)
).expanduser()
if not LOCK_PATH.is_absolute():
    raise ValueError("TORTUOSITY_BENCHMARK_LOCK_DIR must be absolute")


class ProcessGroupCleanupError(RuntimeError):
    """A measurement process group survived cleanup."""


def timestamp():
    """Local wall-clock time with its offset, for the occupancy summaries."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def latest_environment(output_dir):
    """Runtime provenance written by the child that just completed."""
    path = Path(output_dir) / "environment.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("benchmark child wrote no environment record")
    return {key: rows[-1][key] for key in ENVIRONMENT_COLUMNS}


def require_target_result(output_dir, name, case):
    """Require the isolated case of tool `name` to have reached the target."""
    path = Path(output_dir) / "timings" / f"{name}.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["case_id"] == case and row["stop_reason"]
        ]
    if len(rows) != 1 or rows[0]["stop_reason"] != "target_reached":
        raise RuntimeError(f"{name} did not reach the target in isolated output")


def sample_cpu_occupancy(cmd, interval, logpath, env, window=None):
    """Run `cmd`, sampling total CPU occupancy of its process tree until it exits.

    Returns the exit status, the elapsed seconds and the samples, each the
    number of cores busy over one `interval`. `window`, when given, is called
    before and after each interval; a sample is kept only when both calls
    return the same truthy value, which is how a caller keeps the samples
    inside one marked phase of the child.

    The child's output goes to a file rather than to /dev/null: a tool that dies
    on a signal reports only its exit code, and without its stderr there is
    nothing to diagnose from.

    Every process is kept and reused between samples. `cpu_percent` reports the
    share used since that same object's previous call, so its first call always
    returns 0.0; building the child objects afresh each iteration would make
    every call a first call and report a solve that used no CPU at all. This
    matters only when the work happens below the process we launched — a tool run
    through `pixi run` — which is why it went unnoticed while every tool here was
    launched directly.
    """
    with open(logpath, "wb") as log:
        with measurement_process(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        ) as (proc, tracked):
            parent = psutil.Process(proc.pid)
            tracked[parent.pid] = parent
            parent.cpu_percent(None)  # prime; first call always returns 0.0
            samples = []
            started = time.perf_counter()
            while proc.poll() is None:
                before = None if window is None else window()
                time.sleep(interval)
                try:
                    for child in parent.children(recursive=True):
                        if child.pid not in tracked:
                            tracked[child.pid] = child
                            with contextlib.suppress(psutil.Error):
                                child.cpu_percent(None)
                except psutil.Error:
                    pass
                total = 0.0
                for pid, process in list(tracked.items()):
                    try:
                        total += process.cpu_percent(None)
                    except psutil.Error:
                        del tracked[pid]
                after = None if window is None else window()
                if window is None or (before and before == after):
                    samples.append(total / 100.0)
            code = proc.wait()
    return code, time.perf_counter() - started, samples


@contextmanager
def benchmark_measurement_lock():
    """Hold the lock shared by campaign and occupancy measurements."""
    install_termination_handler()
    mask = block_termination_signals()
    acquired = False
    preserve = False
    try:
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            LOCK_PATH.mkdir()
        except FileExistsError as error:
            raise RuntimeError(
                f"another benchmark measurement holds {LOCK_PATH}"
            ) from error
        acquired = True
        pid_path = LOCK_PATH / "pid"
        pid_path.write_text(str(os.getpid()), encoding="ascii")
        restore_signal_mask(mask)
        try:
            yield
        except ProcessGroupCleanupError:
            preserve = True
            raise
    finally:
        block_termination_signals()
        try:
            if acquired and not preserve:
                (LOCK_PATH / "child_pgid").unlink(missing_ok=True)
                (LOCK_PATH / "pid").unlink(missing_ok=True)
                LOCK_PATH.rmdir()
        finally:
            restore_signal_mask(mask)


def register_child_group(pgid):
    """Record the process group whose cleanup protects the measurement lock."""
    (LOCK_PATH / "child_pgid").write_text(str(pgid), encoding="ascii")


def clear_child_group():
    """Clear the registered process group after confirmed disappearance."""
    (LOCK_PATH / "child_pgid").unlink(missing_ok=True)


def _exit_on_termination(signum, _frame):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise SystemExit(128 + signum)


def install_termination_handler():
    """Turn SIGTERM into normal Python unwinding so children are reaped."""
    signal.signal(signal.SIGINT, _exit_on_termination)
    signal.signal(signal.SIGTERM, _exit_on_termination)


def block_termination_signals():
    """Block INT/TERM while a child launch is not yet registered."""
    if os.name == "nt":
        raise RuntimeError(
            "occupancy measurement requires POSIX process-group containment; "
            "run it from Linux or WSL"
        )
    return signal.pthread_sigmask(
        signal.SIG_BLOCK,
        {signal.SIGINT, signal.SIGTERM},
    )


def restore_signal_mask(mask):
    """Restore a signal mask from inside an armed cleanup scope."""
    signal.pthread_sigmask(signal.SIG_SETMASK, mask)


def process_group_kwargs(parent_mask):
    """Popen options that put a measurement child in its own process group."""
    if os.name == "nt":
        raise RuntimeError(
            "occupancy measurement requires POSIX process-group containment; "
            "run it from Linux or WSL"
        )

    def restore_child_mask():
        signal.pthread_sigmask(signal.SIG_SETMASK, parent_mask)

    return {
        "start_new_session": True,
        "preexec_fn": restore_child_mask,
    }


@contextmanager
def measurement_process(cmd, **kwargs):
    """Launch one signal-safe measurement process and contain its descendants."""
    mask = block_termination_signals()
    proc = None
    tracked = {}
    try:
        proc = subprocess.Popen(
            cmd,
            **kwargs,
            **process_group_kwargs(mask),
        )
        register_child_group(proc.pid)
        restore_signal_mask(mask)
        yield proc, tracked
    finally:
        block_termination_signals()
        cleanup_error = None
        try:
            if proc is not None:
                terminate_process_tree(proc, tracked.values())
        except ProcessGroupCleanupError as error:
            cleanup_error = error
        finally:
            if cleanup_error is not None:
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
            restore_signal_mask(mask)
        if cleanup_error is not None:
            raise cleanup_error


def wait_for_process_group(pgid, timeout):
    """Wait until a POSIX process group, including zombies, disappears."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def terminate_process_tree(proc, tracked=(), timeout=10):
    """Terminate and reap a measurement process and retained descendants."""
    by_pid = {process.pid: process for process in tracked}
    try:
        parent = psutil.Process(proc.pid)
        by_pid[parent.pid] = parent
        for child in parent.children(recursive=True):
            by_pid[child.pid] = child
    except psutil.Error:
        pass
    processes = list(by_pid.values())

    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGTERM)
    for process in reversed(processes):
        with contextlib.suppress(psutil.Error):
            process.terminate()
    _, alive = psutil.wait_procs(processes, timeout=timeout)
    if not wait_for_process_group(proc.pid, min(timeout, 2)):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
    for process in alive:
        with contextlib.suppress(psutil.Error):
            process.kill()
    psutil.wait_procs(alive, timeout=timeout)
    if not wait_for_process_group(proc.pid, timeout):
        raise ProcessGroupCleanupError(
            f"process group {proc.pid} survived cleanup; "
            f"measurement lock preserved at {LOCK_PATH}"
        )

    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=timeout)
    if proc.poll() is None:
        proc.kill()
        proc.wait()
    clear_child_group()
