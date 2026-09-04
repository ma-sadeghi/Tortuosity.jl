"""Coordinate benchmark measurements that require exclusive host access."""

import contextlib
import os
import signal
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import psutil

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
