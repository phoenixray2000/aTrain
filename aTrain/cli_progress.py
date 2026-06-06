"""Stderr progress heartbeat for the aTrain CLI.

This module is intentionally dependency-light (standard library only, no
``aTrain_core`` import) so it can be unit-tested without a model or GPU.

The heartbeat polls a shared progress mapping (the same dict that
``aTrain_core`` mutates during transcription/diarization) and emits a single
line whenever the observed ``(task, current, total)`` snapshot *advances*.

Emit-on-advance is deliberate: a hung or stalled run stops updating the
mapping, so the heartbeat falls silent and a downstream *idle/inactivity
timeout* can fire. An unconditional time-based heartbeat would keep emitting
even on a hang and would defeat hang detection.
"""

from __future__ import annotations

import threading
from typing import Callable, Mapping, Tuple

#: Default seconds between heartbeat polls.
DEFAULT_INTERVAL_SEC: float = 10.0

#: Prefix every heartbeat line carries; downstream consumers match on this.
HEARTBEAT_PREFIX: str = "[atrain-progress]"

#: Task value reported by the very first (liveness) line before compute starts.
STARTING_TASK: str = "starting"

# Snapshot is (task, current, total).
_Snapshot = Tuple[str, float, float]


def format_line(task: str, current: float, total: float) -> str:
    """Render the canonical heartbeat line.

    Format (stable contract for collab-runtime):
    ``[atrain-progress] task=<task> current=<current> total=<total>``
    """
    return f"{HEARTBEAT_PREFIX} task={task} current={current} total={total}"


class ProgressHeartbeat:
    """Emit stderr heartbeat lines while a progress mapping advances.

    Parameters
    ----------
    progress:
        A mapping that the transcription backend mutates in place. Expected
        keys are ``"task"``, ``"current"`` and ``"total"``; missing keys are
        treated as empty / zero. May be a plain ``dict`` (CPU / tests) or a
        ``multiprocessing.Manager().dict()`` proxy (GPU child process).
    emit:
        Callable invoked with each formatted line (e.g. a closure that writes
        to stderr).
    interval_sec:
        Seconds between polls. Defaults to :data:`DEFAULT_INTERVAL_SEC`.

    Use as a context manager::

        with ProgressHeartbeat(progress, emit):
            run_the_long_thing()

    or explicitly via :meth:`start` / :meth:`stop`.
    """

    def __init__(
        self,
        progress: Mapping,
        emit: Callable[[str], None],
        interval_sec: float = DEFAULT_INTERVAL_SEC,
    ) -> None:
        self._progress = progress
        self._emit = emit
        self._interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last: _Snapshot | None = None

    def _read_snapshot(self) -> _Snapshot:
        """Read ``(task, round(current, 2), round(total, 2))`` defensively."""
        progress = self._progress
        task = progress.get("task", "") or ""
        current = round(float(progress.get("current", 0.0) or 0.0), 2)
        total = round(float(progress.get("total", 0.0) or 0.0), 2)
        return (str(task), current, total)

    def _emit_snapshot(self, snapshot: _Snapshot) -> None:
        task, current, total = snapshot
        self._emit(format_line(task, current, total))
        self._last = snapshot

    def _run(self) -> None:
        # Initial liveness marker: downstream sees output immediately, before
        # any compute progress is available.
        try:
            self._emit_snapshot((STARTING_TASK, 0.0, 0.0))
        except Exception:
            # Never let an emit failure kill the daemon thread.
            pass

        while not self._stop.wait(self._interval_sec):
            try:
                snapshot = self._read_snapshot()
                # Emit only when the snapshot advanced (changed) since the last
                # emit. A stalled run keeps producing the same snapshot -> no
                # line -> downstream idle-timeout can fire.
                if snapshot != self._last:
                    self._emit_snapshot(snapshot)
            except Exception:
                # A manager hiccup (e.g. proxy momentarily unavailable) must not
                # terminate the heartbeat; just skip this poll.
                continue

    def start(self) -> "ProgressHeartbeat":
        """Start the daemon heartbeat thread. Idempotent."""
        if self._thread is not None:
            return self
        self._stop.clear()
        self._last = None
        self._thread = threading.Thread(
            target=self._run,
            name="atrain-progress-heartbeat",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Signal the thread to stop and join it (bounded wait)."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2)
            self._thread = None

    def __enter__(self) -> "ProgressHeartbeat":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
