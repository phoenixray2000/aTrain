import threading
import time
import unittest

from aTrain.cli_progress import (
    HEARTBEAT_PREFIX,
    STARTING_TASK,
    ProgressHeartbeat,
    format_line,
)


class _RecordingSink:
    """Thread-safe collector for emitted heartbeat lines."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lines: list[str] = []

    def __call__(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)

    def snapshot(self) -> list[str]:
        with self._lock:
            return list(self._lines)

    def heartbeats(self) -> list[str]:
        return [line for line in self.snapshot() if line.startswith(HEARTBEAT_PREFIX)]


def _wait_for(predicate, timeout: float = 2.0, poll: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    return False


class FormatLineTests(unittest.TestCase):
    def test_format_line_is_stable_contract(self):
        self.assertEqual(
            format_line("Transcribe", 12.5, 100.0),
            "[atrain-progress] task=Transcribe current=12.5 total=100.0",
        )


class ProgressHeartbeatTests(unittest.TestCase):
    def test_emits_on_advance_then_silent_on_stall(self):
        progress = {"task": "", "current": 0.0, "total": 0.0}
        sink = _RecordingSink()

        heartbeat = ProgressHeartbeat(progress, sink, interval_sec=0.05)
        heartbeat.start()
        try:
            # Initial liveness marker should appear quickly.
            self.assertTrue(
                _wait_for(lambda: len(sink.heartbeats()) >= 1),
                "expected an initial heartbeat line",
            )
            self.assertEqual(
                sink.heartbeats()[0],
                format_line(STARTING_TASK, 0.0, 0.0),
            )

            # Advance the mapping a few times; each distinct snapshot should be
            # emitted once the poll observes it.
            progress["task"] = "Transcribe"
            progress["total"] = 100.0
            for current in (10.0, 25.0, 60.0):
                progress["current"] = current
                self.assertTrue(
                    _wait_for(
                        lambda c=current: format_line("Transcribe", c, 100.0)
                        in sink.heartbeats()
                    ),
                    f"expected a heartbeat reflecting current={current}",
                )

            # We must have seen multiple advancing lines (starting + >=3 more).
            advancing = sink.heartbeats()
            self.assertGreaterEqual(len(advancing), 4)

            # Now STOP advancing: no new lines should be emitted while stalled.
            count_at_stall = len(sink.heartbeats())
            # Let several poll intervals elapse with an unchanged snapshot.
            time.sleep(0.05 * 6)
            self.assertEqual(
                len(sink.heartbeats()),
                count_at_stall,
                "a stalled (unchanged) mapping must NOT emit further heartbeats",
            )
        finally:
            heartbeat.stop()

        # After stop, the worker thread is gone.
        self.assertFalse(
            any(t.name == "atrain-progress-heartbeat" for t in threading.enumerate())
        )

    def test_context_manager_starts_and_stops(self):
        progress = {"task": "", "current": 0.0, "total": 0.0}
        sink = _RecordingSink()

        with ProgressHeartbeat(progress, sink, interval_sec=0.05):
            self.assertTrue(
                _wait_for(lambda: len(sink.heartbeats()) >= 1),
                "expected at least the initial heartbeat inside the context",
            )

        # Exiting the context joins the thread.
        self.assertFalse(
            any(t.name == "atrain-progress-heartbeat" for t in threading.enumerate())
        )

    def test_emit_failure_does_not_kill_thread(self):
        progress = {"task": "", "current": 0.0, "total": 0.0}
        calls: list[str] = []
        lock = threading.Lock()

        def flaky(line: str) -> None:
            with lock:
                calls.append(line)
            # Fail on the very first (initial) emit to prove the loop survives.
            if len(calls) == 1:
                raise RuntimeError("boom")

        heartbeat = ProgressHeartbeat(progress, flaky, interval_sec=0.05)
        heartbeat.start()
        try:
            progress["task"] = "Transcribe"
            progress["total"] = 50.0
            progress["current"] = 5.0
            self.assertTrue(
                _wait_for(
                    lambda: any(
                        line == format_line("Transcribe", 5.0, 50.0)
                        for line in calls
                    )
                ),
                "heartbeat thread should survive an emit exception and keep polling",
            )
        finally:
            heartbeat.stop()


if __name__ == "__main__":
    unittest.main()
