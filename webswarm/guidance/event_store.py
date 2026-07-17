"""Runtime event storage for guidance.

Each WebSwarm task run owns a GuidanceEventStore containing audit events for web_probing
and subtask_experience. Events support log review only and never modify agent state.
"""

from __future__ import annotations

import threading
import time


# Use a consistent human-readable timestamp in guidance events.
_TS_FMT = "%Y-%m-%d %H:%M:%S"


def _now_str() -> str:
    """Uniform source for all guidance timestamps; returns 'YYYY-MM-DD HH:MM:SS'."""
    return time.strftime(_TS_FMT, time.localtime())


class GuidanceEventStore:
    """Task-level guidance audit-event stream."""

    def __init__(self) -> None:
        self._events: list[dict] = []
        self._lock = threading.RLock()

    def append_event(self, event: dict) -> None:
        """Record one audit event, such as web_probing or subtask_experience."""
        with self._lock:
            self._events.append(event)

    def all_events(self) -> list[dict]:
        with self._lock:
            return list(self._events)

    def stats(self) -> dict:
        with self._lock:
            return {
                "n_events": len(self._events),
            }

    def export(self) -> dict:
        """Export a complete snapshot suitable for inclusion in the task log."""
        with self._lock:
            return {
                "events": list(self._events),
                "stats": self.stats(),
            }
