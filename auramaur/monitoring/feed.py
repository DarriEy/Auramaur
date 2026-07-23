"""Quiet-feed core: noise aggregation, hourly summaries, and a loop watchdog.

The terminal feed used to print every event as it happened — an expired
market-maker quote rendered identically to a dying strategy entry, and the
operator could not tell routine churn from failure. This module gives the
display layer two tools:

* :class:`NoiseAggregator` — counters for high-frequency routine events
  (MM quotes, Claude calls, cache hits, scans). They print nothing in the
  moment; a single summary line is emitted once per interval.
* :class:`LoopWatchdog` — a daemon *thread* that detects a frozen asyncio
  event loop. A blocking sync call inside the loop silences every coroutine
  including any in-loop monitor (the 2026-06-10 freeze lasted 84 minutes
  with zero output); only a thread can still speak when that happens.
"""

from __future__ import annotations

import os
import threading
import time

import structlog

log = structlog.get_logger()

# Counter keys with human labels for the summary line, in display order.
_NOISE_LABELS: dict[str, str] = {
    "claude_call": "claude calls",
    "cache_hit": "cache hits",
    "evidence": "evidence pulls",
    "scan": "scans",
    "mm_quote": "mm quotes",
    "mm_expired": "mm expiries",
    "analyzed": "analyzed",
}


class NoiseAggregator:
    """Counts routine events and renders a periodic one-line summary."""

    def __init__(self, interval_seconds: float = 3600.0):
        self.interval_seconds = interval_seconds
        self._counts: dict[str, int] = {}
        self._started = time.monotonic()
        self._last_flush = time.monotonic()

    def bump(self, key: str, n: int = 1) -> None:
        self._counts[key] = self._counts.get(key, 0) + n

    def due(self) -> bool:
        return (time.monotonic() - self._last_flush) >= self.interval_seconds

    def flush(self) -> str | None:
        """Return the summary line and reset counters, or None if all zero."""
        self._last_flush = time.monotonic()
        if not any(self._counts.values()):
            return None
        parts = [
            f"{self._counts[key]} {label}"
            for key, label in _NOISE_LABELS.items()
            if self._counts.get(key)
        ]
        # Anything bumped under an unlisted key still shows, raw.
        parts += [
            f"{count} {key}"
            for key, count in self._counts.items()
            if count and key not in _NOISE_LABELS
        ]
        self._counts.clear()
        return " | ".join(parts)


class LoopWatchdog(threading.Thread):
    """Daemon thread that alerts when the asyncio event loop stops beating.

    The loop side calls :meth:`beat` from any frequently-running coroutine
    (the order monitor ticks every few seconds). The thread checks staleness
    on its own clock; if the gap exceeds ``stall_seconds`` it emits a loud
    warning through ``alert`` — and a recovery line (with the measured gap)
    once beats resume. Uses ``print``-style callables rather than importing
    the rich console so the display layer stays in charge of rendering.
    """

    def __init__(
        self,
        stall_seconds: float = 180.0,
        check_interval: float = 30.0,
        alert=None,
        hard_exit_seconds: float = 900.0,
    ):
        super().__init__(name="loop-watchdog", daemon=True)
        self.stall_seconds = stall_seconds
        self.check_interval = check_interval
        # A stall this long means exits and risk checks have been dead for
        # its whole duration and self-recovery is no longer worth waiting
        # for: exit nonzero so Docker's restart:on-failure revives the bot
        # (the 2026-07-23 wedges each froze the book 25-35 min and needed
        # manual restarts; the container healthcheck cannot see a stall).
        # Set <= 0 to disable.
        self.hard_exit_seconds = hard_exit_seconds
        self._alert = alert or (lambda msg: None)
        self._last_beat = time.monotonic()
        self._stalled_since: float | None = None
        self._stop = threading.Event()

    def beat(self) -> None:
        self._last_beat = time.monotonic()

    @staticmethod
    def _hard_exit() -> None:  # pragma: no cover - patched in tests
        # os._exit, not sys.exit: the event loop is dead, so graceful
        # shutdown would hang behind the very stall being escaped. State is
        # safe — every completed write is committed, and the WAL recovers
        # anything mid-flight on the next open.
        os._exit(70)

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:  # pragma: no cover - exercised via _check()
        while not self._stop.wait(self.check_interval):
            self._check()

    def _check(self) -> None:
        gap = time.monotonic() - self._last_beat
        if (
            self.hard_exit_seconds > 0
            and gap >= self.hard_exit_seconds
            and not self._stop.is_set()
        ):
            self._alert(
                f"EVENT LOOP DEAD for {gap:.0f}s — exiting so the "
                f"container restart policy can revive the bot."
            )
            log.critical("watchdog.hard_exit", gap_seconds=round(gap))
            self._hard_exit()
            return
        if gap >= self.stall_seconds and self._stalled_since is None:
            self._stalled_since = self._last_beat
            self._alert(
                f"EVENT LOOP STALLED — no heartbeat for {gap:.0f}s "
                f"(blocking call in the loop?). Exits and risk checks are "
                f"NOT running."
            )
            log.error("watchdog.loop_stalled", gap_seconds=round(gap))
        elif gap < self.stall_seconds and self._stalled_since is not None:
            stalled_for = self._last_beat - self._stalled_since
            self._stalled_since = None
            self._alert(f"event loop recovered after ~{stalled_for:.0f}s stall")
            log.warning("watchdog.loop_recovered", stalled_seconds=round(stalled_for))
