"""Persistent daily Claude-call counter — survives restarts, shared cross-process.

The budget counter used to live in-memory on each analyzer instance
(`self._daily_calls`), which broke it twice over:

* Every restart reset it to zero, so the router kept treating the Claude
  budget as fresh — on 2026-06-12 the bot restarted five times and the
  nominal 80-call/day budget was effectively ignored (the Gemini handoff
  at ``claude_budget_threshold`` never engaged).
* Four classes (Claude/Strategic/Ensemble/Agent analyzers) each kept a
  private counter, so the "daily budget" was really up to 4x the
  configured number.

One row per day in the bot's own sqlite file fixes both. Helpers are
synchronous on purpose: sub-millisecond statements, callers sit inside
multi-second LLM calls, and no analyzer constructor has to grow an async
``Database`` dependency.

Connection handling matters here (learned the hard way — #117's timeout
bump didn't stop the lock warnings): the first cut opened a NEW connection
per call and ran ``CREATE TABLE IF NOT EXISTS`` every time — a DDL write
even on the read path — colliding with the bot's long write transactions.
One cached connection per process runs the DDL once and reads never write.

Locking policy (reworked 2026-07-19 after the busy-wait froze the bot): these
helpers run synchronously ON the asyncio event loop, and the bot's own
``Database`` writer only advances when that loop advances — so busy-waiting
here for the bot's write lock can be a SELF-deadlock, and every failed write
used to stall every pillar ~31s (15s busy x2 + 1s sleep). No timeout value
fixes waiting on a lock whose holder you are blocking. So: fail FAST
(250ms), count the missed increment in ``_pending``, and fold it into the
next successful write — the persistent counter stays exact without ever
blocking the loop for more than a beat.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import date

import structlog

log = structlog.get_logger()

_db_path = "auramaur.db"  # same working-directory convention as Database
_conn_cache: sqlite3.Connection | None = None
_lock = threading.Lock()  # the connection is shared cross-thread
_mem_calls = 0
_mem_day = ""
_pending = 0  # increments recorded in memory but not yet persisted
_PENDING_ALARM = 25  # sustained persistence failure — worth a warning


def set_db_path(path: str) -> None:
    """Point the counter at the instance's sqlite file (bot startup)."""
    global _db_path, _mem_calls, _mem_day, _pending
    if path != _db_path:
        _reset_conn()
        _mem_calls, _mem_day, _pending = 0, "", 0
    _db_path = path


def _conn() -> sqlite3.Connection:
    """One cached connection per process; schema created once.

    timeout=0.25 is deliberate — see the locking policy in the module
    docstring: waiting longer cannot help and can freeze the event loop.
    """
    global _conn_cache
    if _conn_cache is None:
        conn = sqlite3.connect(_db_path, timeout=0.25, check_same_thread=False)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS llm_call_counter (
                   day TEXT PRIMARY KEY,
                   claude_calls INTEGER NOT NULL DEFAULT 0
               )"""
        )
        conn.commit()
        _conn_cache = conn
    return _conn_cache


def _reset_conn() -> None:
    global _conn_cache
    if _conn_cache is not None:
        try:
            _conn_cache.close()
        except sqlite3.Error:
            pass
        _conn_cache = None


def calls_today() -> int:
    """Claude calls recorded today across all processes and restarts."""
    today = date.today().isoformat()
    with _lock:
        pending = _pending if _mem_day == today else 0
        try:
            row = _conn().execute(
                "SELECT claude_calls FROM llm_call_counter WHERE day = ?",
                (today,),
            ).fetchone()
            # Pending increments are real calls that haven't landed in the
            # row yet — the budget check must see them.
            return (int(row[0]) if row else 0) + pending
        except sqlite3.Error as e:
            log.debug("call_budget.read_error", error=str(e))
            _reset_conn()
            return _mem_calls if _mem_day == today else 0


def paced_limit(base_limit: int, *, peak_start_hour: int, peak_end_hour: int,
                offpeak_share: float, now_hour: int | None = None) -> int:
    """Time-shaped ceiling on the NON-RESERVED budget.

    The daily counter resets at midnight (UTC on this host), but the bot's
    opportunity flow peaks 12-22 UTC (measured: signals/entries cluster at
    14-17 UTC — US econ prints land 12:30 UTC, then US market hours), and
    under greedy consumption the pool was documented exhausted by ~13:36 UTC
    — dead exactly when the flow arrives. Before the peak window opens,
    non-reserved callers may spend at most ``offpeak_share`` of the pool;
    inside and after it, the full pool. The reserved (pinned) slice is
    untouched — the proven edges were never subject to this limit.

    Pure function (``now_hour`` injectable for tests). Hours are UTC.
    ``offpeak_share >= 1`` or an empty window disables pacing.
    """
    if base_limit <= 0:
        return base_limit
    share = max(0.0, min(1.0, offpeak_share))
    if share >= 1.0 or peak_start_hour >= peak_end_hour:
        return base_limit
    if now_hour is None:
        from datetime import datetime, timezone
        now_hour = datetime.now(timezone.utc).hour
    if now_hour < peak_start_hour:
        return int(base_limit * share)
    return base_limit


def non_reserved_limit(settings) -> int:
    """The shared non-reserved ceiling every unpinned caller checks against:
    (budget - pinned reserve), paced by the peak-window envelope. Returns 0
    when the budget is fully reserved; a non-positive configured budget means
    unlimited (callers already special-case budget <= 0)."""
    nlp = settings.nlp
    base = max(0, nlp.daily_claude_call_budget - nlp.claude_reserve_for_pinned)
    return paced_limit(
        base,
        peak_start_hour=nlp.budget_peak_start_hour_utc,
        peak_end_hour=nlp.budget_peak_end_hour_utc,
        offpeak_share=nlp.budget_offpeak_share,
    )


def record_call() -> int:
    """Count one successful Claude call; returns today's new total.

    Single fast attempt (never a busy-wait — module docstring). On a lock
    miss the increment goes to ``_pending`` and is folded into the next
    successful write, so the persistent counter stays exact.
    """
    global _mem_calls, _mem_day, _pending
    today = date.today().isoformat()
    with _lock:
        if _mem_day != today:
            # The budget is daily; unpersisted increments from yesterday
            # expire with the day they belonged to.
            _mem_day, _mem_calls, _pending = today, 0, 0
        _mem_calls += 1
        try:
            conn = _conn()
            conn.execute(
                """INSERT INTO llm_call_counter (day, claude_calls)
                   VALUES (?, ?)
                   ON CONFLICT(day) DO UPDATE SET
                       claude_calls = claude_calls + ?""",
                (today, 1 + _pending, 1 + _pending),
            )
            conn.commit()
            _pending = 0
            row = conn.execute(
                "SELECT claude_calls FROM llm_call_counter WHERE day = ?",
                (today,),
            ).fetchone()
            return int(row[0])
        except sqlite3.Error as e:
            _reset_conn()
            _pending += 1
            # A miss is now harmless (folded in later) — only sustained
            # failure to persist deserves the health panel's attention.
            if _pending >= _PENDING_ALARM:
                log.warning("call_budget.write_error", error=str(e),
                            pending=_pending)
            else:
                log.debug("call_budget.write_deferred", error=str(e),
                          pending=_pending)
    return _mem_calls
