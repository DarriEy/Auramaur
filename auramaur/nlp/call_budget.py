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
Now one cached connection per process runs the DDL once, reads never
write, and a locked write is retried once before degrading to the
process-local in-memory count (restart-blind, but analysis never breaks).
"""

from __future__ import annotations

import sqlite3
import time
from datetime import date

import structlog

log = structlog.get_logger()

_db_path = "auramaur.db"  # same working-directory convention as Database
_conn_cache: sqlite3.Connection | None = None
_mem_calls = 0
_mem_day = ""


def set_db_path(path: str) -> None:
    """Point the counter at the instance's sqlite file (bot startup)."""
    global _db_path
    if path != _db_path:
        _reset_conn()
    _db_path = path


def _conn() -> sqlite3.Connection:
    """One cached connection per process; schema created once."""
    global _conn_cache
    if _conn_cache is None:
        conn = sqlite3.connect(_db_path, timeout=15, check_same_thread=False)
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
    try:
        row = _conn().execute(
            "SELECT claude_calls FROM llm_call_counter WHERE day = ?",
            (today,),
        ).fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error as e:
        log.debug("call_budget.read_error", error=str(e))
        _reset_conn()
        return _mem_calls if _mem_day == today else 0


def record_call() -> int:
    """Count one successful Claude call; returns today's new total."""
    global _mem_calls, _mem_day
    today = date.today().isoformat()
    if _mem_day != today:
        _mem_day, _mem_calls = today, 0
    _mem_calls += 1
    for attempt in (1, 2):
        try:
            conn = _conn()
            conn.execute(
                """INSERT INTO llm_call_counter (day, claude_calls)
                   VALUES (?, 1)
                   ON CONFLICT(day) DO UPDATE SET
                       claude_calls = claude_calls + 1""",
                (today,),
            )
            conn.commit()
            row = conn.execute(
                "SELECT claude_calls FROM llm_call_counter WHERE day = ?",
                (today,),
            ).fetchone()
            return int(row[0])
        except sqlite3.Error as e:
            _reset_conn()
            if attempt == 1:
                time.sleep(1.0)
                continue
            log.warning("call_budget.write_error", error=str(e))
    return _mem_calls
