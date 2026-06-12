"""Persistent daily Claude-call counter — survives restarts, shared cross-process.

The budget counter used to live in-memory on each analyzer instance
(`self._daily_calls`), which broke it twice over:

* Every restart reset it to zero, so the router kept treating the Claude
  budget as fresh — on 2026-06-12 the bot restarted five times and the
  nominal 80-call/day budget was effectively ignored (the Gemini handoff
  at ``claude_budget_threshold`` never engaged).
* Three classes (ClaudeAnalyzer, StrategicAnalyzer, EnsembleAnalyzer) each
  kept a private counter, so the "daily budget" was really up to 3x the
  configured number.

One row per day in the bot's own sqlite file fixes both. Helpers are
synchronous on purpose: they run a sub-millisecond statement on a
short-lived WAL connection, callers sit inside multi-second LLM calls, and
keeping them sync means no analyzer constructor has to grow an async
``Database`` dependency. Every sqlite failure degrades to a process-local
in-memory count — the budget gets restart-blind again, but analysis never
breaks on a locked/missing file.
"""

from __future__ import annotations

import sqlite3
from datetime import date

import structlog

log = structlog.get_logger()

_db_path = "auramaur.db"  # same working-directory convention as Database
_mem_calls = 0
_mem_day = ""


def set_db_path(path: str) -> None:
    """Point the counter at the instance's sqlite file (bot startup)."""
    global _db_path
    _db_path = path


def _conn() -> sqlite3.Connection:
    # 15s busy-timeout: the bot's scan/settlement cycles hold the write lock
    # for multi-second stretches and 2s lost the race twice within 90s of
    # deploy. Callers sit inside multi-second LLM calls, so waiting is free;
    # losing the write silently undercounts the budget instead.
    conn = sqlite3.connect(_db_path, timeout=15)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS llm_call_counter (
               day TEXT PRIMARY KEY,
               claude_calls INTEGER NOT NULL DEFAULT 0
           )"""
    )
    return conn


def calls_today() -> int:
    """Claude calls recorded today across all processes and restarts."""
    today = date.today().isoformat()
    try:
        with _conn() as conn:
            row = conn.execute(
                "SELECT claude_calls FROM llm_call_counter WHERE day = ?",
                (today,),
            ).fetchone()
            return int(row[0]) if row else 0
    except sqlite3.Error as e:
        log.debug("call_budget.read_error", error=str(e))
        return _mem_calls if _mem_day == today else 0


def record_call() -> int:
    """Count one successful Claude call; returns today's new total."""
    global _mem_calls, _mem_day
    today = date.today().isoformat()
    if _mem_day != today:
        _mem_day, _mem_calls = today, 0
    _mem_calls += 1
    try:
        with _conn() as conn:
            conn.execute(
                """INSERT INTO llm_call_counter (day, claude_calls)
                   VALUES (?, 1)
                   ON CONFLICT(day) DO UPDATE SET
                       claude_calls = claude_calls + 1""",
                (today,),
            )
            row = conn.execute(
                "SELECT claude_calls FROM llm_call_counter WHERE day = ?",
                (today,),
            ).fetchone()
            return int(row[0])
    except sqlite3.Error as e:
        log.warning("call_budget.write_error", error=str(e))
        return _mem_calls
