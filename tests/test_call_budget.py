"""Tests for the persistent daily Claude-call counter.

Regression (2026-06-12): the counter was in-memory per analyzer instance —
five restarts in one day reset it five times, so the 80-call budget and the
Gemini handoff threshold were effectively ignored. Worse, four classes each
kept a private counter, multiplying the real budget by up to 4x.
"""

from __future__ import annotations

from datetime import date

from auramaur.nlp import call_budget


def _use_tmp_db(tmp_path):
    call_budget.set_db_path(str(tmp_path / "test.db"))


def test_counts_persist_across_simulated_restarts(tmp_path):
    _use_tmp_db(tmp_path)
    assert call_budget.calls_today() == 0
    assert call_budget.record_call() == 1
    assert call_budget.record_call() == 2

    # "Restart": a fresh reader against the same file (module state is
    # irrelevant — the DB row is the source of truth).
    call_budget.set_db_path(str(tmp_path / "test.db"))
    assert call_budget.calls_today() == 2


def test_counter_is_shared_not_per_consumer(tmp_path):
    """Four analyzers recording into one row means the budget is global."""
    _use_tmp_db(tmp_path)
    for _ in range(4):
        call_budget.record_call()
    assert call_budget.calls_today() == 4


def test_day_rollover_scopes_to_today(tmp_path):
    _use_tmp_db(tmp_path)
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    conn.execute("CREATE TABLE IF NOT EXISTS llm_call_counter (day TEXT PRIMARY KEY, claude_calls INTEGER NOT NULL DEFAULT 0)")
    conn.execute("INSERT INTO llm_call_counter VALUES ('2020-01-01', 99)")
    conn.commit(); conn.close()
    assert call_budget.calls_today() == 0  # yesterday's row doesn't count
    assert call_budget.record_call() == 1


def test_sqlite_failure_degrades_to_memory(tmp_path):
    """A locked/missing file must never break analysis — falls back to a
    process-local count (restart-blind, but functional)."""
    call_budget.set_db_path(str(tmp_path / "nodir" / "missing.db"))
    call_budget._mem_calls, call_budget._mem_day = 0, ""
    n1 = call_budget.record_call()
    n2 = call_budget.record_call()
    assert (n1, n2) == (1, 2)
    assert call_budget.calls_today() == 2
    assert call_budget._mem_day == date.today().isoformat()
