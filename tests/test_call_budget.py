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


def test_locked_writer_fails_fast_and_folds_count_back(tmp_path):
    """The 2026-07-19 event-loop freeze: a held write lock used to busy-wait
    ~31s ON the loop (a potential self-deadlock — the holder's commit only
    advances when the loop does). A miss must return in a beat, and the
    missed increment must land in the row on the next successful write."""
    import sqlite3
    import time

    _use_tmp_db(tmp_path)
    assert call_budget.record_call() == 1

    blocker = sqlite3.connect(str(tmp_path / "test.db"))
    blocker.execute("BEGIN IMMEDIATE")  # hold the write lock
    t0 = time.monotonic()
    n = call_budget.record_call()  # cannot persist
    elapsed = time.monotonic() - t0
    blocker.rollback()
    blocker.close()

    assert n == 2  # in-memory total is still right
    assert elapsed < 2.0, f"blocked {elapsed:.1f}s — busy-wait is back"
    assert call_budget._pending == 1

    # Budget checks between the miss and the flush must still see the call.
    assert call_budget.calls_today() == 2

    assert call_budget.record_call() == 3  # 1 stored + 1 pending + 1 new
    assert call_budget._pending == 0

    # The ROW carries all three: a fresh reader (restart) agrees.
    call_budget.set_db_path(str(tmp_path / "test.db"))
    assert call_budget.calls_today() == 3


# ---------------------------------------------------------------------------
# Pacing envelope — the non-reserved pool is time-shaped toward the peak
# window (measured flow peaks 12-22 UTC; greedy consumption used to exhaust
# the pool by ~13:36 UTC, dead exactly when the flow arrives).
# ---------------------------------------------------------------------------


def test_paced_limit_offpeak_ration_and_peak_unlock():
    kw = dict(peak_start_hour=12, peak_end_hour=22, offpeak_share=0.4)
    # Overnight: only the ration is spendable.
    assert call_budget.paced_limit(150, now_hour=0, **kw) == 60
    assert call_budget.paced_limit(150, now_hour=11, **kw) == 60
    # Inside and after the window: the full pool.
    assert call_budget.paced_limit(150, now_hour=12, **kw) == 150
    assert call_budget.paced_limit(150, now_hour=17, **kw) == 150
    assert call_budget.paced_limit(150, now_hour=23, **kw) == 150


def test_paced_limit_disable_switches():
    # share >= 1 disables pacing entirely.
    assert call_budget.paced_limit(
        150, peak_start_hour=12, peak_end_hour=22, offpeak_share=1.0,
        now_hour=3) == 150
    # Degenerate window disables pacing.
    assert call_budget.paced_limit(
        150, peak_start_hour=22, peak_end_hour=12, offpeak_share=0.4,
        now_hour=3) == 150
    # Non-positive base passes through (callers special-case budget <= 0).
    assert call_budget.paced_limit(
        0, peak_start_hour=12, peak_end_hour=22, offpeak_share=0.4,
        now_hour=3) == 0


def test_non_reserved_limit_subtracts_reserve_then_paces(monkeypatch):
    from unittest.mock import MagicMock

    s = MagicMock()
    s.nlp.daily_claude_call_budget = 175
    s.nlp.claude_reserve_for_pinned = 25
    s.nlp.budget_peak_start_hour_utc = 12
    s.nlp.budget_peak_end_hour_utc = 22
    s.nlp.budget_offpeak_share = 0.4

    import auramaur.nlp.call_budget as cb

    real = cb.paced_limit
    captured = {}

    def spy(base_limit, **kwargs):
        captured["base"] = base_limit
        return real(base_limit, **kwargs)

    monkeypatch.setattr(cb, "paced_limit", spy)
    limit = cb.non_reserved_limit(s)
    assert captured["base"] == 150          # reserve subtracted first
    assert limit in (60, 150)               # paced by wall clock
