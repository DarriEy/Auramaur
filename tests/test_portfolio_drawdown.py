"""Drawdown gate equity feed: peak ratchet and windowing in note_equity."""

import pytest

from auramaur.db.database import Database
from auramaur.risk.portfolio import PortfolioTracker


async def _tracker():
    db = Database(":memory:")
    await db.connect()
    return db, PortfolioTracker(db)


@pytest.mark.asyncio
async def test_sustained_equity_sets_peak_and_dip_reads_as_drawdown():
    db, tracker = await _tracker()
    for _ in range(3):
        await tracker.note_equity(1000.0)
    assert await tracker.get_drawdown() == pytest.approx(0.0)
    await tracker.note_equity(800.0)
    assert await tracker.get_drawdown() == pytest.approx(20.0)
    await db.close()


@pytest.mark.asyncio
async def test_dip_is_detected_on_the_very_next_tick():
    db, tracker = await _tracker()
    for _ in range(3):
        await tracker.note_equity(1000.0)
    await tracker.note_equity(700.0)
    assert await tracker.get_drawdown() == pytest.approx(30.0)
    await db.close()


@pytest.mark.asyncio
async def test_single_tick_spike_does_not_ratchet_peak():
    db, tracker = await _tracker()
    for _ in range(3):
        await tracker.note_equity(1000.0)
    # One optimistic mark tick, then equity returns to its sustained level:
    # the peak must stay at 1000, not 5000, so no phantom 80% drawdown.
    await tracker.note_equity(5000.0)
    for _ in range(3):
        await tracker.note_equity(1000.0)
    assert await tracker.get_drawdown() == pytest.approx(0.0)
    row = await db.fetchone("SELECT MAX(peak_balance) AS peak FROM daily_stats")
    assert float(row["peak"]) == pytest.approx(1000.0)
    await db.close()


@pytest.mark.asyncio
async def test_sustained_new_high_does_ratchet_peak():
    db, tracker = await _tracker()
    for _ in range(3):
        await tracker.note_equity(1000.0)
    for _ in range(3):
        await tracker.note_equity(1200.0)
    await tracker.note_equity(900.0)
    assert await tracker.get_drawdown() == pytest.approx(25.0)
    await db.close()


@pytest.mark.asyncio
async def test_peak_older_than_window_no_longer_binds():
    db, tracker = await _tracker()
    await db.execute(
        "INSERT INTO daily_stats (date, peak_balance) "
        "VALUES (date('now', '-40 days'), 5000.0)")
    await db.commit()
    for _ in range(3):
        await tracker.note_equity(1000.0)
    assert await tracker.get_drawdown() == pytest.approx(0.0)
    await db.close()


@pytest.mark.asyncio
async def test_recent_persisted_peak_still_binds_across_restart():
    db, tracker = await _tracker()
    await db.execute(
        "INSERT INTO daily_stats (date, peak_balance) "
        "VALUES (date('now', '-5 days'), 2000.0)")
    await db.commit()
    for _ in range(3):
        await tracker.note_equity(1000.0)
    assert await tracker.get_drawdown() == pytest.approx(50.0)
    await db.close()


@pytest.mark.asyncio
async def test_cold_fallback_scopes_to_the_armed_book():
    """With settings attached, the legacy peak+unrealised fallback only sums
    the armed book's positions — paper marks must not read as live drawdown."""

    class _Live:
        is_live = True

    db = Database(":memory:")
    await db.connect()
    tracker = PortfolioTracker(db, settings=_Live())
    await db.execute(
        "INSERT INTO daily_stats (date, peak_balance) VALUES (date('now'), 2000.0)")
    await db.execute(
        """INSERT INTO portfolio
           (market_id, exchange, side, size, avg_price, current_price,
            token, is_paper, updated_at)
           VALUES ('paper1', 'polymarket', 'BUY', 2000, 0.5, 0.2, 'YES', 1,
                   datetime('now'))""")
    await db.commit()
    assert await tracker.get_drawdown() == pytest.approx(0.0)
    await db.close()


@pytest.mark.asyncio
async def test_zero_or_negative_equity_is_ignored():
    db, tracker = await _tracker()
    for _ in range(3):
        await tracker.note_equity(1000.0)
    await tracker.note_equity(0.0)
    await tracker.note_equity(-5.0)
    assert await tracker.get_drawdown() == pytest.approx(0.0)
    await db.close()
