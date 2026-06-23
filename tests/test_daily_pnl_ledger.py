"""Phase 4: the daily-loss gate sources from pnl_ledger (live-only), not the
paper+live-conflated daily_stats.total_pnl.

PortfolioTracker.get_daily_pnl feeds risk/manager.py check_daily_loss, which
blocks LIVE entries. It must count only today's LIVE realized P&L — paper-forced
strategies lose by design and must never gate live trading.
"""

from __future__ import annotations

import asyncio

from auramaur.db.database import Database
from auramaur.risk.portfolio import PortfolioTracker


async def _ledger_row(db, pnl, is_paper, ref, *, today=True):
    realized = "datetime('now')" if today else "'2026-01-01T00:00:00+00:00'"
    await db.execute(
        "INSERT INTO pnl_ledger (market_id, venue, category, strategy_source, kind,"
        " token, qty, pnl, fees, is_paper, source_ref, realized_at)"
        f" VALUES ('m', 'polymarket', 'tech', 'llm', 'sell', 'YES', 1, ?, 0, ?, ?, {realized})",
        (pnl, is_paper, ref),
    )
    await db.commit()


def test_get_daily_pnl_is_live_only_and_today_only():
    async def run():
        db = Database(":memory:")
        await db.connect()
        # Today, LIVE: -5 and +2  -> net -3 (the only rows that should count)
        await _ledger_row(db, -5.0, 0, "live-a")
        await _ledger_row(db, 2.0, 0, "live-b")
        # Today, PAPER: -100 (the by-design paper loss that used to leak in)
        await _ledger_row(db, -100.0, 1, "paper-today")
        # Old day, LIVE: -50 (different day, must not count toward "today")
        await _ledger_row(db, -50.0, 0, "live-old", today=False)

        pnl = await PortfolioTracker(db).get_daily_pnl()
        assert abs(pnl - (-3.0)) < 1e-9, f"expected -3.0 (today live only), got {pnl}"

    asyncio.run(run())


def test_get_daily_pnl_zero_when_no_live_realizations():
    async def run():
        db = Database(":memory:")
        await db.connect()
        # Only a paper realization today — the live gate must read 0.
        await _ledger_row(db, -250.0, 1, "paper-only")
        pnl = await PortfolioTracker(db).get_daily_pnl()
        assert pnl == 0.0

    asyncio.run(run())
