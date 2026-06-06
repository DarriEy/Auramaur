"""Strategy attribution credits the *entry* strategy and ignores open noise.

Two blind spots, analogous to the Kalshi one, are locked in here:
  1. Open positions must NOT be counted as completed trades (they have
     realized_pnl = 0 until they close) — they're reported separately as
     open_positions / unrealized_pnl.
  2. Realized P&L is credited to the strategy that OPENED the position, not the
     order monitor that recorded its exit.
"""

from __future__ import annotations

import asyncio

from auramaur.db.database import Database
from auramaur.monitoring.attribution import PerformanceAttributor


def test_entry_attribution_and_open_vs_closed():
    async def run():
        db = Database(":memory:")
        await db.connect()

        # Market A: opened by news_speed, EXITED by order_monitor, closed +0.9.
        await db.execute(
            "INSERT INTO cost_basis (market_id, token, size, avg_cost, total_cost, "
            "realized_pnl, is_paper) VALUES ('A','YES',0,0,0,0.9,0)")
        await db.execute(
            "INSERT INTO trades (market_id, side, size, price, is_paper, strategy_source, timestamp) "
            "VALUES ('A','BUY',1,0.5,0,'news_speed','2026-06-01T01:00')")
        await db.execute(  # the exit — most recent, must NOT win attribution
            "INSERT INTO trades (market_id, side, size, price, is_paper, strategy_source, timestamp) "
            "VALUES ('A','SELL',1,0.6,0,'order_monitor','2026-06-02T01:00')")

        # Market B: OPEN position opened by news_speed, unrealized +2.
        await db.execute(
            "INSERT INTO cost_basis (market_id, token, size, avg_cost, total_cost, "
            "realized_pnl, is_paper) VALUES ('B','YES',10,0.5,5,0,0)")
        await db.execute(
            "INSERT INTO portfolio (market_id, exchange, side, size, avg_price, "
            "current_price, is_paper) VALUES ('B','polymarket','BUY',10,0.5,0.7,0)")
        await db.execute(
            "INSERT INTO trades (market_id, side, size, price, is_paper, strategy_source, timestamp) "
            "VALUES ('B','BUY',10,0.5,0,'news_speed','2026-06-01T01:00')")
        await db.commit()

        attr = PerformanceAttributor(db=db)
        rows = {r["strategy_source"]: r for r in await attr.get_strategy_summary(is_live=True)}

        # The exit plumbing never claims the P&L.
        assert "order_monitor" not in rows
        ns = rows["news_speed"]
        assert ns["trade_count"] == 1            # only the closed market A
        assert ns["wins"] == 1
        assert abs(ns["total_pnl"] - 0.9) < 1e-9
        assert ns["open_positions"] == 1         # market B, not counted as a trade
        assert abs(ns["unrealized_pnl"] - 2.0) < 1e-9  # (0.7 - 0.5) * 10
        await db.close()

    asyncio.run(run())


def test_order_monitor_only_market_still_visible():
    """A market with no entry strategy still shows under order_monitor."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        await db.execute(
            "INSERT INTO cost_basis (market_id, token, size, avg_cost, total_cost, "
            "realized_pnl, is_paper) VALUES ('C','YES',0,0,0,-5.0,0)")
        await db.execute(
            "INSERT INTO trades (market_id, side, size, price, is_paper, strategy_source, timestamp) "
            "VALUES ('C','SELL',1,0.4,0,'order_monitor','2026-06-02T01:00')")
        await db.commit()

        attr = PerformanceAttributor(db=db)
        rows = {r["strategy_source"]: r for r in await attr.get_strategy_summary(is_live=True)}
        assert "order_monitor" in rows
        assert abs(rows["order_monitor"]["total_pnl"] - (-5.0)) < 1e-9
        await db.close()

    asyncio.run(run())


if __name__ == "__main__":
    test_entry_attribution_and_open_vs_closed()
    test_order_monitor_only_market_still_visible()
    print("ok")
