"""Regression tests for the Kalshi P&L-tracking gap.

Kalshi (and Crypto.com) trade exclusively through the strategic path, which —
unlike the Polymarket-only price-monitor path — never recorded calibration
predictions. As a result Kalshi markets never entered the calibration table,
the resolution tracker (which iterated only calibration) never settled them,
their realized P&L was never booked, and the attribution scorecard had no
per-venue view to surface any of it.

These tests lock in the three fixes:
  1. The resolution tracker settles *held positions*, not just markets with a
     calibration prediction.
  2. get_venue_summary breaks realized/unrealized P&L down by exchange.

The third fix — the strategic path now records a calibration prediction for
every analyzed market (engine._run_cycle_strategic) — closes the root cause
going forward; fix #1 here is the robust safety net that also recovers the
already-open book that predates it.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from auramaur.db.database import Database
from auramaur.exchange.models import Market
from auramaur.monitoring.attribution import PerformanceAttributor
from auramaur.strategy.resolution_tracker import ResolutionTracker


def _make_discovery(market: Market | None):
    disc = AsyncMock()
    disc.get_market = AsyncMock(return_value=market)
    return disc


def test_held_position_without_calibration_settles():
    """A held Kalshi position with NO calibration row still settles.

    This is the core regression: previously check_resolutions only looked at
    the calibration table, so a Kalshi position the bot held would never be
    settled and its realized P&L would never be booked.
    """
    async def run():
        db = Database(":memory:")
        await db.connect()

        mid = "KXTESTMARKET-30JAN01-YES"
        # A live BUY-YES position, entered at 0.40, no calibration prediction.
        await db.execute(
            """INSERT INTO portfolio (market_id, exchange, side, size, avg_price,
                                      current_price, is_paper)
               VALUES (?, 'kalshi', 'BUY', 10, 0.40, 0.99, 0)""",
            (mid,),
        )
        await db.execute(
            "INSERT INTO markets (id, exchange, question, active, last_updated) "
            "VALUES (?, 'kalshi', 'q', 1, datetime('now'))",
            (mid,),
        )
        await db.commit()

        # Sanity: nothing in calibration, so the OLD query would have found this.
        cal = await db.fetchall("SELECT * FROM calibration")
        assert cal == []

        # Market reports resolved YES (inactive + price pinned to 0.99).
        market = Market(
            id=mid, exchange="kalshi", question="q",
            active=False, outcome_yes_price=0.99, outcome_no_price=0.01,
        )
        calibration = AsyncMock()
        tracker = ResolutionTracker(
            db=db, calibration=calibration,
            discoveries={"kalshi": _make_discovery(market)},
        )

        count = await tracker.check_resolutions()
        assert count == 1

        # Position settled: removed from portfolio, P&L booked into cost_basis.
        remaining = await db.fetchall(
            "SELECT * FROM portfolio WHERE market_id = ?", (mid,)
        )
        assert remaining == []
        # BUY YES @0.40, resolves YES (->$1): pnl = (1 - 0.40) * 10 = 6.0
        cb = await db.fetchone(
            "SELECT realized_pnl FROM cost_basis WHERE market_id = ?", (mid,)
        )
        # cost_basis row only exists if it was pre-seeded; settlement updates it
        # when present. Verify daily_stats captured the realized P&L either way.
        stats = await db.fetchone("SELECT total_pnl FROM daily_stats LIMIT 1")
        assert stats is not None
        assert abs(stats["total_pnl"] - 6.0) < 1e-9
        await db.close()

    asyncio.run(run())


def test_venue_summary_breaks_down_by_exchange():
    """get_venue_summary attributes exposure + realized P&L per venue."""
    async def run():
        db = Database(":memory:")
        await db.connect()

        # Open positions: one Kalshi, one Polymarket (live).
        await db.execute(
            "INSERT INTO portfolio (market_id, exchange, side, size, avg_price, "
            "current_price, is_paper) VALUES ('KXFOO-1', 'kalshi', 'BUY', 10, 0.50, 0.60, 0)"
        )
        await db.execute(
            "INSERT INTO portfolio (market_id, exchange, side, size, avg_price, "
            "current_price, is_paper) VALUES ('12345', 'polymarket', 'BUY', 10, 0.50, 0.40, 0)"
        )
        # Realized: Kalshi market resolved +4 (resolution_pnl), Poly sold -2 (cost_basis).
        await db.execute(
            "INSERT INTO markets (id, exchange, question, last_updated) "
            "VALUES ('KXBAR-1', 'kalshi', 'q', datetime('now'))"
        )
        await db.execute(
            "INSERT INTO resolution_pnl (market_id, category, pnl, resolved_at) "
            "VALUES ('KXBAR-1', 'crypto', 4.0, '2026-06-01T00:00')"
        )
        await db.execute(
            "INSERT INTO markets (id, exchange, question, last_updated) "
            "VALUES ('67890', 'polymarket', 'q', datetime('now'))"
        )
        await db.execute(
            "INSERT INTO cost_basis (market_id, token, size, avg_cost, total_cost, "
            "realized_pnl, is_paper) VALUES ('67890', 'YES', 0, 0, 0, -2.0, 0)"
        )
        await db.commit()

        attr = PerformanceAttributor(db=db)
        rows = {r["venue"]: r for r in await attr.get_venue_summary(is_live=True)}

        assert set(rows) == {"kalshi", "polymarket"}
        # Kalshi: exposure 10*0.50=5, unrealized (0.60-0.50)*10=+1, realized +4.
        assert abs(rows["kalshi"]["exposure"] - 5.0) < 1e-9
        assert abs(rows["kalshi"]["unrealized_pnl"] - 1.0) < 1e-9
        assert abs(rows["kalshi"]["realized_pnl"] - 4.0) < 1e-9
        # Polymarket: unrealized (0.40-0.50)*10=-1, realized -2 (sold).
        assert abs(rows["polymarket"]["unrealized_pnl"] - (-1.0)) < 1e-9
        assert abs(rows["polymarket"]["realized_pnl"] - (-2.0)) < 1e-9
        await db.close()

    asyncio.run(run())


def test_venue_summary_ignores_zero_cost_basis():
    """cost_basis.avg_cost=0 (Kraken spec book) must fall back to avg_price.

    Otherwise unrealized = (current - 0) * size books the whole position value
    as a gain (the +$246-vs-actual-(-$3.85) Kraken bug).
    """
    async def run():
        db = Database(":memory:")
        await db.connect()
        await db.execute(
            "INSERT INTO portfolio (market_id, exchange, side, size, avg_price, "
            "current_price, is_paper) VALUES ('XBTUSDC', 'kraken', 'BUY', 10, 0.50, 0.60, 0)"
        )
        # Kraken writes a cost_basis row with avg_cost = 0.
        await db.execute(
            "INSERT INTO cost_basis (market_id, token, size, avg_cost, total_cost, "
            "realized_pnl, is_paper) VALUES ('XBTUSDC', 'YES', 10, 0, 0, 0, 0)"
        )
        await db.commit()
        attr = PerformanceAttributor(db=db)
        rows = {r["venue"]: r for r in await attr.get_venue_summary(is_live=True)}
        # (0.60 - 0.50) * 10 = +1.0, NOT (0.60 - 0) * 10 = +6.0
        assert abs(rows["kraken"]["unrealized_pnl"] - 1.0) < 1e-9
        await db.close()

    asyncio.run(run())


def test_venue_summary_paper_scopes_out_live():
    """Paper mode excludes live Polymarket rows but keeps non-Polymarket venues."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        await db.execute(
            "INSERT INTO portfolio (market_id, exchange, side, size, avg_price, "
            "current_price, is_paper) VALUES ('12345', 'polymarket', 'BUY', 10, 0.50, 0.40, 0)"
        )
        await db.commit()
        attr = PerformanceAttributor(db=db)
        rows = await attr.get_venue_summary(is_live=False)
        # The only row is a live Polymarket position → invisible in paper mode.
        assert all(r["venue"] != "polymarket" for r in rows)
        await db.close()

    asyncio.run(run())


if __name__ == "__main__":
    test_held_position_without_calibration_settles()
    test_venue_summary_breaks_down_by_exchange()
    test_venue_summary_paper_scopes_out_live()
    print("ok")
