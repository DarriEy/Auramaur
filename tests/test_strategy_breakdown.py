"""Strategy Compare attribution for open-book unrealized P&L."""

import pytest

from auramaur.db.database import Database
from auramaur.web.queries import strategy_breakdown


@pytest.mark.asyncio
async def test_unrealized_uses_entry_strategy_and_token_cost_basis():
    db = Database(":memory:")
    await db.connect()
    try:
        await db.execute(
            """INSERT INTO portfolio
               (market_id, exchange, side, size, avg_price, current_price,
                category, token, token_id, is_paper)
               VALUES ('m1', 'polymarket', 'BUY', 10, .40, .60,
                       'tech', 'YES', 'yes-1', 1)""")
        await db.execute(
            """INSERT INTO cost_basis
               (market_id, token, token_id, size, avg_cost, total_cost,
                realized_pnl, is_paper)
               VALUES ('m1', 'YES', 'yes-1', 10, .50, 5, 0, 1)""")
        await db.execute(
            """INSERT INTO trades
               (market_id, timestamp, side, size, price, is_paper, status,
                strategy_source)
               VALUES ('m1', '2026-01-01', 'BUY', 10, .50, 1, 'filled',
                       'weather_temp')""")
        await db.execute(
            """INSERT INTO trades
               (market_id, timestamp, side, size, price, is_paper, status,
                strategy_source)
               VALUES ('m1', '2026-01-02', 'BUY', 0, .60, 1, 'filled',
                       'order_monitor')""")
        await db.commit()

        rows = {r["strategy"]: r for r in await strategy_breakdown(db, 1)}
        assert rows["weather_temp"]["open_positions"] == 1
        assert rows["weather_temp"]["unrealized"] == pytest.approx(1.0)
        assert "order_monitor" not in rows
    finally:
        await db.close()
