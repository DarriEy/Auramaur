"""Category lineage must survive venue reconciliation and reporting."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from auramaur.broker.sync import PositionSyncer
from auramaur.db.database import Database
from auramaur.exchange.models import LivePosition
from auramaur.risk.portfolio import PortfolioTracker
from auramaur.web.queries import category_exposure


async def _seed_empty_portfolio_with_classified_market(db: Database) -> None:
    await db.execute(
        """INSERT INTO markets (id, exchange, question, category, active, last_updated)
           VALUES ('m1', 'polymarket', 'Will Bitcoin rise?', 'crypto', 1,
                   datetime('now'))""")
    await db.execute(
        """INSERT INTO portfolio
           (market_id, exchange, side, size, avg_price, current_price,
            unrealized_pnl, category, token, token_id, is_paper, updated_at)
           VALUES ('m1', 'polymarket', 'BUY', 10, .4, .5, 1, '', 'YES',
                   'tok', 1, datetime('now'))""")
    await db.commit()


@pytest.mark.asyncio
async def test_portfolio_risk_read_falls_back_to_market_category():
    db = Database(":memory:")
    await db.connect()
    try:
        await _seed_empty_portfolio_with_classified_market(db)
        positions = await PortfolioTracker(db).get_positions(is_paper=True)
        assert positions[0].category == "crypto"
        assert await PortfolioTracker(db).get_category_exposure(is_paper=True) == {
            "crypto": 100.0,
        }
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_web_category_exposure_falls_back_to_market_category():
    db = Database(":memory:")
    await db.connect()
    try:
        await _seed_empty_portfolio_with_classified_market(db)
        rows = await category_exposure(db, 1)
        assert rows == [{"category": "crypto", "positions": 1, "value": 5.0}]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_reconciler_does_not_erase_existing_category():
    db = Database(":memory:")
    await db.connect()
    try:
        await _seed_empty_portfolio_with_classified_market(db)
        await db.execute(
            "UPDATE portfolio SET category = 'crypto' WHERE market_id = 'm1'")
        await db.commit()

        syncer = PositionSyncer.__new__(PositionSyncer)
        syncer._settings = MagicMock(is_live=False)
        syncer._db = db
        syncer._exchange = SimpleNamespace(get_order_book=AsyncMock())
        await syncer._reconcile([LivePosition(
            market_id="m1", size=10, avg_cost=.4, current_price=.5,
            category="",
        )])

        row = await db.fetchone(
            "SELECT category FROM portfolio WHERE market_id = 'm1'")
        assert row["category"] == "crypto"
    finally:
        await db.close()
