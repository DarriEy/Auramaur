"""Reconciler-merge must never persist a live held position at a <=0 mark.

A reconciler position whose token price can't be resolved (e.g. the markets
row lacks clob tokens, so side-resolution fails) arrives at current_price 0.
_sync_live floors such marks to avg_cost; _merge_new_positions (the additive
reconciler path) used to write the raw 0 straight through — a phantom -100%
that distorts the risk gates. It must apply the same floor.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from auramaur.broker.sync import PositionSyncer
from auramaur.db.database import Database
from auramaur.exchange.models import LivePosition, TokenType


def _syncer(db: Database) -> PositionSyncer:
    syncer = PositionSyncer.__new__(PositionSyncer)
    syncer._settings = MagicMock()
    syncer._settings.is_live = True  # is_paper_flag -> 0 (live rows)
    syncer._db = db
    syncer._exchange = SimpleNamespace(get_order_book=AsyncMock())
    syncer._paper = MagicMock()
    syncer._pnl = MagicMock()
    return syncer


async def _row(db: Database, market_id: str) -> dict:
    return await db.fetchone(
        "SELECT current_price, unrealized_pnl FROM portfolio "
        "WHERE market_id = ? AND is_paper = 0",
        (market_id,),
    )


def test_zero_mark_floors_to_avg_cost():
    """A reconciler position at current_price 0 is floored to avg_cost and its
    phantom -100% unrealized is zeroed; a healthy mark passes through intact."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            syncer = _syncer(db)
            stranded = LivePosition(
                market_id="mkt-stranded", token_id="tok-no", token=TokenType.NO,
                size=12.1, avg_cost=0.82, current_price=0.0,
            )
            healthy = LivePosition(
                market_id="999", token_id="tok-yes", token=TokenType.YES,
                size=10.0, avg_cost=0.40, current_price=0.50,
            )
            await syncer._merge_new_positions([stranded, healthy])

            s = await _row(db, "mkt-stranded")
            assert abs(s["current_price"] - 0.82) < 1e-9, s["current_price"]
            assert abs(s["unrealized_pnl"]) < 1e-9, s["unrealized_pnl"]

            h = await _row(db, "999")
            assert abs(h["current_price"] - 0.50) < 1e-9, h["current_price"]
            assert abs(h["unrealized_pnl"] - 1.0) < 1e-9, h["unrealized_pnl"]
        finally:
            await db.close()

    asyncio.run(run())
