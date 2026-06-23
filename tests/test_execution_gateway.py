"""Tests for ExecutionGateway (Phase 0 of the execution refactor).

The gateway is the extracted, reusable form of the engine's canonical entry
tail (route -> place -> record). These lock in that a submitted intent records
the fill, updates cost basis, and mirrors to trades — the behavior every
strategy will inherit when it routes through the gateway in later phases.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from auramaur.broker.execution_gateway import ExecutionGateway, TradeIntent
from auramaur.broker.pnl import PnLTracker
from auramaur.db.database import Database
from auramaur.exchange.models import (
    Market, Order, OrderResult, OrderSide, Signal, TokenType,
)
from config.settings import Settings


def _signal(market_id="m1"):
    return Signal(
        market_id=market_id, claude_prob=0.62, claude_confidence="HIGH",
        market_prob=0.50, edge=12.0, strategy_source="llm",
    )


def _paper_exchange(order: Order, result: OrderResult):
    ex = MagicMock()
    ex.prepare_order = MagicMock(return_value=order)
    ex.place_order = AsyncMock(return_value=result)
    return ex


def _gateway(db, exchange):
    return ExecutionGateway(
        router=None, exchange=exchange, exchange_name="polymarket",
        settings=Settings(), db=db, pnl_tracker=PnLTracker(db, Settings()),
    )


def test_submit_records_fill_cost_basis_and_trade():
    async def run():
        db = Database(":memory:")
        await db.connect()
        await db.execute(
            "INSERT INTO markets (id, exchange, question, category, active, last_updated)"
            " VALUES ('m1', 'polymarket', 'Q?', 'tech', 1, datetime('now'))"
        )
        await db.commit()

        order = Order(market_id="m1", exchange="polymarket", token_id="tok",
                      side=OrderSide.BUY, token=TokenType.YES, size=20.0, price=0.50)
        result = OrderResult(order_id="p1", market_id="m1", status="paper",
                             filled_size=20.0, filled_price=0.50, is_paper=True)
        gw = _gateway(db, _paper_exchange(order, result))

        res = await gw.submit(TradeIntent(signal=_signal(), market=Market(id="m1", question="Q?"),
                                          size_dollars=10.0))

        # Outcome surfaced
        assert res.status == "paper"
        assert res.result is result
        assert res.fill is not None and res.fill.size == 20.0

        # Fill + cost basis recorded by the real PnLTracker
        fills = await db.fetchall("SELECT * FROM fills WHERE market_id='m1'")
        assert len(fills) == 1
        cb = await db.fetchall("SELECT token, size FROM cost_basis WHERE market_id='m1'")
        assert len(cb) == 1
        assert dict(cb[0])["token"] == "YES" and dict(cb[0])["size"] == 20.0

        # Mirrored to legacy trades with the strategy attribution
        trades = await db.fetchall("SELECT side, status, strategy_source FROM trades WHERE market_id='m1'")
        assert len(trades) == 1
        assert dict(trades[0])["status"] == "filled"  # 'paper' maps to 'filled' in trades
        assert dict(trades[0])["strategy_source"] == "llm"

    asyncio.run(run())


def test_submit_skips_on_build_failure_without_recording():
    async def run():
        db = Database(":memory:")
        await db.connect()
        ex = MagicMock()
        ex.prepare_order = MagicMock(return_value=None)  # build failure
        ex.place_order = AsyncMock()
        gw = ExecutionGateway(router=None, exchange=ex, exchange_name="polymarket",
                              settings=Settings(), db=db, pnl_tracker=PnLTracker(db, Settings()))

        res = await gw.submit(TradeIntent(signal=_signal(), market=Market(id="m1", question="Q?"),
                                          size_dollars=10.0))

        assert res.status == "skipped"
        assert res.result is None
        ex.place_order.assert_not_awaited()
        # Nothing recorded, and the market is blocked from immediate retry.
        assert await db.fetchall("SELECT * FROM fills") == []
        drops = await db.fetchall("SELECT market_id FROM order_build_drops WHERE market_id='m1'")
        assert len(drops) == 1

    asyncio.run(run())


def test_rejected_order_returns_result_not_none():
    """A rejected submission still returns the OrderResult (not skipped), so
    callers relying on the legacy 'None == not submitted' contract behave."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        order = Order(market_id="m1", exchange="polymarket", token_id="tok",
                      side=OrderSide.BUY, token=TokenType.YES, size=20.0, price=0.50)
        rejected = OrderResult(order_id="ERROR", market_id="m1", status="rejected")
        gw = _gateway(db, _paper_exchange(order, rejected))

        res = await gw.submit(TradeIntent(signal=_signal(), market=Market(id="m1", question="Q?"),
                                          size_dollars=10.0))
        assert res.status == "rejected"
        assert res.result is rejected           # not None
        assert res.fill is None                 # nothing filled
        assert await db.fetchall("SELECT * FROM fills") == []

    asyncio.run(run())


def test_gateway_lazy_rebuild_tracks_late_bound_collaborators():
    """The engine binds exchange_name / pnl after init; the lazy _gateway
    property must rebuild when they change."""
    async def run():
        from auramaur.strategy.engine import TradingEngine as TE
        # Exercise the real property against a stand-in carrying the attributes
        # it reads (avoids spinning up a full TradingEngine).
        holder = MagicMock()
        holder._exec_gateway = None
        holder.router = None
        holder.exchange = MagicMock()
        holder.settings = Settings()
        holder.db = MagicMock()
        holder.exchange_name = "polymarket"
        holder._components_pnl = None
        gw1 = TE._gateway.fget(holder)
        assert gw1.exchange_name == "polymarket"
        # Same collaborators -> same instance.
        assert TE._gateway.fget(holder) is gw1
        # Late-bound pnl changes -> rebuild.
        holder._components_pnl = PnLTracker(Database(":memory:"), Settings())
        gw2 = TE._gateway.fget(holder)
        assert gw2 is not gw1

    asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover
    pass
