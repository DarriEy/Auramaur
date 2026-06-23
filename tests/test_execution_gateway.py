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


def _leg(market_id, *, order_side=OrderSide.BUY, place: OrderResult, cancel=False):
    """A per-leg exchange: prepare_order -> a built order, place_order -> place."""
    order = Order(market_id=market_id, exchange="polymarket", token_id="tok",
                  side=order_side, token=TokenType.YES, size=20.0, price=0.40)
    ex = MagicMock()
    ex.prepare_order = MagicMock(return_value=order)
    ex.place_order = AsyncMock(return_value=place)
    if cancel:
        ex.cancel_order = AsyncMock()
    return ex


def _paired_gateway(db):
    return ExecutionGateway(
        router=None, exchange=MagicMock(), exchange_name="polymarket",
        settings=Settings(), db=db, pnl_tracker=PnLTracker(db, Settings()),
    )


def _intent(market_id):
    return TradeIntent(signal=_signal(market_id), market=Market(id=market_id, question="Q?"),
                       size_dollars=10.0)


def test_submit_paired_records_both_legs():
    async def run():
        db = Database(":memory:")
        await db.connect()
        ex_a = _leg("a", place=OrderResult(order_id="pa", market_id="a", status="paper",
                                           filled_size=20.0, filled_price=0.40, is_paper=True))
        ex_b = _leg("b", place=OrderResult(order_id="pb", market_id="b", status="paper",
                                           filled_size=20.0, filled_price=0.40, is_paper=True))
        gw = _paired_gateway(db)
        res_a, res_b = await gw.submit_paired(
            _intent("a"), _intent("b"),
            exchange_a=ex_a, exchange_name_a="polymarket",
            exchange_b=ex_b, exchange_name_b="kalshi")
        assert res_a.status == "paper" and res_b.status == "paper"
        fills = await db.fetchall("SELECT market_id FROM fills")
        assert {dict(f)["market_id"] for f in fills} == {"a", "b"}

    asyncio.run(run())


def test_submit_paired_leg_a_rejected_never_places_b():
    async def run():
        db = Database(":memory:")
        await db.connect()
        ex_a = _leg("a", place=OrderResult(order_id="ERROR", market_id="a", status="rejected"))
        ex_b = _leg("b", place=OrderResult(order_id="pb", market_id="b", status="paper",
                                           filled_size=20.0, filled_price=0.40, is_paper=True))
        gw = _paired_gateway(db)
        res_a, res_b = await gw.submit_paired(
            _intent("a"), _intent("b"),
            exchange_a=ex_a, exchange_name_a="polymarket",
            exchange_b=ex_b, exchange_name_b="polymarket")
        assert res_a.status == "rejected"
        assert res_b.status == "skipped"
        ex_b.place_order.assert_not_awaited()  # both-or-nothing: B never attempted
        assert await db.fetchall("SELECT * FROM fills") == []

    asyncio.run(run())


def test_submit_paired_leg_b_fail_unwinds_pending_a():
    async def run():
        db = Database(":memory:")
        await db.connect()
        # A is a LIVE pending order; B rejects -> A must be cancelled.
        ex_a = _leg("a", place=OrderResult(order_id="live-a", market_id="a", status="pending",
                                           filled_size=0.0, is_paper=False), cancel=True)
        ex_b = _leg("b", place=OrderResult(order_id="ERROR", market_id="b", status="rejected"))
        gw = _paired_gateway(db)
        res_a, res_b = await gw.submit_paired(
            _intent("a"), _intent("b"),
            exchange_a=ex_a, exchange_name_a="polymarket",
            exchange_b=ex_b, exchange_name_b="polymarket")
        assert res_a.status == "pending" and res_b.status == "rejected"
        ex_a.cancel_order.assert_awaited_once_with("live-a")

    asyncio.run(run())


def test_record_external_fill_paper_records_fill_and_trade():
    """The arb scanner places legs concurrently, then records each via
    record_external_fill — a paper fill must land in fills + trades, attributed
    to the arb strategy (not 'order_monitor')."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        order = Order(market_id="arb1", exchange="polymarket", token_id="tok",
                      side=OrderSide.BUY, token=TokenType.YES, size=20.0, price=0.40)
        result = OrderResult(order_id="ax", market_id="arb1", status="paper",
                             filled_size=20.0, filled_price=0.40, is_paper=True)
        gw = _gateway(db, MagicMock())  # exchange unused — order already placed
        res = await gw.record_external_fill(
            order, result, strategy_source="cross_exchange_arb", exchange_name="polymarket")
        assert res.status == "paper" and res.fill is not None
        assert len(await db.fetchall("SELECT 1 FROM fills WHERE market_id='arb1'")) == 1
        t = await db.fetchall("SELECT strategy_source FROM trades WHERE market_id='arb1'")
        assert len(t) == 1 and dict(t[0])["strategy_source"] == "cross_exchange_arb"

    asyncio.run(run())


def test_record_external_fill_live_pending_defers_to_monitor():
    async def run():
        db = Database(":memory:")
        await db.connect()
        order = Order(market_id="arb2", exchange="polymarket", token_id="tok",
                      side=OrderSide.BUY, token=TokenType.YES, size=20.0, price=0.40)
        result = OrderResult(order_id="live-ax", market_id="arb2", status="pending",
                             filled_size=0.0, is_paper=False)
        gw = _gateway(db, MagicMock())
        res = await gw.record_external_fill(
            order, result, strategy_source="cross_exchange_arb", exchange_name="polymarket")
        assert res.status == "pending"
        assert await db.fetchall("SELECT 1 FROM fills") == []  # monitor records it
        t = await db.fetchall("SELECT status FROM trades WHERE order_id='live-ax'")
        assert len(t) == 1 and dict(t[0])["status"] == "pending"

    asyncio.run(run())


def test_submit_exit_paper_records_fill_and_trade():
    async def run():
        db = Database(":memory:")
        await db.connect()
        sell = Order(market_id="x", exchange="polymarket", token_id="tok",
                     side=OrderSide.SELL, token=TokenType.YES, size=20.0, price=0.60)
        place = OrderResult(order_id="px", market_id="x", status="paper",
                            filled_size=20.0, filled_price=0.60, is_paper=True)
        ex = MagicMock(); ex.place_order = AsyncMock(return_value=place)
        gw = ExecutionGateway(router=None, exchange=MagicMock(), exchange_name="polymarket",
                              settings=Settings(), db=db, pnl_tracker=PnLTracker(db, Settings()))
        res = await gw.submit_exit(sell, exchange=ex, exchange_name="polymarket")
        assert res.status == "paper"
        assert len(await db.fetchall("SELECT 1 FROM fills WHERE market_id='x'")) == 1
        t = await db.fetchall("SELECT strategy_source, side FROM trades WHERE market_id='x'")
        assert len(t) == 1
        assert dict(t[0])["strategy_source"] == "exit" and dict(t[0])["side"] == "SELL"

    asyncio.run(run())


def test_submit_exit_live_pending_writes_trades_but_defers_fill_to_monitor():
    """A live exit is pending at placement — submit_exit writes the pending
    trades row (so the monitor UPDATEs it, not INSERTs 'order_monitor') but does
    NOT record the fill; the monitor records it once when it fills."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        sell = Order(market_id="x", exchange="polymarket", token_id="tok",
                     side=OrderSide.SELL, token=TokenType.YES, size=20.0, price=0.60)
        place = OrderResult(order_id="live-x", market_id="x", status="pending",
                            filled_size=0.0, is_paper=False)
        ex = MagicMock(); ex.place_order = AsyncMock(return_value=place)
        gw = ExecutionGateway(router=None, exchange=MagicMock(), exchange_name="polymarket",
                              settings=Settings(), db=db, pnl_tracker=PnLTracker(db, Settings()))
        res = await gw.submit_exit(sell, exchange=ex, exchange_name="polymarket")
        assert res.status == "pending"
        # No fill recorded yet (the monitor will, once).
        assert await db.fetchall("SELECT 1 FROM fills") == []
        # A pending trades row exists keyed by order_id, so the monitor's
        # UPDATE ... WHERE order_id finds it (no 'order_monitor' duplicate).
        t = await db.fetchall("SELECT status, order_id, strategy_source FROM trades WHERE order_id='live-x'")
        assert len(t) == 1
        assert dict(t[0])["status"] == "pending" and dict(t[0])["strategy_source"] == "exit"

    asyncio.run(run())


def test_submit_paired_build_failure_places_neither():
    async def run():
        db = Database(":memory:")
        await db.connect()
        ex_a = _leg("a", place=OrderResult(order_id="pa", market_id="a", status="paper",
                                           filled_size=20.0, filled_price=0.40, is_paper=True))
        ex_a.prepare_order = MagicMock(return_value=None)  # A can't be built
        ex_b = _leg("b", place=OrderResult(order_id="pb", market_id="b", status="paper",
                                           filled_size=20.0, filled_price=0.40, is_paper=True))
        gw = _paired_gateway(db)
        res_a, res_b = await gw.submit_paired(
            _intent("a"), _intent("b"),
            exchange_a=ex_a, exchange_name_a="polymarket",
            exchange_b=ex_b, exchange_name_b="polymarket")
        assert res_a.status == "skipped" and res_b.status == "skipped"
        ex_a.place_order.assert_not_awaited()
        ex_b.place_order.assert_not_awaited()  # neither leg placed

    asyncio.run(run())
