"""Router taker-or-skip: BUY entries execute at the ask or not at all.

Passive entries (bid+1 tick) filled 25-29% of the time before the TTL
reaper killed them, and a resting bid only fills when flow moves against
the thesis — the survivors are adversely selected. The realizable price
for an entry is the ask: when the edge measured there doesn't clear the
minimum-edge floor (or the ask is past the cents cap), the router raises
UnmarketableSignal instead of posting a coin-flip maker quote.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from auramaur.broker.router import SmartOrderRouter, UnmarketableSignal
from auramaur.exchange.models import (
    Market,
    Order,
    OrderBook,
    OrderBookLevel,
    OrderSide,
    OrderType,
    Signal,
    TokenType,
)


def _book(best_bid: float | None, best_ask: float | None) -> OrderBook:
    return OrderBook(
        bids=[OrderBookLevel(price=best_bid, size=500.0)] if best_bid else [],
        asks=[OrderBookLevel(price=best_ask, size=500.0)] if best_ask else [],
    )


def _settings(max_cross_cents: int = 4, min_edge_pct: float = 2.5) -> MagicMock:
    settings = MagicMock()
    settings.execution.entry_max_cross_cents = max_cross_cents
    settings.risk.min_edge_pct = min_edge_pct
    return settings


def _router(book: OrderBook, base_price: float, settings=None,
            side: OrderSide = OrderSide.BUY) -> SmartOrderRouter:
    exchange = MagicMock()
    exchange.prepare_order = MagicMock(
        return_value=Order(
            market_id="m1",
            exchange="polymarket",
            token_id="tok",
            side=side,
            token=TokenType.YES,
            size=10.0,
            price=base_price,
            dry_run=True,
        )
    )
    exchange.get_order_book = AsyncMock(return_value=book)
    return SmartOrderRouter(settings if settings is not None else _settings(), exchange)


def _signal(edge: float) -> Signal:
    return Signal(
        market_id="m1",
        claude_prob=0.6,
        claude_confidence="HIGH",
        market_prob=0.45,
        edge=edge,
    )


@pytest.mark.asyncio
async def test_crosses_to_ask_within_budget():
    """Cheap cross + plenty of edge -> lift the ask as a plain (taker) limit."""
    router = _router(_book(0.20, 0.24), base_price=0.22)
    order = await router.route(_signal(edge=10.0), Market(id="m1", question="Q?"), 10.0, False)
    assert order is not None
    assert order.order_type == OrderType.LIMIT
    assert order.price == 0.24
    assert order.post_only is False


@pytest.mark.asyncio
async def test_skips_when_edge_at_ask_below_floor():
    """Crossing would eat the edge below the floor -> skip, don't post a
    passive quote. Cross cost is 2 pts; edge 4.0 leaves 2.0 < 2.5 floor."""
    router = _router(_book(0.20, 0.24), base_price=0.22)
    with pytest.raises(UnmarketableSignal, match="edge at the ask"):
        await router.route(_signal(edge=4.0), Market(id="m1", question="Q?"), 10.0, False)


@pytest.mark.asyncio
async def test_skips_when_ask_past_cents_cap():
    """Ask further than entry_max_cross_cents above reference -> skip
    (8c chase vs a 4c cap), even though the proportional edge would survive."""
    router = _router(_book(0.20, 0.30), base_price=0.22)
    with pytest.raises(UnmarketableSignal, match="above"):
        await router.route(_signal(edge=20.0), Market(id="m1", question="Q?"), 10.0, False)


@pytest.mark.asyncio
async def test_skips_on_dead_book():
    """No asks at all (dead or one-sided market) -> skip. The old fallback
    posted a 'market' order at the stale reference, which on the CLOB is
    just another resting limit that dies at TTL."""
    router = _router(_book(0.20, None), base_price=0.22)
    with pytest.raises(UnmarketableSignal, match="no asks"):
        await router.route(_signal(edge=10.0), Market(id="m1", question="Q?"), 10.0, False)


@pytest.mark.asyncio
async def test_takes_price_improvement_when_ask_below_reference():
    """Ask below the reference price is free improvement -> take it; the
    negative cross cost ADDS to the realizable edge (4 + 2 = 6 >= 2.5)."""
    router = _router(_book(0.18, 0.20), base_price=0.22)
    order = await router.route(_signal(edge=4.0), Market(id="m1", question="Q?"), 10.0, False)
    assert order is not None
    assert order.order_type == OrderType.LIMIT
    assert order.price == 0.20
    assert order.post_only is False


@pytest.mark.asyncio
async def test_high_urgency_prices_at_real_ask():
    """edge > 40 used to submit at the stale reference price, which just
    rested on the book until the TTL reaper killed it. It must price at the
    actual ask to be marketable (cents cap waived above 40% edge)."""
    router = _router(_book(0.40, 0.55), base_price=0.40)
    order = await router.route(_signal(edge=56.8), Market(id="m1", question="Q?"), 10.0, False)
    assert order is not None
    assert order.order_type == OrderType.LIMIT
    assert order.price == 0.55
    assert order.post_only is False


@pytest.mark.asyncio
async def test_high_urgency_still_skips_below_floor():
    """Even urgent orders skip when the ask is past the price where edge
    hits the floor. The old behavior posted at the budget price (0.60),
    which rested below the 0.90 ask — unfillable."""
    router = _router(_book(0.10, 0.90), base_price=0.20)
    with pytest.raises(UnmarketableSignal, match="edge at the ask"):
        await router.route(_signal(edge=42.5), Market(id="m1", question="Q?"), 10.0, False)


@pytest.mark.asyncio
async def test_engine_records_unmarketable_skip():
    """The engine converts UnmarketableSignal into a recorded drop (30-min
    cooldown in order_build_drops) and places no order — for paper AND live,
    so paper books can't graduate on fills live could never get."""
    from auramaur.db.database import Database
    from auramaur.strategy.engine import TradingEngine

    db = Database(":memory:")
    await db.connect()
    try:
        settings = MagicMock()
        settings.is_live = False
        router = MagicMock()
        router.route = AsyncMock(
            side_effect=UnmarketableSignal("edge at the ask 1.80% below minimum 2.26%")
        )
        exchange = MagicMock()
        exchange.place_order = AsyncMock()

        engine = TradingEngine(
            settings=settings, db=db, discovery=MagicMock(),
            aggregator=MagicMock(), analyzer=MagicMock(), cache=MagicMock(),
            risk_manager=MagicMock(), exchange=exchange, router=router,
        )

        result = await engine._build_and_place_order(
            _signal(edge=4.0), Market(id="m1", question="Q?"), 5.0
        )

        assert result is None
        exchange.place_order.assert_not_awaited()
        row = await db.fetchone(
            "SELECT reason, blocked_until FROM order_build_drops WHERE market_id='m1'"
        )
        assert row is not None
        assert "unmarketable" in row["reason"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_sell_keeps_passive_path():
    """SELLs (exits) are not gated — they post one tick inside the book as
    maker orders. 'No fill' is not an acceptable outcome for an exit."""
    router = _router(_book(0.40, 0.50), base_price=0.45, side=OrderSide.SELL)
    order = await router.route(_signal(edge=10.0), Market(id="m1", question="Q?"), 10.0, False)
    assert order is not None
    assert order.order_type == OrderType.LIMIT
    assert order.price == 0.49  # best_ask - 1 tick
    assert order.post_only is True
