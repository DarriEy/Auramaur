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
            side: OrderSide = OrderSide.BUY,
            token: TokenType = TokenType.YES) -> SmartOrderRouter:
    exchange = MagicMock()
    exchange.prepare_order = MagicMock(
        return_value=Order(
            market_id="m1",
            exchange="polymarket",
            token_id="tok",
            side=side,
            token=token,
            size=10.0,
            price=base_price,
            dry_run=True,
        )
    )
    exchange.get_order_book = AsyncMock(return_value=book)
    return SmartOrderRouter(settings if settings is not None else _settings(), exchange)


def _signal(edge: float, claude_prob: float = 0.6, market_prob: float = 0.45) -> Signal:
    return Signal(
        market_id="m1",
        claude_prob=claude_prob,
        claude_confidence="HIGH",
        market_prob=market_prob,
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


# ---------------------------------------------------------------------------
# Depth-aware routing (phase 1): trim to in-budget capacity, sweep pricing
# ---------------------------------------------------------------------------

def _depth_settings(*, max_cross_cents=4, min_edge_pct=2.5,
                    cap_frac=0.5, min_fill=0.5) -> MagicMock:
    s = MagicMock()
    s.execution.entry_max_cross_cents = max_cross_cents
    s.execution.depth_aware_routing = True
    s.execution.book_capacity_fraction = cap_frac
    s.execution.min_fill_fraction = min_fill
    s.risk.min_edge_pct = min_edge_pct
    return s


def _depth_router(book: OrderBook, base_price: float, order_size: float,
                  settings: MagicMock) -> SmartOrderRouter:
    exchange = MagicMock()
    exchange.prepare_order = MagicMock(return_value=Order(
        market_id="m1", exchange="polymarket", token_id="tok",
        side=OrderSide.BUY, token=TokenType.YES, size=order_size,
        price=base_price, dry_run=True))
    exchange.get_order_book = AsyncMock(return_value=book)
    return SmartOrderRouter(settings, exchange)


@pytest.mark.asyncio
async def test_depth_trims_size_to_capacity():
    """Order bigger than the in-budget depth allows -> trim to capacity_fraction
    of it, price at the (single-level) sweep price."""
    book = OrderBook(bids=[OrderBookLevel(price=0.48, size=500.0)],
                     asks=[OrderBookLevel(price=0.50, size=40.0)])
    router = _depth_router(book, base_price=0.49, order_size=30.0,
                           settings=_depth_settings(cap_frac=0.5, min_fill=0.5))
    order = await router.route(_signal(edge=10.0), Market(id="m1", question="Q?"), 30.0, False)
    assert order is not None
    assert order.size == 20.0          # depth 40 * 0.5 capacity
    assert order.price == 0.50         # fits the 0.50 level -> ask


@pytest.mark.asyncio
async def test_depth_skips_when_below_min_fill():
    """In-budget capacity below min_fill_fraction of the requested size -> skip."""
    book = OrderBook(bids=[OrderBookLevel(price=0.48, size=500.0)],
                     asks=[OrderBookLevel(price=0.50, size=40.0)])
    router = _depth_router(book, base_price=0.49, order_size=50.0,
                           settings=_depth_settings(cap_frac=0.5, min_fill=0.5))
    # capacity = 40*0.5 = 20 < min_fill(0.5)*50 = 25 -> skip
    with pytest.raises(UnmarketableSignal, match="absorbs only"):
        await router.route(_signal(edge=10.0), Market(id="m1", question="Q?"), 50.0, False)


@pytest.mark.asyncio
async def test_depth_sweep_prices_at_marginal_level():
    """A size needing two levels prices the limit at the deeper (marginal)
    level, not the best ask -- so it can actually sweep to fill."""
    book = OrderBook(bids=[OrderBookLevel(price=0.48, size=500.0)],
                     asks=[OrderBookLevel(price=0.50, size=10.0),
                           OrderBookLevel(price=0.55, size=50.0)])
    router = _depth_router(book, base_price=0.49, order_size=40.0,
                           settings=_depth_settings(max_cross_cents=10, cap_frac=1.0, min_fill=0.5))
    order = await router.route(_signal(edge=12.0), Market(id="m1", question="Q?"), 40.0, False)
    assert order is not None
    assert order.price == 0.55         # marginal sweep level, not the 0.50 ask
    # Size is re-derived from the dollar intent (40 tokens * 0.49 reference)
    # at each price lift: 19.6 / 0.50 = 39.2 at the ask, then 19.6 / 0.55 =
    # 35.64 at the sweep price — the notional stays at the intent instead of
    # inflating with the limit.
    assert order.size == 35.64


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
    assert order.size == 10.0  # cheaper fill keeps the token count (shrink-only)


@pytest.mark.asyncio
async def test_high_urgency_prices_at_real_ask():
    """edge > 40 used to submit at the stale reference price, which just
    rested on the book until the TTL reaper killed it. It must price at the
    actual ask to be marketable (cents cap waived above 40 points of edge,
    with fair value clearing the ask). The signal is coherent: fair 0.95 vs
    reference 0.40 implies the 55-point edge it claims."""
    router = _router(_book(0.40, 0.55), base_price=0.40)
    order = await router.route(
        _signal(edge=55.0, claude_prob=0.95), Market(id="m1", question="Q?"), 10.0, False)
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
async def test_relative_edge_clamped_to_fair_implied_gap():
    """A producer that emits a RELATIVE edge (a few points of gap on a cheap
    market reads as a huge percentage) used to both clear the floor and waive
    the cents cap, letting the router pay a dead-book ask at a multiple of
    fair value. The edge basis is now clamped to the gap the signal's own
    fair value implies (fair 0.13 vs reference 0.08 = 5 points, not 47), so
    crossing to a 0.42 ask fails the min-edge floor outright."""
    router = _router(_book(0.05, 0.42), base_price=0.08)
    with pytest.raises(UnmarketableSignal, match="edge at the ask"):
        await router.route(
            _signal(edge=47.0, claude_prob=0.13, market_prob=0.08),
            Market(id="m1", question="Q?"), 10.0, False,
        )


@pytest.mark.asyncio
async def test_waiver_holds_when_fair_clears_ask():
    """A genuine high-conviction entry still waives the cents cap: fair 0.90
    clears the 0.50 ask by far more than the floor. The size is re-derived
    at the crossed price (10 tokens * 0.40 ref = $4 intent -> 8 at 0.50)."""
    router = _router(_book(0.40, 0.50), base_price=0.40)
    order = await router.route(
        _signal(edge=55.0, claude_prob=0.90), Market(id="m1", question="Q?"), 10.0, False)
    assert order is not None
    assert order.price == 0.50
    assert order.size == 8.0


@pytest.mark.asyncio
async def test_waiver_uses_no_token_fair_value():
    """When the bought token is NO, the waiver measures fair = 1 - claude_prob
    against the NO ask."""
    router = _router(_book(0.40, 0.50), base_price=0.40, token=TokenType.NO)
    order = await router.route(
        _signal(edge=55.0, claude_prob=0.10), Market(id="m1", question="Q?"), 10.0, False)
    assert order is not None
    assert order.price == 0.50
    assert order.size == 8.0


@pytest.mark.asyncio
async def test_cross_rederives_size_from_dollar_intent():
    """Crossing up re-derives the token count from the dollar intent at the
    limit price: 10 tokens sized at 0.22 ($2.20) become 9.17 at the 0.24 ask
    instead of 10 tokens costing $2.40."""
    router = _router(_book(0.20, 0.24), base_price=0.22)
    order = await router.route(_signal(edge=10.0), Market(id="m1", question="Q?"), 10.0, False)
    assert order is not None
    assert order.price == 0.24
    assert order.size == 9.17


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
