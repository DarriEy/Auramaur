"""Router cross-to-ask: marketable entries within the edge budget.

A maker quote at bid+1 tick only fills if flow comes to us before the TTL
reaper cancels the order — on quiet books most entries died unfilled (and
silently). When lifting the ask costs little relative to the signal's edge,
the router should pay for certainty instead.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from auramaur.broker.router import SmartOrderRouter
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


def _book(best_bid: float, best_ask: float) -> OrderBook:
    return OrderBook(
        bids=[OrderBookLevel(price=best_bid, size=500.0)],
        asks=[OrderBookLevel(price=best_ask, size=500.0)],
    )


def _settings(max_cross_cents: int = 4, min_edge_pct: float = 2.5) -> MagicMock:
    settings = MagicMock()
    settings.execution.entry_max_cross_cents = max_cross_cents
    settings.risk.min_edge_pct = min_edge_pct
    return settings


def _router(book: OrderBook, base_price: float, settings=None) -> SmartOrderRouter:
    exchange = MagicMock()
    exchange.prepare_order = MagicMock(
        return_value=Order(
            market_id="m1",
            exchange="polymarket",
            token_id="tok",
            side=OrderSide.BUY,
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
async def test_no_cross_when_edge_too_thin():
    """Crossing would eat the edge below the floor -> stay maker at bid+1."""
    router = _router(_book(0.20, 0.24), base_price=0.22)
    # Cross cost is 2 pts; edge 4.0 - min_edge 2.5 leaves only 1.5 pts budget.
    order = await router.route(_signal(edge=4.0), Market(id="m1", question="Q?"), 10.0, False)
    assert order is not None
    assert order.order_type == OrderType.LIMIT
    assert order.price == 0.21  # best_bid + 1 tick
    assert order.post_only is True


@pytest.mark.asyncio
async def test_no_cross_when_spread_too_wide():
    """Ask further than entry_max_cross_cents from our quote -> stay maker."""
    router = _router(_book(0.20, 0.30), base_price=0.22)
    order = await router.route(_signal(edge=20.0), Market(id="m1", question="Q?"), 10.0, False)
    assert order is not None
    assert order.order_type == OrderType.LIMIT
    assert order.price == 0.21
    assert order.post_only is True


@pytest.mark.asyncio
async def test_high_urgency_prices_at_real_ask():
    """edge > 40 used to submit at the stale reference price, which just
    rested on the book until the TTL reaper killed it. It must price at the
    actual ask to be marketable."""
    router = _router(_book(0.40, 0.55), base_price=0.40)
    order = await router.route(_signal(edge=56.8), Market(id="m1", question="Q?"), 10.0, False)
    assert order is not None
    assert order.order_type == OrderType.MARKET
    assert order.price == 0.55
    assert order.post_only is False


@pytest.mark.asyncio
async def test_high_urgency_capped_by_edge_budget():
    """Even urgent orders never pay past the price where edge hits the floor."""
    router = _router(_book(0.10, 0.90), base_price=0.20)
    order = await router.route(_signal(edge=42.5), Market(id="m1", question="Q?"), 10.0, False)
    assert order is not None
    assert order.order_type == OrderType.MARKET
    # budget = 0.20 + (42.5 - 2.5)/100 = 0.60, well below the 0.90 ask
    assert order.price == 0.60
