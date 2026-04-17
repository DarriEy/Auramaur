"""post_only wiring: Order model, router defaults, and BBO-join on tight spread."""

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


def test_order_post_only_defaults_false():
    o = Order(market_id="m", side=OrderSide.BUY, size=10.0, price=0.5)
    assert o.post_only is False


def test_order_post_only_set_true():
    o = Order(market_id="m", side=OrderSide.BUY, size=10.0, price=0.5, post_only=True)
    assert o.post_only is True


def test_limit_price_improves_inside_wide_spread():
    book = _book(0.40, 0.45)  # 5-cent spread — improvement at 0.41 is safe
    price = SmartOrderRouter._compute_limit_price(book, OrderSide.BUY, TokenType.YES)
    assert price == 0.41


def test_limit_price_joins_bbo_on_one_cent_spread_buy():
    # 1-cent spread: best_bid + 0.01 == best_ask, would cross on post_only.
    # Should fall back to joining the bid queue at 0.50.
    book = _book(0.50, 0.51)
    price = SmartOrderRouter._compute_limit_price(book, OrderSide.BUY, TokenType.YES)
    assert price == 0.50


def test_limit_price_joins_bbo_on_one_cent_spread_sell():
    book = _book(0.50, 0.51)
    price = SmartOrderRouter._compute_limit_price(book, OrderSide.SELL, TokenType.YES)
    assert price == 0.51


@pytest.mark.asyncio
async def test_router_sets_post_only_on_limit_orders():
    """Router should set post_only=True on any LIMIT order it emits."""
    settings = MagicMock()
    exchange = MagicMock()
    exchange.prepare_order = MagicMock(
        return_value=Order(
            market_id="m1",
            exchange="polymarket",
            token_id="tok",
            side=OrderSide.BUY,
            token=TokenType.YES,
            size=10.0,
            price=0.45,
            dry_run=True,
        )
    )
    exchange.get_order_book = AsyncMock(return_value=_book(0.40, 0.50))

    router = SmartOrderRouter(settings, exchange)

    market = Market(id="m1", question="Q?", liquidity=5000)
    signal = Signal(
        market_id="m1",
        claude_prob=0.6,
        claude_confidence="HIGH",
        market_prob=0.45,
        edge=15.0,
    )

    order = await router.route(signal, market, size_dollars=10.0, is_live=False)
    assert order is not None
    assert order.order_type == OrderType.LIMIT
    assert order.post_only is True


@pytest.mark.asyncio
async def test_router_does_not_set_post_only_on_market_orders():
    """High-urgency edge -> MARKET, which should not carry post_only."""
    settings = MagicMock()
    exchange = MagicMock()
    exchange.prepare_order = MagicMock(
        return_value=Order(
            market_id="m2",
            exchange="polymarket",
            token_id="tok",
            side=OrderSide.BUY,
            token=TokenType.YES,
            size=10.0,
            price=0.50,
            dry_run=True,
        )
    )
    exchange.get_order_book = AsyncMock(return_value=_book(0.49, 0.51))

    router = SmartOrderRouter(settings, exchange)

    market = Market(id="m2", question="Q?", liquidity=5000)
    signal = Signal(
        market_id="m2",
        claude_prob=0.9,
        claude_confidence="HIGH",
        market_prob=0.50,
        edge=50.0,  # > 40 triggers MARKET
    )

    order = await router.route(signal, market, size_dollars=10.0, is_live=False)
    assert order is not None
    assert order.order_type == OrderType.MARKET
    assert order.post_only is False
