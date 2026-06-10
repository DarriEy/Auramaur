"""Marketable exit pricing: SELLs cross down to the real bid within budget.

A SELL posted at the position's snapshot price sits at/above the ask, rests
until the TTL reaper cancels it, and the cleared exit suppression re-posts
it next pass — the Obama winner looped like that for 2 days (35 cancelled
SELLs, no fill). Exits must take the bid when it's within the cross-down
budget, or at worst post at the budget floor inside the spread.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from auramaur.bot import AuramaurBot
from auramaur.exchange.models import (
    OrderBook,
    OrderBookLevel,
    OrderResult,
    TokenType,
)


def _book(best_bid: float | None, best_ask: float | None) -> OrderBook:
    return OrderBook(
        bids=[OrderBookLevel(price=best_bid, size=500.0)] if best_bid else [],
        asks=[OrderBookLevel(price=best_ask, size=500.0)] if best_ask else [],
    )


def _setup(book: OrderBook, current_price: float = 0.90):
    settings = MagicMock()
    settings.is_live = False
    settings.execution.exit_max_cross_cents = 3

    bot = AuramaurBot(settings=settings)
    db = AsyncMock()
    db.fetchone = AsyncMock(return_value={"token_id": "tok-1"})
    bot._components = {"db": db}

    exchange = SimpleNamespace(
        get_order_book=AsyncMock(return_value=book),
        place_order=AsyncMock(return_value=OrderResult(
            order_id="x1", market_id="m1", status="pending", is_paper=False,
        )),
    )
    pos = SimpleNamespace(
        market_id="m1",
        size=100.0,
        current_price=current_price,
        token=TokenType.YES,
        unrealized_pnl=76.0,
    )
    reason = SimpleNamespace(value="profit_target")
    alerts = SimpleNamespace(send=AsyncMock())
    return bot, pos, reason, exchange, alerts


@pytest.mark.asyncio
async def test_exit_takes_bid_within_budget():
    bot, pos, reason, exchange, alerts = _setup(_book(0.88, 0.91), current_price=0.90)

    ok = await bot._execute_poly_exit(pos, reason, AsyncMock(), exchange, alerts)

    assert ok is True
    order = exchange.place_order.await_args.args[0]
    assert order.price == 0.88  # bid is 2 cents below snapshot, within the 3-cent budget


@pytest.mark.asyncio
async def test_exit_caps_cross_at_budget_floor():
    bot, pos, reason, exchange, alerts = _setup(_book(0.80, 0.91), current_price=0.90)

    ok = await bot._execute_poly_exit(pos, reason, AsyncMock(), exchange, alerts)

    assert ok is True
    order = exchange.place_order.await_args.args[0]
    assert order.price == 0.87  # bid too deep — post at snapshot minus 3-cent budget


@pytest.mark.asyncio
async def test_exit_keeps_snapshot_price_without_book():
    bot, pos, reason, exchange, alerts = _setup(_book(None, None), current_price=0.90)

    ok = await bot._execute_poly_exit(pos, reason, AsyncMock(), exchange, alerts)

    assert ok is True
    order = exchange.place_order.await_args.args[0]
    assert order.price == 0.90
