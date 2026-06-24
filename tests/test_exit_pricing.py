"""Marketable exit pricing: SELLs cross down to the real bid, or skip.

A SELL only fills by crossing down to the real bid. Posted at the snapshot
price — or anywhere inside the spread — it sits above the bid, rests until the
TTL reaper cancels it, and the cleared exit suppression re-posts it next pass:
a held winner looped that way unfilled for days against a wide-spread book. So
exits take the bid when it's within the slippage band and above the junk floor;
otherwise they skip, and returning False leaves the monitor's suppression set
so the retry backs off instead of looping.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from auramaur.components import Components
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


def _setup(book: OrderBook, current_price: float = 0.90, max_slip_cents: int = 10):
    settings = MagicMock()
    settings.is_live = False
    settings.execution.exit_max_cross_cents = max_slip_cents
    settings.execution.exit_min_bid_price = 0.05

    bot = AuramaurBot(settings=settings)
    db = AsyncMock()
    db.fetchone = AsyncMock(return_value={"token_id": "tok-1"})
    # pnl_tracker present-but-None preserves the prior `.get()` → None behavior
    # now that the exit reads it via the typed accessor.
    bot._components = Components({"db": db, "pnl_tracker": None})

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
async def test_exit_takes_bid_within_band():
    """Bid 2c below the snapshot, well within the band — cross to the bid."""
    bot, pos, reason, exchange, alerts = _setup(_book(0.88, 0.91), current_price=0.90)

    ok = await bot._execute_poly_exit(pos, reason, AsyncMock(), exchange, alerts)

    assert ok is True
    order = exchange.place_order.await_args.args[0]
    assert order.price == 0.88  # the real bid, marketable


@pytest.mark.asyncio
async def test_exit_crosses_to_bid_not_a_resting_floor():
    """A bid 5c under the snapshot, inside a 10c band, fills at the bid (0.85)
    — never at a 'budget floor' (0.87) that would only rest above the bid."""
    bot, pos, reason, exchange, alerts = _setup(_book(0.85, 0.95), current_price=0.90)

    ok = await bot._execute_poly_exit(pos, reason, AsyncMock(), exchange, alerts)

    assert ok is True
    order = exchange.place_order.await_args.args[0]
    assert order.price == 0.85  # crosses to the bid, not 0.87


@pytest.mark.asyncio
async def test_exit_skips_when_bid_below_band():
    """Bid 0.70 is 20c under the 0.90 mark — beyond the 10c band. Selling here
    would dump well under the mark, so skip and back off (the wide-spread case)."""
    bot, pos, reason, exchange, alerts = _setup(_book(0.70, 0.95), current_price=0.90)

    ok = await bot._execute_poly_exit(pos, reason, AsyncMock(), exchange, alerts)

    assert ok is False
    exchange.place_order.assert_not_awaited()


@pytest.mark.asyncio
async def test_exit_skips_junk_bid_below_floor():
    """A near-zero bid within the band is still junk: redeem at resolution
    rather than dump into a 2c buyer."""
    bot, pos, reason, exchange, alerts = _setup(_book(0.02, 0.20), current_price=0.12)

    ok = await bot._execute_poly_exit(pos, reason, AsyncMock(), exchange, alerts)

    assert ok is False
    exchange.place_order.assert_not_awaited()


@pytest.mark.asyncio
async def test_exit_skips_when_book_has_no_bid():
    """An empty bid side has nothing to cross into — skip, don't rest a SELL
    at the snapshot that can only TTL-cancel."""
    bot, pos, reason, exchange, alerts = _setup(_book(None, None), current_price=0.90)

    ok = await bot._execute_poly_exit(pos, reason, AsyncMock(), exchange, alerts)

    assert ok is False
    exchange.place_order.assert_not_awaited()


@pytest.mark.asyncio
async def test_exit_best_effort_when_book_fetch_errors():
    """A genuine book fetch error (network/API) is the transient path, not the
    wide-spread loop — fall back to posting at the snapshot price."""
    bot, pos, reason, exchange, alerts = _setup(_book(0.88, 0.91), current_price=0.90)
    exchange.get_order_book = AsyncMock(side_effect=RuntimeError("clob timeout"))

    ok = await bot._execute_poly_exit(pos, reason, AsyncMock(), exchange, alerts)

    assert ok is True
    order = exchange.place_order.await_args.args[0]
    assert order.price == 0.90


@pytest.mark.asyncio
async def test_exit_skips_when_mark_contradicts_book():
    """Snapshot $0.90 but the token's book asks $0.13: the mark is fiction
    (wrong-side label, stale price). Skip the order entirely."""
    bot, pos, reason, exchange, alerts = _setup(_book(0.07, 0.13), current_price=0.90)

    ok = await bot._execute_poly_exit(pos, reason, AsyncMock(), exchange, alerts)

    assert ok is False
    exchange.place_order.assert_not_awaited()
