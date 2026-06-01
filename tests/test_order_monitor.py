"""Tests for bot-level pending order monitoring."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from auramaur.bot import AuramaurBot
from auramaur.exchange.models import Order, OrderResult, OrderSide, TokenType


@pytest.mark.asyncio
async def test_order_monitor_records_live_fill_once():
    settings = MagicMock()
    settings.execution.limit_order_ttl_seconds = 60

    bot = AuramaurBot(settings=settings)
    bot._running = True

    order = Order(
        market_id="m1",
        exchange="polymarket",
        token_id="tok_yes",
        token=TokenType.YES,
        side=OrderSide.BUY,
        size=10,
        price=0.50,
        dry_run=False,
    )
    exchange = SimpleNamespace(
        _live_pending={"live-1": order},
        get_order_status=AsyncMock(
            return_value=OrderResult(
                order_id="live-1",
                market_id="m1",
                status="filled",
                filled_size=10,
                filled_price=0.52,
                is_paper=False,
            )
        ),
    )
    paper = SimpleNamespace(
        pending_orders=[],
        check_fills=AsyncMock(return_value=[]),
        cancel_expired=AsyncMock(return_value=0),
    )
    pnl_tracker = AsyncMock()
    db_cursor = MagicMock()
    db_cursor.rowcount = 1
    db = AsyncMock()
    db.execute = AsyncMock(return_value=db_cursor)
    db.commit = AsyncMock()

    bot._components = {
        "paper": paper,
        "exchange": exchange,
        "discovery": AsyncMock(),
        "pnl_tracker": pnl_tracker,
        "db": db,
    }

    async def stop_after_loop(_seconds):
        bot._running = False

    with patch("asyncio.sleep", new=AsyncMock(side_effect=stop_after_loop)):
        await bot._task_order_monitor()

    pnl_tracker.record_fill.assert_awaited_once()
    fill = pnl_tracker.record_fill.await_args.args[0]
    assert fill.order_id == "live-1"
    assert fill.size == 10
    assert fill.price == 0.52
    assert fill.is_paper is False
    assert "live-1" not in exchange._live_pending
    db.execute.assert_awaited()
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_order_monitor_ttl_cancels_stale_live_order():
    """A live limit order still resting past the TTL is cancelled to free balance.

    Live GTC orders never auto-expire on-chain, so without this they'd lock
    collateral forever. Mirrors the paper cancel_expired behaviour.
    """
    settings = MagicMock()
    settings.execution.limit_order_ttl_seconds = 60

    bot = AuramaurBot(settings=settings)
    bot._running = True

    stale_order = Order(
        market_id="m1",
        exchange="polymarket",
        token_id="tok_yes",
        token=TokenType.YES,
        side=OrderSide.BUY,
        size=10,
        price=0.50,
        dry_run=False,
    )
    # Placed 10 minutes ago -> older than the 60s TTL.
    stale_order.created_at = datetime.now(timezone.utc) - timedelta(minutes=10)

    exchange = SimpleNamespace(
        _live_pending={"live-1": stale_order},
        # Still open/unfilled -> non-terminal status, so the TTL branch runs.
        get_order_status=AsyncMock(return_value=OrderResult(
            order_id="live-1", market_id="m1", status="pending",
            filled_size=0, filled_price=0.50, is_paper=False,
        )),
        cancel_order=AsyncMock(return_value=True),
    )
    paper = SimpleNamespace(
        pending_orders=[],
        check_fills=AsyncMock(return_value=[]),
        cancel_expired=AsyncMock(return_value=0),
    )
    bot._components = {
        "paper": paper,
        "exchange": exchange,
        "discovery": AsyncMock(),
        "pnl_tracker": AsyncMock(),
        "db": AsyncMock(),
    }

    async def stop_after_loop(_seconds):
        bot._running = False

    with patch("asyncio.sleep", new=AsyncMock(side_effect=stop_after_loop)):
        await bot._task_order_monitor()

    exchange.cancel_order.assert_awaited_once_with("live-1")
    assert "live-1" not in exchange._live_pending


@pytest.mark.asyncio
async def test_order_monitor_keeps_fresh_live_order():
    """A live order younger than the TTL is left resting (not cancelled)."""
    settings = MagicMock()
    settings.execution.limit_order_ttl_seconds = 60

    bot = AuramaurBot(settings=settings)
    bot._running = True

    fresh_order = Order(
        market_id="m1", exchange="polymarket", token_id="tok_yes",
        token=TokenType.YES, side=OrderSide.BUY, size=10, price=0.50, dry_run=False,
    )
    fresh_order.created_at = datetime.now(timezone.utc)  # just placed

    exchange = SimpleNamespace(
        _live_pending={"live-1": fresh_order},
        get_order_status=AsyncMock(return_value=OrderResult(
            order_id="live-1", market_id="m1", status="pending",
            filled_size=0, filled_price=0.50, is_paper=False,
        )),
        cancel_order=AsyncMock(return_value=True),
    )
    paper = SimpleNamespace(
        pending_orders=[],
        check_fills=AsyncMock(return_value=[]),
        cancel_expired=AsyncMock(return_value=0),
    )
    bot._components = {
        "paper": paper, "exchange": exchange, "discovery": AsyncMock(),
        "pnl_tracker": AsyncMock(), "db": AsyncMock(),
    }

    async def stop_after_loop(_seconds):
        bot._running = False

    with patch("asyncio.sleep", new=AsyncMock(side_effect=stop_after_loop)):
        await bot._task_order_monitor()

    exchange.cancel_order.assert_not_called()
    assert "live-1" in exchange._live_pending


@pytest.mark.asyncio
async def test_order_monitor_polls_all_live_exchanges():
    settings = MagicMock()
    settings.execution.limit_order_ttl_seconds = 60

    bot = AuramaurBot(settings=settings)
    bot._running = True

    poly_order = Order(
        market_id="poly-m1",
        exchange="polymarket",
        token_id="poly_yes",
        token=TokenType.YES,
        side=OrderSide.BUY,
        size=5,
        price=0.40,
        dry_run=False,
    )
    kalshi_order = Order(
        market_id="KXTEST",
        exchange="kalshi",
        token_id="KXTEST",
        token=TokenType.YES,
        side=OrderSide.BUY,
        size=3,
        price=0.55,
        dry_run=False,
    )

    poly = SimpleNamespace(
        _live_pending={"poly-1": poly_order},
        get_order_status=AsyncMock(return_value=OrderResult(
            order_id="poly-1",
            market_id="poly-m1",
            status="filled",
            filled_size=5,
            filled_price=0.41,
            is_paper=False,
        )),
    )
    kalshi = SimpleNamespace(
        _live_pending={"kalshi-1": kalshi_order},
        get_order_status=AsyncMock(return_value=OrderResult(
            order_id="kalshi-1",
            market_id="KXTEST",
            status="filled",
            filled_size=3,
            filled_price=0.56,
            is_paper=False,
        )),
    )
    paper = SimpleNamespace(
        pending_orders=[],
        check_fills=AsyncMock(return_value=[]),
        cancel_expired=AsyncMock(return_value=0),
    )
    pnl_tracker = AsyncMock()
    db_cursor = MagicMock()
    db_cursor.rowcount = 1
    db = AsyncMock()
    db.execute = AsyncMock(return_value=db_cursor)
    db.commit = AsyncMock()

    bot._components = {
        "paper": paper,
        "exchange": poly,
        "exchanges": {"polymarket": poly, "kalshi": kalshi},
        "discovery": AsyncMock(),
        "pnl_tracker": pnl_tracker,
        "db": db,
    }

    async def stop_after_loop(_seconds):
        bot._running = False

    with patch("asyncio.sleep", new=AsyncMock(side_effect=stop_after_loop)):
        await bot._task_order_monitor()

    assert pnl_tracker.record_fill.await_count == 2
    assert poly._live_pending == {}
    assert kalshi._live_pending == {}
    poly.get_order_status.assert_awaited_once_with("poly-1")
    kalshi.get_order_status.assert_awaited_once_with("kalshi-1")
