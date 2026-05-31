"""Tests for bot-level pending order monitoring."""

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
