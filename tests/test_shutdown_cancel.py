"""Graceful shutdown cancels resting live orders.

GTC orders outlive the process: every restart inherited the prior session's
market-maker quotes and in-flight entries, locking collateral until the
startup reconciler + TTL reaper mopped them up minutes later.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from auramaur.bot import AuramaurBot


def _bot_with(components: dict) -> AuramaurBot:
    bot = AuramaurBot(settings=MagicMock())
    bot._components = components
    bot._lock_file = None
    return bot


@pytest.mark.asyncio
async def test_shutdown_cancels_all_resting_live_orders():
    poly = SimpleNamespace(
        _live_pending={"mm-bid-1": object(), "mm-ask-1": object()},
        cancel_order=AsyncMock(return_value=True),
    )
    kalshi = SimpleNamespace(
        _live_pending={"kx-1": object()},
        cancel_order=AsyncMock(return_value=True),
    )
    db = AsyncMock()
    bot = _bot_with({
        "exchange": poly,
        "exchanges": {"polymarket": poly, "kalshi": kalshi},
        "db": db,
    })

    await bot.shutdown()

    assert poly.cancel_order.await_count == 2
    kalshi.cancel_order.assert_awaited_once_with("kx-1")
    update_calls = [
        c for c in db.execute.await_args_list
        if c.args[0].startswith("UPDATE trades SET status = 'cancelled'")
    ]
    assert {c.args[1][0] for c in update_calls} == {"mm-bid-1", "mm-ask-1", "kx-1"}
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_shutdown_survives_cancel_failures():
    """A cancel that errors or is refused must not break shutdown or write a
    false 'cancelled' status — the next session's reconciler inherits it."""
    poly = SimpleNamespace(
        _live_pending={"bad-1": object(), "refused-1": object()},
        cancel_order=AsyncMock(side_effect=[Exception("api down"), False]),
    )
    db = AsyncMock()
    bot = _bot_with({"exchange": poly, "db": db})

    await bot.shutdown()  # must not raise

    update_calls = [
        c for c in db.execute.await_args_list
        if c.args[0].startswith("UPDATE trades SET status = 'cancelled'")
    ]
    assert update_calls == []


@pytest.mark.asyncio
async def test_shutdown_noop_without_live_orders():
    poly = SimpleNamespace(_live_pending={}, cancel_order=AsyncMock())
    bot = _bot_with({"exchange": poly, "db": AsyncMock()})

    await bot.shutdown()

    poly.cancel_order.assert_not_awaited()
