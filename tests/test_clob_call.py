"""clob_call: blocking py_clob_client calls must not freeze the event loop.

The SDK is synchronous HTTP with no default timeout. Run bare on the loop,
one stalled socket froze the entire live bot for 84 minutes on 2026-06-10
(stalled get_market inside the reconciler) — no exits, no risk checks, no
kill-switch polling. clob_call pushes calls to a worker thread behind a
lock (the SDK is not thread-safe) and bounds them with a deadline.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from auramaur.exchange.client import PolymarketClient
from unittest.mock import MagicMock


def _client() -> PolymarketClient:
    client = PolymarketClient.__new__(PolymarketClient)
    client._clob_lock = asyncio.Lock()
    client._call_timeout = 0.2
    return client


@pytest.mark.asyncio
async def test_clob_call_returns_result():
    client = _client()
    assert await client.clob_call(lambda a, b: a + b, 2, 3) == 5


@pytest.mark.asyncio
async def test_clob_call_times_out_instead_of_hanging():
    client = _client()
    with pytest.raises(asyncio.TimeoutError):
        await client.clob_call(time.sleep, 1)


@pytest.mark.asyncio
async def test_event_loop_stays_responsive_during_blocking_call():
    """The whole point: a blocking SDK call must not stop the loop."""
    client = _client()
    client._call_timeout = 2.0
    ticks = 0

    async def ticker():
        nonlocal ticks
        for _ in range(5):
            await asyncio.sleep(0.02)
            ticks += 1

    blocking = asyncio.create_task(client.clob_call(time.sleep, 0.3))
    await ticker()
    await blocking
    # Run bare on the loop, time.sleep(0.3) would have frozen the ticker;
    # off-loop, all ticks land while the call is still blocking.
    assert ticks == 5


@pytest.mark.asyncio
async def test_clob_call_serializes_not_thread_safe_sdk():
    client = _client()
    client._call_timeout = 2.0
    concurrent = 0
    peak = 0

    def tracked():
        nonlocal concurrent, peak
        concurrent += 1
        peak = max(peak, concurrent)
        time.sleep(0.02)
        concurrent -= 1

    await asyncio.gather(*[client.clob_call(tracked) for _ in range(4)])
    assert peak == 1


@pytest.mark.asyncio
async def test_timeout_releases_lock_for_next_call():
    client = _client()
    with pytest.raises(asyncio.TimeoutError):
        await client.clob_call(time.sleep, 1)
    # The abandoned call must not leave the lock held.
    assert await asyncio.wait_for(client.clob_call(lambda: "ok"), 1.0) == "ok"


@pytest.mark.asyncio
async def test_get_order_book_survives_hanging_sdk():
    """A hung book fetch degrades to an empty book, not a frozen bot."""
    client = _client()
    client._init_clob_client = lambda: None
    client._clob_client = MagicMock()
    client._clob_client.get_order_book = lambda token_id: time.sleep(1)

    book = await client.get_order_book("tok-1")
    assert book.bids == [] and book.asks == []
