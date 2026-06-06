"""Regression tests for Kalshi live-order tracking.

Kalshi orders were inserted at placement with status 'pending' and never
reconciled, because KalshiClient didn't expose the `_live_pending` interface
the order monitor keys on — so the monitor skipped Kalshi entirely and the
rows stayed 'pending' forever (48 such rows had accumulated).

These tests lock in:
  1. KalshiClient exposes `_live_pending` (so the monitor no longer skips it).
  2. reconcile_open_orders rehydrates resting orders across restarts.
  3. reconcile_pending_kalshi_orders cleans up the stale historical rows.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from auramaur.broker.order_reconcile import reconcile_pending_kalshi_orders
from auramaur.db.database import Database
from auramaur.exchange.kalshi import KalshiClient
from auramaur.exchange.models import OrderResult, OrderSide, TokenType


def _client(is_live: bool = True) -> KalshiClient:
    settings = MagicMock()
    settings.is_live = is_live
    return KalshiClient(settings=settings, paper_trader=MagicMock())


def test_client_exposes_live_pending():
    """The monitor's `hasattr(client, "_live_pending")` guard must pass."""
    client = _client()
    assert hasattr(client, "_live_pending")
    assert client._live_pending == {}


@pytest.mark.asyncio
async def test_reconcile_open_orders_rehydrates_resting():
    client = _client(is_live=True)
    client._init_api = MagicMock()
    client._portfolio_api = MagicMock()  # normally set by _init_api
    client._call_raw = AsyncMock(return_value=json.dumps({
        "orders": [
            {"order_id": "o1", "status": "resting", "ticker": "KXFOO-1",
             "action": "buy", "side": "yes", "yes_price": 60,
             "remaining_count": 5, "created_time": "2026-06-01T00:00:00Z"},
            {"order_id": "o2", "status": "resting", "ticker": "KXBAR-1",
             "action": "buy", "side": "no", "yes_price": 30,
             "remaining_count": 3, "created_time": "2026-06-01T00:00:00Z"},
            # Non-resting orders must be ignored.
            {"order_id": "o3", "status": "executed", "ticker": "KXBAZ-1",
             "action": "buy", "side": "yes", "yes_price": 50},
        ]
    }))

    count = await client.reconcile_open_orders()
    assert count == 2
    assert set(client._live_pending) == {"o1", "o2"}

    o1 = client._live_pending["o1"]
    assert o1.token == TokenType.YES and o1.side == OrderSide.BUY
    assert abs(o1.price - 0.60) < 1e-9 and o1.size == 5

    # NO order: its own price is the complement of the yes_price.
    o2 = client._live_pending["o2"]
    assert o2.token == TokenType.NO
    assert abs(o2.price - 0.70) < 1e-9


@pytest.mark.asyncio
async def test_reconcile_open_orders_noop_when_not_live():
    client = _client(is_live=False)
    client._call_raw = AsyncMock()  # should never be called
    assert await client.reconcile_open_orders() == 0
    client._call_raw.assert_not_called()


def _make_exchange(status_by_oid: dict[str, str]):
    """Mock exchange whose get_order_status returns mapped statuses.

    A missing oid raises (simulating an order the venue no longer knows).
    """
    exch = MagicMock()

    async def _status(oid):
        if oid not in status_by_oid:
            raise RuntimeError("order not found")
        return OrderResult(
            order_id=oid, market_id="m", status=status_by_oid[oid],
            filled_size=10 if status_by_oid[oid] == "filled" else 0,
            filled_price=0.5 if status_by_oid[oid] == "filled" else 0,
            is_paper=False,
        )

    exch.get_order_status = AsyncMock(side_effect=_status)
    return exch


def test_reconcile_pending_orders_updates_terminal_only():
    async def run():
        db = Database(":memory:")
        await db.connect()
        rows = [
            ("KX1", "o-filled"),
            ("KX2", "o-cancelled"),
            ("KX3", "o-resting"),    # still open → get_order_status maps to 'pending'
            ("KX4", "o-gone"),       # venue 404 → expired
        ]
        for mid, oid in rows:
            await db.execute(
                "INSERT INTO trades (market_id, side, size, price, is_paper, "
                "order_id, status, exchange, strategy_source) "
                "VALUES (?, 'BUY', 1, 0.4, 0, ?, 'pending', 'kalshi', 'llm')",
                (mid, oid),
            )
        await db.commit()

        # A resting Kalshi order surfaces via get_order_status as 'pending'
        # (the client maps resting -> pending); 'o-gone' is absent -> raises -> expired.
        exch = _make_exchange({
            "o-filled": "filled", "o-cancelled": "cancelled", "o-resting": "pending",
        })

        # Dry-run writes nothing.
        res = await reconcile_pending_kalshi_orders(db, exch, dry_run=True)
        assert res.scanned == 4
        assert res.updated == 3  # filled, cancelled, expired
        assert res.still_pending == 1
        n_pending = (await db.fetchone(
            "SELECT COUNT(*) c FROM trades WHERE status='pending'"))["c"]
        assert n_pending == 4  # nothing written yet

        # Write mode persists terminal statuses.
        res = await reconcile_pending_kalshi_orders(db, exch, dry_run=False)
        assert res.by_status == {"filled": 1, "cancelled": 1, "expired": 1}
        statuses = {
            r["order_id"]: r["status"]
            for r in await db.fetchall("SELECT order_id, status FROM trades")
        }
        assert statuses["o-filled"] == "filled"
        assert statuses["o-cancelled"] == "cancelled"
        assert statuses["o-gone"] == "expired"
        assert statuses["o-resting"] == "pending"
        # Filled row took on the venue's filled size/price.
        filled = await db.fetchone(
            "SELECT size, price FROM trades WHERE order_id='o-filled'")
        assert filled["size"] == 10 and abs(filled["price"] - 0.5) < 1e-9
        await db.close()

    asyncio.run(run())


if __name__ == "__main__":
    test_client_exposes_live_pending()
    asyncio.run(test_reconcile_open_orders_rehydrates_resting())
    asyncio.run(test_reconcile_open_orders_noop_when_not_live())
    test_reconcile_pending_orders_updates_terminal_only()
    print("ok")
