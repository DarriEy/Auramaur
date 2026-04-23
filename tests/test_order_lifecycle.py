"""Tests for live order lifecycle management."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Stub py_clob_client so tests can exercise the live-order path without the
# optional C-extension package installed.
_clob_stub = MagicMock()
_clob_stub.order_builder.constants.BUY = "BUY"
_clob_stub.order_builder.constants.SELL = "SELL"
for mod_name in (
    "py_clob_client",
    "py_clob_client.client",
    "py_clob_client.clob_types",
    "py_clob_client.order_builder",
    "py_clob_client.order_builder.constants",
):
    sys.modules.setdefault(mod_name, _clob_stub)

from auramaur.exchange.client import PolymarketClient
from auramaur.exchange.models import Order, OrderResult, OrderSide
from auramaur.exchange.paper import PaperTrader


@pytest.fixture
def mock_settings():
    s = MagicMock()
    s.is_live = True
    s.auramaur_live = True
    s.execution.live = True
    s.polygon_private_key = "0xfake"
    s.polymarket_proxy_address = "0xproxy"
    s.polymarket_api_key = "key"
    s.polymarket_api_secret = "secret"
    s.polymarket_passphrase = "pass"
    return s


@pytest.fixture
def mock_paper():
    db = AsyncMock()
    db.fetchall = AsyncMock(return_value=[])
    db.fetchone = AsyncMock(return_value={"net": 0})
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    return PaperTrader(db=db, initial_balance=1000.0)


@pytest.fixture
def client(mock_settings, mock_paper):
    return PolymarketClient(settings=mock_settings, paper_trader=mock_paper)


@pytest.mark.asyncio
async def test_get_order_status_filled(client):
    """Mock CLOB returns filled status."""
    mock_clob = MagicMock()
    mock_clob.get_order.return_value = {
        "status": "matched",
        "size_matched": 10.0,
        "price": 0.55,
        "market": "m1",
    }
    client._clob_client = mock_clob

    result = await client.get_order_status("order123")
    assert result.status == "filled"
    assert result.filled_size == 10.0
    assert result.is_paper is False


@pytest.mark.asyncio
async def test_cancel_order_success(client):
    """Mock cancel succeeds."""
    mock_clob = MagicMock()
    mock_clob.cancel.return_value = None
    client._clob_client = mock_clob
    client._live_pending["order123"] = Order(
        market_id="m1", side=OrderSide.BUY, size=10, price=0.5
    )

    success = await client.cancel_order("order123")
    assert success is True
    assert "order123" not in client._live_pending


@pytest.mark.asyncio
async def test_poll_until_terminal_fills(client):
    """Pending then filled on second poll."""
    call_count = 0

    async def mock_get_status(order_id):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return OrderResult(
                order_id=order_id, market_id="m1", status="pending", is_paper=False
            )
        return OrderResult(
            order_id=order_id, market_id="m1", status="filled",
            filled_size=10.0, filled_price=0.55, is_paper=False
        )

    client.get_order_status = mock_get_status
    client._live_pending["order123"] = Order(
        market_id="m1", side=OrderSide.BUY, size=10, price=0.5
    )

    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await client.poll_until_terminal("order123", timeout=60, poll_interval=1)
    assert result.status == "filled"
    assert "order123" not in client._live_pending


@pytest.mark.asyncio
async def test_poll_timeout_cancels(client):
    """Always pending → cancel called on timeout."""
    async def mock_get_status(order_id):
        return OrderResult(
            order_id=order_id, market_id="m1", status="pending", is_paper=False
        )

    cancel_called = False
    original_order = Order(market_id="m1", side=OrderSide.BUY, size=10, price=0.5)

    async def mock_cancel(order_id):
        nonlocal cancel_called
        cancel_called = True
        return True

    client.get_order_status = mock_get_status
    client.cancel_order = mock_cancel
    client._live_pending["order123"] = original_order

    import time
    with patch("asyncio.sleep", new_callable=AsyncMock), \
         patch("time.monotonic", side_effect=[0, 0, 0.5, 1.0, 1.5, 999]):
        result = await client.poll_until_terminal("order123", timeout=1, poll_interval=0.1)

    assert result.status == "cancelled"
    assert cancel_called


@pytest.mark.asyncio
async def test_live_order_pending_size_zero(client):
    """Live pending orders must have filled_size=0."""
    mock_clob = MagicMock()
    mock_clob.create_and_post_order.return_value = {"orderID": "live-123"}
    client._clob_client = mock_clob

    order = Order(
        market_id="m1", token_id="tok1", side=OrderSide.BUY,
        size=10, price=0.5, dry_run=False
    )

    with patch.object(type(client), "_is_live_enabled", return_value=True), \
         patch("pathlib.Path.exists", return_value=False):
        result = await client.place_order(order)

    assert result.status == "pending"
    assert result.filled_size == 0
    assert result.is_paper is False
    assert "live-123" in client._live_pending
