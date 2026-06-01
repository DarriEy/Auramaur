"""Tests for live order lifecycle management."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Stub py_clob_client_v2 so tests can exercise the live-order path without the
# optional C-extension package installed.
_clob_stub = MagicMock()
_clob_stub.order_builder.constants.BUY = "BUY"
_clob_stub.order_builder.constants.SELL = "SELL"
for mod_name in (
    "py_clob_client_v2",
    "py_clob_client_v2.client",
    "py_clob_client_v2.clob_types",
    "py_clob_client_v2.order_builder",
    "py_clob_client_v2.order_builder.constants",
):
    sys.modules.setdefault(mod_name, _clob_stub)

from auramaur.exchange.client import PolymarketClient
from auramaur.exchange.models import Order, OrderResult, OrderSide
from auramaur.exchange.paper import PaperTrader
from auramaur.broker.sync import PositionSyncer


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


async def _place_live_buy(client, size=10, price=0.5):
    order = Order(market_id="m1", token_id="tok1", side=OrderSide.BUY,
                  size=size, price=price, dry_run=False)
    with patch.object(type(client), "_is_live_enabled", return_value=True), \
         patch("pathlib.Path.exists", return_value=False):
        return await client.place_order(order)


@pytest.mark.asyncio
async def test_buy_skipped_counts_orphan_open_orders(client):
    """Orphan-proofing: a resting BUY from a prior session (NOT in _live_pending)
    is still counted via the CLOB's authoritative open-order list.

    Gross $26.28, orphan BUY locks 51.78 @ $0.50 = $25.89 -> ~$0.39 free, so a
    $5 order must be skipped even though _live_pending is empty.
    """
    mock_clob = MagicMock()
    mock_clob.get_balance_allowance.return_value = {"balance": 26275397}  # $26.28
    mock_clob.get_open_orders.return_value = [
        {"side": "BUY", "price": "0.5", "original_size": "51.78", "size_matched": "0"},
    ]
    client._clob_client = mock_clob
    assert client._live_pending == {}  # no in-memory record of the orphan

    result = await _place_live_buy(client)

    assert result.status == "rejected"
    assert result.order_id == "INSUFFICIENT_BALANCE"
    mock_clob.create_and_post_order.assert_not_called()


@pytest.mark.asyncio
async def test_buy_guard_counts_only_unfilled_portion(client):
    """Reservation uses original_size - size_matched (the still-resting part)."""
    mock_clob = MagicMock()
    mock_clob.get_balance_allowance.return_value = {"balance": 26275397}  # $26.28
    # 100 @ $0.50 with 60 already filled -> only 40 @ $0.50 = $20 reserved.
    mock_clob.get_open_orders.return_value = [
        {"side": "BUY", "price": "0.5", "original_size": "100", "size_matched": "60"},
    ]
    mock_clob.create_and_post_order.return_value = {"orderID": "live-ok"}
    client._clob_client = mock_clob

    # $6.28 free; a $5 order fits, a $7 order does not.
    assert (await _place_live_buy(client, size=10, price=0.5)).status == "pending"
    assert (await _place_live_buy(client, size=14, price=0.5)).order_id == "INSUFFICIENT_BALANCE"


@pytest.mark.asyncio
async def test_buy_guard_excludes_sell_open_orders(client):
    """SELL orders lock conditional tokens, not USDC -> not counted as reserved."""
    mock_clob = MagicMock()
    mock_clob.get_balance_allowance.return_value = {"balance": 26275397}  # $26.28
    mock_clob.get_open_orders.return_value = [
        {"side": "SELL", "price": "0.5", "original_size": "1000", "size_matched": "0"},
    ]
    mock_clob.create_and_post_order.return_value = {"orderID": "live-ok"}
    client._clob_client = mock_clob

    result = await _place_live_buy(client)
    assert result.status == "pending"
    assert result.order_id == "live-ok"


@pytest.mark.asyncio
async def test_buy_guard_falls_back_to_live_pending_when_open_orders_unavailable(client):
    """If the authoritative open-orders query fails, fall back to _live_pending."""
    mock_clob = MagicMock()
    mock_clob.get_balance_allowance.return_value = {"balance": 26275397}  # $26.28
    mock_clob.get_open_orders.side_effect = RuntimeError("api down")
    client._clob_client = mock_clob
    client._live_pending["resting"] = Order(
        market_id="m0", token_id="t0", side=OrderSide.BUY, size=51.78, price=0.5,
    )

    result = await _place_live_buy(client)
    assert result.order_id == "INSUFFICIENT_BALANCE"


@pytest.mark.asyncio
async def test_reconcile_open_orders_imports_orphans(client):
    """Startup reconcile pulls CLOB open orders into _live_pending, stamped with
    their real age so the monitor's TTL-cancel reaps stale ones."""
    mock_clob = MagicMock()
    mock_clob.get_open_orders.return_value = [
        {"id": "orphan-1", "side": "BUY", "price": "0.5",
         "original_size": "20", "size_matched": "0",
         "asset_id": "tokA", "market": "mktA", "created_at": 1000},
        {"id": "orphan-2", "side": "SELL", "price": "0.7",
         "original_size": "5", "size_matched": "0",
         "asset_id": "tokB", "market": "mktB", "created_at": 2000},
    ]
    client._clob_client = mock_clob

    with patch.object(type(client), "_is_live_enabled", return_value=True):
        n = await client.reconcile_open_orders()

    assert n == 2
    assert set(client._live_pending) == {"orphan-1", "orphan-2"}
    buy = client._live_pending["orphan-1"]
    assert buy.side == OrderSide.BUY
    assert buy.size == 20 and buy.price == 0.5
    assert buy.token_id == "tokA"
    # created_at parsed from the order's unix timestamp (not "now")
    assert buy.created_at.year == 1970


@pytest.mark.asyncio
async def test_reconcile_open_orders_noop_when_not_live(client):
    """Reconcile is a no-op (and never calls the CLOB) when live trading is off."""
    mock_clob = MagicMock()
    client._clob_client = mock_clob
    with patch.object(type(client), "_is_live_enabled", return_value=False):
        n = await client.reconcile_open_orders()
    assert n == 0
    mock_clob.get_open_orders.assert_not_called()


@pytest.mark.asyncio
async def test_buy_proceeds_when_free_collateral_sufficient(client):
    """With no resting orders, the full gross balance is spendable -> order sent."""
    mock_clob = MagicMock()
    mock_clob.get_balance_allowance.return_value = {"balance": 26275397}  # $26.28
    mock_clob.get_open_orders.return_value = []
    mock_clob.create_and_post_order.return_value = {"orderID": "live-ok"}
    client._clob_client = mock_clob

    result = await _place_live_buy(client)

    assert result.status == "pending"
    assert result.order_id == "live-ok"


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


@pytest.mark.asyncio
async def test_live_balance_does_not_cancel_resting_orders(client, mock_paper):
    """Balance checks should not globally cancel open maker/limit orders."""
    mock_clob = MagicMock()
    mock_clob.get_balance_allowance.return_value = {"balance": 123_000_000}
    client._clob_client = mock_clob
    client._init_clob_client = MagicMock()

    syncer = PositionSyncer(
        settings=client._settings,
        db=AsyncMock(),
        exchange=client,
        paper=mock_paper,
        pnl=AsyncMock(),
    )

    balance = await syncer._get_live_balance()

    assert balance == 123.0
    mock_clob.cancel_all.assert_not_called()


@pytest.mark.asyncio
async def test_live_balance_nets_out_open_buy_orders(client, mock_paper):
    """Spendable balance subtracts collateral reserved by resting BUYs (not
    SELLs) so the engine sizes against genuinely-available cash, not gross."""
    mock_clob = MagicMock()
    mock_clob.get_balance_allowance.return_value = {"balance": 26_275_397}  # $26.28 gross
    mock_clob.get_open_orders.return_value = [
        {"side": "BUY", "price": "0.5", "original_size": "51.78", "size_matched": "0"},  # $25.89 locked
        {"side": "SELL", "price": "0.9", "original_size": "100", "size_matched": "0"},   # excluded
    ]
    client._clob_client = mock_clob
    client._init_clob_client = MagicMock()

    syncer = PositionSyncer(
        settings=client._settings,
        db=AsyncMock(),
        exchange=client,
        paper=mock_paper,
        pnl=AsyncMock(),
    )

    balance = await syncer._get_live_balance()

    assert balance == pytest.approx(0.39, abs=0.01)  # 26.28 - 25.89
