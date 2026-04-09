"""Tests for the triple-gate safety model at the execution layer.

The three gates that must ALL be open for a live order:
1. AURAMAUR_LIVE=true environment variable
2. execution.live=true in config
3. dry_run=False on the order

If ANY gate is closed, the order must route to paper trading.
If the kill switch file exists, the order must be blocked entirely.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from auramaur.exchange.client import PolymarketClient
from auramaur.exchange.models import Order, OrderSide
from auramaur.exchange.paper import PaperTrader


def _make_settings(auramaur_live: bool = False, execution_live: bool = False) -> MagicMock:
    settings = MagicMock()
    settings.auramaur_live = auramaur_live
    settings.execution.live = execution_live
    settings.kill_switch_active = False
    # is_live requires both env + config and no kill switch
    settings.is_live = auramaur_live and execution_live
    return settings


def _make_order(dry_run: bool = True) -> Order:
    return Order(
        market_id="test-market-123",
        side=OrderSide.BUY,
        size=10.0,
        price=0.55,
        dry_run=dry_run,
    )


@pytest.fixture
def paper_trader():
    paper = MagicMock(spec=PaperTrader)
    paper.execute = AsyncMock(return_value=MagicMock(
        order_id="PAPER-abc",
        market_id="test-market-123",
        status="paper",
        filled_size=10.0,
        filled_price=0.55,
        is_paper=True,
    ))
    return paper


# ---------------------------------------------------------------------------
# Gate 1: AURAMAUR_LIVE env var
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_env_gate_closed_routes_to_paper(paper_trader):
    """When AURAMAUR_LIVE=false, order must go to paper regardless of other gates."""
    settings = _make_settings(auramaur_live=False, execution_live=True)
    client = PolymarketClient(settings, paper_trader)

    order = _make_order(dry_run=False)
    with patch.object(Path, "exists", return_value=False):
        result = await client.place_order(order)

    paper_trader.execute.assert_awaited_once_with(order)
    assert result.is_paper


# ---------------------------------------------------------------------------
# Gate 2: execution.live config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_config_gate_closed_routes_to_paper(paper_trader):
    """When execution.live=false, order must go to paper regardless of other gates."""
    settings = _make_settings(auramaur_live=True, execution_live=False)
    client = PolymarketClient(settings, paper_trader)

    order = _make_order(dry_run=False)
    with patch.object(Path, "exists", return_value=False):
        result = await client.place_order(order)

    paper_trader.execute.assert_awaited_once_with(order)
    assert result.is_paper


# ---------------------------------------------------------------------------
# Gate 3: dry_run per-order flag
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dry_run_true_routes_to_paper(paper_trader):
    """When dry_run=True, order must go to paper even if both global gates are open."""
    settings = _make_settings(auramaur_live=True, execution_live=True)
    client = PolymarketClient(settings, paper_trader)

    order = _make_order(dry_run=True)
    with patch.object(Path, "exists", return_value=False):
        result = await client.place_order(order)

    paper_trader.execute.assert_awaited_once_with(order)
    assert result.is_paper


# ---------------------------------------------------------------------------
# All gates closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_gates_closed_routes_to_paper(paper_trader):
    """When all gates are closed, order must go to paper."""
    settings = _make_settings(auramaur_live=False, execution_live=False)
    client = PolymarketClient(settings, paper_trader)

    order = _make_order(dry_run=True)
    with patch.object(Path, "exists", return_value=False):
        result = await client.place_order(order)

    paper_trader.execute.assert_awaited_once_with(order)
    assert result.is_paper


# ---------------------------------------------------------------------------
# All gates open → live path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_gates_open_attempts_live(paper_trader):
    """When all three gates are open, order must NOT go to paper (attempts live)."""
    settings = _make_settings(auramaur_live=True, execution_live=True)
    client = PolymarketClient(settings, paper_trader)

    order = _make_order(dry_run=False)
    order.token_id = "0xabc123"

    with patch.object(Path, "exists", return_value=False), \
         patch.object(client, "_init_clob_client") as mock_init:
        # Mock the CLOB client to avoid real API calls
        mock_clob = MagicMock()
        mock_clob.create_and_post_order.return_value = {"orderID": "LIVE-xyz"}
        client._clob_client = mock_clob

        result = await client.place_order(order)

    # Paper trader must NOT have been called
    paper_trader.execute.assert_not_awaited()
    assert result.is_paper is False
    assert result.order_id == "LIVE-xyz"
    assert result.status == "pending"


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kill_switch_blocks_order(paper_trader):
    """When KILL_SWITCH file exists, order must be blocked (not paper, not live)."""
    settings = _make_settings(auramaur_live=True, execution_live=True)
    client = PolymarketClient(settings, paper_trader)

    order = _make_order(dry_run=False)
    with patch.object(Path, "exists", return_value=True):
        result = await client.place_order(order)

    # Neither paper nor live should execute
    paper_trader.execute.assert_not_awaited()
    assert result.order_id == "BLOCKED"
    assert result.status == "rejected"


@pytest.mark.asyncio
async def test_kill_switch_blocks_even_paper(paper_trader):
    """Kill switch blocks ALL orders, including paper trades."""
    settings = _make_settings(auramaur_live=False, execution_live=False)
    client = PolymarketClient(settings, paper_trader)

    order = _make_order(dry_run=True)
    with patch.object(Path, "exists", return_value=True):
        result = await client.place_order(order)

    paper_trader.execute.assert_not_awaited()
    assert result.order_id == "BLOCKED"
    assert result.status == "rejected"


# ---------------------------------------------------------------------------
# Default order safety
# ---------------------------------------------------------------------------


def test_order_defaults_to_dry_run():
    """Order model must default to dry_run=True (paper trading)."""
    order = Order(market_id="x", side=OrderSide.BUY, size=1, price=0.5)
    assert order.dry_run is True


# ---------------------------------------------------------------------------
# Combinatorial: every gate combination
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "auramaur_live, execution_live, dry_run, expect_paper",
    [
        (False, False, True, True),
        (False, False, False, True),
        (False, True, True, True),
        (False, True, False, True),
        (True, False, True, True),
        (True, False, False, True),
        (True, True, True, True),
        # (True, True, False) is the only combo that goes live — tested above
    ],
)
async def test_any_closed_gate_routes_to_paper(
    paper_trader, auramaur_live, execution_live, dry_run, expect_paper
):
    """Every combination with at least one closed gate must route to paper."""
    settings = _make_settings(auramaur_live=auramaur_live, execution_live=execution_live)
    client = PolymarketClient(settings, paper_trader)

    order = _make_order(dry_run=dry_run)
    with patch.object(Path, "exists", return_value=False):
        result = await client.place_order(order)

    assert result.is_paper is True
    paper_trader.execute.assert_awaited_once()
