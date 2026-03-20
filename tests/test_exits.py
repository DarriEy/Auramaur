"""Tests for position exit strategy (stop-loss, profit target, edge erosion, time decay)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from auramaur.exchange.models import ExitReason, OrderSide, Position
from auramaur.risk.portfolio import PortfolioTracker


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.fetchall = AsyncMock(return_value=[])
    db.fetchone = AsyncMock(return_value=None)
    return db


@pytest.fixture
def settings():
    s = MagicMock()
    s.execution.stop_loss_pct = 30.0
    s.execution.profit_target_pct = 50.0
    s.execution.edge_erosion_min_pct = 2.0
    s.execution.time_decay_hours = 12.0
    return s


def _make_position(market_id: str, avg_price: float, size: float = 10.0,
                   side: OrderSide = OrderSide.BUY) -> dict:
    """Return a DB row dict for a position."""
    return {
        "market_id": market_id,
        "side": side.value,
        "size": size,
        "avg_price": avg_price,
        "current_price": avg_price,
        "category": "test",
    }


def _make_market(market_id: str, yes_price: float, end_date: datetime | None = None,
                 active: bool = True):
    m = MagicMock()
    m.id = market_id
    m.outcome_yes_price = yes_price
    m.end_date = end_date
    m.active = active
    return m


@pytest.mark.asyncio
async def test_stop_loss_triggered(mock_db, settings):
    """Position at -35% → STOP_LOSS."""
    mock_db.fetchall = AsyncMock(return_value=[
        _make_position("m1", avg_price=0.60),
    ])
    gamma = AsyncMock()
    # Current price dropped to 0.39 → loss = (0.39-0.60)*10 = -2.10, cost = 6.0, pct = -35%
    gamma.get_market = AsyncMock(return_value=_make_market("m1", 0.39))

    tracker = PortfolioTracker(db=mock_db)
    exits = await tracker.check_exits(settings, gamma)
    assert len(exits) == 1
    assert exits[0][1] == ExitReason.STOP_LOSS


@pytest.mark.asyncio
async def test_profit_target_triggered(mock_db, settings):
    """Position at +55% → PROFIT_TARGET."""
    mock_db.fetchall = AsyncMock(return_value=[
        _make_position("m1", avg_price=0.40),
    ])
    gamma = AsyncMock()
    # Current price rose to 0.62 → gain = (0.62-0.40)*10 = 2.20, cost = 4.0, pct = +55%
    gamma.get_market = AsyncMock(return_value=_make_market("m1", 0.62))

    tracker = PortfolioTracker(db=mock_db)
    exits = await tracker.check_exits(settings, gamma)
    assert len(exits) == 1
    assert exits[0][1] == ExitReason.PROFIT_TARGET


@pytest.mark.asyncio
async def test_edge_erosion_triggered(mock_db, settings):
    """Price converged close to 1.0 → EDGE_EROSION."""
    mock_db.fetchall = AsyncMock(return_value=[
        _make_position("m1", avg_price=0.90),
    ])
    gamma = AsyncMock()
    # Current price at 0.99 → remaining edge = (1.0-0.99)*100 = 1.0% < 2.0%
    # PnL = (0.99-0.90)*10 = 0.90, cost = 9.0, pct = +10% (not triggering stop/profit)
    gamma.get_market = AsyncMock(return_value=_make_market("m1", 0.99))

    tracker = PortfolioTracker(db=mock_db)
    exits = await tracker.check_exits(settings, gamma)
    assert len(exits) == 1
    assert exits[0][1] == ExitReason.EDGE_EROSION


@pytest.mark.asyncio
async def test_time_decay_triggered(mock_db, settings):
    """6h to resolution + small edge → TIME_DECAY."""
    mock_db.fetchall = AsyncMock(return_value=[
        _make_position("m1", avg_price=0.90),
    ])
    gamma = AsyncMock()
    # Current price at 0.96 → remaining edge = 4.0% (< 5.0%)
    # End date 6h from now (< 12h)
    # PnL pct = +6.7% (not triggering stop/profit)
    end_date = datetime.now(timezone.utc) + timedelta(hours=6)
    gamma.get_market = AsyncMock(return_value=_make_market("m1", 0.96, end_date=end_date))

    tracker = PortfolioTracker(db=mock_db)
    exits = await tracker.check_exits(settings, gamma)
    assert len(exits) == 1
    assert exits[0][1] == ExitReason.TIME_DECAY


@pytest.mark.asyncio
async def test_no_exit_healthy_position(mock_db, settings):
    """All thresholds safe → empty list."""
    mock_db.fetchall = AsyncMock(return_value=[
        _make_position("m1", avg_price=0.50),
    ])
    gamma = AsyncMock()
    # Current price at 0.55 → PnL = +10%, remaining edge = 45%, no end date
    gamma.get_market = AsyncMock(return_value=_make_market("m1", 0.55))

    tracker = PortfolioTracker(db=mock_db)
    exits = await tracker.check_exits(settings, gamma)
    assert len(exits) == 0
