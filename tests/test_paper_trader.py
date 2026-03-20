"""Tests for paper trading simulator."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from auramaur.exchange.models import Order, OrderSide
from auramaur.exchange.paper import PaperTrader


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.fetchall = AsyncMock(return_value=[])
    db.fetchone = AsyncMock(return_value={"net": 0})
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    return db


@pytest.fixture
def paper(mock_db):
    return PaperTrader(db=mock_db, initial_balance=1000.0)


@pytest.mark.asyncio
async def test_buy_reduces_balance(paper):
    order = Order(market_id="m1", side=OrderSide.BUY, size=10, price=0.5)
    result = await paper.execute(order)
    assert result.status == "paper"
    assert result.is_paper is True
    assert paper.balance == 995.0  # 1000 - 10*0.5


@pytest.mark.asyncio
async def test_sell_increases_balance(paper):
    # First buy
    await paper.execute(Order(market_id="m1", side=OrderSide.BUY, size=10, price=0.5))
    # Then sell
    result = await paper.execute(Order(market_id="m1", side=OrderSide.SELL, size=10, price=0.6))
    assert paper.balance == pytest.approx(1001.0)  # 1000 - 5 + 6


@pytest.mark.asyncio
async def test_insufficient_balance(paper):
    order = Order(market_id="m1", side=OrderSide.BUY, size=10000, price=0.5)
    result = await paper.execute(order)
    assert result.status == "rejected"


@pytest.mark.asyncio
async def test_position_tracking(paper):
    await paper.execute(Order(market_id="m1", side=OrderSide.BUY, size=10, price=0.5))
    assert "m1" in paper.positions
    assert paper.positions["m1"].size == 10


@pytest.mark.asyncio
async def test_position_closed_on_full_sell(paper):
    await paper.execute(Order(market_id="m1", side=OrderSide.BUY, size=10, price=0.5))
    await paper.execute(Order(market_id="m1", side=OrderSide.SELL, size=10, price=0.6))
    assert "m1" not in paper.positions
