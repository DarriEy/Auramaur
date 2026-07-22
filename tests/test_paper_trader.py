"""Tests for paper trading simulator."""

import pytest
from unittest.mock import AsyncMock

from auramaur.exchange.models import Order, OrderSide
from auramaur.exchange.paper import PaperTrader


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.fetchall = AsyncMock(return_value=[])
    # _compute_balance reads pnl_ledger ("p") + cost_basis ("c"); both empty here
    # so spendable == initial_balance, keeping the incremental-arithmetic tests valid.
    db.fetchone = AsyncMock(return_value={"p": 0, "c": 0})
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


# ---------------------------------------------------------------------------
# Spendable balance from authoritative state (the drain-bug fix)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compute_balance_from_ledger_and_open_cost():
    """Spendable = initial + realized paper P&L - cost tied up in OPEN paper
    positions — NOT the old SUM(trades) that drained to ~0."""
    from auramaur.db.database import Database
    db = Database(":memory:")
    await db.connect()
    # $300 tied up in open paper positions; +$50 realized paper P&L.
    await db.execute("INSERT INTO cost_basis (market_id, token, size, avg_cost, "
                     "total_cost, is_paper) VALUES ('a','YES',100,2.0,200,1)")
    await db.execute("INSERT INTO cost_basis (market_id, token, size, avg_cost, "
                     "total_cost, is_paper) VALUES ('b','YES',100,1.0,100,1)")
    await db.execute("INSERT INTO pnl_ledger (market_id, venue, category, "
                     "strategy_source, kind, token, qty, pnl, fees, is_paper, "
                     "source_ref) VALUES ('c','polymarket','x','llm','sell','YES',1,50,0,1,'r1')")
    # LIVE rows must NOT count.
    await db.execute("INSERT INTO cost_basis (market_id, token, size, avg_cost, "
                     "total_cost, is_paper) VALUES ('live','YES',100,5.0,500,0)")
    await db.commit()

    pt = PaperTrader(db=db, initial_balance=1000.0)
    bal = await pt._compute_balance()
    assert bal == pytest.approx(1000 + 50 - 300)  # 750
    await db.close()


@pytest.mark.asyncio
async def test_resolved_position_does_not_drain_balance():
    """The OLD bug: a BUY debited forever and resolution never credited, so the
    wallet drained. Now a resolved position (gone from open cost_basis, its P&L
    in the ledger) leaves the balance healthy — no drain."""
    from auramaur.db.database import Database
    db = Database(":memory:")
    await db.connect()
    # A position that was bought for $100 and resolved to a $30 loss: it is NOT
    # in open cost_basis (closed), and its -30 sits in the ledger.
    await db.execute("INSERT INTO pnl_ledger (market_id, venue, category, "
                     "strategy_source, kind, token, qty, pnl, fees, is_paper, "
                     "source_ref) VALUES ('done','polymarket','x','llm','sell','YES',1,-30,0,1,'r1')")
    await db.commit()
    pt = PaperTrader(db=db, initial_balance=1000.0)
    # Spendable = 1000 - 30 = 970 (NOT 1000 - 100 the BUY cost): cash recovered.
    assert await pt._compute_balance() == pytest.approx(970)
    await db.close()

@pytest.mark.asyncio
async def test_insufficient_balance_reports_amount_and_suppresses_unchanged_retry(paper):
    paper._compute_balance = AsyncMock(side_effect=[5.72, 5.72, 20.0])
    order = Order(market_id="cash-blocked", side=OrderSide.BUY, size=10, price=1.0)

    first = await paper.execute(order)
    repeated = await paper.execute(order)
    after_cash_change = await paper.execute(order)

    assert first.status == "rejected"
    assert first.error_message == (
        "insufficient paper balance: $5.72, requires $10.00")
    assert first.order_id.startswith("PAPER-")
    assert repeated.status == "rejected"
    assert repeated.order_id == "SKIP_CASH"
    assert repeated.error_message == first.error_message
    assert after_cash_change.status == "paper"
    assert paper.trade_count == 1
