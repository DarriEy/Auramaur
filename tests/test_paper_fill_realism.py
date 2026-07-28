"""Paper fills must not credit a resting order the market never traded through.

Measured 2026-07-28: llm crossed on 14% of its fills and every other strategy
on 0%, so essentially every paper maker fill was an instant credit at the bid —
a fill live would never have granted. The evidence layer compensated by
stamping them 'synthetic' (uncountable), which left maker strategies
permanently ungraduatable.
"""

import pytest

from auramaur.db.database import Database
from auramaur.exchange.models import Order, OrderSide
from auramaur.exchange.paper import PaperTrader


def _order(price, bid, ask, side=OrderSide.BUY, decision_id=None):
    return Order(market_id="m1", token_id="tok", side=side, size=10.0,
                 price=price, best_bid=bid, best_ask=ask,
                 decision_id=decision_id)


async def _trader():
    db = Database(":memory:")
    await db.connect()
    return PaperTrader(db, initial_balance=1_000.0), db


@pytest.mark.asyncio
async def test_a_resting_order_is_queued_not_filled():
    """BUY at the bid does not cross, so live would not have filled it."""
    paper, db = await _trader()
    result = await paper.execute(_order(0.82, 0.82, 0.83))
    assert result.status == "pending"
    assert result.filled_size == 0
    assert len(paper.pending_orders) == 1
    await db.close()


@pytest.mark.asyncio
async def test_a_marketable_order_still_fills_immediately():
    """Lifting the ask is a real fill and must stay immediate."""
    paper, db = await _trader()
    result = await paper.execute(_order(0.83, 0.82, 0.83))
    assert result.status == "paper"
    assert result.filled_size == 10.0
    assert paper.pending_orders == []
    await db.close()


@pytest.mark.asyncio
async def test_an_order_with_no_book_keeps_the_old_behaviour():
    """marketable is None when the book is unknown — never guess and rest."""
    paper, db = await _trader()
    result = await paper.execute(
        Order(market_id="m1", side=OrderSide.BUY, size=10.0, price=0.5))
    assert result.status == "paper"
    await db.close()


@pytest.mark.asyncio
async def test_a_queued_order_fills_only_on_trade_through():
    """Touching the limit is not evidence our queue position executed."""
    paper, db = await _trader()
    await paper.execute(_order(0.82, 0.82, 0.83, decision_id=7))

    # Market sits AT the limit: no fill.
    assert await paper.check_fills({"m1": 0.82}) == []
    assert len(paper.pending_orders) == 1
    # Market moves the wrong way: no fill.
    assert await paper.check_fills({"m1": 0.90}) == []
    # Market trades THROUGH the bid: now it would have filled.
    filled = await paper.check_fills({"m1": 0.80})
    assert len(filled) == 1
    result, order = filled[0]
    assert result.status == "filled"
    assert order.decision_id == 7        # attributable back to its decision
    assert paper.pending_orders == []
    await db.close()


@pytest.mark.asyncio
async def test_a_sell_rests_and_trades_through_the_other_way():
    paper, db = await _trader()
    await paper.execute(_order(0.90, 0.88, 0.90, side=OrderSide.SELL))
    assert paper.pending_orders, "SELL above the bid should rest"
    assert await paper.check_fills({"m1": 0.89}) == []
    assert len(await paper.check_fills({"m1": 0.95})) == 1
    await db.close()


@pytest.mark.asyncio
async def test_the_toggle_restores_immediate_fills():
    """execution.paper_defer_resting_fills=False must need no code change."""
    paper, db = await _trader()
    paper._defer_resting = False
    result = await paper.execute(_order(0.82, 0.82, 0.83))
    assert result.status == "paper" and result.filled_size == 10.0
    await db.close()


@pytest.mark.asyncio
async def test_check_fills_does_not_requeue_what_it_just_filled():
    """check_fills calls execute(force=True); without that the fill would be
    re-queued forever because it is still non-marketable."""
    paper, db = await _trader()
    await paper.execute(_order(0.82, 0.82, 0.83))
    filled = await paper.check_fills({"m1": 0.80})
    assert len(filled) == 1
    assert paper.pending_orders == []
    assert await paper.check_fills({"m1": 0.80}) == []
    await db.close()
