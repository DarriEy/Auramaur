"""A resting or refused order is not a position.

Two shapes of the same defect. (1) PaperTrader.check_fills stamped
status="filled" over an execute() that REFUSED for insufficient paper cash —
booking a size-0/price-0 fill and marking the decision executable with
"trade_through" evidence, which graduation.credible_fill_evidence accepts. (2)
Four pillars wrote a full-size portfolio row on status="pending", which is what
every live Polymarket place_order returns and what maker-priced paper orders
always defer to. Both feed the ledger the graduation ladder reads.

econ_indicator:227, long_horizon:400 and informed_flow_pillar:221 already had
the guard, with a comment explaining it.
"""

import inspect

import pytest

from auramaur.db.database import Database
from auramaur.exchange.models import Order, OrderSide, TokenType
from auramaur.exchange.paper import PaperTrader


def _order(price=0.82, size=100.0):
    return Order(market_id="m1", exchange="polymarket", token_id="t1",
                 side=OrderSide.BUY, token=TokenType.YES,
                 size=size, price=price, dry_run=True)


@pytest.mark.asyncio
async def test_refused_trade_through_is_not_reported_as_a_fill():
    """The reported case: a rested BUY trades through on a book that cannot
    fund it. execute(force=True) still hits the balance gate — `force` only
    bypasses the marketability check."""
    db = Database(":memory:")
    await db.connect()
    trader = PaperTrader(db, initial_balance=5.0)   # far below 100 * 0.82
    order = _order()
    trader.submit_limit_order(order)
    assert trader.pending_orders, "order should be resting"

    filled = await trader.check_fills({"m1": 0.80})   # trades through

    assert filled == [], "a refusal must not be handed over as a fill"
    # And it must not vanish: it neither filled nor rested before this fix.
    assert trader.pending_orders, "the unfilled order must stay resting"


@pytest.mark.asyncio
async def test_funded_trade_through_still_fills():
    db = Database(":memory:")
    await db.connect()
    trader = PaperTrader(db, initial_balance=1000.0)
    order = _order()
    trader.submit_limit_order(order)

    filled = await trader.check_fills({"m1": 0.80})

    assert len(filled) == 1
    result, _ = filled[0]
    assert result.status == "filled"
    assert result.filled_size > 0
    assert trader.pending_orders == []


def test_order_monitor_paper_branch_guards_like_the_live_branch():
    from auramaur import bot_order_monitor
    src = inspect.getsource(bot_order_monitor)
    paper_arm = src.split("_record_deferred_paper_fills", 1)[1].split("async def", 1)[0]
    assert 'result.status != "filled" or result.filled_size <= 0' in paper_arm


@pytest.mark.parametrize("module,func", [
    ("auramaur.strategy.bias_harvest", "_try_enter"),
    ("auramaur.strategy.agent_trader", "_try_enter"),
    ("auramaur.strategy.entailment_arb", "_enter_pair"),
])
def test_pillars_do_not_record_a_resting_order_as_a_position(module, func):
    import importlib
    mod = importlib.import_module(module)
    cls = next(v for k, v in vars(mod).items()
               if isinstance(v, type) and hasattr(v, func))
    src = inspect.getsource(getattr(cls, func))
    assert 'status != "pending"' in src and "filled_size > 0" in src


def test_oddlot_does_not_book_an_unfilled_share_order():
    from auramaur.strategy import oddlot_tender
    cls = next(v for v in vars(oddlot_tender).values()
               if isinstance(v, type) and hasattr(v, "run_once"))
    src = inspect.getsource(cls)
    guard = src.split("_record_entry(ticker", 1)[0]
    assert 'result.status != "pending" and result.filled_size > 0' in guard
