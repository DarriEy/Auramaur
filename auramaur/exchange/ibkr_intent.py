"""Present an IBKR instrument as the intent shape ExecutionGateway reads.

The gateway is very nearly venue-agnostic. Measured on 2026-07-27, it touches
prediction-market concepts on nine lines and reads exactly six fields off a
``TradeIntent``::

    market.id  market.category  market.neg_risk_market_id
    signal.claude_prob  signal.market_prob  signal.strategy_source

Everything else flows through ``Order`` — already generic — and
``exchange.place_order(order)``, which the IBKR clients implement. So the IBKR
books do not need a gateway refactor to gain decision capture, pnl_ledger
booking, the aggregate market-cap guard and the force_paper contract; they need
an adapter that fills in those six fields, plus a ``prepare_order`` for
``_build_order`` to call. That is what this module is.

The three prediction-market couplings degrade benignly for an instrument:

* ``order.token.value`` keys the aggregate market-cap guard. Left at the YES
  default it becomes one cap per instrument, which is the semantics we want.
* the ``orderbook_snapshots`` lookup in decision capture finds nothing and is
  already null-guarded.
* ``neg_risk_market_id`` is optional; the event family falls back to the id.

Nothing here decides whether a trade is live. ``dry_run`` is passed in by the
caller and remains governed by the book's ``paper`` flag and, later, the
graduation ladder.
"""

from __future__ import annotations

from auramaur.exchange.models import (
    Confidence,
    Market,
    Order,
    OrderSide,
    OrderType,
    Signal,
)

# An instrument is one market to the gateway, and its book is the "category"
# the graduation ladder cells on. `ibkr_fx_paper x fx` reads the way
# `llm x politics_us` does.
_VENUE = "ibkr"


def instrument_market(spec, *, mark: float) -> Market:
    """The instrument as a Market, carrying only what the gateway reads."""
    return Market(
        id=f"{_VENUE}:{spec.key}",
        exchange=_VENUE,
        question=spec.description or spec.symbol,
        category=spec.asset_class or spec.book.value,
        active=True,
        outcome_yes_price=mark,
        outcome_no_price=max(0.0, 1.0 - mark) if 0.0 < mark < 1.0 else 0.0,
    )


def instrument_signal(spec, *, strategy_source: str, mark: float,
                      fair: float | None, side: OrderSide,
                      rationale: str = "") -> Signal:
    """The entry as a Signal.

    ``fair`` is the strategy's own view of value. Directional price books do not
    produce a probability, so callers pass None and the signal carries the mark
    for both — the decision is still captured, but it asserts no forecast edge.
    That distinction matters downstream: the ladder's second bar is a
    market-Brier-edge LCB, and a book that never forecast anything must not be
    scored as though it did (see docs/ibkr_graduation_spec.md).
    """
    reference = float(mark)
    return Signal(
        market_id=f"{_VENUE}:{spec.key}",
        market_question=spec.description or spec.symbol,
        claude_prob=float(fair) if fair is not None else reference,
        claude_confidence=Confidence.MEDIUM,
        market_prob=reference,
        edge=(float(fair) - reference) * 100.0 if fair is not None else 0.0,
        evidence_summary=rationale[:500],
        recommended_side=side,
        strategy_source=strategy_source,
    )


def prepare_instrument_order(spec, *, side: OrderSide, quantity: float,
                             price: float, is_live: bool,
                             strategy_source: str) -> Order | None:
    """Build the Order the gateway will place. None when it is not placeable.

    Quantity is in contracts/shares, not dollars: an IBKR instrument's size is
    not a token count and must not be re-derived from a dollar stake the way a
    CLOB order is.
    """
    if quantity <= 0 or price <= 0:
        return None
    return Order(
        market_id=f"{_VENUE}:{spec.key}",
        exchange=_VENUE,
        token_id=spec.key,
        side=side,
        size=float(quantity),
        price=float(price),
        order_type=OrderType.LIMIT,
        dry_run=not is_live,
        source=strategy_source,
    )
