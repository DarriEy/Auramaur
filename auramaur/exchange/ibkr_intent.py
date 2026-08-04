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

from datetime import date
import uuid

from auramaur.exchange.models import (
    Confidence,
    Market,
    Order,
    OrderResult,
    OrderSide,
    OrderType,
    Signal,
)

# An instrument is one market to the gateway, and its book is the "category"
# the graduation ladder cells on. `ibkr_fx_paper x fx` reads the way
# `llm x politics_us` does.
_VENUE = "ibkr"

# Every market_id this adapter mints starts here. It is the ONE handle the rest
# of the system has for "this row is broker exposure, not a prediction-market
# contract" — cost_basis has no venue column, so the shared paper wallet
# (auramaur/exchange/paper.py) scopes itself off this prefix. Changing it
# without changing that query silently drains the prediction-market paper book.
INSTRUMENT_ID_PREFIX = f"{_VENUE}:"


def instrument_market_id(spec, cell: str = "") -> str:
    """The gateway/ledger market_id for one instrument in one book cell.

    ``cell`` separates concurrently-running arms of the same book. It is not
    cosmetic: ``cost_basis`` is keyed by (market_id, is_paper, token), so two
    arms holding SPY under one market_id MERGE into a single basis row — their
    realized P&L, the aggregate exposure guard, and the ledger's
    earliest-entrant strategy attribution all then belong to whichever arm
    bought first.
    """
    return (f"{INSTRUMENT_ID_PREFIX}{cell}:{spec.key}" if cell
            else f"{INSTRUMENT_ID_PREFIX}{spec.key}")


def instrument_event_family(spec, cell: str, session_date: str) -> str:
    """The independent-evidence family for one forecast.

    Left at the market_id, an (arm, symbol) pair is ONE family for all time:
    28 ETF symbols give 28 families ever, against min_paired_forecasts of 30.
    The book could never graduate, two families short, no matter how good its
    forecasts were.

    Each forecast is nonetheless a genuinely separate event, so the family has
    to advance with time — but not per forecast. Consecutive daily forecasts on
    one symbol share four of their five sessions, and counting them as
    independent is exactly the overcounting the family mechanism exists to
    prevent. Bucketing by ISO week collapses a symbol's overlapping forecasts
    into one family per week and lets genuinely fresh windows count.

    Approximate at the boundary, and deliberately so: a Friday forecast runs
    into the next week. It never treats two forecasts from the same week as
    independent, which is the direction that matters — it under-counts
    evidence rather than inventing it.
    """
    try:
        year, week, _ = date.fromisoformat(str(session_date)[:10]).isocalendar()
        bucket = f"{year}W{week:02d}"
    except ValueError:
        bucket = str(session_date)[:10]
    return f"{instrument_market_id(spec, cell)}:{bucket}"


def instrument_market(spec, *, mark: float, cell: str = "",
                      event_family: str = "") -> Market:
    """The instrument as a Market, carrying only what the gateway reads.

    ``event_family`` rides on ``neg_risk_market_id`` because that is the field
    ``_capture_decision`` already reads for the family, falling back to the id.
    Routing it this way needs no change to the shared gateway and cannot affect
    a prediction-market intent, which never sets it for an instrument.
    """
    return Market(
        id=instrument_market_id(spec, cell),
        exchange=_VENUE,
        question=spec.description or spec.symbol,
        category=spec.asset_class or spec.book.value,
        active=True,
        outcome_yes_price=mark,
        outcome_no_price=max(0.0, 1.0 - mark) if 0.0 < mark < 1.0 else 0.0,
        neg_risk_market_id=event_family or "",
    )


def instrument_signal(spec, *, strategy_source: str, mark: float,
                      fair: float | None, side: OrderSide,
                      rationale: str = "", cell: str = "") -> Signal:
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
        market_id=instrument_market_id(spec, cell),
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
                             strategy_source: str, cell: str = "",
                             usd_per_point: float = 1.0,
                             usd_capital_per_unit: float | None = None
                             ) -> Order | None:
    """Build the Order the gateway will place. None when it is not placeable.

    Quantity is in contracts/shares, not dollars: an IBKR instrument's size is
    not a token count and must not be re-derived from a dollar stake the way a
    CLOB order is.

    ``price`` is in the instrument's own currency; ``usd_per_point`` is its
    contract multiplier times its FX rate, and the Order carries the PRODUCT.
    That matters because ``PnLTracker.record_fill`` realizes
    ``(price - avg_cost) * size`` with no notion of either: booking a raw local
    price would have recorded $0.02 where the FX book records $21.61, and fed
    that fiction to the graduation ladder's net-P&L bar. Both default to the
    identity, so an ETF or a prediction contract is byte-identical to before.

    ``usd_capital_per_unit`` is committed capital, which for a leveraged
    instrument is not notional — see ``Order.capital_ratio``.
    """
    if quantity <= 0 or price <= 0 or usd_per_point <= 0:
        return None
    return Order(
        market_id=instrument_market_id(spec, cell),
        exchange=_VENUE,
        token_id=spec.key,
        side=side,
        size=float(quantity),
        # USD per unit, so every downstream rail (cost_basis, pnl_ledger, the
        # aggregate cap) speaks one currency.
        price=float(price) * float(usd_per_point),
        order_type=OrderType.LIMIT,
        dry_run=not is_live,
        source=strategy_source,
        usd_capital_per_unit=(None if usd_capital_per_unit is None
                              else float(usd_capital_per_unit)),
    )


class SimulatedInstrumentExchange:
    """A local fill simulator that satisfies the gateway's exchange protocol.

    This is NOT a broker adapter. It holds no connection, no credentials and no
    live branch, and it refuses outright to carry an order that is not
    ``dry_run``. That is what lets a structurally paper-only book
    (``ExecutionMode.PAPER_SIMULATED`` — "local fills ONLY, no order path
    exists, at any gate") route through ``ExecutionGateway`` for decision
    capture and pnl_ledger booking without acquiring an order path: the
    gateway's ``exchange.place_order`` call lands here and goes no further.

    The book has already decided its own fill (price, quantity, reference) by
    the time it reaches the gateway, so ``stage`` hands that exact order over
    rather than letting the gateway re-derive one. ``order_id`` is the book's
    own fill reference, which makes ``fills.order_id`` join straight back to
    the venue-native fill row and gives ``PnLTracker.record_fill`` a real
    idempotency key.
    """

    def __init__(self, exchange_name: str = _VENUE) -> None:
        self.exchange_name = exchange_name
        self._staged: tuple[Order, str] | None = None

    def stage(self, order: Order, *, order_id: str = "") -> None:
        """Hand the already-decided order to the next ``gateway.submit``."""
        self._staged = (order, order_id or f"IBKR-SIM-{uuid.uuid4().hex[:12]}")

    def prepare_order(self, signal, market, size_dollars: float,
                      is_live: bool) -> Order | None:
        """The gateway's build hook. Returns the staged order, or None."""
        if self._staged is None:
            return None
        return self._staged[0]

    async def place_order(self, order: Order) -> OrderResult:
        """Simulate the fill locally. Never reaches a broker, by construction."""
        if not order.dry_run:
            # A simulated venue must never be the thing that "executes" a live
            # order. Reaching here means a caller cleared the live gates and
            # then routed to the simulator — refuse rather than record a
            # fictional live fill.
            raise RuntimeError(
                "SimulatedInstrumentExchange cannot carry a live order: "
                f"{order.market_id}")
        staged = self._staged
        self._staged = None
        order_id = (staged[1] if staged is not None and staged[0] is order
                    else f"IBKR-SIM-{uuid.uuid4().hex[:12]}")
        return OrderResult(
            order_id=order_id,
            market_id=order.market_id,
            status="paper",
            filled_size=order.size,
            filled_price=order.price,
            is_paper=True,
        )
