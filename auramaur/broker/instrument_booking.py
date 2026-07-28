"""Book an already-executed IBKR fill onto the shared rails.

Deliberately NOT in ``auramaur/exchange``. That package is the registered
execution boundary and tests/test_exposure_registry.py skips it, so a
``gateway.submit`` callsite living there is invisible to the exposure audit.
This is application-level booking logic and must be auditable.

ONE path for both IBKR books. They previously had near-duplicate copies and
drifted into three separate defects: the multiasset side stored a USD price in
``decision_snapshots.fair_probability``, stamped every fill ``synthetic`` (so
no decision could count under ``require_executable_fills``), and used a
per-instrument event family that caps a book at 28 families forever against a
30-family minimum. The ETF side had all three fixed. Hence one copy.
"""

from __future__ import annotations

from dataclasses import dataclass

from auramaur.broker.execution_gateway import TradeIntent
from auramaur.broker.ledger import record_ledger_event
from auramaur.exchange.ibkr_intent import (
    instrument_event_family, instrument_market, instrument_market_id,
    instrument_signal, prepare_instrument_order,
)
from auramaur.exchange.models import OrderSide

__all__ = ["InstrumentFill", "book_instrument_fill"]


@dataclass(frozen=True)
class InstrumentFill:
    """One already-executed fill, described for the shared booking path.

    ``fair`` and ``reference`` are PROBABILITIES, and the type matters. The
    multiasset book previously passed its USD mark straight through, so
    decision_snapshots stored fair_probability = 4334.45 for SHEL.L — a price
    in a probability column. It degraded safely (fair == reference gives a
    Brier edge of exactly zero, which is the intended "no forecast claimed")
    but it is not a thing anyone reading that table could interpret. A price
    book now passes fair=None and leaves reference at the neutral 0.5.
    """
    spec: object
    cell: str
    strategy_source: str
    side: OrderSide
    quantity: float
    price: float
    fee_usd: float
    fill_ref: str
    session_date: str
    bid: float = 0.0
    ask: float = 0.0
    fair: float | None = None
    reference: float = 0.5
    usd_per_point: float = 1.0
    usd_capital_per_unit: float | None = None
    rationale: str = ""


async def book_instrument_fill(db, gateway, simulator, fill: InstrumentFill,
                               *, log=None) -> bool:
    """Register an already-executed fill on the shared rails. Returns success.

    ONE path for both IBKR books. They previously had near-duplicate copies of
    this, and they drifted: the ETF book gained a real benchmark, an
    orderbook_snapshots write and a week-bucketed event family, while the
    multiasset book kept storing prices as probabilities, stamping every fill
    'synthetic', and using a per-instrument family that caps the book at 28
    families forever. Three bugs from one duplication, so there is now one copy.

    Runs AFTER the book's own venue-native accounting and cannot veto it: this
    is evidence, so a failure here must lose evidence rather than change what
    the book traded.
    """
    if db is None or getattr(gateway, "pnl_tracker", None) is None:
        return False
    spec, cell = fill.spec, fill.cell
    market_id = instrument_market_id(spec, cell)
    try:
        await db.execute(
            """INSERT OR IGNORE INTO markets
               (id, exchange, question, category, active,
                outcome_yes_price, outcome_no_price, last_updated)
               VALUES (?, 'ibkr', ?, ?, 0, 0, 0, datetime('now'))""",
            (market_id, f"{cell}: {spec.description or spec.symbol}",
             spec.asset_class or ""))
        order = prepare_instrument_order(
            spec, side=fill.side, quantity=fill.quantity, price=fill.price,
            is_live=False, strategy_source=fill.strategy_source, cell=cell,
            usd_per_point=fill.usd_per_point,
            usd_capital_per_unit=fill.usd_capital_per_unit)
        if order is None:
            return False
        # The real BBO. Without it _capture_decision finds no book, so the fill
        # cannot be judged executable and is stamped 'synthetic' — which is not
        # in graduation.credible_fill_evidence, making the decision uncountable
        # under require_executable_fills. Both books have a genuine bid/ask.
        if fill.bid > 0 and fill.ask > 0:
            await db.execute(
                """INSERT INTO orderbook_snapshots
                   (market_id, token_id, exchange, best_bid, best_ask, mid,
                    recorded_at)
                   VALUES (?, ?, 'ibkr', ?, ?, ?, datetime('now'))""",
                (market_id, spec.key, fill.bid, fill.ask,
                 (fill.bid + fill.ask) / 2))
        simulator.stage(order, order_id=fill.fill_ref)
        result = await gateway.submit(TradeIntent(
            signal=instrument_signal(
                spec, strategy_source=fill.strategy_source,
                mark=fill.reference, fair=fill.fair, side=fill.side, cell=cell,
                rationale=fill.rationale),
            market=instrument_market(
                spec, mark=fill.reference, cell=cell,
                event_family=instrument_event_family(
                    spec, cell, fill.session_date)),
            size_dollars=fill.quantity * fill.price * fill.usd_per_point,
            force_paper=True))
        if result.status not in ("paper", "filled", "partial"):
            if log is not None:
                log.error("ibkr.fill_not_booked", cell=cell, key=spec.key,
                          status=result.status, reason=result.reason,
                          detail="venue-native fill stands; ladder evidence lost")
            return False
        await record_ledger_event(
            db, market_id=market_id, kind="commission", token="YES",
            qty=fill.quantity, pnl=-float(fill.fee_usd), fees=0.0,
            is_paper=True, source_ref=f"{fill.fill_ref}:commission")
        return True
    except Exception as exc:  # noqa: BLE001 - evidence, never the money path
        if log is not None:
            log.error("ibkr.fill_booking_failed", cell=cell,
                      key=getattr(spec, "key", "?"), error=str(exc)[:200])
        return False
