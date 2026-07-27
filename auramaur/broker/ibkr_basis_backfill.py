"""Give pre-existing IBKR paper positions a cost basis on the shared rails.

Increment 2 routed the IBKR books' fills through ``ExecutionGateway`` so their
realized P&L lands in ``pnl_ledger`` where ``GraduationLadder`` reads it. Only
fills made AFTER that change write a ``cost_basis`` row, and positions opened
before it have none.

That is not cosmetic. ``PnLTracker.record_fill`` caps a SELL at the tracked
position size (``pnl.sell_exceeds_position``), so a legacy position exits
against a size of zero: ``pnl = (price - 0) * 0 - fee`` books only the negative
commission and the entire gross P&L of that round trip is lost. As of
2026-07-27 that is nine open positions — three FX, six global_etf — and the
book's whole realized history to date is a single round trip at $0.00 gross, so
those nine would be most of the first real evidence the ladder ever sees.

The basis is stored in USD per unit (``avg_cost * multiplier * fx_to_usd``)
because that is the convention increment 2 established for every gateway rail:
``record_fill`` realizes ``(price - avg_cost) * size`` with no notion of a
contract multiplier or an FX rate, so a local-currency basis would book a
1000x-wrong number for the FX book.

Idempotent by construction: a position that already has a basis row is skipped,
never overwritten. Overwriting would silently restate an open position's cost.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BasisRow:
    """One planned cost_basis insert, in the gateway's USD convention."""
    market_id: str
    size: float
    usd_avg_cost: float
    description: str
    category: str

    @property
    def total_cost(self) -> float:
        return self.size * self.usd_avg_cost


def plan_multiasset_basis(positions, existing: set[str]) -> list[BasisRow]:
    """Rows for open ibkr_paper_positions that have no basis yet.

    ``positions`` are mappings with book, instrument_key, quantity, multiplier,
    fx_to_usd and avg_cost. ``existing`` is the set of market_ids already in
    cost_basis — membership means skip, so re-running is a no-op.
    """
    planned = []
    for row in positions:
        size = float(row["quantity"] or 0)
        multiplier = float(row["multiplier"] or 1) or 1.0
        fx = float(row["fx_to_usd"] or 1) or 1.0
        avg = float(row["avg_cost"] or 0)
        book = str(row["book"])
        key = str(row["instrument_key"])
        market_id = f"ibkr:{book}:{key}"
        if size <= 0 or avg <= 0 or market_id in existing:
            continue
        planned.append(BasisRow(
            market_id=market_id, size=size, usd_avg_cost=avg * multiplier * fx,
            description=f"{book}: {key}", category=book))
    return planned


def plan_etf_basis(positions, existing: set[str]) -> list[BasisRow]:
    """Rows for open ibkr_etf_positions. Multiplier and FX are always 1 here —
    the ETF arms trade USD-denominated US listings — so the USD basis is the
    stored average cost unchanged."""
    planned = []
    for row in positions:
        size = float(row["quantity"] or 0)
        avg = float(row["avg_cost"] or 0)
        alias = str(row["model_alias"])
        symbol = str(row["symbol"])
        market_id = f"ibkr:{alias}:{symbol}"
        if size <= 0 or avg <= 0 or market_id in existing:
            continue
        planned.append(BasisRow(
            market_id=market_id, size=size, usd_avg_cost=avg,
            description=f"{alias}: {symbol}", category=alias))
    return planned


async def load_existing_basis(db) -> set[str]:
    rows = await db.fetchall(
        "SELECT market_id FROM cost_basis WHERE market_id LIKE 'ibkr:%'")
    return {str(row["market_id"]) for row in rows or []}


async def apply_basis(db, planned: list[BasisRow]) -> int:
    """Insert the planned rows. Returns the number written.

    A `markets` row accompanies each: the ledger resolves venue and category
    from it, and without one the pnl_ledger entry lands venue-blank and cannot
    be scoped. active=0 — a continuously-priced broker instrument is not a
    resolvable prediction market.
    """
    written = 0
    for row in planned:
        await db.execute(
            """INSERT OR IGNORE INTO markets
               (id, exchange, question, category, active,
                outcome_yes_price, outcome_no_price, last_updated)
               VALUES (?, 'ibkr', ?, ?, 0, 0, 0, datetime('now'))""",
            (row.market_id, row.description, row.category))
        await db.execute(
            """INSERT OR IGNORE INTO cost_basis
               (market_id, token, token_id, size, avg_cost, total_cost,
                realized_pnl, is_paper, updated_at)
               VALUES (?, 'YES', '', ?, ?, ?, 0, 1, datetime('now'))""",
            (row.market_id, row.size, row.usd_avg_cost, row.total_cost))
        written += 1
    await db.commit()
    return written
