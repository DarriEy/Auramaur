"""Web-only breakdown queries, merged into each book's state by the broker.

These live here rather than in the cockpit because the TUI doesn't render
them; the shared core (positions/P&L) stays in ``monitoring.cockpit`` so the
headline numbers can never diverge between the two views.
"""

from __future__ import annotations

from auramaur.web.db import ReadOnlyDatabase


async def strategy_breakdown(db: ReadOnlyDatabase, is_paper_flag: int) -> list[dict]:
    """Realized P&L per strategy from the authoritative pnl_ledger."""
    rows = await db.fetchall(
        """SELECT COALESCE(NULLIF(strategy_source, ''), 'llm') AS strategy,
                  COUNT(*) AS entries,
                  SUM(pnl) AS pnl,
                  SUM(fees) AS fees
           FROM pnl_ledger
           WHERE is_paper = ?
           GROUP BY strategy
           ORDER BY pnl DESC""",
        (is_paper_flag,),
    )
    return [
        {"strategy": r["strategy"], "entries": r["entries"],
         "pnl": r["pnl"] or 0.0, "fees": r["fees"] or 0.0}
        for r in rows
    ]


async def category_exposure(db: ReadOnlyDatabase, is_paper_flag: int) -> list[dict]:
    """Open exposure (position value at mark) per market category."""
    rows = await db.fetchall(
        """SELECT COALESCE(NULLIF(category, ''), 'uncategorized') AS category,
                  COUNT(*) AS positions,
                  SUM(COALESCE(current_price, avg_price) * size) AS value
           FROM portfolio
           WHERE is_paper = ?
           GROUP BY category
           ORDER BY value DESC""",
        (is_paper_flag,),
    )
    return [
        {"category": r["category"], "positions": r["positions"], "value": r["value"] or 0.0}
        for r in rows
    ]


async def kraken_paper_positions(db: ReadOnlyDatabase) -> list[dict]:
    """The Kraken directional paper book — its own table, invisible to the
    portfolio query, so it must be surfaced explicitly (paper view only)."""
    rows = await db.fetchall(
        """SELECT strategy, pair, quantity, entry_price, peak_gain_pct, opened_at
           FROM kraken_paper_positions
           ORDER BY opened_at DESC"""
    )
    return [
        {"strategy": r["strategy"], "pair": r["pair"], "quantity": r["quantity"],
         "entry_price": r["entry_price"], "peak_gain_pct": r["peak_gain_pct"],
         "opened_at": r["opened_at"]}
        for r in rows
    ]
