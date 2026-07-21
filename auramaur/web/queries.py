"""Web-only breakdown queries, merged into each book's state by the broker.

These live here rather than in the cockpit because the TUI doesn't render
them; the shared core (positions/P&L) stays in ``monitoring.cockpit`` so the
headline numbers can never diverge between the two views.
"""

from __future__ import annotations

from datetime import datetime, timezone

from auramaur.web.db import ReadOnlyDatabase


async def venue_balances(db: ReadOnlyDatabase) -> dict[str, dict]:
    """Venue cash as recorded by the bot's balance recorder, with an honest
    age. The dashboard never asks the venues itself — it holds no credentials
    by construction — so a stopped bot shows as a growing age, not a lie.

    Degrades to ``{}`` when the table doesn't exist yet (bot not restarted
    since the schema gained ``venue_balances``): the panel just omits balances
    rather than taking the whole envelope into the error path.
    """
    try:
        rows = await db.fetchall(
            "SELECT venue, detail, fetched_at FROM venue_balances ORDER BY venue")
    except Exception:
        return {}
    now = datetime.now(timezone.utc)
    out: dict[str, dict] = {}
    for r in rows:
        try:
            age = max(0.0, (now - datetime.fromisoformat(r["fetched_at"])).total_seconds())
        except (ValueError, TypeError):  # unparseable or naive timestamp
            age = None
        out[r["venue"]] = {"detail": r["detail"], "age_seconds": age}
    return out


async def ibkr_paper_books(db: ReadOnlyDatabase) -> list[dict]:
    """Per-book IBKR paper summary: open positions, unrealized, latest mark.

    Degrades to ``[]`` before the multiasset schema exists (older DBs) —
    the venues panel simply shows the IBKR balance row alone.
    """
    try:
        rows = await db.fetchall(
            """SELECT p.book,
                      COUNT(*) AS positions,
                      SUM(COALESCE(p.unrealized_pnl_usd, 0)) AS unrealized,
                      (SELECT m.equity_usd FROM ibkr_paper_daily_marks m
                        WHERE m.book = p.book
                        ORDER BY m.mark_date DESC LIMIT 1) AS equity
               FROM ibkr_paper_positions p
               GROUP BY p.book ORDER BY p.book""")
    except Exception:
        return []
    return [
        {"book": r["book"], "positions": r["positions"],
         "unrealized": r["unrealized"] or 0.0,
         "equity": r["equity"]}
        for r in rows
    ]


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


async def local_llm_stats(db: ReadOnlyDatabase) -> dict:
    """Local Ollama tier health: last-24h calls by purpose plus distiller
    output. Book-independent, like ``venue_balances``.

    Degrades to ``{}`` when the tables don't exist yet (bot not restarted
    since schema v35) so the panel just omits the tier rather than erroring.
    """
    try:
        rows = await db.fetchall(
            """SELECT purpose,
                      COUNT(*) AS calls,
                      SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS ok,
                      SUM(CASE WHEN status IN ('parse_error', 'timeout',
                                               'request_error', 'api_error')
                               THEN 1 ELSE 0 END) AS errors,
                      CAST(AVG(duration_ms) AS INTEGER) AS avg_ms,
                      SUM(prompt_tokens) AS prompt_tokens,
                      SUM(output_tokens) AS output_tokens
               FROM local_llm_calls
               WHERE created_at >= datetime('now', '-24 hours')
               GROUP BY purpose ORDER BY purpose""")
        claims = await db.fetchone(
            """SELECT COUNT(*) AS n, MAX(created_at) AS latest
               FROM distilled_claims
               WHERE created_at >= datetime('now', '-24 hours')""")
    except Exception:
        return {}
    return {
        "purposes": {
            r["purpose"]: {
                "calls": r["calls"], "ok": r["ok"], "errors": r["errors"],
                "avg_ms": r["avg_ms"],
                "prompt_tokens": r["prompt_tokens"] or 0,
                "output_tokens": r["output_tokens"] or 0,
            }
            for r in rows
        },
        "claims_24h": claims["n"] if claims else 0,
        "last_claim_at": claims["latest"] if claims else None,
    }
