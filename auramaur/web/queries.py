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
            "SELECT venue, detail, available, equity, fetched_at FROM venue_balances ORDER BY venue")
    except Exception:
        return {}
    now = datetime.now(timezone.utc)
    out: dict[str, dict] = {}
    for r in rows:
        try:
            age = max(0.0, (now - datetime.fromisoformat(r["fetched_at"])).total_seconds())
        except (ValueError, TypeError):  # unparseable or naive timestamp
            age = None
        out[r["venue"]] = {"detail": r["detail"], "age_seconds": age,
                           "available": r["available"], "equity": r["equity"],
                           "fetched_at": r["fetched_at"]}
    return out


async def venue_reconciliation(db: ReadOnlyDatabase) -> dict:
    """Compare the latest venue-native snapshot with the live DB projection."""
    try:
        summary = await db.fetchone(
            """SELECT COUNT(CASE WHEN redeemable=0 THEN 1 END) AS venue_count,
                      MAX(fetched_at) AS fetched_at,
                      COALESCE(SUM(CASE WHEN redeemable=0 THEN current_value ELSE 0 END),0)
                          AS venue_value
                 FROM venue_positions WHERE venue='polymarket'""")
        dbrow = await db.fetchone(
            """SELECT COUNT(*) AS db_count,
                      COALESCE(SUM(size * COALESCE(current_price,avg_price)),0) AS db_value
                 FROM portfolio WHERE exchange='polymarket' AND is_paper=0""")
        missing = await db.fetchall(
            """SELECT v.asset_id,v.title,v.outcome,v.size
                 FROM venue_positions v LEFT JOIN portfolio p
                   ON p.token_id=v.asset_id AND p.exchange='polymarket' AND p.is_paper=0
                WHERE v.venue='polymarket' AND v.redeemable=0
                  AND p.token_id IS NULL ORDER BY v.current_value DESC""")
        extra = await db.fetchall(
            """SELECT p.token_id,p.market_id,p.token,p.size
                 FROM portfolio p LEFT JOIN venue_positions v
                   ON v.asset_id=p.token_id AND v.venue='polymarket' AND v.redeemable=0
                WHERE p.exchange='polymarket' AND p.is_paper=0 AND v.asset_id IS NULL""")
        size_mismatches = await db.fetchall(
            """SELECT v.asset_id,v.title,v.outcome,v.size AS venue_size,p.size AS db_size
                 FROM venue_positions v JOIN portfolio p
                   ON p.token_id=v.asset_id AND p.exchange='polymarket' AND p.is_paper=0
                WHERE v.venue='polymarket' AND v.redeemable=0
                  AND ABS(v.size-p.size)>0.001
                ORDER BY ABS(v.size-p.size) DESC""")
    except Exception:
        return {"available": False, "in_sync": False, "venue_count": 0,
                "db_count": 0, "missing": [], "extra": [],
                "size_mismatches": [], "fetched_at": None}
    return {
        "available": summary["fetched_at"] is not None,
        "in_sync": not missing and not extra and not size_mismatches,
        "venue_count": summary["venue_count"], "db_count": dbrow["db_count"],
        "venue_value": summary["venue_value"], "db_value": dbrow["db_value"],
        "fetched_at": summary["fetched_at"],
        "missing": [dict(r) for r in missing], "extra": [dict(r) for r in extra],
        "size_mismatches": [dict(r) for r in size_mismatches],
    }


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


async def strategy_heartbeats(db: ReadOnlyDatabase) -> dict[str, dict]:
    """Per-strategy liveness written by the bot's run_pillar_once wrapper.

    Keyed by strategy name; tolerant of the table not existing yet (older
    deployments) — absence degrades to the pre-heartbeat behavior."""
    try:
        rows = await db.fetchall(
            """SELECT strategy, last_beat_at, status, entries, cycles,
                      interval_seconds, detail,
                      (julianday('now') - julianday(last_beat_at)) * 86400.0
                          AS age_seconds
               FROM strategy_heartbeats""")
    except Exception:
        return {}
    out: dict[str, dict] = {}
    for r in rows or []:
        out[r["strategy"]] = {
            "last_beat_at": r["last_beat_at"],
            "status": r["status"],
            "entries": r["entries"],
            "cycles": r["cycles"],
            "interval_seconds": r["interval_seconds"],
            "age_seconds": r["age_seconds"],
            "detail": r["detail"] or "",
        }
    return out


async def category_exposure(db: ReadOnlyDatabase, is_paper_flag: int) -> list[dict]:
    """Open exposure (position value at mark) per market category."""
    rows = await db.fetchall(
        """SELECT COALESCE(NULLIF(p.category, ''), NULLIF(m.category, ''),
                            'other') AS category,
                  COUNT(*) AS positions,
                  SUM(COALESCE(p.current_price, p.avg_price) * p.size) AS value
           FROM portfolio p
           LEFT JOIN markets m ON m.id = p.market_id
           WHERE p.is_paper = ?
           GROUP BY 1
           ORDER BY value DESC""",
        (is_paper_flag,),
    )
    return [
        {"category": r["category"], "positions": r["positions"], "value": r["value"] or 0.0}
        for r in rows
    ]


async def performance_history(db: ReadOnlyDatabase, days: int = 14) -> list[dict]:
    """Recent daily account marks for compact, honest trend context."""
    try:
        rows = await db.fetchall(
            """SELECT date,total_pnl,trades_count,wins,losses,max_drawdown,peak_balance
                 FROM daily_stats ORDER BY date DESC LIMIT ?""",
            (days,),
        )
    except Exception:
        return []
    return [dict(r) for r in reversed(rows)]


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


async def intelligence_eval_summary(db: ReadOnlyDatabase) -> list[dict]:
    """Per-arm resolved scorecard from the unified evidence view (schema
    v37). Mirrors EvaluationStore.summary() but through the read-only web
    connection; degrades to [] when the view does not exist yet so the
    panel omits itself rather than erroring."""
    try:
        rows = await db.fetchall(
            """WITH ranked AS (
                 SELECT u.*, ROW_NUMBER() OVER (
                   PARTITION BY u.arm,u.event_family
                   ORDER BY u.observed_at ASC,u.forecast_key ASC) AS rn
                 FROM unified_forecast_evidence u
                 WHERE u.stream='intelligence_eval' AND u.outcome IS NOT NULL
               )
               SELECT arm AS arm_name,model,COUNT(*) AS forecasts,
                      AVG((probability-outcome)*(probability-outcome)) AS brier,
                      AVG((market_probability-outcome)*
                          (market_probability-outcome)) AS market_brier,
                      SUM(abstained) AS abstains
                 FROM ranked WHERE rn=1
                GROUP BY arm,model ORDER BY brier ASC""")
    except Exception:
        return []
    return [
        {
            "arm": r["arm_name"], "model": r["model"],
            "forecasts": r["forecasts"],
            "brier": round(r["brier"], 4) if r["brier"] is not None else None,
            "market_brier": (round(r["market_brier"], 4)
                             if r["market_brier"] is not None else None),
            "abstains": r["abstains"] or 0,
        }
        for r in rows
    ]

