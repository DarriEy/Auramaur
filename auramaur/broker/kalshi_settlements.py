"""Kalshi settlements sweep — book venue-reported settlements into the ledger.

Kalshi realized P&L was NEVER recorded (discovered 2026-06-12): the
resolution tracker keyed off ``market.status``, a field no Market object
carried, and the position syncer reconciled settled positions out of the
``portfolio`` table before any P&L could be booked. Wins and losses simply
evaporated into the cash balance, unmeasured — which made the "is Kalshi
worth its fees?" question unanswerable.

The venue's ``GET /portfolio/settlements`` feed is authoritative and
complete, so one idempotent sweep serves as both the LIVE booking path
(runs every resolution cycle; new settlements get ledger rows) and the
HISTORICAL backfill (the first run walks the full feed). The ledger's
``source_ref`` uniqueness (``kalshi-settle:{ticker}:{settled_time}``) makes
re-runs no-ops.

Cost basis comes from our own records (cost_basis, falling back to filled
trades); a settlement whose cost we never recorded is reported but NOT
booked — a fabricated cost would poison the venue scorecard this sweep
exists to build. daily_stats is intentionally not touched: backfilled
settlements belong to their own (past) days, and pnl_ledger carries
``realized_at`` for that.
"""

from __future__ import annotations

import structlog

from auramaur.broker.ledger import record_ledger_event
from auramaur.db.database import Database

log = structlog.get_logger()

_MAX_PAGES = 25  # 200/page → up to 5000 settlements; far beyond current history


def _field(obj, *names, default=None):
    """Read the current fixed-point field first, with legacy compatibility."""
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if value not in (None, ""):
            return value
    return default


async def sweep_kalshi_settlements(
    db: Database, kalshi_client, *, dry_run: bool = False,
    max_pages: int = _MAX_PAGES,
) -> list[dict]:
    """Book venue-reported Kalshi settlements missing from the ledger.

    Returns the bookings performed (or planned, when ``dry_run``); entries
    whose cost basis is unknown carry ``"booked": False`` and a reason.
    """
    settlements = await _fetch_all_settlements(kalshi_client, max_pages)
    if not settlements:
        return []

    results: list[dict] = []
    for st in settlements:
        ticker = str(_field(st, "ticker", default="") or "")
        settled_time = _field(st, "settled_time")
        result = str(_field(st, "market_result", "result", default="") or "").lower()
        revenue_cents = float(_field(st, "revenue", default=0) or 0)
        yes_count = float(_field(st, "yes_count_fp", "yes_count", default=0) or 0)
        no_count = float(_field(st, "no_count_fp", "no_count", default=0) or 0)
        fee = float(_field(st, "fee_cost", default=0) or 0)
        if not ticker or settled_time is None:
            continue

        settled_iso = settled_time.isoformat()
        source_ref = f"kalshi-settle:{ticker}:{settled_iso}"
        if await db.fetchone(
                "SELECT 1 FROM pnl_ledger WHERE source_ref = ?", (source_ref,)):
            continue  # already booked (idempotent re-runs)

        token = "YES" if yes_count >= no_count else "NO"
        qty = max(yes_count, no_count)
        revenue = revenue_cents / 100.0
        if qty <= 0 or result not in ("yes", "no"):
            log.error("kalshi_settlements.schema_mismatch", ticker=ticker,
                      result=result, yes_count=yes_count, no_count=no_count)
            results.append({
                "ticker": ticker, "result": result, "qty": qty,
                "revenue": revenue, "pnl": None, "settled": settled_iso,
                "booked": False, "reason": "invalid settlement schema",
            })
            continue

        venue_cost = float(_field(
            st,
            "yes_total_cost_dollars" if token == "YES" else "no_total_cost_dollars",
            default=0,
        ) or 0)
        cost = await _cost_basis(db, ticker)
        if cost is None and venue_cost > 0:
            cost = venue_cost
        if cost is None:
            log.warning("kalshi_settlements.no_cost_basis", ticker=ticker,
                        revenue=revenue, settled=settled_iso)
            results.append({
                "ticker": ticker, "result": result, "qty": qty,
                "revenue": revenue, "pnl": None, "settled": settled_iso,
                "booked": False, "reason": "no cost basis on record",
            })
            continue

        pnl = round(revenue - cost - fee, 4)
        results.append({
            "ticker": ticker, "result": result, "qty": qty,
            "revenue": revenue, "fees": fee, "pnl": pnl, "settled": settled_iso,
            "booked": not dry_run, "reason": "",
        })
        if dry_run:
            continue

        await record_ledger_event(
            db, market_id=ticker, kind="settlement", token=token, qty=qty,
            pnl=pnl, fees=fee, is_paper=False, source_ref=source_ref,
            realized_at=settled_iso,
        )
        # Close out our records the way _settle_position does, minus
        # daily_stats (see module docstring).
        await db.execute(
            """UPDATE cost_basis SET realized_pnl = realized_pnl + ?, size = 0,
               updated_at = datetime('now')
               WHERE market_id = ? AND is_paper = 0""",
            (pnl, ticker),
        )
        await db.execute(
            "DELETE FROM portfolio WHERE market_id = ? AND is_paper = 0",
            (ticker,),
        )
        await db.commit()
        log.info("kalshi_settlements.booked", ticker=ticker, pnl=pnl,
                 settled=settled_iso)

    return results


async def _fetch_all_settlements(kalshi_client, max_pages: int) -> list:
    """Walk the paginated settlements feed via the client's SDK."""
    # The SDK client is lazily constructed; outside the bot's order paths
    # nothing has triggered it yet (the first CLI dry-run returned a silent
    # empty list this way — silence must not look like "nothing to settle").
    init = getattr(kalshi_client, "_init_api", None)
    if callable(init):
        try:
            init()
        except Exception as e:
            log.warning("kalshi_settlements.init_error", error=str(e))
            return []
    api = getattr(kalshi_client, "_portfolio_api", None)
    call = getattr(kalshi_client, "_call", None)
    if api is None or call is None:
        log.warning("kalshi_settlements.client_unavailable")
        return []
    out: list = []
    cursor = None
    for _ in range(max_pages):
        try:
            resp = await call(api.get_settlements, limit=200, cursor=cursor)
        except Exception as e:
            log.warning("kalshi_settlements.fetch_error", error=str(e))
            break
        batch = getattr(resp, "settlements", None) or []
        out.extend(batch)
        cursor = getattr(resp, "cursor", None)
        if not cursor or not batch:
            break
    return out


async def _cost_basis(db: Database, ticker: str) -> float | None:
    """Dollar cost of the settled position from our own records.

    cost_basis is authoritative when it still carries size; otherwise fall
    back to net filled BUY notional from trades. Returns None when neither
    source knows the position — the caller reports instead of guessing.
    """
    row = await db.fetchone(
        """SELECT size, avg_cost FROM cost_basis
           WHERE market_id = ? AND is_paper = 0 AND size > 0""",
        (ticker,),
    )
    if row is not None:
        return round(float(row["size"]) * float(row["avg_cost"]), 4)
    trow = await db.fetchone(
        """SELECT SUM(CASE WHEN side = 'BUY' THEN size * price
                           ELSE -size * price END) AS net
           FROM trades WHERE market_id = ? AND is_paper = 0
           AND status = 'filled'""",
        (ticker,),
    )
    if trow is not None and trow["net"] is not None and trow["net"] > 0:
        return round(float(trow["net"]), 4)
    return None
