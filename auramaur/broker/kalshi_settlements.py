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

The settlement record does NOT say which side we held: on the current API
``yes_count_fp``/``no_count_fp`` both mirror the same pair count, legacy
``revenue`` is always 0, and ``value`` is just the YES settlement price in
cents (verified against live payloads, 2026-07-19). Payout therefore comes
from records that know the side: our own ``cost_basis`` (token + size +
cost), or — when we have no record — the venue's per-side cost fields IF the
history is single-sided (exactly one side ever bought, so the held side is
unambiguous). A two-sided history with no own record is reported but NOT
booked — a guessed side would poison the venue scorecard this sweep exists
to build. daily_stats is intentionally not touched: backfilled settlements
belong to their own (past) days, and pnl_ledger carries ``realized_at``.
"""

from __future__ import annotations

import json
from datetime import datetime

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


def _iso(ts) -> str:
    """Normalize settled_time to a stable ISO string for source_ref identity.

    The typed SDK gave a datetime; the raw-JSON path gives a string like
    ``2026-07-19T14:00:00Z``. Both must produce the same ref for the same
    settlement or idempotency breaks across code versions.
    """
    if hasattr(ts, "isoformat"):
        return ts.isoformat()
    s = str(ts)
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return s


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
        yes_count = float(_field(st, "yes_count_fp", "yes_count", default=0) or 0)
        no_count = float(_field(st, "no_count_fp", "no_count", default=0) or 0)
        fee = float(_field(st, "fee_cost", default=0) or 0)
        if not ticker or settled_time is None:
            continue

        settled_iso = _iso(settled_time)
        source_ref = f"kalshi-settle:{ticker}:{settled_iso}"
        if await db.fetchone(
                "SELECT 1 FROM pnl_ledger WHERE source_ref = ?", (source_ref,)):
            continue  # already booked (idempotent re-runs)

        venue_count = max(yes_count, no_count)
        if venue_count <= 0 or result not in ("yes", "no"):
            log.error("kalshi_settlements.schema_mismatch", ticker=ticker,
                      result=result, yes_count=yes_count, no_count=no_count)
            results.append({
                "ticker": ticker, "result": result, "qty": venue_count,
                "payout": None, "pnl": None, "settled": settled_iso,
                "booked": False, "reason": "invalid settlement schema",
            })
            continue

        # Which side we HELD is not in the record (module docstring) — take it
        # from our own cost_basis, else from the venue's per-side costs when
        # the history is single-sided and the side therefore unambiguous.
        basis = ""
        own = await _own_basis(db, ticker)
        if own is not None:
            token, qty, cost = own
            basis = "own_records"
            if abs(qty - venue_count) > max(1.0, venue_count * 0.01):
                log.warning("kalshi_settlements.count_mismatch", ticker=ticker,
                            own_qty=qty, venue_count=venue_count)
        else:
            y_cost = float(_field(st, "yes_total_cost_dollars", default=0) or 0)
            n_cost = float(_field(st, "no_total_cost_dollars", default=0) or 0)
            if (y_cost > 0) != (n_cost > 0):
                token = "YES" if y_cost > 0 else "NO"
                qty = venue_count
                cost = y_cost if y_cost > 0 else n_cost
                basis = "venue_single_sided"
            else:
                log.warning("kalshi_settlements.held_side_unknown", ticker=ticker,
                            yes_cost=y_cost, no_cost=n_cost, settled=settled_iso)
                results.append({
                    "ticker": ticker, "result": result, "qty": venue_count,
                    "payout": None, "pnl": None, "settled": settled_iso,
                    "booked": False,
                    "reason": "held side unknowable (no own basis; two-sided venue history)",
                })
                continue

        payout = round(qty, 4) if token.lower() == result else 0.0
        pnl = round(payout - cost - fee, 4)
        results.append({
            "ticker": ticker, "result": result, "qty": qty, "payout": payout,
            "fees": fee, "pnl": pnl, "settled": settled_iso, "basis": basis,
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
    """Walk the paginated settlements feed as raw JSON dicts.

    Deliberately NOT the typed SDK call: kalshi-python 2.1.x's generated
    ``Settlement`` model predates the current API contract and its
    ``from_dict`` drops every field it doesn't know — exactly the
    ``market_result``/``*_fp``/cost fields the consumer above reads. Fetching
    through it silently blanked all settlements (100% schema_mismatch, zero
    ever booked). The raw ``*_without_preload_content`` variant returns the
    venue's JSON untouched, the same pattern ``sync_positions`` uses.
    """
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
    call_raw = getattr(kalshi_client, "_call_raw", None)
    if api is None or call_raw is None:
        log.warning("kalshi_settlements.client_unavailable")
        return []
    out: list = []
    cursor = None
    for _ in range(max_pages):
        try:
            kwargs = {"limit": 200}
            if cursor:
                kwargs["cursor"] = cursor
            raw = await call_raw(
                api.get_settlements_without_preload_content, **kwargs)
            data = json.loads(raw)
        except Exception as e:
            log.warning("kalshi_settlements.fetch_error", error=str(e))
            break
        batch = data.get("settlements") or []
        out.extend(batch)
        cursor = data.get("cursor")
        if not cursor or not batch:
            break
    return out


async def _own_basis(db: Database, ticker: str) -> tuple[str, float, float] | None:
    """(token, size, dollar_cost) of the settled position from our records.

    Only cost_basis qualifies: it is the one store that knows WHICH side we
    held, and payout depends on the side. (The old net-filled-trades fallback
    was removed with the payout rework — trades rows carry no token, so a
    cost without a side can no longer price a settlement.) Returns None when
    we have no record — the caller decides between the venue single-sided
    fallback and reporting.
    """
    row = await db.fetchone(
        """SELECT token, size, avg_cost FROM cost_basis
           WHERE market_id = ? AND is_paper = 0 AND size > 0""",
        (ticker,),
    )
    if row is None:
        return None
    size = float(row["size"])
    return (str(row["token"] or "YES"), size, round(size * float(row["avg_cost"]), 4))
