"""Measure what execution actually costs, from the broker's own fills.

Every cost figure in the IBKR viability screen is currently ASSUMED — the
commission schedule, the spread, the slippage — and those assumptions decide
the whole tradeable universe. At USD 800 notional the difference between 4bps
and 32bps is the difference between FX being the best instrument available and
nothing clearing its costs at all. A handful of deliberate small fills replaces
the guesses with measurements.

READ-ONLY BY CONSTRUCTION. This module asks IBKR what happened and records it.
It has no order path and must never grow one — placement is a separate,
separately-gated concern.

The output is shaped to feed `ibkr_edge_economics.round_trip_cost_bps()`
directly, so the screen can be re-run on measured numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

log = structlog.get_logger()


# IBKR secType -> the cost class the screen reasons in. Grouped by what
# actually drives cost: commission structure, minimum ticket, and whether a
# currency conversion is implied.
_SEC_TYPE_CLASS = {"CASH": "fx", "STK": "equity", "OPT": "option",
                   "FUT": "future", "BOND": "bond"}
_EU_CURRENCIES = {"EUR", "GBP", "CHF", "SEK", "NOK", "DKK"}
_ASIA_CURRENCIES = {"JPY", "HKD", "SGD", "AUD", "KRW"}


def venue_class(sec_type: str, currency: str) -> str:
    """Cost class for a fill: what schedule and frictions it is subject to.

    Region matters for equities and only for equities — a EUR-denominated
    stock carries a minimum ticket and a conversion that a US one does not,
    and GBP adds stamp duty on the buy. FX and options do not split this way.
    """
    base = _SEC_TYPE_CLASS.get((sec_type or "").upper(), "other")
    if base != "equity":
        return base
    cur = (currency or "").upper()
    if cur in _EU_CURRENCIES:
        return "eu_equity"
    if cur in _ASIA_CURRENCIES:
        return "asia_equity"
    return "us_equity"


@dataclass(frozen=True)
class VenueCosts:
    """Measured cost parameters for one venue class, in the shape
    `round_trip_cost_bps()` consumes."""

    venue_class: str
    fills: int
    commission_usd: float | None   # mean per fill; None = not yet reported
    commission_bps: float | None   # mean, as bps of that fill's notional
    commission_unknown: int        # fills whose commission report has not landed
    slippage_bps: float | None     # mean |filled - mid| / mid; None if no mids
    mids_available: int
    mean_notional: float


async def ingest_fills(db, ib, *, probe_label: str = "",
                       mids: dict[str, float] | None = None) -> int:
    """Record IBKR executions into `cost_observations`. Returns rows written.

    Idempotent on `execId`, so it is safe to run repeatedly — IBKR only
    returns the current session's executions, and re-ingesting must not
    double-count.

    `mids` maps orderRef -> mid price at submission. Supplied by the operator
    because it CANNOT be recovered afterwards: the fill records what was paid,
    never what was quoted at the moment of submission, and without it slippage
    is unrecoverable.
    """
    mids = mids or {}
    fills = await ib.reqExecutionsAsync()
    # UPSERT makes rowcount 1 for an update too, so "written" would count
    # refreshes as new rows. Classify against what is already stored.
    existing = {r["exec_id"] for r in
                (await db.fetchall("SELECT exec_id FROM cost_observations") or [])}
    written = 0
    updated = 0
    for fill in fills:
        ex, con = fill.execution, fill.contract
        report = getattr(fill, "commissionReport", None)
        commission = getattr(report, "commission", None) if report else None
        # IBKR sends the commission report as a separate message; a fill read
        # before it lands has none. Store NULL rather than 0 — a zero
        # commission is a claim, and a wrong one.
        if commission is not None and float(commission) == 0.0:
            commission = None
        shares = abs(float(ex.shares or 0))
        price = float(ex.price or 0)
        cls = venue_class(con.secType, con.currency)
        # UPSERT, not INSERT OR IGNORE. The commission report is a SEPARATE
        # async message, so a fill ingested moments after execution has none
        # — and with IGNORE it would stay NULL forever, since re-ingesting
        # would skip the row. Backfill it when it arrives, but never
        # overwrite a known commission with NULL.
        await db.execute(
            """INSERT INTO cost_observations
                 (exec_id, account, symbol, sec_type, exchange, currency,
                  venue_class, side, shares, price, notional, commission,
                  commission_currency, mid_at_submit, order_ref, probe_label,
                  filled_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(exec_id) DO UPDATE SET
                   commission = COALESCE(excluded.commission, commission),
                   commission_currency = CASE
                       WHEN excluded.commission IS NOT NULL
                       THEN excluded.commission_currency
                       ELSE commission_currency END,
                   mid_at_submit = COALESCE(excluded.mid_at_submit, mid_at_submit)""",
            (ex.execId, ex.acctNumber or "", con.symbol or "",
             con.secType or "", ex.exchange or "", con.currency or "",
             cls, (ex.side or "").upper(), shares, price, shares * price,
             float(commission) if commission is not None else None,
             getattr(report, "currency", "") if report else "",
             mids.get(ex.orderRef or ""), ex.orderRef or "", probe_label,
             str(fill.time)))
        if ex.execId in existing:
            updated += 1
        else:
            written += 1
            existing.add(ex.execId)
    await db.commit()
    log.info("execution_costs.ingested", seen=len(fills), written=written,
             refreshed=updated, probe_label=probe_label)
    return written


async def measure(db, venue: str | None = None) -> list[VenueCosts]:
    """Measured cost parameters per venue class.

    Commission is averaged in BOTH dollars and bps because the two answer
    different questions: dollars exposes a fixed minimum (the term that
    dominates at small size), bps exposes the marginal rate. Reporting only
    one of them is how a $1 floor gets mistaken for a 0.1% rate.
    """
    where, args = "", []
    if venue:
        where, args = "WHERE venue_class = ?", [venue]
    rows = await db.fetchall(
        f"""SELECT venue_class,
                   COUNT(*) AS fills,
                   AVG(commission) AS comm_usd,
                   SUM(CASE WHEN commission IS NULL THEN 1 ELSE 0 END) AS comm_unknown,
                   AVG(commission / NULLIF(notional,0)) * 10000 AS comm_bps,
                   AVG(notional) AS mean_notional,
                   SUM(CASE WHEN mid_at_submit IS NOT NULL THEN 1 ELSE 0 END) AS mids,
                   AVG(CASE WHEN mid_at_submit IS NOT NULL AND mid_at_submit > 0
                            THEN ABS(price - mid_at_submit) / mid_at_submit * 10000
                       END) AS slip_bps
              FROM cost_observations
              {where}
             GROUP BY venue_class ORDER BY venue_class""", tuple(args))
    out = []
    for r in rows or []:
        out.append(VenueCosts(
            venue_class=r["venue_class"], fills=int(r["fills"] or 0),
            # None, never 0.0: an unreported commission is UNKNOWN, and
            # rendering it as zero claims the trade was free.
            commission_usd=(float(r["comm_usd"])
                            if r["comm_usd"] is not None else None),
            commission_bps=(float(r["comm_bps"])
                            if r["comm_bps"] is not None else None),
            commission_unknown=int(r["comm_unknown"] or 0),
            slippage_bps=(float(r["slip_bps"]) if r["slip_bps"] is not None
                          else None),
            mids_available=int(r["mids"] or 0),
            mean_notional=float(r["mean_notional"] or 0.0),
        ))
    return out
