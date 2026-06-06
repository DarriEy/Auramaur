"""One-time reconciliation of stale Kalshi order rows.

Kalshi orders inserted at placement carry status ``pending``. The order
monitor is meant to flip them to a terminal status once the venue fills or
cancels them — but Kalshi historically wasn't monitored at all (its client
didn't expose ``_live_pending``), so those rows accumulated as permanent
``pending`` cruft even though the underlying orders long since executed,
cancelled, or expired on-venue.

This re-queries each pending Kalshi order against the venue and updates the
``trades`` row to match. It is ledger hygiene only — it does NOT record fills
or mutate P&L (realized P&L already flows from fills/sync/resolution), so it is
safe to run repeatedly and cannot double-count.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

log = structlog.get_logger()

# Statuses we consider terminal — anything here moves a row out of 'pending'.
_TERMINAL = {"filled", "cancelled", "expired", "rejected"}


@dataclass
class ReconcileResult:
    scanned: int = 0
    updated: int = 0
    still_pending: int = 0
    by_status: dict[str, int] = field(default_factory=dict)


async def reconcile_pending_kalshi_orders(
    db, exchange, *, dry_run: bool = True
) -> ReconcileResult:
    """Reconcile every ``pending`` live Kalshi trade against the venue.

    Args:
        db: Database handle.
        exchange: A KalshiClient exposing ``get_order_status(order_id)``.
        dry_run: When True (default) compute the changes but don't write.

    Returns:
        ReconcileResult summarising what was (or would be) updated.
    """
    rows = await db.fetchall(
        """SELECT id, order_id, market_id FROM trades
           WHERE exchange = 'kalshi' AND status = 'pending'
             AND order_id IS NOT NULL AND order_id != ''
             AND order_id NOT IN ('unknown', 'ERROR', 'SKIP_DUP', 'BLOCKED')"""
    )

    result = ReconcileResult(scanned=len(rows))
    for r in rows:
        oid = r["order_id"]
        size: float | None = None
        price: float | None = None
        try:
            status_result = await exchange.get_order_status(oid)
            status = status_result.status
            if status == "filled":
                if status_result.filled_size > 0:
                    size = status_result.filled_size
                if status_result.filled_price > 0:
                    price = status_result.filled_price
        except Exception:
            # The venue no longer knows this order (too old / pruned). It is by
            # definition no longer resting and can never fill — treat as expired.
            status = "expired"

        if status not in _TERMINAL:
            result.still_pending += 1
            continue

        result.updated += 1
        result.by_status[status] = result.by_status.get(status, 0) + 1
        if not dry_run:
            if size is not None and price is not None:
                await db.execute(
                    "UPDATE trades SET status = ?, size = ?, price = ? WHERE id = ?",
                    (status, size, price, r["id"]),
                )
            else:
                await db.execute(
                    "UPDATE trades SET status = ? WHERE id = ?", (status, r["id"]),
                )

    if not dry_run:
        await db.commit()

    log.info(
        "kalshi.order_reconcile",
        scanned=result.scanned,
        updated=result.updated,
        still_pending=result.still_pending,
        by_status=result.by_status,
        dry_run=dry_run,
    )
    return result
