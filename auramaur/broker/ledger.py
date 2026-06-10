"""Unified realized-P&L ledger.

One row per realization event — a SELL fill or a market settlement — carrying
venue, entry strategy, category, token, quantity, P&L and fees. This is THE
source of truth for realized money; the legacy sources it unifies
(``cost_basis.realized_pnl`` for sells, ``resolution_pnl`` for whole-market
oracle results) overlap for any market that was partially sold and then
resolved, which is why every consumer had to know which one to trust where.

Write paths (forward):
  * :meth:`PnLTracker.record_fill` — SELL branch (``kind='sell'``)
  * :meth:`ResolutionTracker._settle_position` — residual position at
    resolution (``kind='settlement'``)

``source_ref`` is globally unique by construction (``fill:<fills.id>`` /
``settle:<market>:<token>:<mode>``) and the column is UNIQUE, so writes are
idempotent (INSERT OR IGNORE) and the backfill can re-run safely alongside
forward writes.
"""

from __future__ import annotations

import structlog

from auramaur.db.database import Database

log = structlog.get_logger()

# Quote currencies that identify a Kraken spot pair used as a market_id
# (the directional book records fills with market_id == pair, and those
# pairs have no row in ``markets``).
_KRAKEN_QUOTES = ("USDC", "USDT", "ZUSD", "USD", "EUR", "ZEUR")

# Entry-strategy resolution — same semantics as
# auramaur.monitoring.attribution._entry_strategy_expr: credit the strategy
# that DECIDED to open the position (earliest non-order_monitor trade, then
# earliest such signal, then order_monitor as a last resort).
_ENTRY_STRATEGY_SQL = """COALESCE(
    (SELECT t.strategy_source FROM trades t
     WHERE t.market_id = ?
       AND t.strategy_source IS NOT NULL
       AND t.strategy_source != 'order_monitor'
     ORDER BY t.timestamp ASC LIMIT 1),
    (SELECT s.strategy_source FROM signals s
     WHERE s.market_id = ?
       AND s.strategy_source IS NOT NULL
       AND s.strategy_source != 'order_monitor'
     ORDER BY s.timestamp ASC LIMIT 1),
    (SELECT t.strategy_source FROM trades t
     WHERE t.market_id = ?
       AND t.strategy_source IS NOT NULL
     ORDER BY t.timestamp ASC LIMIT 1),
    ''
)"""


def _looks_like_kraken_pair(market_id: str) -> bool:
    return (
        market_id.isupper()
        and market_id.isalnum()
        and any(market_id.endswith(q) for q in _KRAKEN_QUOTES)
    )


def _looks_like_us_ticker(market_id: str) -> bool:
    """Bare 1-5 letter uppercase symbol (checked AFTER the kraken-pair test,
    so 6+ char pairs like XBTUSDC never reach this)."""
    return market_id.isupper() and market_id.isalpha() and 1 <= len(market_id) <= 5


async def _market_context(db: Database, market_id: str) -> tuple[str, str, str]:
    """Resolve ``(venue, category, strategy_source)`` for a realization."""
    row = await db.fetchone(
        "SELECT exchange, category FROM markets WHERE id = ?", (market_id,)
    )
    if row is not None:
        venue = row["exchange"] or ""
        category = row["category"] or ""
    elif _looks_like_kraken_pair(market_id):
        # Kraken directional book: market_id is the spot pair, no markets row.
        # 'kraken_spot' matches the calibration category used for its signals.
        return "kraken", "kraken_spot", "kraken_directional"
    elif _looks_like_us_ticker(market_id):
        # IBKR equity fill without a markets row (the pillars insert one, so
        # this is a safety net for direct/manual fills). Strategy still
        # resolves via the entry-strategy SQL below.
        venue, category = "ibkr", "ibkr_equity"
    else:
        venue, category = "", ""

    srow = await db.fetchone(
        f"SELECT {_ENTRY_STRATEGY_SQL} AS src",
        (market_id, market_id, market_id),
    )
    strategy = (srow["src"] if srow else "") or ""
    return venue, category, strategy


async def record_ledger_event(
    db: Database,
    *,
    market_id: str,
    kind: str,
    token: str,
    qty: float,
    pnl: float,
    fees: float,
    is_paper: bool,
    source_ref: str,
    realized_at: str | None = None,
) -> None:
    """Insert one realization row; idempotent on ``source_ref``.

    Never raises — a ledger failure must not break order/settlement flow.
    """
    try:
        venue, category, strategy = await _market_context(db, market_id)
        if realized_at is None:
            await db.execute(
                """INSERT OR IGNORE INTO pnl_ledger
                   (market_id, venue, category, strategy_source, kind, token,
                    qty, pnl, fees, is_paper, source_ref)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (market_id, venue, category, strategy, kind, token,
                 qty, pnl, fees, 1 if is_paper else 0, source_ref),
            )
        else:
            await db.execute(
                """INSERT OR IGNORE INTO pnl_ledger
                   (market_id, venue, category, strategy_source, kind, token,
                    qty, pnl, fees, is_paper, source_ref, realized_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (market_id, venue, category, strategy, kind, token,
                 qty, pnl, fees, 1 if is_paper else 0, source_ref, realized_at),
            )
        await db.commit()
    except Exception as e:  # pragma: no cover - defensive
        log.error("ledger.write_failed", market_id=market_id,
                  source_ref=source_ref, error=str(e))


# ----------------------------------------------------------------------
# Backfill — reconstruct history from fills + calibration outcomes
# ----------------------------------------------------------------------

async def backfill_ledger(db: Database) -> dict[str, int]:
    """Reconstruct ledger rows for past realizations.

    * Every SELL fill becomes a ``kind='sell'`` row: P&L from a
      weighted-average cost walk over that market's fills in timestamp
      order, per (market_id, token, is_paper) — the same accounting
      PnLTracker applies forward.
    * Every resolved market (calibration outcome) with tokens still held
      at resolution becomes a ``kind='settlement'`` row: residual size
      marked to the $1/$0 payout against its walked average cost.

    Idempotent: rows are keyed by source_ref and inserted OR IGNOREd, so
    re-running (or running alongside forward writes) cannot double-count.
    Positions still open and unresolved produce no rows.
    """
    outcomes = {
        r["market_id"]: r["actual_outcome"]
        for r in await db.fetchall(
            "SELECT market_id, actual_outcome FROM calibration "
            "WHERE actual_outcome IS NOT NULL"
        )
    }
    resolved_at = {
        r["market_id"]: r["resolved_at"]
        for r in await db.fetchall(
            "SELECT market_id, resolved_at FROM calibration "
            "WHERE actual_outcome IS NOT NULL AND resolved_at IS NOT NULL"
        )
    }

    fills = await db.fetchall(
        """SELECT id, market_id, side, token, size, price, fee, is_paper, timestamp
           FROM fills ORDER BY market_id, is_paper, timestamp, id"""
    )

    written = {"sell": 0, "settlement": 0}
    # Walk per (market_id, is_paper, token): running (size, avg_cost)
    basis: dict[tuple[str, int, str], tuple[float, float]] = {}

    for f in fills:
        token = f["token"] or "YES"
        key = (f["market_id"], int(f["is_paper"]), token)
        size, avg = basis.get(key, (0.0, 0.0))
        if f["side"] == "BUY":
            new_size = size + f["size"]
            avg = ((avg * size) + f["price"] * f["size"]) / new_size if new_size > 0 else 0.0
            basis[key] = (new_size, avg)
        else:
            sell_size = min(f["size"], size)
            pnl = (f["price"] - avg) * sell_size - (f["fee"] or 0.0)
            basis[key] = (size - sell_size, avg if size - sell_size > 0 else 0.0)
            await record_ledger_event(
                db,
                market_id=f["market_id"],
                kind="sell",
                token=token,
                qty=sell_size,
                pnl=pnl,
                fees=f["fee"] or 0.0,
                is_paper=bool(f["is_paper"]),
                source_ref=f"fill:{f['id']}",
                realized_at=f["timestamp"],
            )
            written["sell"] += 1

    # Settlement rows: residual tokens at resolution marked to payout.
    for (market_id, is_paper, token), (size, avg) in basis.items():
        if size <= 0.01 or market_id not in outcomes:
            continue
        outcome = outcomes[market_id]
        won = (token == "YES" and outcome == 1) or (token == "NO" and outcome == 0)
        payout = 1.0 if won else 0.0
        pnl = (payout - avg) * size
        await record_ledger_event(
            db,
            market_id=market_id,
            kind="settlement",
            token=token,
            qty=size,
            pnl=pnl,
            fees=0.0,
            is_paper=bool(is_paper),
            source_ref=f"settle:{market_id}:{token}:{is_paper}",
            realized_at=resolved_at.get(market_id),
        )
        written["settlement"] += 1

    log.info("ledger.backfill_done", **written)
    return written
