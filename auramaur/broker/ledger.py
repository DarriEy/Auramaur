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

# Token-scoped entry-strategy resolution: credit the strategy that opened
# THIS side of the market. Market-level attribution (below) credits every
# realization on a market to its earliest entrant — with 40+ markets carrying
# trades from 2+ strategies, one strategy's win on the OPPOSITE token landed
# in another's cell (observed: an llm NO win booked into an agent arm's cell
# because the arm's YES entry was 3 minutes earlier). trades has no token
# column, so the token comes from the paired fill (same order_id), scoped to
# the realization's mode. Same-token stacking by multiple strategies still
# collapses to the earliest entrant — cost_basis is genuinely merged there.
_TOKEN_ENTRY_STRATEGY_SQL = """
    SELECT t.strategy_source FROM trades t
    JOIN fills f ON f.order_id = t.order_id
    WHERE f.market_id = ?
      AND UPPER(f.token) = UPPER(?)
      AND f.is_paper = ?
      AND t.strategy_source IS NOT NULL
      AND t.strategy_source != 'order_monitor'
      AND t.strategy_source != ''
    ORDER BY f.timestamp ASC LIMIT 1
"""

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


# Settlement-size sanity bound. A settlement realizes tokens the position
# tables say we hold — but those tables are not append-only truth: the
# 2026-07-23/24 all-paper window let instant-filled MM quote inventory (no
# trades/fills ancestry, blended ~mid price) persist into `portfolio`, and
# each such row settled into the cell of the earliest REAL entrant on the
# market (280 tokens booked against a 13.7-token entry: -$138.10 into
# term_structure's record). A named strategy is therefore only credited when
# the settled quantity is explainable by that mode's actually-placed BUYs;
# anything larger books to `phantom_unattributed` so a per-cell record can
# never again be moved by inventory nobody ordered. The bound is deliberately
# loose (stacked entries, partial exits, rounding all fit under 2x + 5) —
# the incident class overshoots by 5-20x.
_SETTLEMENT_QTY_MULT = 2.0
_SETTLEMENT_QTY_SLACK = 5.0
PHANTOM_STRATEGY = "phantom_unattributed"


async def _settlement_qty_explained(
    db: Database, market_id: str, qty: float, is_paper: bool,
) -> bool:
    """True when ``qty`` is coverable by this mode's placed (non-cancelled)
    BUY trades on the market. Order size bounds fill size from above, so a
    pending/partial order only ever loosens the bound — safe direction."""
    row = await db.fetchone(
        "SELECT COALESCE(SUM(size), 0) AS bought FROM trades "
        "WHERE market_id = ? AND side = 'BUY' AND is_paper = ? "
        "AND status NOT IN ('cancelled', 'rejected')",
        (market_id, 1 if is_paper else 0),
    )
    bought = float(row["bought"]) if row else 0.0
    return qty <= bought * _SETTLEMENT_QTY_MULT + _SETTLEMENT_QTY_SLACK


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


async def _market_context(
    db: Database, market_id: str, token: str = "", is_paper: bool | None = None,
) -> tuple[str, str, str]:
    """Resolve ``(venue, category, strategy_source)`` for a realization.

    When the realization's ``token`` and mode are known (they always are on
    the ledger write path), the strategy is resolved for THAT side of the
    market first; the market-level earliest-entrant lookup is the fallback
    for legacy rows without a paired fill (kraken pairs, manual fills).
    """
    row = await db.fetchone(
        "SELECT exchange, category FROM markets WHERE id = ?", (market_id,)
    )
    if row is not None:
        venue = row["exchange"] or ""
        category = row["category"] or ""
    elif market_id.startswith("coinbase:"):
        return "coinbase", "coinbase_spot", "coinbase_paper"
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

    strategy = ""
    if token and is_paper is not None:
        srow = await db.fetchone(
            _TOKEN_ENTRY_STRATEGY_SQL,
            (market_id, token, 1 if is_paper else 0),
        )
        strategy = (srow["strategy_source"] if srow else "") or ""
    if not strategy:
        srow = await db.fetchone(
            f"SELECT {_ENTRY_STRATEGY_SQL} AS src",
            (market_id, market_id, market_id),
        )
        strategy = (srow["src"] if srow else "") or ""
    if not strategy:
        # Instrument-book synthetic ids ENCODE their owner:
        # instrument_market_id() builds "ibkr:<book>:<key>" and the pillar's
        # name is "ibkr_<book>_paper". Fallback only — a position entered
        # before the shared booking path has no entry trade to derive from,
        # and the sell's own trade row can land milliseconds after this
        # lookup (observed 2026-08-04: the XLV/VNQ drain exits realized as
        # strategy_source='' 0.2s before their trade rows committed).
        # Anything the entry lookups already resolve is untouched.
        parts = market_id.split(":")
        if len(parts) == 3 and parts[0] == "ibkr" and parts[1]:
            strategy = f"ibkr_{parts[1]}_paper"
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
        venue, category, strategy = await _market_context(
            db, market_id, token=token, is_paper=is_paper)
        if kind == "settlement" and strategy and not await _settlement_qty_explained(
                db, market_id, qty, is_paper):
            log.warning(
                "ledger.phantom_settlement_quarantined",
                market_id=market_id, qty=round(qty, 2),
                displaced_strategy=strategy, is_paper=is_paper,
            )
            strategy = PHANTOM_STRATEGY
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
