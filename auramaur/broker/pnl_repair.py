"""Repair helpers for realized P&L ledgers.

These functions are intentionally side-effect free unless ``dry_run`` is
False.  They let maintenance commands backfill the dollar P&L ledger from
already-recorded fills and rebuild category attribution from that ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from auramaur.db.database import Database

log = structlog.get_logger()


@dataclass
class MarketPnL:
    market_id: str
    category: str
    outcome: int
    pnl: float
    buy_cost: float
    sell_proceeds: float
    fees: float
    payout: float


@dataclass
class PnLRepairResult:
    scanned_markets: int = 0
    markets_with_fills: int = 0
    written_resolution_rows: int = 0
    total_pnl: float = 0.0
    by_category: dict[str, float] = field(default_factory=dict)


def _fill_side(side: str | None) -> str:
    return (side or "").upper()


def _fill_token(token: str | None) -> str:
    token = (token or "YES").upper()
    return token if token in ("YES", "NO") else "YES"


async def compute_market_resolution_pnl(
    db: Database,
    market_id: str,
    outcome: int,
    category: str = "",
    *,
    is_paper: int = 0,
) -> MarketPnL | None:
    """Compute realized resolution P&L for one market from persisted fills.

    Returns ``None`` if the market has no fills in the requested paper/live
    namespace.  The formula mirrors ``CalibrationTracker._persist_realized_pnl``:

    ``sell proceeds - buy cost - fees + winning-token payout``.
    """
    fills = await db.fetchall(
        """SELECT side, token, size, price, fee
           FROM fills
           WHERE market_id = ? AND is_paper = ?""",
        (market_id, is_paper),
    )
    if not fills:
        return None

    buy_cost = 0.0
    sell_proceeds = 0.0
    fees = 0.0
    net: dict[str, float] = {}

    for fill in fills:
        side = _fill_side(fill["side"])
        token = _fill_token(fill["token"])
        size = float(fill["size"] or 0.0)
        price = float(fill["price"] or 0.0)
        fees += float(fill["fee"] or 0.0)

        if side == "BUY":
            buy_cost += size * price
            net[token] = net.get(token, 0.0) + size
        elif side == "SELL":
            sell_proceeds += size * price
            net[token] = net.get(token, 0.0) - size

    payout = 0.0
    for token, size in net.items():
        if size <= 0.01:
            continue
        if (token == "YES" and outcome == 1) or (token == "NO" and outcome == 0):
            payout += size

    pnl = sell_proceeds - buy_cost - fees + payout
    return MarketPnL(
        market_id=market_id,
        category=category or "",
        outcome=outcome,
        pnl=pnl,
        buy_cost=buy_cost,
        sell_proceeds=sell_proceeds,
        fees=fees,
        payout=payout,
    )


async def backfill_resolution_pnl(
    db: Database,
    *,
    is_paper: int = 0,
    dry_run: bool = True,
) -> PnLRepairResult:
    """Backfill ``resolution_pnl`` from latest resolved calibration outcomes."""
    rows = await db.fetchall(
        """
        SELECT c.market_id,
               c.actual_outcome,
               COALESCE(NULLIF(c.category, ''), NULLIF(m.category, ''), '') AS category
        FROM calibration c
        LEFT JOIN markets m ON m.id = c.market_id
        WHERE c.actual_outcome IS NOT NULL
          AND c.id IN (
              SELECT MAX(id)
              FROM calibration
              WHERE actual_outcome IS NOT NULL
              GROUP BY market_id
          )
        """
    )

    result = PnLRepairResult(scanned_markets=len(rows))
    for row in rows:
        market_pnl = await compute_market_resolution_pnl(
            db,
            row["market_id"],
            int(row["actual_outcome"]),
            row["category"] or "",
            is_paper=is_paper,
        )
        if market_pnl is None:
            continue

        result.markets_with_fills += 1
        result.total_pnl += market_pnl.pnl
        cat = market_pnl.category or "other"
        result.by_category[cat] = result.by_category.get(cat, 0.0) + market_pnl.pnl

        if not dry_run:
            await db.execute(
                """INSERT OR REPLACE INTO resolution_pnl
                   (market_id, category, pnl, resolved_at)
                   VALUES (?, ?, ?, datetime('now'))""",
                (market_pnl.market_id, market_pnl.category, market_pnl.pnl),
            )
            result.written_resolution_rows += 1

    if not dry_run:
        await db.commit()

    log.info(
        "pnl.backfill_resolution_pnl",
        dry_run=dry_run,
        scanned_markets=result.scanned_markets,
        markets_with_fills=result.markets_with_fills,
        written=result.written_resolution_rows,
        total_pnl=round(result.total_pnl, 2),
    )
    return result


async def rebuild_category_stats_from_resolution_pnl(
    db: Database,
    *,
    dry_run: bool = True,
) -> PnLRepairResult:
    """Rebuild dollar attribution columns in ``category_stats``.

    This updates only dollar-P&L fields: ``total_pnl``, ``trade_count``,
    ``win_count``, and ``avg_edge``.  Existing Brier scores and Kelly
    multipliers remain owned by the feedback loop.
    """
    rows = await db.fetchall(
        """
        SELECT COALESCE(NULLIF(r.category, ''), NULLIF(m.category, ''), 'other') AS category,
               COUNT(*) AS trade_count,
               SUM(r.pnl) AS total_pnl,
               SUM(CASE WHEN r.pnl > 0 THEN 1 ELSE 0 END) AS win_count,
               AVG(COALESCE(s.edge, 0)) AS avg_edge
        FROM resolution_pnl r
        LEFT JOIN markets m ON m.id = r.market_id
        LEFT JOIN (
            SELECT market_id, edge
            FROM signals
            WHERE id IN (SELECT MAX(id) FROM signals GROUP BY market_id)
        ) s ON s.market_id = r.market_id
        GROUP BY 1
        """
    )

    result = PnLRepairResult(
        scanned_markets=sum(int(r["trade_count"] or 0) for r in rows),
        markets_with_fills=sum(int(r["trade_count"] or 0) for r in rows),
        total_pnl=sum(float(r["total_pnl"] or 0.0) for r in rows),
        by_category={r["category"]: float(r["total_pnl"] or 0.0) for r in rows},
    )

    if dry_run:
        return result

    for row in rows:
        await db.execute(
            """INSERT INTO category_stats
               (category, total_pnl, trade_count, win_count, avg_edge, updated_at)
               VALUES (?, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(category) DO UPDATE SET
                   total_pnl = excluded.total_pnl,
                   trade_count = excluded.trade_count,
                   win_count = excluded.win_count,
                   avg_edge = excluded.avg_edge,
                   updated_at = excluded.updated_at""",
            (
                row["category"],
                float(row["total_pnl"] or 0.0),
                int(row["trade_count"] or 0),
                int(row["win_count"] or 0),
                float(row["avg_edge"] or 0.0),
            ),
        )
    await db.commit()

    log.info(
        "pnl.rebuild_category_stats",
        categories=len(rows),
        total_pnl=round(result.total_pnl, 2),
    )
    return result
