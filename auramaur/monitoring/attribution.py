"""Performance attribution — tracks per-category PnL and suggests allocation weights."""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from auramaur.db.database import Database

log = structlog.get_logger()


class PerformanceAttributor:
    """Tracks per-category PnL and computes Kelly multipliers based on demonstrated edge."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def record_trade_result(
        self,
        category: str,
        pnl: float,
        edge: float,
    ) -> None:
        """Record a trade result for attribution tracking."""
        row = await self._db.fetchone(
            "SELECT * FROM category_stats WHERE category = ?", (category,)
        )
        now = datetime.now(timezone.utc).isoformat()

        if row is None:
            win = 1 if pnl > 0 else 0
            await self._db.execute(
                """INSERT INTO category_stats (category, total_pnl, trade_count, win_count, avg_edge, updated_at)
                   VALUES (?, ?, 1, ?, ?, ?)""",
                (category, pnl, win, edge, now),
            )
        else:
            new_count = row["trade_count"] + 1
            new_pnl = row["total_pnl"] + pnl
            new_wins = row["win_count"] + (1 if pnl > 0 else 0)
            new_avg_edge = ((row["avg_edge"] * row["trade_count"]) + edge) / new_count
            await self._db.execute(
                """UPDATE category_stats
                   SET total_pnl = ?, trade_count = ?, win_count = ?, avg_edge = ?, updated_at = ?
                   WHERE category = ?""",
                (new_pnl, new_count, new_wins, new_avg_edge, now, category),
            )
        await self._db.commit()
        log.debug("attribution.recorded", category=category, pnl=pnl)

    async def compute_kelly_multipliers(self) -> dict[str, float]:
        """Compute Kelly multipliers per category based on demonstrated edge.

        Categories with positive EV: multiplier up to 1.5x
        Categories with negative EV: multiplier down to 0.25x
        New categories with insufficient data: 1.0x (neutral)

        Returns:
            Dict mapping category to kelly multiplier.
        """
        rows = await self._db.fetchall(
            "SELECT * FROM category_stats WHERE trade_count >= 5"
        )

        multipliers: dict[str, float] = {}
        for row in rows:
            category = row["category"]
            win_rate = row["win_count"] / row["trade_count"] if row["trade_count"] > 0 else 0.5
            avg_pnl_per_trade = row["total_pnl"] / row["trade_count"] if row["trade_count"] > 0 else 0

            if avg_pnl_per_trade > 0:
                # Positive EV: scale up, cap at 1.5x
                # Linear scale: 1.0 at breakeven, 1.5 at strong +EV
                mult = min(1.5, 1.0 + win_rate - 0.5)
            else:
                # Negative EV: scale down, floor at 0.25x
                mult = max(0.25, 1.0 + (win_rate - 0.5) * 1.5)

            multipliers[category] = round(mult, 2)

            # Store the multiplier
            await self._db.execute(
                "UPDATE category_stats SET kelly_multiplier = ? WHERE category = ?",
                (multipliers[category], category),
            )

        await self._db.commit()
        log.info("attribution.multipliers", multipliers=multipliers)
        return multipliers

    async def get_category_stats(self) -> list[dict]:
        """Return all category stats as dicts."""
        rows = await self._db.fetchall(
            "SELECT * FROM category_stats ORDER BY total_pnl DESC"
        )
        return [dict(row) for row in rows]

    async def get_accuracy_and_kelly_maps(self) -> tuple[dict[str, float | None], dict[str, float]]:
        """Return accuracy and kelly multiplier maps for display."""
        cal_rows = await self._db.fetchall(
            """SELECT category,
                      COUNT(*) AS n,
                      SUM(CASE WHEN (predicted_prob > 0.5 AND actual_outcome = 1)
                               OR (predicted_prob <= 0.5 AND actual_outcome = 0)
                          THEN 1 ELSE 0 END) AS correct
               FROM calibration
               WHERE actual_outcome IS NOT NULL AND category != ''
               GROUP BY category"""
        )
        accuracy_map: dict[str, float | None] = {
            r["category"]: r["correct"] / r["n"] if r["n"] > 0 else None
            for r in cal_rows
        }

        stats_rows = await self._db.fetchall("SELECT category, kelly_multiplier FROM category_stats")
        kelly_map: dict[str, float] = {
            r["category"]: float(r["kelly_multiplier"] or 1.0) for r in stats_rows
        }

        return accuracy_map, kelly_map

    async def get_category_lookup(self) -> dict[str, str]:
        """Return market_id -> category mapping from markets and portfolio tables."""
        rows = await self._db.fetchall(
            """SELECT id, category FROM markets
               WHERE category IS NOT NULL AND category != ''"""
        )
        lookup = {r["id"]: r["category"] for r in rows}
        rows2 = await self._db.fetchall(
            """SELECT market_id, category FROM portfolio
               WHERE category IS NOT NULL AND category != ''"""
        )
        for r in rows2:
            lookup.setdefault(r["market_id"], r["category"])
        return lookup

    async def get_category_summary(self, *, is_live: bool = False) -> list[dict]:
        """Return per-category summary combining portfolio data with calibration stats."""
        paper_flag = 0 if is_live else 1
        portfolio_rows = await self._db.fetchall(
            """SELECT COALESCE(NULLIF(p.category, ''), NULLIF(m.category, ''), 'other') AS cat,
                      COUNT(*) AS positions,
                      SUM(p.size * p.avg_price) AS exposure,
                      SUM(
                          (COALESCE(p.current_price, p.avg_price)
                           - COALESCE(cb.avg_cost, p.avg_price)) * p.size
                      ) AS unrealized_pnl
               FROM portfolio p
               LEFT JOIN markets m ON p.market_id = m.id
               LEFT JOIN cost_basis cb ON p.market_id = cb.market_id
                                       AND cb.is_paper = p.is_paper
               WHERE p.is_paper = ? OR p.exchange != 'polymarket'
               GROUP BY cat
               ORDER BY exposure DESC""",
            (paper_flag,),
        )

        stats_rows = await self._db.fetchall("SELECT * FROM category_stats")
        stats_map = {r["category"]: dict(r) for r in stats_rows}

        cal_rows = await self._db.fetchall(
            """SELECT category,
                      COUNT(*) AS n,
                      SUM(CASE WHEN (predicted_prob > 0.5 AND actual_outcome = 1)
                               OR (predicted_prob <= 0.5 AND actual_outcome = 0)
                          THEN 1 ELSE 0 END) AS correct
               FROM calibration
               WHERE actual_outcome IS NOT NULL AND category != ''
               GROUP BY category"""
        )
        accuracy_map = {
            r["category"]: r["correct"] / r["n"] if r["n"] > 0 else None
            for r in cal_rows
        }

        result = []
        for row in portfolio_rows:
            cat = row["cat"]
            stats = stats_map.get(cat, {})
            result.append({
                "category": cat,
                "positions": row["positions"],
                "exposure": row["exposure"] or 0,
                "unrealized_pnl": row["unrealized_pnl"] or 0,
                "accuracy": accuracy_map.get(cat),
                "kelly_multiplier": stats.get("kelly_multiplier", 1.0),
            })

        return result

    async def get_kelly_multiplier(self, category: str) -> float:
        """Get the Kelly multiplier for a specific category. Returns 1.0 if unknown."""
        row = await self._db.fetchone(
            "SELECT kelly_multiplier FROM category_stats WHERE category = ?",
            (category,),
        )
        if row is None or row["kelly_multiplier"] is None:
            return 1.0
        return float(row["kelly_multiplier"])
