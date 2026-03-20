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

    async def get_kelly_multiplier(self, category: str) -> float:
        """Get the Kelly multiplier for a specific category. Returns 1.0 if unknown."""
        row = await self._db.fetchone(
            "SELECT kelly_multiplier FROM category_stats WHERE category = ?",
            (category,),
        )
        if row is None or row["kelly_multiplier"] is None:
            return 1.0
        return float(row["kelly_multiplier"])
