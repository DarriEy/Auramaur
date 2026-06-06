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

        # category_stats.total_pnl only covers oracle-resolved markets
        # (resolution_pnl). Positions closed by *selling* book their realized
        # P&L solely in cost_basis.realized_pnl, which this view used to drop —
        # under-reporting the loss. Sum that bucket per category, excluding
        # already-resolved markets so it isn't double-counted, mirroring the
        # union the cockpit and get_strategy_summary already use.
        resolved_exclusion = (
            "AND cb.market_id NOT IN (SELECT market_id FROM resolution_pnl)"
            if is_live else ""
        )
        sold_rows = await self._db.fetchall(
            f"""SELECT COALESCE(NULLIF(p.category, ''), NULLIF(m.category, ''), 'other') AS cat,
                      SUM(cb.realized_pnl) AS realized_sold
               FROM cost_basis cb
               LEFT JOIN portfolio p ON cb.market_id = p.market_id
                                     AND p.is_paper = cb.is_paper
               LEFT JOIN markets m ON cb.market_id = m.id
               WHERE cb.is_paper = ? {resolved_exclusion}
               GROUP BY cat""",
            (paper_flag,),
        )
        sold_map = {r["cat"]: (r["realized_sold"] or 0) for r in sold_rows}

        result = []
        seen: set[str] = set()
        for row in portfolio_rows:
            cat = row["cat"]
            seen.add(cat)
            stats = stats_map.get(cat, {})
            result.append({
                "category": cat,
                "positions": row["positions"],
                "exposure": row["exposure"] or 0,
                "unrealized_pnl": row["unrealized_pnl"] or 0,
                "realized_pnl": (stats.get("total_pnl", 0) or 0) + sold_map.get(cat, 0),
                "accuracy": accuracy_map.get(cat),
                "kelly_multiplier": stats.get("kelly_multiplier", 1.0),
            })

        # Categories that exist only as sold-to-close positions (no open
        # portfolio row) would otherwise vanish along with their realized P&L.
        for cat, realized_sold in sold_map.items():
            if cat in seen or not realized_sold:
                continue
            stats = stats_map.get(cat, {})
            result.append({
                "category": cat,
                "positions": 0,
                "exposure": 0,
                "unrealized_pnl": 0,
                "realized_pnl": (stats.get("total_pnl", 0) or 0) + realized_sold,
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

    async def get_venue_summary(self, *, is_live: bool = False) -> list[dict]:
        """Per-venue (exchange) summary: exposure, unrealized and realized P&L.

        Mirrors ``get_category_summary`` but groups by exchange so each trading
        venue (polymarket, kalshi, cryptodotcom, …) is visible on its own.
        Without this, a venue whose settlements lag or whose realized P&L is
        small is indistinguishable from one that is genuinely flat — the exact
        blind spot that hid Kalshi's unbooked settlements. Realized P&L is
        attributed to a venue via ``markets.exchange`` (falling back to the
        portfolio row, then "polymarket"), combining the same two authoritative
        sources the category/strategy views use: ``resolution_pnl`` for
        oracle-resolved markets (live) and ``cost_basis.realized_pnl`` for
        positions closed by selling.
        """
        paper_flag = 0 if is_live else 1
        portfolio_rows = await self._db.fetchall(
            """SELECT COALESCE(NULLIF(p.exchange, ''), 'polymarket') AS venue,
                      COUNT(*) AS positions,
                      SUM(p.size * p.avg_price) AS exposure,
                      SUM(
                          (COALESCE(p.current_price, p.avg_price)
                           - COALESCE(cb.avg_cost, p.avg_price)) * p.size
                      ) AS unrealized_pnl
               FROM portfolio p
               LEFT JOIN cost_basis cb ON p.market_id = cb.market_id
                                       AND cb.is_paper = p.is_paper
               WHERE (p.is_paper = ? OR p.exchange != 'polymarket') AND p.size > 0
               GROUP BY venue""",
            (paper_flag,),
        )

        # resolution_pnl is live-only; for the rest use cost_basis, excluding
        # already-resolved markets so they aren't double-counted (mirrors
        # get_strategy_summary).
        resolved_union = (
            "SELECT market_id, pnl AS realized FROM resolution_pnl UNION ALL "
            if is_live else ""
        )
        resolved_exclusion = (
            "AND market_id NOT IN (SELECT market_id FROM resolution_pnl)"
            if is_live else ""
        )
        realized_rows = await self._db.fetchall(
            f"""SELECT venue,
                       COALESCE(SUM(realized), 0) AS realized_pnl,
                       SUM(CASE WHEN realized > 0 THEN 1 ELSE 0 END) AS wins,
                       SUM(CASE WHEN realized < 0 THEN 1 ELSE 0 END) AS losses,
                       COUNT(*) AS resolved_count
                FROM (
                    SELECT mp.realized,
                           COALESCE(
                               (SELECT m.exchange FROM markets m WHERE m.id = mp.market_id),
                               (SELECT p.exchange FROM portfolio p
                                WHERE p.market_id = mp.market_id LIMIT 1),
                               'polymarket'
                           ) AS venue
                    FROM (
                        {resolved_union}
                        SELECT market_id, realized_pnl AS realized FROM cost_basis
                        WHERE is_paper = ? {resolved_exclusion}
                    ) mp
                )
                GROUP BY venue""",
            (paper_flag,),
        )
        realized_map = {r["venue"]: dict(r) for r in realized_rows}

        result: list[dict] = []
        seen: set[str] = set()
        for row in portfolio_rows:
            venue = row["venue"]
            seen.add(venue)
            rz = realized_map.get(venue, {})
            result.append({
                "venue": venue,
                "positions": row["positions"],
                "exposure": row["exposure"] or 0,
                "unrealized_pnl": row["unrealized_pnl"] or 0,
                "realized_pnl": rz.get("realized_pnl", 0) or 0,
                "wins": rz.get("wins", 0) or 0,
                "losses": rz.get("losses", 0) or 0,
                "resolved_count": rz.get("resolved_count", 0) or 0,
            })

        # Venues with only resolved/sold history (no open positions) would
        # otherwise vanish along with their realized P&L.
        for venue, rz in realized_map.items():
            if venue in seen:
                continue
            result.append({
                "venue": venue,
                "positions": 0,
                "exposure": 0,
                "unrealized_pnl": 0,
                "realized_pnl": rz.get("realized_pnl", 0) or 0,
                "wins": rz.get("wins", 0) or 0,
                "losses": rz.get("losses", 0) or 0,
                "resolved_count": rz.get("resolved_count", 0) or 0,
            })

        result.sort(key=lambda r: r["exposure"], reverse=True)
        return result

    @staticmethod
    def _entry_strategy_expr(alias: str) -> str:
        """SQL that resolves a market's *entry* strategy.

        Attribution must credit the strategy that DECIDED to open a position,
        not the last thing to touch it. The old version used the most-recent
        ``trades.strategy_source``, which meant exits — recorded by the order
        monitor as ``strategy_source = 'order_monitor'`` — stole every closed
        position's P&L from the strategy that actually opened it. Here we take
        the earliest non-``order_monitor`` trade, fall back to the earliest such
        signal, and only as a last resort accept ``order_monitor`` itself (for
        markets that genuinely have no other source).
        """
        return f"""COALESCE(
            (SELECT t.strategy_source FROM trades t
             WHERE t.market_id = {alias}.market_id
               AND t.strategy_source IS NOT NULL
               AND t.strategy_source != 'order_monitor'
             ORDER BY t.timestamp ASC LIMIT 1),
            (SELECT s.strategy_source FROM signals s
             WHERE s.market_id = {alias}.market_id
               AND s.strategy_source IS NOT NULL
               AND s.strategy_source != 'order_monitor'
             ORDER BY s.timestamp ASC LIMIT 1),
            (SELECT t.strategy_source FROM trades t
             WHERE t.market_id = {alias}.market_id
               AND t.strategy_source IS NOT NULL
             ORDER BY t.timestamp ASC LIMIT 1)
        )"""

    async def get_strategy_summary(self, *, is_live: bool = False) -> list[dict]:
        """Per-strategy performance, attributed to the strategy that opened each
        position.

        Two earlier blind spots are fixed here:

        * **Open positions were counted as completed trades.** ``cost_basis``
          carries a row for every position, open or closed, with
          ``realized_pnl = 0`` until it closes. Summing them blind made a
          strategy actively building a book (e.g. news_speed, 21 open / 1
          closed) read as "21 trades, 0 wins, $0.00". ``trade_count`` now counts
          only *closed* markets — ``cost_basis`` rows with ``size = 0`` plus
          oracle-resolved markets — and the open book is reported separately as
          ``open_positions`` / ``unrealized_pnl``.

        * **Exits stole the wins.** See :meth:`_entry_strategy_expr` — P&L is now
          credited to the entry strategy, not ``order_monitor``.

        Realized P&L still comes from the two authoritative sources:
        ``resolution_pnl`` (oracle-resolved, live) and ``cost_basis.realized_pnl``
        (closed by selling).
        """
        paper_flag = 0 if is_live else 1
        # resolution_pnl is live-only; for the rest use cost_basis, excluding
        # already-resolved markets so they aren't double-counted.
        resolved_union = (
            "SELECT market_id, pnl AS realized FROM resolution_pnl UNION ALL "
            if is_live else ""
        )
        resolved_exclusion = (
            "AND market_id NOT IN (SELECT market_id FROM resolution_pnl)"
            if is_live else ""
        )
        realized_strat = self._entry_strategy_expr("mp")
        # Realized arm: closed positions only (cost_basis.size = 0) + resolved.
        realized_rows = await self._db.fetchall(
            f"""SELECT strat AS strategy_source,
                       COUNT(*) AS trade_count,
                       COALESCE(SUM(realized), 0) AS total_pnl,
                       SUM(CASE WHEN realized > 0 THEN 1 ELSE 0 END) AS wins,
                       SUM(CASE WHEN realized < 0 THEN 1 ELSE 0 END) AS losses
                FROM (
                    SELECT mp.realized, {realized_strat} AS strat
                    FROM (
                        {resolved_union}
                        SELECT market_id, realized_pnl AS realized FROM cost_basis
                        WHERE is_paper = ? AND size = 0 {resolved_exclusion}
                    ) mp
                )
                WHERE strat IS NOT NULL
                GROUP BY strat""",
            (paper_flag,),
        )

        # Open arm: live/non-polymarket open positions, with unrealized P&L,
        # attributed to the same entry strategy.
        open_strat = self._entry_strategy_expr("p")
        open_rows = await self._db.fetchall(
            f"""SELECT strat AS strategy_source,
                       COUNT(*) AS open_positions,
                       COALESCE(SUM(unrealized), 0) AS unrealized_pnl
                FROM (
                    SELECT (COALESCE(p.current_price, p.avg_price)
                            - COALESCE(cb.avg_cost, p.avg_price)) * p.size AS unrealized,
                           p.market_id, {open_strat} AS strat
                    FROM portfolio p
                    LEFT JOIN cost_basis cb ON p.market_id = cb.market_id
                                            AND cb.is_paper = p.is_paper
                    WHERE (p.is_paper = ? OR p.exchange != 'polymarket') AND p.size > 0
                ) op
                WHERE strat IS NOT NULL
                GROUP BY strat""",
            (paper_flag,),
        )

        merged: dict[str, dict] = {}
        for r in realized_rows:
            merged[r["strategy_source"]] = {
                "strategy_source": r["strategy_source"],
                "trade_count": r["trade_count"],
                "total_pnl": r["total_pnl"] or 0,
                "wins": r["wins"] or 0,
                "losses": r["losses"] or 0,
                "open_positions": 0,
                "unrealized_pnl": 0,
            }
        for r in open_rows:
            entry = merged.setdefault(r["strategy_source"], {
                "strategy_source": r["strategy_source"],
                "trade_count": 0, "total_pnl": 0, "wins": 0, "losses": 0,
                "open_positions": 0, "unrealized_pnl": 0,
            })
            entry["open_positions"] = r["open_positions"]
            entry["unrealized_pnl"] = r["unrealized_pnl"] or 0

        return sorted(merged.values(), key=lambda r: r["total_pnl"])
