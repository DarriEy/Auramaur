"""Portfolio exposure tracking backed by the SQLite database."""

from __future__ import annotations

from datetime import date, datetime, timezone

import structlog

from auramaur.db.database import Database
from auramaur.exchange.models import ExitReason, OrderSide, Position

log = structlog.get_logger()


class PortfolioTracker:
    """Reads and writes portfolio / daily-stats tables to provide exposure
    information consumed by the risk checks and the Kelly sizer."""

    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    async def get_positions(self) -> list[Position]:
        """Return all current open positions."""
        rows = await self.db.fetchall("SELECT * FROM portfolio")
        return [
            Position(
                market_id=row["market_id"],
                side=OrderSide(row["side"]),
                size=row["size"],
                avg_price=row["avg_price"],
                current_price=row["current_price"] or 0.0,
                category=row["category"] or "",
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Category exposure
    # ------------------------------------------------------------------

    async def get_category_exposure(self) -> dict[str, float]:
        """Return the percentage of portfolio notional per category.

        Notional = size * avg_price for each position.
        """
        positions = await self.get_positions()
        if not positions:
            return {}

        category_notional: dict[str, float] = {}
        total = 0.0
        for pos in positions:
            notional = pos.size * pos.avg_price
            total += notional
            category_notional[pos.category] = category_notional.get(pos.category, 0.0) + notional

        if total == 0:
            return {}

        return {cat: (val / total) * 100.0 for cat, val in category_notional.items()}

    # ------------------------------------------------------------------
    # Correlated markets
    # ------------------------------------------------------------------

    async def get_correlated_markets(self, market_id: str) -> list[str]:
        """Return other open-position market IDs in the same category as
        *market_id*.  If *market_id* is not yet in the portfolio we look up
        the category from the markets table instead."""
        # Determine category
        row = await self.db.fetchone(
            "SELECT category FROM portfolio WHERE market_id = ?", (market_id,)
        )
        if row is None:
            row = await self.db.fetchone(
                "SELECT category FROM markets WHERE id = ?", (market_id,)
            )
        if row is None or not row["category"]:
            return []

        category = row["category"]
        rows = await self.db.fetchall(
            "SELECT market_id FROM portfolio WHERE category = ? AND market_id != ?",
            (category, market_id),
        )
        correlated = [r["market_id"] for r in rows]

        # Also check semantic relationships
        rel_rows = await self.db.fetchall(
            """SELECT market_id_a, market_id_b FROM market_relationships
               WHERE (market_id_a = ? OR market_id_b = ?) AND strength >= 0.5""",
            (market_id, market_id),
        )
        for r in rel_rows:
            related_id = r["market_id_b"] if r["market_id_a"] == market_id else r["market_id_a"]
            # Only count if we have an open position in the related market
            pos_row = await self.db.fetchone(
                "SELECT market_id FROM portfolio WHERE market_id = ?", (related_id,)
            )
            if pos_row and related_id not in correlated:
                correlated.append(related_id)

        return correlated

    # ------------------------------------------------------------------
    # PnL
    # ------------------------------------------------------------------

    async def get_daily_pnl(self) -> float:
        """Return today's realised PnL from the daily_stats table."""
        today = date.today().isoformat()
        row = await self.db.fetchone(
            "SELECT total_pnl FROM daily_stats WHERE date = ?", (today,)
        )
        return float(row["total_pnl"]) if row else 0.0

    # ------------------------------------------------------------------
    # Drawdown
    # ------------------------------------------------------------------

    async def get_drawdown(self) -> float:
        """Return current drawdown from peak as a percentage.

        Peak is stored in ``daily_stats.peak_balance``.  Current balance is
        ``peak_balance + today's unrealised PnL``.
        """
        # Get the most recent peak balance
        row = await self.db.fetchone(
            "SELECT peak_balance FROM daily_stats ORDER BY date DESC LIMIT 1"
        )
        if row is None or row["peak_balance"] is None or row["peak_balance"] == 0:
            return 0.0

        peak = float(row["peak_balance"])

        # Sum unrealised PnL across open positions
        positions = await self.get_positions()
        unrealised = sum(p.unrealized_pnl for p in positions)

        current = peak + unrealised
        if peak <= 0:
            return 0.0

        drawdown_pct = ((peak - current) / peak) * 100.0
        return max(drawdown_pct, 0.0)

    # ------------------------------------------------------------------
    # Exit checks
    # ------------------------------------------------------------------

    async def check_exits(
        self,
        settings,
        discovery_client,
    ) -> list[tuple[Position, ExitReason]]:
        """Check all positions for exit conditions.

        Returns a list of (position, reason) tuples for positions that
        should be exited.
        """
        positions = await self.get_positions()
        exits: list[tuple[Position, ExitReason]] = []

        for pos in positions:
            # Refresh current price from discovery client
            try:
                market = await discovery_client.get_market(pos.market_id)
                if market is None:
                    continue
                pos.current_price = market.outcome_yes_price
            except Exception as e:
                log.debug("check_exits.price_error", market_id=pos.market_id, error=str(e))
                continue

            cost_basis = pos.avg_price * pos.size
            if cost_basis == 0:
                continue

            # Unrealized PnL percentage of cost basis
            pnl_pct = (pos.unrealized_pnl / cost_basis) * 100.0

            # 1. Stop-loss
            if pnl_pct <= -settings.execution.stop_loss_pct:
                exits.append((pos, ExitReason.STOP_LOSS))
                continue

            # 2. Profit target
            if pnl_pct >= settings.execution.profit_target_pct:
                exits.append((pos, ExitReason.PROFIT_TARGET))
                continue

            # 3. Edge erosion: remaining price distance < threshold
            if pos.side == OrderSide.BUY:
                remaining_edge = (1.0 - pos.current_price) * 100.0
            else:
                remaining_edge = pos.current_price * 100.0

            if remaining_edge < settings.execution.edge_erosion_min_pct:
                exits.append((pos, ExitReason.EDGE_EROSION))
                continue

            # 4. Time decay: market resolves soon AND edge has shrunk below 5%
            if market.end_date is not None:
                end = market.end_date if market.end_date.tzinfo else market.end_date.replace(tzinfo=timezone.utc)
                hours_left = (end - datetime.now(timezone.utc)).total_seconds() / 3600.0
                if hours_left <= settings.execution.time_decay_hours and remaining_edge < 5.0:
                    exits.append((pos, ExitReason.TIME_DECAY))
                    continue

        return exits

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    async def update_position(self, position: Position) -> None:
        """Insert or replace a position row in the portfolio table."""
        await self.db.execute(
            """
            INSERT INTO portfolio (market_id, side, size, avg_price, current_price, category, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(market_id) DO UPDATE SET
                side = excluded.side,
                size = excluded.size,
                avg_price = excluded.avg_price,
                current_price = excluded.current_price,
                category = excluded.category,
                updated_at = excluded.updated_at
            """,
            (
                position.market_id,
                position.side.value,
                position.size,
                position.avg_price,
                position.current_price,
                position.category,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        await self.db.commit()
        log.info(
            "portfolio.position_updated",
            market_id=position.market_id,
            side=position.side.value,
            size=position.size,
        )
