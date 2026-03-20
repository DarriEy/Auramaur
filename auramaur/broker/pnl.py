"""P&L tracking with FIFO cost basis accounting."""

from __future__ import annotations

from datetime import date, datetime, timezone

import structlog

from auramaur.db.database import Database
from auramaur.exchange.models import Fill, LivePosition, OrderSide

log = structlog.get_logger()


class PnLTracker:
    """Tracks realized and unrealized P&L from fills using weighted-average
    cost basis accounting.

    Every fill is persisted to the ``fills`` table.  A running cost basis
    is maintained in the ``cost_basis`` table so that realized P&L can be
    computed on sells and unrealized P&L can be derived from current prices.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Fill recording
    # ------------------------------------------------------------------

    async def record_fill(self, fill: Fill) -> None:
        """Record a fill and update cost basis.

        BUY:  increases position size, adjusts weighted-average cost.
        SELL: decreases position size, realizes P&L at
              ``(sell_price - avg_cost) * size``.
        """
        # 1. Persist the fill
        await self._db.execute(
            """INSERT INTO fills
               (order_id, market_id, token_id, side, token, size, price, fee, is_paper, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                fill.order_id,
                fill.market_id,
                fill.token_id,
                fill.side.value,
                fill.token.value,
                fill.size,
                fill.price,
                fill.fee,
                1 if fill.is_paper else 0,
                fill.timestamp.isoformat(),
            ),
        )

        # 2. Fetch current cost basis (if any)
        row = await self._db.fetchone(
            "SELECT size, avg_cost, total_cost, realized_pnl FROM cost_basis WHERE market_id = ?",
            (fill.market_id,),
        )

        old_size: float = float(row["size"]) if row else 0.0
        old_avg_cost: float = float(row["avg_cost"]) if row else 0.0
        old_total_cost: float = float(row["total_cost"]) if row else 0.0
        realized_pnl: float = float(row["realized_pnl"]) if row else 0.0

        fill_cost = fill.price * fill.size

        if fill.side == OrderSide.BUY:
            # 3. BUY — increase position, recalculate weighted average
            new_size = old_size + fill.size
            new_total_cost = old_total_cost + fill_cost
            new_avg_cost = new_total_cost / new_size if new_size > 0 else 0.0
        else:
            # 4. SELL — realize P&L, reduce position
            sell_size = fill.size
            if sell_size > old_size:
                log.warning(
                    "pnl.sell_exceeds_position",
                    market_id=fill.market_id,
                    fill_size=fill.size,
                    position_size=old_size,
                    capped_to=old_size,
                )
                sell_size = old_size
            pnl = (fill.price - old_avg_cost) * sell_size
            realized_pnl += pnl
            new_size = old_size - sell_size
            # Average cost stays the same for remaining shares
            new_avg_cost = old_avg_cost if new_size > 0 else 0.0
            new_total_cost = new_avg_cost * new_size

            log.info(
                "pnl.realized",
                market_id=fill.market_id,
                pnl=round(pnl, 4),
                fill_price=fill.price,
                avg_cost=old_avg_cost,
                size=fill.size,
            )

        # Clamp negative sizes to zero (shouldn't happen, but be safe)
        if new_size < 0:
            log.warning("pnl.negative_size_clamped", market_id=fill.market_id, size=new_size)
            new_size = 0.0
            new_total_cost = 0.0
            new_avg_cost = 0.0

        # 5. Upsert cost_basis
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """INSERT INTO cost_basis
               (market_id, token, token_id, size, avg_cost, total_cost, realized_pnl, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(market_id) DO UPDATE SET
                   size = excluded.size,
                   avg_cost = excluded.avg_cost,
                   total_cost = excluded.total_cost,
                   realized_pnl = excluded.realized_pnl,
                   updated_at = excluded.updated_at""",
            (
                fill.market_id,
                fill.token.value,
                fill.token_id,
                new_size,
                new_avg_cost,
                new_total_cost,
                realized_pnl,
                now,
            ),
        )
        await self._db.commit()

        log.info(
            "pnl.fill_recorded",
            market_id=fill.market_id,
            side=fill.side.value,
            size=fill.size,
            price=fill.price,
            new_position_size=new_size,
            avg_cost=round(new_avg_cost, 4),
        )

    # ------------------------------------------------------------------
    # Cost basis queries
    # ------------------------------------------------------------------

    async def get_cost_basis(self, market_id: str) -> tuple[float, float]:
        """Return ``(avg_cost, size)`` for a market.

        Returns ``(0.0, 0.0)`` if no cost basis exists.
        """
        row = await self._db.fetchone(
            "SELECT avg_cost, size FROM cost_basis WHERE market_id = ?",
            (market_id,),
        )
        if row is None:
            return 0.0, 0.0
        return float(row["avg_cost"]), float(row["size"])

    # ------------------------------------------------------------------
    # P&L calculations
    # ------------------------------------------------------------------

    async def get_unrealized_pnl(self, positions: list[LivePosition]) -> float:
        """Sum ``(current_price - avg_cost) * size`` across all positions."""
        total = 0.0
        for pos in positions:
            avg_cost, size = await self.get_cost_basis(pos.market_id)
            if size > 0:
                total += (pos.current_price - avg_cost) * size
            else:
                # Fall back to the position's own avg_cost if no DB record
                total += pos.unrealized_pnl
        return total

    async def get_realized_pnl(self, start_date: str | None = None) -> float:
        """Sum ``realized_pnl`` from the cost_basis table.

        If *start_date* is provided (ISO format ``YYYY-MM-DD``), only include
        rows updated on or after that date.
        """
        if start_date:
            row = await self._db.fetchone(
                "SELECT COALESCE(SUM(realized_pnl), 0) as total FROM cost_basis WHERE updated_at >= ?",
                (start_date,),
            )
        else:
            row = await self._db.fetchone(
                "SELECT COALESCE(SUM(realized_pnl), 0) as total FROM cost_basis",
            )
        return float(row["total"]) if row else 0.0

    async def get_total_pnl(self, positions: list[LivePosition]) -> float:
        """Return realized + unrealized P&L."""
        realized = await self.get_realized_pnl()
        unrealized = await self.get_unrealized_pnl(positions)
        return realized + unrealized

    async def get_daily_pnl(self, positions: list[LivePosition]) -> float:
        """Today's realized P&L + change in unrealized.

        Daily realized comes from fills recorded today.  The unrealized
        component is the current mark-to-market of open positions.
        """
        today = date.today().isoformat()
        realized_today = await self.get_realized_pnl(start_date=today)
        unrealized = await self.get_unrealized_pnl(positions)
        return realized_today + unrealized
