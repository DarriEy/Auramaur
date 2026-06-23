"""P&L tracking with FIFO cost basis accounting."""

from __future__ import annotations

from datetime import date, datetime, timezone

import structlog

from auramaur.db.database import Database
from auramaur.exchange.models import Fill, LivePosition, OrderSide, TokenType
from config.settings import Settings

log = structlog.get_logger()


class PnLTracker:
    """Tracks realized and unrealized P&L from fills using weighted-average
    cost basis accounting.

    Every fill is persisted to the ``fills`` table.  A running cost basis
    is maintained in the ``cost_basis`` table so that realized P&L can be
    computed on sells and unrealized P&L can be derived from current prices.

    Queries are scoped to the current mode (paper vs live) via ``settings``
    so paper cost_basis rows don't bleed into live PnL reporting.
    """

    def __init__(self, db: Database, settings: Settings) -> None:
        self._db = db
        self._settings = settings

    def _mode_flag(self) -> int:
        return 0 if self._settings.is_live else 1

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
        cursor = await self._db.execute(
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
        fill_rowid = cursor.lastrowid

        # 2. Fetch current cost basis for the same paper/live mode AND token.
        # cost_basis is keyed by (market_id, is_paper, token); reading without
        # the is_paper filter would let paper fills consume a live row's basis,
        # and reading without the token filter would let a NO sell realize
        # against the YES side's basis when both coexist (post-#78 PKs).
        is_paper_flag = 1 if fill.is_paper else 0
        row = await self._db.fetchone(
            "SELECT size, avg_cost, total_cost, realized_pnl FROM cost_basis"
            " WHERE market_id = ? AND is_paper = ? AND token = ?",
            (fill.market_id, is_paper_flag, fill.token.value),
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
            pnl = (fill.price - old_avg_cost) * sell_size - fill.fee
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

            # Update daily_stats for REPORTING only (CLI / cockpit). NOTE:
            # total_pnl here is mode-MIXED (this runs for paper + live fills),
            # so it must NOT gate live trading — the daily-loss risk gate now
            # sources today's LIVE realized P&L from pnl_ledger (is_paper=0) via
            # PortfolioTracker.get_daily_pnl. peak_balance (drawdown) still uses
            # daily_stats, which is a balance tracker, not realized P&L.
            try:
                today = date.today().isoformat()
                await self._db.execute(
                    """INSERT INTO daily_stats (date, total_pnl, trades_count, wins, losses)
                       VALUES (?, ?, 1, ?, ?)
                       ON CONFLICT(date) DO UPDATE SET
                           total_pnl = total_pnl + excluded.total_pnl,
                           trades_count = trades_count + 1,
                           wins = wins + excluded.wins,
                           losses = losses + excluded.losses""",
                    (today, pnl, 1 if pnl > 0 else 0, 1 if pnl < 0 else 0),
                )
            except Exception as e:
                log.debug("pnl.daily_stats_error", error=str(e))

            # Unified realized-P&L ledger: one row per realization event,
            # idempotent on fill rowid (the backfill uses the same ref).
            from auramaur.broker.ledger import record_ledger_event
            await record_ledger_event(
                self._db,
                market_id=fill.market_id,
                kind="sell",
                token=fill.token.value,
                qty=sell_size,
                pnl=pnl,
                fees=fill.fee,
                is_paper=fill.is_paper,
                source_ref=f"fill:{fill_rowid}",
                realized_at=fill.timestamp.isoformat(),
            )

        # Clamp negative sizes to zero (shouldn't happen, but be safe)
        if new_size < 0:
            log.warning("pnl.negative_size_clamped", market_id=fill.market_id, size=new_size)
            new_size = 0.0
            new_total_cost = 0.0
            new_avg_cost = 0.0

        # 5. Upsert cost_basis — composite PK (market_id, is_paper) keeps
        # paper and live rows independent.
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """INSERT INTO cost_basis
               (market_id, token, token_id, size, avg_cost, total_cost, realized_pnl, is_paper, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(market_id, is_paper, token) DO UPDATE SET
                   token = excluded.token,
                   token_id = excluded.token_id,
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
                is_paper_flag,
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
        """Return ``(avg_cost, size)`` for a market's dominant position in the
        current mode.

        cost_basis is keyed by ``(market_id, token, is_paper)``, so a market
        may have more than one row (a real two-sided position, or — pre-v20 —
        casing-split duplicates). Pick deterministically: the largest open
        size, then the most recently updated. Returns ``(0.0, 0.0)`` if no
        cost basis exists.
        """
        row = await self._db.fetchone(
            "SELECT avg_cost, size FROM cost_basis WHERE market_id = ? AND is_paper = ?"
            " ORDER BY size DESC, updated_at DESC LIMIT 1",
            (market_id, self._mode_flag()),
        )
        if row is None:
            return 0.0, 0.0
        return float(row["avg_cost"]), float(row["size"])

    async def get_token_info(self, market_id: str) -> tuple[TokenType, str]:
        """Return ``(token_type, token_id)`` for a market's dominant position.

        Same deterministic disambiguation as :meth:`get_cost_basis` (largest
        open size, then most recently updated) and case-insensitive token
        parsing via :meth:`TokenType.from_str`, so it stays aligned with the
        row that method returns. Returns ``(TokenType.YES, "")`` if none.
        """
        row = await self._db.fetchone(
            "SELECT token, token_id FROM cost_basis WHERE market_id = ? AND is_paper = ?"
            " ORDER BY size DESC, updated_at DESC LIMIT 1",
            (market_id, self._mode_flag()),
        )
        if row is None:
            return TokenType.YES, ""
        return TokenType.from_str(row["token"]), row["token_id"] or ""

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
        """Realized P&L from the authoritative ``pnl_ledger`` for the current mode.

        Reporting read (feeds the displayed session P&L). Sources the unified
        ledger rather than summing ``cost_basis.realized_pnl``, which
        under-counted (zeroed rows + settlements with no basis row). If
        *start_date* (ISO ``YYYY-MM-DD``) is given, include realizations on or
        after that date — ``realized_at`` is the realization time, which is the
        correct basis for a since-date sum.
        """
        flag = self._mode_flag()
        if start_date:
            row = await self._db.fetchone(
                "SELECT COALESCE(SUM(pnl), 0) as total FROM pnl_ledger WHERE realized_at >= ? AND is_paper = ?",
                (start_date, flag),
            )
        else:
            row = await self._db.fetchone(
                "SELECT COALESCE(SUM(pnl), 0) as total FROM pnl_ledger WHERE is_paper = ?",
                (flag,),
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

    # ------------------------------------------------------------------
    # Attribution breakdowns
    # ------------------------------------------------------------------

    async def get_pnl_by_exchange(self) -> dict[str, float]:
        """Realized P&L grouped by venue, from the authoritative pnl_ledger.

        Uses the ledger's own ``venue`` column — no cost_basis→markets join, so
        it's immune to markets aging out (which left 'unknown'-bucketed rows).
        """
        flag = self._mode_flag()
        rows = await self._db.fetchall(
            """SELECT COALESCE(NULLIF(venue, ''), 'unknown') as exchange,
                      COALESCE(SUM(pnl), 0) as total
               FROM pnl_ledger WHERE is_paper = ? GROUP BY venue""",
            (flag,),
        )
        return {r["exchange"]: float(r["total"]) for r in (rows or [])}

    async def get_pnl_by_category(self) -> dict[str, float]:
        """Realized P&L grouped by category, from the authoritative pnl_ledger."""
        flag = self._mode_flag()
        rows = await self._db.fetchall(
            """SELECT COALESCE(NULLIF(category, ''), 'unknown') as category,
                      COALESCE(SUM(pnl), 0) as total
               FROM pnl_ledger WHERE is_paper = ? GROUP BY category""",
            (flag,),
        )
        return {r["category"]: float(r["total"]) for r in (rows or [])}
