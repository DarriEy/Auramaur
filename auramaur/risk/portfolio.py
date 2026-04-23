"""Portfolio exposure tracking backed by the SQLite database."""

from __future__ import annotations

from datetime import date, datetime, timezone

import structlog

from auramaur.db.database import Database
from auramaur.exchange.models import ExitReason, OrderSide, Position, TokenType

log = structlog.get_logger()


class PortfolioTracker:
    """Reads and writes portfolio / daily-stats tables to provide exposure
    information consumed by the risk checks and the Kelly sizer."""

    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    async def get_positions(self, exchange: str | None = None) -> list[Position]:
        """Return all current open positions, optionally filtered by exchange."""
        if exchange:
            rows = await self.db.fetchall(
                "SELECT * FROM portfolio WHERE exchange = ?", (exchange,),
            )
        else:
            rows = await self.db.fetchall("SELECT * FROM portfolio")
        positions = []
        for row in rows:
            keys = row.keys()
            # token/token_id columns added in schema v6; handle older DBs
            token_str = row["token"] if "token" in keys else "YES"
            token_id = row["token_id"] if "token_id" in keys else ""
            exchange_name = row["exchange"] if "exchange" in keys else "polymarket"
            positions.append(Position(
                market_id=row["market_id"],
                exchange=exchange_name or "polymarket",
                side=OrderSide(row["side"]),
                size=row["size"],
                avg_price=row["avg_price"],
                current_price=row["current_price"] or 0.0,
                category=row["category"] or "",
                token=TokenType(token_str) if token_str else TokenType.YES,
                token_id=token_id or "",
            ))
        return positions

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

    # Weight applied to same-category positions that have no semantic
    # relationship.  Category alone is weak evidence of correlation.
    CATEGORY_WEIGHT = 0.3

    async def get_correlated_markets(self, market_id: str) -> float:
        """Return a *weighted* correlation score for *market_id*.

        Semantic relationships (strength >= 0.5) count at full weight.
        Same-category positions without a semantic link count at
        ``CATEGORY_WEIGHT`` (default 0.3) each.  This prevents a large
        number of unrelated same-category positions from blocking trades.
        """
        # Determine category
        row = await self.db.fetchone(
            "SELECT category FROM portfolio WHERE market_id = ?", (market_id,)
        )
        if row is None:
            row = await self.db.fetchone(
                "SELECT category FROM markets WHERE id = ?", (market_id,)
            )
        if row is None or not row["category"]:
            return 0.0

        category = row["category"]

        # --- Semantic relationships (full weight) ---
        rel_rows = await self.db.fetchall(
            """SELECT market_id_a, market_id_b, strength
               FROM market_relationships
               WHERE (market_id_a = ? OR market_id_b = ?) AND strength >= 0.5""",
            (market_id, market_id),
        )
        semantic_ids: dict[str, float] = {}
        for r in rel_rows:
            related_id = (
                r["market_id_b"] if r["market_id_a"] == market_id else r["market_id_a"]
            )
            # Only count if we hold a position in the related market
            pos_row = await self.db.fetchone(
                "SELECT market_id FROM portfolio WHERE market_id = ?",
                (related_id,),
            )
            if pos_row:
                # Use the relationship strength as weight (0.5–1.0)
                semantic_ids[related_id] = max(
                    semantic_ids.get(related_id, 0.0), float(r["strength"])
                )

        # --- Same-category positions (discounted weight) ---
        cat_rows = await self.db.fetchall(
            "SELECT market_id FROM portfolio WHERE category = ? AND market_id != ?",
            (category, market_id),
        )

        score = 0.0
        for r in cat_rows:
            mid = r["market_id"]
            if mid in semantic_ids:
                # Already counted at full semantic weight
                score += semantic_ids[mid]
            else:
                # Category-only: weak signal
                score += self.CATEGORY_WEIGHT

        # Add any semantic relationships outside the category (cross-category)
        for mid, strength in semantic_ids.items():
            if not any(r["market_id"] == mid for r in cat_rows):
                score += strength

        return round(score, 1)

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
        exchange: str | None = None,
    ) -> list[tuple[Position, ExitReason]]:
        """Check positions for exit conditions.

        If ``exchange`` is provided, only positions on that exchange are
        evaluated and ``discovery_client`` must correspond to that exchange.
        Multi-exchange callers should invoke this once per exchange.

        Returns a list of (position, reason) tuples for positions that
        should be exited.

        Exit hierarchy (evaluated in order):
        1. Stop-loss — hard floor, prevent catastrophic loss
        2. Trailing stop — lock in gains after a peak
        3. Profit target — take profits at threshold
        4. Edge erosion — price converging toward resolution boundary
        5. Time decay — market expiring soon with thin edge remaining
        """
        positions = await self.get_positions(exchange=exchange)
        exits: list[tuple[Position, ExitReason]] = []

        # Load peak prices for trailing stop calculation
        peak_prices = await self._get_peak_prices()

        for pos in positions:
            # Refresh current price from discovery client
            try:
                market = await discovery_client.get_market(pos.market_id)
                if market is None:
                    continue
                # Use the correct price for the token we actually hold
                if pos.token == TokenType.NO:
                    pos.current_price = (
                        market.outcome_no_price
                        if market.outcome_no_price > 0.01
                        else 1.0 - market.outcome_yes_price
                    )
                else:
                    pos.current_price = market.outcome_yes_price
            except Exception as e:
                log.debug("check_exits.price_error", market_id=pos.market_id, error=str(e))
                continue

            cost_basis = pos.avg_price * pos.size
            if cost_basis == 0:
                continue

            # Unrealized PnL percentage of cost basis
            pnl_pct = (pos.unrealized_pnl / cost_basis) * 100.0

            # Track peak PnL for trailing stop
            peak_pnl_pct = peak_prices.get(pos.market_id, pnl_pct)
            if pnl_pct > peak_pnl_pct:
                peak_pnl_pct = pnl_pct
                await self._update_peak_price(pos.market_id, pnl_pct)

            # 1. Stop-loss — hard floor
            if pnl_pct <= -settings.execution.stop_loss_pct:
                exits.append((pos, ExitReason.STOP_LOSS))
                continue

            # 2. Trailing stop — if position was up 12%+ but has dropped
            #    back 45%+ of peak gains, exit to lock in profit.
            if peak_pnl_pct >= 12.0:
                drawdown_from_peak = peak_pnl_pct - pnl_pct
                if drawdown_from_peak > peak_pnl_pct * 0.45:
                    log.info(
                        "exit.trailing_stop",
                        market_id=pos.market_id,
                        peak_pnl=round(peak_pnl_pct, 1),
                        current_pnl=round(pnl_pct, 1),
                    )
                    exits.append((pos, ExitReason.PROFIT_TARGET))
                    continue

            # 3. Profit target — time-aware: tighten near expiry, widen early
            profit_target = settings.execution.profit_target_pct  # default +50%
            if market.end_date is not None:
                end_dt = market.end_date if market.end_date.tzinfo else market.end_date.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                # Look up when position was first entered
                entry_row = await self.db.fetchone(
                    "SELECT MIN(timestamp) as first_entry FROM trades WHERE market_id = ?",
                    (pos.market_id,),
                )
                if entry_row and entry_row["first_entry"]:
                    entry_dt = datetime.fromisoformat(entry_row["first_entry"]).replace(tzinfo=timezone.utc)
                else:
                    entry_dt = now  # fallback: assume just entered

                total_lifetime = (end_dt - entry_dt).total_seconds()
                elapsed = (now - entry_dt).total_seconds()

                if total_lifetime > 0:
                    lifetime_fraction_remaining = 1.0 - (elapsed / total_lifetime)
                    if lifetime_fraction_remaining > 0.50:
                        # Early in position lifetime — let winners run
                        profit_target = 75.0
                    elif lifetime_fraction_remaining < 0.10:
                        # Near expiry — take what you can
                        profit_target = 25.0
                    # else: keep default (50%)

            if pnl_pct >= profit_target:
                exits.append((pos, ExitReason.PROFIT_TARGET))
                continue

            # 4. Edge erosion — price converging on resolution boundary
            #    measures how much room is left for the position to pay out
            if pos.side == OrderSide.BUY:
                # Bought YES: need price to go to 1.0
                remaining_upside = (1.0 - pos.current_price) * 100.0
                # True edge erosion: compare current profit margin to entry
                entry_upside = (1.0 - pos.avg_price) * 100.0
                erosion = entry_upside - remaining_upside if entry_upside > 0 else 0
            else:
                # Bought NO / sold YES: need price to go to 0.0
                remaining_upside = pos.current_price * 100.0
                entry_upside = pos.avg_price * 100.0
                erosion = entry_upside - remaining_upside if entry_upside > 0 else 0

            # Exit if remaining upside is tiny (near resolution boundary)
            if remaining_upside < settings.execution.edge_erosion_min_pct:
                exits.append((pos, ExitReason.EDGE_EROSION))
                continue

            # 5. Time decay — market expiring soon with thin edge
            if market.end_date is not None:
                end = market.end_date if market.end_date.tzinfo else market.end_date.replace(tzinfo=timezone.utc)
                hours_left = (end - datetime.now(timezone.utc)).total_seconds() / 3600.0
                if hours_left <= settings.execution.time_decay_hours and remaining_upside < 5.0:
                    exits.append((pos, ExitReason.TIME_DECAY))
                    continue

        return exits

    async def _get_peak_prices(self) -> dict[str, float]:
        """Load tracked peak PnL percentages for trailing stop."""
        try:
            rows = await self.db.fetchall(
                "SELECT market_id, peak_pnl_pct FROM position_peaks"
            )
            return {r["market_id"]: r["peak_pnl_pct"] for r in rows}
        except Exception:
            # Table might not exist yet — will be created on first write
            return {}

    async def _update_peak_price(self, market_id: str, peak_pnl_pct: float) -> None:
        """Track the highest PnL percentage reached for trailing stop."""
        try:
            await self.db.execute(
                """INSERT INTO position_peaks (market_id, peak_pnl_pct, updated_at)
                   VALUES (?, ?, datetime('now'))
                   ON CONFLICT(market_id) DO UPDATE SET
                       peak_pnl_pct = MAX(excluded.peak_pnl_pct, position_peaks.peak_pnl_pct),
                       updated_at = excluded.updated_at""",
                (market_id, peak_pnl_pct),
            )
            await self.db.commit()
        except Exception as e:
            log.debug("peak_price.update_error", error=str(e))

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    async def update_position(self, position: Position) -> None:
        """Insert or replace a position row in the portfolio table."""
        await self.db.execute(
            """
            INSERT INTO portfolio (market_id, exchange, side, size, avg_price, current_price, category, token, token_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(market_id) DO UPDATE SET
                exchange = excluded.exchange,
                side = excluded.side,
                size = excluded.size,
                avg_price = excluded.avg_price,
                current_price = excluded.current_price,
                category = excluded.category,
                token = excluded.token,
                token_id = excluded.token_id,
                updated_at = excluded.updated_at
            """,
            (
                position.market_id,
                position.exchange,
                position.side.value,
                position.size,
                position.avg_price,
                position.current_price,
                position.category,
                position.token.value,
                position.token_id,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        await self.db.commit()
        log.info(
            "portfolio.position_updated",
            market_id=position.market_id,
            exchange=position.exchange,
            side=position.side.value,
            size=position.size,
        )
