"""Auto-detect market resolutions and feed into calibration loop.

Periodically checks markets with pending predictions (calibration entries
where actual_outcome IS NULL), re-fetches them from the appropriate exchange,
and records the outcome when the market has resolved.  This closes the
feedback loop so the bot's Platt scaling improves over time.
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from auramaur.db.database import Database
from auramaur.exchange.models import Market
from auramaur.exchange.protocols import MarketDiscovery
from auramaur.nlp.calibration import CalibrationTracker

log = structlog.get_logger()


class ResolutionTracker:
    """Periodically checks if markets with pending predictions have resolved."""

    def __init__(
        self,
        db: Database,
        calibration: CalibrationTracker,
        discoveries: dict[str, MarketDiscovery],
    ) -> None:
        self._db = db
        self._calibration = calibration
        self._discoveries = discoveries

    async def check_resolutions(self) -> int:
        """Check all pending predictions for resolutions.

        Returns the count of newly resolved markets.
        """
        # Get market_ids with predictions but no resolution yet.
        # Join against markets table to get the exchange, falling back to
        # "polymarket" when the row is missing (e.g. paper-only runs).
        rows = await self._db.fetchall(
            """SELECT DISTINCT c.market_id, m.exchange
               FROM calibration c
               LEFT JOIN markets m ON c.market_id = m.id
               WHERE c.actual_outcome IS NULL"""
        )

        resolved_count = 0
        for row in rows:
            market_id = row["market_id"]
            exchange = row["exchange"] or "polymarket"

            discovery = self._discoveries.get(exchange)
            if discovery is None:
                # Try all discoveries — the market might be discoverable
                # through another exchange client.
                for ex_name, disc in self._discoveries.items():
                    try:
                        market = await disc.get_market(market_id)
                        if market is not None:
                            discovery = disc
                            exchange = ex_name
                            break
                    except Exception:
                        continue
                if discovery is None:
                    continue

            try:
                market = await discovery.get_market(market_id)
                if market is None:
                    continue

                outcome = self._detect_resolution(market, exchange)
                if outcome is None:
                    continue

                # Feed into calibration — triggers online Platt update
                await self._calibration.record_resolution(market_id, outcome)

                # Compute realized PnL from the portfolio position and clean up
                await self._settle_position(market_id, outcome)

                # Mark the market as inactive in our DB so other queries
                # (e.g. depth research, correlation) stop considering it.
                await self._db.execute(
                    "UPDATE markets SET active = 0, outcome_yes_price = ?, outcome_no_price = ? WHERE id = ?",
                    (market.outcome_yes_price, market.outcome_no_price, market_id),
                )
                await self._db.commit()

                log.info(
                    "resolution.detected",
                    market_id=market_id,
                    exchange=exchange,
                    outcome="YES" if outcome else "NO",
                    question=market.question[:60] if market.question else "",
                )
                resolved_count += 1

            except Exception as e:
                log.debug(
                    "resolution.check_error",
                    market_id=market_id,
                    error=str(e),
                )

        if resolved_count > 0:
            log.info("resolution.cycle_complete", newly_resolved=resolved_count)

        return resolved_count

    # ------------------------------------------------------------------
    # Resolution detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_resolution(market: Market, exchange: str) -> bool | None:
        """Determine if a market has resolved and what the outcome was.

        Returns:
            True  — resolved YES
            False — resolved NO
            None  — not yet resolved / ambiguous
        """
        yes_price = market.outcome_yes_price

        # Price convergence is the strongest signal.  Resolved markets go
        # to ~0 or ~1.  Check this FIRST because some APIs (Gamma/Polymarket)
        # keep the ``active`` flag True even after a market has fully resolved
        # and prices have converged.
        if yes_price >= 0.95:
            return True  # Resolved YES
        if yes_price <= 0.05:
            return False  # Resolved NO

        # Market still trading at a non-extreme price and flagged active
        # by the exchange — genuinely unresolved.
        if market.active:
            return None

        # Kalshi-specific: markets have an explicit status field.  If the
        # market is settled/finalized but the price didn't clearly converge
        # (rare edge case for multi-outcome events), use price as tiebreak.
        if exchange == "kalshi":
            status = getattr(market, "status", None)
            if status in ("settled", "finalized"):
                return yes_price > 0.5

        # Market inactive but price ambiguous — can't determine cleanly.
        return None

    # ------------------------------------------------------------------
    # Position settlement
    # ------------------------------------------------------------------

    async def _settle_position(self, market_id: str, outcome: bool) -> None:
        """Clean up the portfolio entry and log realized PnL for a resolved market.

        Computes the realized profit/loss from the position, records it in
        the daily_stats table, and removes the portfolio row.
        """
        pos_row = await self._db.fetchone(
            "SELECT * FROM portfolio WHERE market_id = ?",
            (market_id,),
        )
        if pos_row is None:
            # No position — we only had a calibration prediction, not a trade.
            return

        entry_price = pos_row["avg_price"]
        size = pos_row["size"]
        side = pos_row["side"]
        token = pos_row.get("token", "YES")

        # Settlement price: YES resolves to $1, NO resolves to $0
        if token == "NO":
            # Holding NO tokens: worth $1 if outcome is NO, $0 if YES
            exit_price = 0.0 if outcome else 1.0
        else:
            # Holding YES tokens: worth $1 if outcome is YES, $0 if NO
            exit_price = 1.0 if outcome else 0.0

        if side == "BUY":
            pnl = (exit_price - entry_price) * size
        else:
            pnl = (entry_price - exit_price) * size

        # Record PnL in daily stats
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
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
            log.debug("resolution.daily_stats_error", error=str(e))

        # Update cost_basis realized PnL
        try:
            await self._db.execute(
                """UPDATE cost_basis
                   SET realized_pnl = realized_pnl + ?, size = 0, updated_at = datetime('now')
                   WHERE market_id = ?""",
                (pnl, market_id),
            )
        except Exception as e:
            log.debug("resolution.cost_basis_error", error=str(e))

        # Remove from portfolio
        await self._db.execute(
            "DELETE FROM portfolio WHERE market_id = ?",
            (market_id,),
        )

        # Remove peak tracking
        await self._db.execute(
            "DELETE FROM position_peaks WHERE market_id = ?",
            (market_id,),
        )

        await self._db.commit()

        log.info(
            "resolution.position_settled",
            market_id=market_id,
            side=side,
            token=token,
            entry_price=round(entry_price, 4),
            exit_price=round(exit_price, 4),
            size=round(size, 2),
            pnl=round(pnl, 2),
        )
