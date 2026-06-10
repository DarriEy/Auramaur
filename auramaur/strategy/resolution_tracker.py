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
        # Resolve two groups of markets:
        #   1. Markets we have an open prediction for (calibration), and
        #   2. Markets we currently hold a position in (portfolio).
        # The portfolio arm is essential: a held position must settle when it
        # resolves even if no calibration prediction was ever recorded for it.
        # Kalshi/Crypto.com positions historically fell into exactly this gap —
        # their strategic-path predictions weren't logged, so they never entered
        # calibration, were never checked here, and their realized P&L was never
        # booked. Driving settlement off open positions makes it robust for any
        # venue regardless of how the prediction was (or wasn't) recorded.
        # The exchange is taken from the markets join for the calibration arm
        # and from the portfolio row directly for the position arm; MAX() picks
        # a non-NULL value when a market appears in both, falling back to
        # "polymarket" below when neither knows the venue.
        rows = await self._db.fetchall(
            """SELECT market_id, MAX(exchange) AS exchange FROM (
                   SELECT c.market_id AS market_id, m.exchange AS exchange
                   FROM calibration c
                   LEFT JOIN markets m ON c.market_id = m.id
                   WHERE c.actual_outcome IS NULL
                   UNION ALL
                   SELECT p.market_id AS market_id, p.exchange AS exchange
                   FROM portfolio p
                   WHERE p.size > 0
               )
               GROUP BY market_id"""
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

        Resolution is only declared when the exchange itself has signalled
        closure.  The earlier implementation checked price convergence
        before ``market.active`` and could mark a still-trading market at
        95%/5% as resolved — which then settled the portfolio position
        prematurely and fed a fake outcome into calibration.
        """
        # Kalshi exposes an explicit settlement status — trust it first.
        if exchange == "kalshi":
            status = getattr(market, "status", None)
            if status in ("settled", "finalized"):
                return market.outcome_yes_price > 0.5

        # For everything else, the exchange must have flagged the market
        # inactive (closed/resolved on Polymarket's side).  A still-active
        # market is by definition not resolved, regardless of price.
        if market.active:
            return None

        # Inactive + tightly converged price → resolution.  The threshold
        # is intentionally tight (0.99/0.01) to avoid declaring a winner
        # on markets that closed early or paused with non-trivial spread.
        yes_price = market.outcome_yes_price
        if yes_price >= 0.99:
            return True
        if yes_price <= 0.01:
            return False

        # Inactive but price ambiguous — wait for clearer signal.
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

        # aiosqlite.Row supports __getitem__ but not .get(); normalise to a
        # plain dict so we can safely use defaults for columns added in
        # later migrations (token, is_paper).
        pos = dict(pos_row)
        entry_price = pos["avg_price"]
        size = pos["size"]
        side = pos["side"]
        token = pos.get("token") or "YES"
        is_paper_flag = int(pos.get("is_paper", 1))

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

        # Update cost_basis realized PnL — scoped to the same paper/live mode
        # AND token as the settled position, so a paper resolution can't zero a
        # live row's cost basis and settling one side can't zero the other
        # (YES+NO can coexist since the #78 PKs).
        try:
            await self._db.execute(
                """UPDATE cost_basis
                   SET realized_pnl = realized_pnl + ?, size = 0, updated_at = datetime('now')
                   WHERE market_id = ? AND is_paper = ? AND token = ?""",
                (pnl, market_id, is_paper_flag, token),
            )
        except Exception as e:
            log.debug("resolution.cost_basis_error", error=str(e))

        # Unified realized-P&L ledger: settlement of the residual position.
        # Same source_ref scheme as the backfill, so re-running either is a
        # no-op rather than a double-count.
        from auramaur.broker.ledger import record_ledger_event
        await record_ledger_event(
            self._db,
            market_id=market_id,
            kind="settlement",
            token=token,
            qty=size,
            pnl=pnl,
            fees=0.0,
            is_paper=bool(is_paper_flag),
            source_ref=f"settle:{market_id}:{token}:{is_paper_flag}",
        )

        # Remove from portfolio — scoped by is_paper so a paper resolution
        # doesn't delete a live position for the same market (and vice versa).
        await self._db.execute(
            "DELETE FROM portfolio WHERE market_id = ? AND is_paper = ?",
            (market_id, is_paper_flag),
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
