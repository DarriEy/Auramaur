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
        calibration: CalibrationTracker | None,
        discoveries: dict[str, MarketDiscovery],
        proxy_address: str = "",
    ) -> None:
        self._db = db
        self._calibration = calibration
        self._discoveries = discoveries
        self._proxy_address = proxy_address

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
        # The cost_basis arm catches held legs the portfolio view is missing:
        # a row dropped mid-sync, or a settled leg whose cost_basis size never
        # zeroed (so it resurrects every cycle). Driving settlement off actual
        # holdings — not just the portfolio snapshot — closes the gap where such
        # a leg never books and lingers at a $0 mark.
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
                   UNION ALL
                   SELECT cb.market_id AS market_id, m.exchange AS exchange
                   FROM cost_basis cb
                   LEFT JOIN markets m ON cb.market_id = m.id
                   WHERE cb.size > 0
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

        # Venue-truth sweep: the loop above can't settle markets the Gamma
        # API no longer returns (archived) or whose active flag is stale —
        # those positions sat at phantom marks forever (March sports legs
        # were marked $0 on BOTH sides, -100% "unrealized" on each). The
        # data-api per-token resolution state is authoritative and covers
        # exactly the rows the loop skipped.
        if self._proxy_address:
            try:
                settled = await self.settle_via_venue(self._proxy_address)
                resolved_count += len(settled)
            except Exception as e:
                # Warning, not debug: a silently-failing sweep looks identical
                # to "nothing to settle" and the stuck positions never clear.
                log.warning("resolution.venue_sweep_error", error=str(e))

        if resolved_count > 0:
            log.info("resolution.cycle_complete", newly_resolved=resolved_count)

        return resolved_count

    # ------------------------------------------------------------------
    # Venue-truth settlement (Polymarket data-api)
    # ------------------------------------------------------------------

    async def settle_via_venue(
        self, proxy_address: str, *, dry_run: bool = False,
    ) -> list[dict]:
        """Settle live Polymarket positions the venue itself says resolved.

        Polymarket's data-api reports per held token whether the position is
        redeemable (oracle-confirmed) or effectively resolved (curPrice
        pinned to 0/1, pending oracle), keyed by asset id == our
        portfolio.token_id. Settlement goes through the standard idempotent
        path (ledger source_ref dedupes), scoped to the exact token row so
        YES+NO pairs settle leg by leg.

        Returns the settlements performed (or planned, when ``dry_run``).
        """
        from auramaur.broker.redeemer import fetch_redeemable_positions

        if not proxy_address:
            return []
        venue_positions = await fetch_redeemable_positions(proxy_address)
        if not venue_positions:
            return []

        rows = await self._db.fetchall(
            """SELECT p.market_id, p.token, p.token_id, p.size, p.avg_price
               FROM portfolio p WHERE p.is_paper = 0 AND p.size > 0""")
        by_token_id = {r["token_id"]: dict(r) for r in (rows or [])
                       if r["token_id"]}

        settlements: list[dict] = []
        for vp in venue_positions:
            row = by_token_id.get(vp.asset_id)
            if row is None:
                continue
            # Already settled in a prior pass (the wallet still holds the
            # tokens until redemption, so the syncer can resurrect the row).
            # Only the ledger insert is idempotent — re-running settlement
            # would re-add the P&L to daily_stats and cost_basis.
            held_token = row["token"] or "YES"
            ref = f"settle:{row['market_id']}:{held_token}:0"
            if await self._db.fetchone(
                    "SELECT 1 FROM pnl_ledger WHERE source_ref = ?", (ref,)):
                continue
            # cur_price/is_winner are relative to OUR held side; map back to
            # the market-level YES outcome the settlement path expects.
            held_yes = (row["token"] or "YES") == "YES"
            outcome_yes = vp.is_winner if held_yes else (not vp.is_winner)
            exit_price = 1.0 if vp.is_winner else 0.0
            settlements.append({
                "market_id": row["market_id"],
                "title": vp.title,
                "token": row["token"] or "YES",
                "size": row["size"],
                "avg_price": row["avg_price"],
                "pnl": (exit_price - row["avg_price"]) * row["size"],
                "status": vp.status,
                "outcome_yes": outcome_yes,
            })
            if dry_run:
                continue
            await self._settle_position(
                row["market_id"], outcome_yes,
                token_scope=row["token"] or "YES", is_paper_scope=0,
            )
            await self._db.execute(
                "UPDATE markets SET active = 0 WHERE id = ?",
                (row["market_id"],),
            )
            await self._db.commit()
            log.info(
                "resolution.venue_settled",
                market_id=row["market_id"],
                token=row["token"],
                status=vp.status,
                title=vp.title[:60],
            )
        return settlements

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
        # (market.status/.result only exist as of the Market-model fields
        # added 2026-06-12; before that getattr always returned None and no
        # Kalshi position ever settled.) Prefer the venue's explicit result
        # side over price inference.
        if exchange == "kalshi":
            status = getattr(market, "status", None)
            if status in ("settled", "finalized"):
                result = getattr(market, "result", "")
                if result in ("yes", "no"):
                    return result == "yes"
                return market.outcome_yes_price > 0.5

        # The exchange's active flag is the primary resolution signal, but on
        # Polymarket it lags the actual outcome: a market stays active=True for
        # a while after resolving, price pinned to 0/1. A market is therefore
        # eligible for the price check when it is inactive OR already past its
        # end_date. A still-trading market — active AND before its end_date — is
        # never resolved regardless of price; that guard is what stops an early
        # 95/5 market from being settled prematurely and feeding a fake outcome
        # into calibration. (Paper positions settle ONLY through this path, so
        # the stale active flag was stranding them indefinitely.)
        # `closed` is the venue's explicit "trading has ended" flag and flips
        # before the lagging `active` flag does. Without it, a market that
        # resolved while still flagged active=True with a future end_date — the
        # exact shape of a held losing leg pinned to 0/1 — never became eligible
        # and lingered in the portfolio at a $0 mark, its realized loss unbooked.
        end = getattr(market, "end_date", None)
        if end is not None and end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        past_end_date = end is not None and datetime.now(timezone.utc) > end
        closed = bool(getattr(market, "closed", False))
        if market.active and not past_end_date and not closed:
            return None

        # Eligible (inactive, or past end_date) + tightly converged price →
        # resolution.  The threshold is intentionally tight (0.99/0.01) so a
        # market that merely closed/paused with a non-trivial spread isn't
        # declared a winner.
        yes_price = market.outcome_yes_price
        if yes_price >= 0.99:
            return True
        if yes_price <= 0.01:
            return False

        # Eligible but price ambiguous (e.g. closed awaiting oracle) — wait.
        return None

    # ------------------------------------------------------------------
    # Position settlement
    # ------------------------------------------------------------------

    async def _settle_position(self, market_id: str, outcome: bool,
                               token_scope: str | None = None,
                               is_paper_scope: int | None = None) -> None:
        """Clean up the portfolio entry and log realized PnL for a resolved market.

        Computes the realized profit/loss from the position, records it in
        the daily_stats table, and removes the portfolio row. When
        ``token_scope`` / ``is_paper_scope`` are given the settlement is
        scoped to that exact row (the venue sweep settles per held token);
        without them the legacy market-wide behavior is preserved.
        """
        # Token comparisons are case-insensitive throughout settlement: the
        # cost_basis table stores the venue's mixed-case label ("Yes"/"No")
        # while the portfolio/ledger use the upper-cased TokenType ("YES"/"NO").
        # A case-sensitive match left cost_basis.size un-zeroed on settlement,
        # so _sync_live resurrected the position every cycle (only _drop_settled
        # kept the portfolio clean), and realized_pnl never accrued.
        where = "market_id = ?"
        params: list = [market_id]
        if token_scope is not None:
            where += " AND UPPER(token) = UPPER(?)"
            params.append(token_scope)
        if is_paper_scope is not None:
            where += " AND is_paper = ?"
            params.append(is_paper_scope)
        pos_row = await self._db.fetchone(
            f"SELECT * FROM portfolio WHERE {where}",
            tuple(params),
        )
        # aiosqlite.Row supports __getitem__ but not .get(); normalise to a
        # plain dict so we can safely use defaults for columns added in
        # later migrations (token, is_paper).
        if pos_row is not None:
            pos = dict(pos_row)
            entry_price = pos["avg_price"]
            size = pos["size"]
            side = pos["side"]
            token = pos.get("token") or "YES"
            is_paper_flag = int(pos.get("is_paper", 1))
        else:
            # No portfolio row — fall back to cost_basis, the holdings source of
            # truth. The portfolio view can briefly drop a row mid-sync (so a
            # settlement that raced it never booked), or a settled leg's row was
            # removed while its cost_basis size stayed non-zero (resurrecting it
            # every cycle). Settling off cost_basis closes both gaps; the
            # idempotency guard below stops an already-booked leg double-counting.
            cb_where = "market_id = ? AND size > 0"
            cb_params: list = [market_id]
            if token_scope is not None:
                cb_where += " AND UPPER(token) = UPPER(?)"
                cb_params.append(token_scope)
            if is_paper_scope is not None:
                cb_where += " AND is_paper = ?"
                cb_params.append(is_paper_scope)
            cb_row = await self._db.fetchone(
                f"SELECT * FROM cost_basis WHERE {cb_where} ORDER BY size DESC",
                tuple(cb_params),
            )
            if cb_row is None:
                # Neither a position nor a holding — only a calibration prediction.
                return
            cb = dict(cb_row)
            entry_price = cb["avg_cost"]
            size = cb["size"]
            side = "BUY"  # Polymarket holdings are always long
            token = cb.get("token") or "YES"
            is_paper_flag = int(cb.get("is_paper", 1))

        # Normalise to the upper-cased TokenType used by the portfolio/ledger, so
        # the source_ref, cost_basis match, and exit-price logic are all
        # consistent regardless of the venue's mixed-case cost_basis label.
        token = (token or "YES").upper()

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

        # Idempotency: the ledger is INSERT-OR-IGNORE on source_ref, but
        # daily_stats and cost_basis.realized_pnl are running totals that WOULD
        # double-count on a re-settle. So only book the P&L when this leg isn't
        # already in the ledger; an already-booked leg still falls through to the
        # cleanup below (zero its residual cost_basis size, drop the portfolio
        # row) so it stops resurrecting.
        source_ref = f"settle:{market_id}:{token}:{is_paper_flag}"
        already_booked = await self._db.fetchone(
            "SELECT 1 FROM pnl_ledger WHERE source_ref = ?", (source_ref,)
        )

        if already_booked is None:
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

            # Unified realized-P&L ledger: settlement of the residual position.
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
                source_ref=source_ref,
            )
            cost_basis_pnl_delta = pnl
        else:
            cost_basis_pnl_delta = 0.0

        # Zero the cost_basis size (always) and accrue realized P&L only when we
        # booked above. Scoped to the same paper/live mode AND token as the
        # settled position, so a paper resolution can't zero a live row's cost
        # basis and settling one side can't zero the other (YES+NO can coexist
        # since the #78 PKs).
        try:
            await self._db.execute(
                """UPDATE cost_basis
                   SET realized_pnl = realized_pnl + ?, size = 0, updated_at = datetime('now')
                   WHERE market_id = ? AND is_paper = ? AND UPPER(token) = UPPER(?)""",
                (cost_basis_pnl_delta, market_id, is_paper_flag, token),
            )
        except Exception as e:
            log.debug("resolution.cost_basis_error", error=str(e))

        # Remove from portfolio — scoped by is_paper so a paper resolution
        # doesn't delete a live position for the same market (and vice versa).
        # When the caller settled a specific token, only that row goes: a
        # YES+NO pair (mergeable) settles leg by leg.
        if token_scope is not None:
            await self._db.execute(
                "DELETE FROM portfolio WHERE market_id = ? AND is_paper = ? "
                "AND UPPER(token) = UPPER(?)",
                (market_id, is_paper_flag, token_scope),
            )
        else:
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
