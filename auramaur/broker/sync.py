"""Position syncer — queries ground-truth positions from exchange or paper trader."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

import structlog

from auramaur.broker.pnl import PnLTracker
from auramaur.db.database import Database
from auramaur.exchange.client import PolymarketClient
from auramaur.exchange.models import LivePosition, OrderSide, TokenType
from auramaur.exchange.paper import PaperTrader
from config.settings import Settings

log = structlog.get_logger()


class Syncer(Protocol):
    """Common interface implemented by both exchange syncers."""

    exchange_name: str

    async def sync(self) -> list[LivePosition]: ...
    async def get_cash_balance(self) -> float: ...


class PositionSyncer:
    """Synchronises positions between the exchange (live or paper) and
    the local database.

    This is the single source of truth for "what do we actually hold?"
    It queries the CLOB API (live) or the PaperTrader (paper) and
    reconciles the ``portfolio`` table to match.
    """

    exchange_name = "polymarket"

    def __init__(
        self,
        settings: Settings,
        db: Database,
        exchange: PolymarketClient,
        paper: PaperTrader,
        pnl: PnLTracker,
    ) -> None:
        self._settings = settings
        self._db = db
        self._exchange = exchange
        self._paper = paper
        self._pnl = pnl

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def sync(self) -> list[LivePosition]:
        """Full sync: query the exchange for actual positions and reconcile
        the local database.

        Returns the canonical list of ``LivePosition`` objects.
        """
        if self._settings.is_live:
            positions = await self._sync_live()
        else:
            positions = await self._sync_paper()

        await self._reconcile(positions)

        log.info(
            "sync.complete",
            mode="live" if self._settings.is_live else "paper",
            position_count=len(positions),
        )
        return positions

    async def get_cash_balance(self) -> float:
        """Query available USDC balance.

        Live:  queries the CLOB API for on-chain balance.
        Paper: returns ``PaperTrader.balance``.
        """
        if self._settings.is_live:
            return await self._get_live_balance()
        return self._paper.balance

    # ------------------------------------------------------------------
    # Live sync
    # ------------------------------------------------------------------

    async def _sync_live(self) -> list[LivePosition]:
        """Build positions from cost_basis table (populated by PnLTracker fills).

        The cost_basis table is the most reliable source since it tracks
        every fill we've made.  We enrich with current market prices
        from the markets table (refreshed each scan cycle).
        """
        positions: list[LivePosition] = []

        try:
            # Cost basis is our ground truth — populated by every recorded fill.
            # Join against markets table to get current prices and category.
            # Filter is_paper=0 so paper fills don't leak into live sync.
            rows = await self._db.fetchall(
                """SELECT cb.market_id, cb.token, cb.token_id, cb.size, cb.avg_cost,
                          m.outcome_yes_price, m.outcome_no_price, m.category,
                          m.clob_token_yes, m.clob_token_no
                   FROM cost_basis cb
                   LEFT JOIN markets m ON cb.market_id = m.id
                   WHERE cb.size > 0 AND cb.is_paper = 0"""
            )

            for row in rows:
                market_id = row["market_id"]
                raw_token = (row["token"] or "YES").upper()
                token_id = row["token_id"] or ""
                yes_price = float(row["outcome_yes_price"] or 0)
                no_price = float(row["outcome_no_price"] or 0)

                # Resolve which SIDE the held token is. Matching token_id
                # against the market's CLOB token ids is authoritative; the
                # outcome label only works when it literally is YES/NO. For
                # anything else ("Something"/"Nothing", "Up"/"Down", team
                # names) the old YES-default marked the position at the wrong
                # outcome's price — e.g. a low-priced held side marked at its
                # complement's high price, a phantom gain whose exit the bot
                # then chased for days.
                direct_price: float | None = None
                if token_id and token_id == (row["clob_token_no"] or ""):
                    token = TokenType.NO
                elif token_id and token_id == (row["clob_token_yes"] or ""):
                    token = TokenType.YES
                elif raw_token in ("YES", "NO"):
                    token = TokenType(raw_token)
                else:
                    # Side unknown — price the token we actually hold straight
                    # off its own book instead of guessing an outcome.
                    token = TokenType.YES
                    direct_price = await self._price_held_token(token_id)
                    if direct_price is None:
                        log.warning(
                            "sync.unresolved_token_side",
                            market_id=market_id,
                            token_label=raw_token,
                        )

                # Use the correct price for the token we hold
                if direct_price is not None:
                    current_price = direct_price
                elif token == TokenType.NO:
                    current_price = no_price if no_price > 0.01 else (1.0 - yes_price) if yes_price > 0 else 0.0
                else:
                    current_price = yes_price

                # No usable market price — a stale or never-scanned market row
                # carries 0 for both sides. Fall back to avg_cost so an ACTIVE
                # held position isn't marked $0: a phantom -100% loss that
                # distorts portfolio value, the risk gates, and Kelly. Mirrors
                # the reconciler floor; real book/market marks still win when
                # present (this only fires when current_price resolved to 0).
                if current_price <= 0:
                    current_price = float(row["avg_cost"] or 0)

                category = row["category"] or ""

                positions.append(
                    LivePosition(
                        market_id=market_id,
                        token_id=row["token_id"] or "",
                        token=token,
                        size=float(row["size"]),
                        avg_cost=float(row["avg_cost"]),
                        current_price=current_price,
                        category=category,
                    ),
                )

            log.info("sync.live.done", positions=len(positions))

        except Exception as e:
            log.error("sync.live.error", error=str(e))

        return positions

    async def _price_held_token(self, token_id: str) -> float | None:
        """Mark an outcome token directly off its own CLOB book.

        Used when neither the token-id match nor the YES/NO label can tell
        which side we hold. The book midpoint is what the token is actually
        worth right now; on a one-sided book fall back to the bid — the
        price an exit could realize. Returns None when there is no book.
        """
        if not token_id:
            return None
        try:
            book = await self._exchange.get_order_book(token_id)
        except Exception as e:
            log.debug("sync.token_book_error", token_id=token_id[:20], error=str(e))
            return None
        mid = book.midpoint
        if mid is not None:
            return mid
        return book.best_bid

    async def _get_live_balance(self) -> float:
        """Return *spendable* pUSD: gross collateral minus collateral already
        reserved by resting BUY orders.

        ``get_balance_allowance`` reports gross collateral (open orders are not
        netted out), so without this subtraction the engine sizes Kelly bets and
        approves trades against cash that is actually locked in open orders —
        only for the submit-time guard to then reject them. Netting here makes
        sizing/allocation deploy only what is genuinely available.
        """
        self._exchange._init_clob_client()
        client = self._exchange._clob_client
        try:
            from py_clob_client_v2 import BalanceAllowanceParams, AssetType
            resp = client.get_balance_allowance(
                BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=2)
            )
            gross = int(resp.get("balance", 0)) / 1e6 if isinstance(resp, dict) else 0.0
        except Exception as e:
            log.error("sync.balance.error", error=str(e))
            return 0.0

        reserved = 0.0
        reserve_fn = getattr(self._exchange, "_reserved_buy_collateral_from_open_orders", None)
        if reserve_fn is not None:
            reserved = reserve_fn() or 0.0
        available = max(0.0, gross - reserved)
        log.info("sync.balance.computed", available=round(available, 2),
                 gross=round(gross, 2), reserved=round(reserved, 2))
        return available

    # ------------------------------------------------------------------
    # Paper sync
    # ------------------------------------------------------------------

    async def _sync_paper(self) -> list[LivePosition]:
        """Convert ``PaperTrader.positions`` dict to ``LivePosition`` list."""
        positions: list[LivePosition] = []

        for market_id, pos in self._paper.positions.items():
            # Look up cost basis from PnLTracker for more accurate avg cost
            # and to get the token type/ID
            avg_cost, cb_size = await self._pnl.get_cost_basis(market_id)
            token, token_id = await self._pnl.get_token_info(market_id)
            if cb_size <= 0:
                # Fall back to paper trader's own tracking
                avg_cost = pos.avg_price

            positions.append(
                LivePosition(
                    market_id=market_id,
                    token=token,
                    token_id=token_id,
                    size=pos.size,
                    avg_cost=avg_cost,
                    current_price=pos.current_price,
                    category=pos.category,
                ),
            )

        log.info("sync.paper.done", positions=len(positions))
        return positions

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    async def _settled_keys(self, is_paper_flag: int) -> set[tuple[str, str]]:
        """(market_id, token) pairs whose settlement is already in the ledger.

        A settled-but-unredeemed position still sits in the wallet, so every
        sync resurrected it into portfolio at a $0 mark — a phantom -100%
        "unrealized" loss on a leg whose P&L is already realized (and, once
        the venue sweep re-settled the zombie, daily_stats/cost_basis would
        double-count). Settled keys stay out of portfolio; redeem-check
        tracks the wallet side until on-chain redemption empties it.
        """
        try:
            rows = await self._db.fetchall(
                "SELECT source_ref FROM pnl_ledger WHERE kind = 'settlement'")
        except Exception:
            return set()
        keys: set[tuple[str, str]] = set()
        suffix = f":{is_paper_flag}"
        for r in rows or []:
            ref = r["source_ref"] or ""
            # source_ref format: settle:{market_id}:{token}:{is_paper}
            if ref.startswith("settle:") and ref.endswith(suffix):
                parts = ref.split(":")
                if len(parts) == 4:
                    keys.add((parts[1], parts[2]))
        return keys

    def _drop_settled(self, positions: list[LivePosition],
                      settled: set[tuple[str, str]]) -> list[LivePosition]:
        kept = []
        for pos in positions:
            token = pos.token.value if pos.token else "YES"
            if (pos.market_id, token) in settled:
                log.debug("sync.skip_settled", market_id=pos.market_id,
                          token=token)
                continue
            kept.append(pos)
        return kept

    async def _merge_new_positions(self, positions: list[LivePosition]) -> None:
        """Add newly discovered positions to the portfolio table.

        This is additive-only — it never deletes existing rows.
        Used to merge positions found by the reconciler that aren't
        in cost_basis (e.g. manual buys on Polymarket).
        """
        now = datetime.now(timezone.utc).isoformat()
        is_paper_flag = 0 if self._settings.is_live else 1
        positions = self._drop_settled(
            positions, await self._settled_keys(is_paper_flag))
        for pos in positions:
            side = OrderSide.BUY.value
            token = pos.token.value if pos.token else "YES"
            token_id = pos.token_id or ""
            await self._db.execute(
                """INSERT INTO portfolio
                   (market_id, exchange, side, size, avg_price, current_price,
                    unrealized_pnl, category, token, token_id, is_paper, updated_at)
                   VALUES (?, 'polymarket', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(market_id, is_paper, token) DO UPDATE SET
                       exchange = excluded.exchange,
                       size = excluded.size,
                       avg_price = excluded.avg_price,
                       current_price = excluded.current_price,
                       unrealized_pnl = excluded.unrealized_pnl,
                       token = excluded.token,
                       token_id = excluded.token_id,
                       is_paper = excluded.is_paper,
                       updated_at = excluded.updated_at""",
                (pos.market_id, side, pos.size, pos.avg_cost,
                 pos.current_price, pos.unrealized_pnl, pos.category,
                 token, token_id, is_paper_flag, now),
            )
        await self._db.commit()
        log.info("sync.merge_new", count=len(positions))

    async def _reconcile(self, positions: list[LivePosition]) -> None:
        """Update the ``portfolio`` table to match the live position list.

        Positions not present in *positions* are deleted.  Positions in
        *positions* are inserted or updated.

        Scoped to the current mode (paper vs live) so a live sync
        doesn't wipe paper-mode rows and vice versa.
        """
        # Empty positions means cost_basis has no rows for the current mode —
        # almost always because the user traded live outside this bot, not
        # because every holding went to zero. Skip the delete step and let
        # the caller's reconciler pass (_merge_new_positions) populate from
        # on-chain ground truth. Without this guard every portfolio_monitor
        # tick wipes then re-adds the full live portfolio.
        if not positions:
            log.debug("sync.reconcile.skip_empty")
            return

        now = datetime.now(timezone.utc).isoformat()
        is_paper_flag = 0 if self._settings.is_live else 1

        # Settled-but-unredeemed legs must not resurrect: dropping them from
        # the incoming list ALSO routes their existing portfolio rows into
        # the stale-delete below, so zombies get purged on the next sync.
        positions = self._drop_settled(
            positions, await self._settled_keys(is_paper_flag))
        if not positions:
            log.debug("sync.reconcile.skip_all_settled")
            return

        # Scope reconciliation to Polymarket rows in the CURRENT mode only.
        # Mixing modes would let a live sync (which returns empty when the
        # account has no live fills yet) wipe rows from a prior paper run.
        db_rows = await self._db.fetchall(
            "SELECT market_id FROM portfolio WHERE exchange = 'polymarket' AND is_paper = ?",
            (is_paper_flag,),
        )
        db_market_ids = {row["market_id"] for row in db_rows}
        live_market_ids = {pos.market_id for pos in positions}

        # DELETE positions no longer held (Polymarket, current mode only)
        stale = db_market_ids - live_market_ids
        for market_id in stale:
            await self._db.execute(
                "DELETE FROM portfolio WHERE market_id = ? AND exchange = 'polymarket' AND is_paper = ?",
                (market_id, is_paper_flag),
            )
            log.info("sync.reconcile.removed", market_id=market_id)

        # INSERT or UPDATE positions from exchange
        for pos in positions:
            # Determine side — we always BUY on Polymarket, but keep BUY as default
            side = OrderSide.BUY.value
            token = pos.token.value if pos.token else "YES"
            token_id = pos.token_id or ""

            await self._db.execute(
                """INSERT INTO portfolio
                   (market_id, exchange, side, size, avg_price, current_price,
                    unrealized_pnl, category, token, token_id, is_paper, updated_at)
                   VALUES (?, 'polymarket', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(market_id, is_paper, token) DO UPDATE SET
                       exchange = excluded.exchange,
                       side = excluded.side,
                       size = excluded.size,
                       avg_price = excluded.avg_price,
                       current_price = excluded.current_price,
                       unrealized_pnl = excluded.unrealized_pnl,
                       category = excluded.category,
                       token = excluded.token,
                       token_id = excluded.token_id,
                       is_paper = excluded.is_paper,
                       updated_at = excluded.updated_at""",
                (
                    pos.market_id,
                    side,
                    pos.size,
                    pos.avg_cost,
                    pos.current_price,
                    pos.unrealized_pnl,
                    pos.category,
                    token,
                    token_id,
                    is_paper_flag,
                    now,
                ),
            )

        await self._db.commit()

        if stale:
            log.info("sync.reconcile.cleanup", removed=len(stale))


class KalshiPositionSyncer:
    """Position syncer for Kalshi.

    Delegates to ``KalshiClient.sync_positions()`` which queries the Kalshi
    portfolio API and upserts rows into the ``portfolio`` table, then reads
    those rows back as ``LivePosition`` objects so the rest of the system
    (check_exits, allocator, UI) can treat Kalshi uniformly with Polymarket.
    """

    exchange_name = "kalshi"

    def __init__(self, settings: Settings, db: Database, exchange, paper=None) -> None:
        self._settings = settings
        self._db = db
        self._exchange = exchange
        self._paper = paper

    async def sync(self) -> list[LivePosition]:
        if not self._settings.is_live and self._paper is not None:
            return await self._sync_paper()

        try:
            await self._exchange.sync_positions(self._db)
        except Exception as e:
            log.error("sync.kalshi.error", error=str(e))
            return []

        is_paper_flag = 0 if self._settings.is_live else 1
        rows = await self._db.fetchall(
            """SELECT p.market_id, p.token, p.size, p.avg_price, p.current_price,
                      p.category
               FROM portfolio p
               WHERE p.exchange = 'kalshi' AND p.size > 0 AND p.is_paper = ?""",
            (is_paper_flag,),
        )

        positions: list[LivePosition] = []
        for row in rows:
            raw_token = (row["token"] or "YES").upper()
            token = TokenType(raw_token) if raw_token in ("YES", "NO") else TokenType.YES
            positions.append(
                LivePosition(
                    market_id=row["market_id"],
                    token=token,
                    token_id=row["market_id"],  # Kalshi uses ticker as token id
                    size=float(row["size"] or 0),
                    avg_cost=float(row["avg_price"] or 0),
                    current_price=float(row["current_price"] or 0),
                    category=row["category"] or "",
                ),
            )

        log.info("sync.kalshi.done", positions=len(positions))
        return positions

    async def _sync_paper(self) -> list[LivePosition]:
        """Sync paper positions: persist PaperTrader state to portfolio table."""
        now = datetime.now(timezone.utc).isoformat()
        positions: list[LivePosition] = []
        current_ids: set[str] = set()

        for market_id, pos in self._paper.positions.items():
            token = pos.token if pos.token else TokenType.YES
            token_id = pos.token_id or market_id
            category = getattr(pos, "category", "") or ""
            current_ids.add(market_id)

            await self._db.execute(
                """INSERT INTO portfolio
                   (market_id, exchange, side, size, avg_price, current_price,
                    unrealized_pnl, category, token, token_id, is_paper, updated_at)
                   VALUES (?, 'kalshi', ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                   ON CONFLICT(market_id, is_paper, token) DO UPDATE SET
                       exchange = excluded.exchange,
                       side = excluded.side,
                       size = excluded.size,
                       avg_price = excluded.avg_price,
                       current_price = excluded.current_price,
                       unrealized_pnl = excluded.unrealized_pnl,
                       category = excluded.category,
                       token = excluded.token,
                       token_id = excluded.token_id,
                       is_paper = excluded.is_paper,
                       updated_at = excluded.updated_at""",
                (
                    market_id,
                    pos.side.value,
                    pos.size,
                    pos.avg_price,
                    pos.current_price,
                    pos.unrealized_pnl,
                    category,
                    token.value,
                    token_id,
                    now,
                ),
            )
            positions.append(
                LivePosition(
                    market_id=market_id,
                    token=token,
                    token_id=token_id,
                    size=pos.size,
                    avg_cost=pos.avg_price,
                    current_price=pos.current_price,
                    category=category,
                ),
            )

        if current_ids:
            placeholders = ",".join("?" * len(current_ids))
            await self._db.execute(
                f"DELETE FROM portfolio WHERE exchange='kalshi' AND is_paper=1"
                f" AND market_id NOT IN ({placeholders})",
                tuple(current_ids),
            )
        else:
            await self._db.execute(
                "DELETE FROM portfolio WHERE exchange='kalshi' AND is_paper=1"
            )

        await self._db.commit()
        log.info("sync.kalshi.paper.done", positions=len(positions))
        return positions

    async def get_cash_balance(self) -> float:
        if not self._settings.is_live and self._paper is not None:
            return self._paper.balance
        try:
            return await self._exchange.get_balance()
        except Exception as e:
            log.debug("sync.kalshi.balance_error", error=str(e))
            return 0.0
