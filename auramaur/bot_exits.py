"""Exit execution — extracted from AuramaurBot as a mixin (Phase 5 split).

Pure structural move: the poly / Kalshi / IBKR exit paths and their helpers
live here as ExitExecutionMixin, mixed into AuramaurBot. Behavior is unchanged;
the methods still operate on the bot's ``self`` (components, settings, exit
suppression sets, the lazy exit gateway). Method-local imports moved with the
methods; only the module-level order types + logger are needed at module scope.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from auramaur.exchange.models import Order, OrderSide, OrderType, TokenType

if TYPE_CHECKING:
    from auramaur.exchange.protocols import ExchangeClient, MarketDiscovery
    from auramaur.monitoring.alerts import AlertManager

log = structlog.get_logger()


class ExitExecutionMixin:
    """Exit-execution methods for :class:`AuramaurBot` (see module docstring)."""

    def _clear_exit_suppression(self, exchange_name: str, order, status: str) -> None:
        """Release the exit-retry suppression for a terminated exit order.

        The portfolio monitor sets ``exit:{exchange}:{market_id}`` in
        ``_exit_pending`` (or ``_exit_failures``) when it places an exit sell so
        it won't spam duplicates while the order rests. When that order reaches a
        terminal state the order monitor only clears the exchange-level
        ``_live_pending`` dict — without this, a cancelled/expired exit would
        keep the bot-level key set and suppress all future exits for that market
        until restart. Mirror the key construction in the portfolio monitor and
        clear both sets for SELL (exit) orders, on any terminal status.
        """
        if order is None or getattr(order, "side", None) != OrderSide.SELL:
            return
        exit_key = f"exit:{exchange_name}:{order.market_id}"
        self._exit_pending.discard(exit_key)
        self._exit_failures.discard(exit_key)
        log.debug("exit.suppression_cleared", exit_key=exit_key, status=status)

    @property
    def _exit_gateway(self):
        """Lazily build the ExecutionGateway used for exits.

        Exits price the SELL themselves and skip risk, so this is only used via
        ``submit_exit`` (prebuilt order, per-call exchange). The gateway's own
        exchange/router are unused for that path; db + pnl_tracker come from the
        wired components. Rebuilt if the pnl_tracker reference changes.
        """
        db = self._components.db
        pnl = self._components.pnl_tracker
        gw = self._exit_gateway_obj
        if gw is None or gw.db is not db or gw.pnl_tracker is not pnl:
            from auramaur.broker.execution_gateway import ExecutionGateway
            gw = ExecutionGateway(
                router=None, exchange=None, exchange_name="",
                settings=self.settings, db=db, pnl_tracker=pnl,
            )
            self._exit_gateway_obj = gw
        return gw

    async def _execute_poly_exit(
        self,
        pos,
        reason,
        discovery: MarketDiscovery,
        exchange: ExchangeClient,
        alerts: AlertManager,
    ) -> bool:
        """Execute an exit for a Polymarket position.

        Returns True if the sell was accepted (or a retry is worth attempting
        next cycle), False if we should stop retrying this position.
        """
        # Resolve the real token_id: reconciler → cost_basis → Gamma
        token_id = ""
        reconciler_comp = self._components.reconciler
        if reconciler_comp and self.settings.is_live:
            try:
                for rp in await reconciler_comp.reconcile():
                    if rp.market_id == pos.market_id or rp.question == pos.market_id:
                        token_id = rp.token_id
                        log.info("exit.token_from_reconciler",
                                 market_id=pos.market_id, token_id=token_id[:20])
                        break
            except Exception as e:
                log.debug("exit.reconciler_error", error=str(e))

        if not token_id:
            try:
                # _execute_poly_exit only fires for live positions; scope to
                # is_paper=0 so we can't pick up a stale paper-mode token_id.
                row = await self._components.db.fetchone(
                    "SELECT token_id FROM cost_basis WHERE market_id = ? AND size > 0 AND is_paper = 0",
                    (pos.market_id,),
                )
                if row and row["token_id"]:
                    token_id = row["token_id"]
            except Exception:
                pass

        if not token_id:
            market_data = await discovery.get_market(pos.market_id)
            if market_data:
                if pos.token == TokenType.NO:
                    token_id = market_data.clob_token_no or market_data.clob_token_yes
                else:
                    token_id = market_data.clob_token_yes or market_data.clob_token_no

        if not token_id:
            log.debug("exit.no_token", market_id=pos.market_id)
            return False

        # Cancel stale sell orders so balance is free
        if hasattr(exchange, "cancel_open_orders_for_token"):
            try:
                await exchange.cancel_open_orders_for_token(token_id)
            except Exception as e:
                log.debug("exit.pre_cancel_error", error=str(e))

        # On-chain balance ground truth
        sell_size = pos.size
        try:
            from py_clob_client_v2 import BalanceAllowanceParams, AssetType
            await exchange.clob_call(exchange._init_clob_client)
            bal = await exchange.clob_call(
                exchange._clob_client.get_balance_allowance,
                BalanceAllowanceParams(
                    asset_type=AssetType.CONDITIONAL,
                    token_id=token_id,
                    signature_type=2,
                ),
            )
            onchain = int(bal.get("balance", 0)) / 1e6
            if onchain < sell_size:
                log.info("exit.size_adjusted",
                         market_id=pos.market_id,
                         db_size=sell_size, onchain=onchain)
                sell_size = onchain
        except Exception as e:
            log.debug("exit.balance_check_error", error=str(e))

        if sell_size < 5:
            if self.settings.is_live and sell_size <= 0.01:
                await self._prune_zero_onchain_poly_position(pos.market_id)
            log.debug("exit.too_small", market_id=pos.market_id, size=sell_size)
            return False
        if pos.current_price < 0.01:
            log.debug("exit.near_zero", market_id=pos.market_id, price=pos.current_price)
            return False

        sell_price = max(0.01, min(0.99, round(pos.current_price, 2)))
        # Marketable exit: a SELL only fills by crossing down to the real bid.
        # Pricing at the snapshot — or anywhere inside the spread — rests above
        # the bid, TTL-cancels, and the cleared exit suppression re-posts it
        # next pass: a held winner looped that way unfilled for days against a
        # wide-spread book.
        # So take the bid outright when it's within the slippage band and above
        # the junk floor; otherwise skip. Returning False (no order placed)
        # leaves the portfolio monitor's exit-failure suppression set, which
        # backs the retry off until restart or resolution instead of looping.
        try:
            max_slip = float(self.settings.execution.exit_max_cross_cents) / 100.0
        except Exception:
            max_slip = 0.10
        try:
            min_bid = float(self.settings.execution.exit_min_bid_price)
        except Exception:
            min_bid = 0.05
        try:
            book = await exchange.get_order_book(token_id)
        except Exception as e:
            # Genuine fetch error (network/API) — best-effort: post at the
            # snapshot. This is the transient path, not the wide-spread loop.
            log.debug("exit.book_unavailable", market_id=pos.market_id, error=str(e))
            book = None
        if book is not None:
            best_bid = book.best_bid
            best_ask = book.best_ask
            # Mark-vs-book sanity: when the snapshot sits far above the best
            # ask, the mark is fiction (mislabeled side, stale price) — even
            # the bid is meaningless. Clearer signal than bid_too_thin.
            if best_ask is not None and (sell_price - max_slip) > best_ask + 0.10:
                log.warning(
                    "exit.mark_book_divergence",
                    market_id=pos.market_id,
                    snapshot=sell_price,
                    best_ask=best_ask,
                    best_bid=best_bid,
                )
                return False
            # No buyers at all — nothing to cross into; redeem at resolution.
            if best_bid is None or best_bid <= 0:
                log.warning(
                    "exit.no_bid", market_id=pos.market_id, snapshot=sell_price,
                )
                return False
            # Junk bid below the absolute floor — redeeming beats dumping.
            if best_bid < min_bid:
                log.warning(
                    "exit.bid_below_floor",
                    market_id=pos.market_id,
                    snapshot=sell_price,
                    bid=best_bid,
                    floor=min_bid,
                )
                return False
            # Bid too far under the mark to accept the slippage — back off.
            if best_bid < sell_price - max_slip:
                log.warning(
                    "exit.bid_too_thin",
                    market_id=pos.market_id,
                    snapshot=sell_price,
                    bid=best_bid,
                    max_slip=max_slip,
                )
                return False
            # Cross to the bid: a marketable SELL that actually fills.
            if best_bid < sell_price:
                log.info(
                    "exit.priced_to_bid",
                    market_id=pos.market_id,
                    snapshot=sell_price,
                    bid=best_bid,
                    price=round(best_bid, 2),
                )
            sell_price = max(0.01, min(0.99, round(best_bid, 2)))
        sell_order = Order(
            market_id=pos.market_id,
            token_id=token_id,
            side=OrderSide.SELL,
            size=sell_size,
            price=sell_price,
            order_type=OrderType.LIMIT,
            dry_run=not self.settings.is_live,
            source="exit",
        )
        res = await self._exit_gateway.submit_exit(
            sell_order, exchange=exchange, exchange_name="polymarket")
        if res.status == "rejected":
            log.warning("exit.sell_failed", market_id=pos.market_id)
            return False

        await alerts.send(
            f"Exit {reason.value} (poly): {pos.market_id[:12]} "
            f"size={pos.size:.2f} pnl={pos.unrealized_pnl:+.2f}",
            level="warning",
        )
        return True

    async def _prune_zero_onchain_poly_position(self, market_id: str) -> None:
        """Remove stale live Polymarket rows after an on-chain zero balance check."""
        db = self._components.db
        if db is None:
            return
        try:
            pf_cur = await db.execute(
                """DELETE FROM portfolio
                   WHERE market_id = ? AND exchange = 'polymarket' AND is_paper = 0""",
                (market_id,),
            )
            cb_cur = await db.execute(
                "DELETE FROM cost_basis WHERE market_id = ? AND is_paper = 0",
                (market_id,),
            )
            await db.execute("DELETE FROM position_peaks WHERE market_id = ?", (market_id,))
            await db.commit()
            log.info(
                "exit.stale_zero_pruned",
                market_id=market_id,
                portfolio=getattr(pf_cur, "rowcount", 0),
                cost_basis=getattr(cb_cur, "rowcount", 0),
            )
        except Exception as e:
            log.debug("exit.stale_zero_prune_error", market_id=market_id, error=str(e))

    async def _execute_kalshi_exit(
        self,
        pos,
        reason,
        discovery: MarketDiscovery,
        exchange: ExchangeClient,
        alerts: AlertManager,
    ) -> bool:
        """Execute an exit for a Kalshi position.

        Kalshi is ticker-based with direct YES/NO sells, so we just build a
        SELL signal with ``exit_token`` set and let the exchange's
        ``prepare_order`` do the rest.
        """
        from auramaur.exchange.models import Confidence, Signal

        market = await discovery.get_market(pos.market_id)
        if market is None:
            log.debug("exit.no_market", market_id=pos.market_id)
            return False

        exit_signal = Signal(
            market_id=pos.market_id,
            market_question=market.question,
            claude_prob=0.5,
            claude_confidence=Confidence.MEDIUM,
            market_prob=0.5,
            edge=10.0,
            evidence_summary=f"Exit: {reason.value}",
            recommended_side=OrderSide.SELL,
            exit_token=pos.token,
        )

        # Notional to feed into prepare_order sizing
        notional = pos.size * max(pos.current_price, 0.01)
        order = exchange.prepare_order(exit_signal, market, notional, self.settings.is_live)
        if order is None:
            return False

        # Never sell more than we hold
        order.size = min(order.size, pos.size)
        if not market.fractional_trading_enabled:
            # Legacy holdings can carry fractional sizes; a non-fractional
            # market rejects a fractional count at the API, stranding the exit.
            order.size = float(int(order.size))
        minimum = 0.01 if market.fractional_trading_enabled else 1.0
        if order.size < minimum:
            return False

        order.source = "exit"
        res = await self._exit_gateway.submit_exit(
            order, exchange=exchange, exchange_name="kalshi")
        if res.status == "rejected":
            return False

        await alerts.send(
            f"Exit {reason.value} (kalshi): {pos.market_id} "
            f"size={pos.size:.0f} pnl={pos.unrealized_pnl:+.2f}",
            level="warning",
        )
        return True

    async def _execute_ibkr_exit(self, pos, reason, discovery, exchange, alerts) -> bool:
        """Close a held IBKR option position by selling the exact contract.

        Unlike the prediction-venue exits (which reframe a fresh SELL signal), an
        option position is closed by selling the specific contract we hold.
        ``pos.token_id`` carries that contract (conId:action:right:strike:expiry)
        from prepare_order, so we build a direct SELL for it rather than routing
        through the reframer (which would open a new position).
        """
        from auramaur.exchange.models import Order, OrderType

        if not pos.token_id or pos.token_id.count(":") < 4:
            log.debug("exit.ibkr.no_contract", market_id=pos.market_id)
            return False
        if pos.size < 1:
            return False

        order = Order(
            market_id=pos.market_id,
            exchange="ibkr",
            token_id=pos.token_id,
            side=OrderSide.SELL,
            token=pos.token,
            size=float(int(pos.size)),
            price=max(pos.current_price or 0.0, 0.01),
            order_type=OrderType.LIMIT,
            dry_run=not self.settings.is_live,
            source="exit",
        )
        # DIRECT_EQUITY exception (the one named direct placement in the exit
        # path): IBKR options are off the prediction-market ExecutionGateway and
        # carry their own fill/position accounting, so this exit places directly
        # rather than via submit_exit (poly/kalshi exits go through the gateway).
        # test_exit_path_only_ibkr_places_directly locks this as the SOLE exit
        # bypass — a direct place_order on any other exit method fails the guard.
        result = await exchange.place_order(order)
        from auramaur.monitoring.display import show_order
        show_order(
            result.status, result.order_id, "SELL", order.size, order.price,
            result.is_paper, exchange="ibkr", error_message=result.error_message,
            market_id=pos.market_id,
        )
        if result.status == "rejected":
            return False
        await alerts.send(
            f"Exit {reason.value} (ibkr): {pos.market_id} "
            f"contracts={pos.size:.0f} pnl={pos.unrealized_pnl:+.2f}",
            level="warning",
        )
        return True

