"""Polymarket CLOB client wrapper — ALL real money flows through this module."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from auramaur.killswitch import kill_switch_present

import structlog

from auramaur.exchange.models import (
    Market, Order, OrderBook, OrderBookLevel, OrderResult, OrderSide, OrderType, Signal, TokenType,
)
from auramaur.exchange.paper import PaperTrader

log = structlog.get_logger()

# Map CLOB API status strings to our OrderResult status literals
_CLOB_STATUS_MAP: dict[str, str] = {
    "matched": "filled",
    "filled": "filled",
    "live": "pending",
    "open": "pending",
    "delayed": "pending",
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "expired": "expired",
}


class PolymarketClient:
    """
    Wraps py-clob-client for order execution.

    Safety: This is the ONLY module that can place real orders.
    Three gates must ALL be true for live trading:
    1. AURAMAUR_LIVE=true env var
    2. execution.live=true in config
    3. dry_run=False on the order
    """

    def __init__(self, settings, paper_trader: PaperTrader):
        self._settings = settings
        self._paper = paper_trader
        self._clob_client = None  # Lazy init only when live trading
        self._live_pending: dict[str, Order] = {}  # order_id -> Order for tracking
        # Cache of real on-chain positions: asset_id -> net token balance
        self._real_positions: dict[str, dict] = {}
        self._positions_loaded = False
        self._geoblocked = False  # Set on first 403 geoblock; routes to paper for session
        # market_id -> set of CLOB token ids. Initialized here (not lazily) so
        # get_sellable_token_id() can read it safely even before any market
        # tokens have been registered.
        self._market_token_map: dict[str, set[str]] = {}
        # Serializes py_clob_client calls (the SDK is not thread-safe).
        self._clob_lock = asyncio.Lock()
        # Deadline for any single CLOB HTTP call; instance attr so tests can
        # shrink it.
        self._call_timeout = 20.0

    async def clob_call(self, fn, *args, timeout: float | None = None, **kwargs):
        """Run a blocking py_clob_client call off the event loop, with a deadline.

        The SDK is synchronous HTTP with no default timeout. Called directly
        from async code, one stalled socket freezes the entire event loop —
        no exits, no risk checks, no kill-switch polling (2026-06-10: a
        stalled get_market call froze the live bot for 84 minutes until ^C).
        Worker thread + wait_for means a stall costs one abandoned thread and
        a TimeoutError instead. The SDK is not thread-safe, so calls
        serialize behind one lock; a timed-out call releases the lock and
        abandons its thread.
        """
        async with self._clob_lock:
            return await asyncio.wait_for(
                asyncio.to_thread(fn, *args, **kwargs),
                timeout if timeout is not None else self._call_timeout,
            )

    def _load_real_positions(self) -> None:
        """Load actual on-chain positions from CLOB trade history."""
        if self._positions_loaded:
            return
        self._init_clob_client()
        try:
            proxy = self._settings.polymarket_proxy_address.lower()
            trades = self._clob_client.get_trades()
            if not trades:
                return

            positions: dict[str, dict] = {}
            for t in trades:
                if t.get("status") != "CONFIRMED":
                    continue
                asset_id = None
                side = None
                size = 0.0

                # Check if we're the maker
                for mo in t.get("maker_orders", []):
                    if mo.get("maker_address", "").lower() == proxy:
                        asset_id = mo["asset_id"]
                        side = mo["side"]
                        size = float(mo["matched_amount"])
                        break

                # Or the taker
                if not asset_id and t.get("trader_side") == "TAKER":
                    asset_id = t["asset_id"]
                    side = t["side"]
                    size = float(t["size"])

                if asset_id and side:
                    if asset_id not in positions:
                        positions[asset_id] = {
                            "market": t.get("market", ""),
                            "outcome": t.get("outcome", ""),
                            "net": 0.0,
                        }
                    if side == "BUY":
                        positions[asset_id]["net"] += size
                    else:
                        positions[asset_id]["net"] -= size

            # Only keep positions with positive balance
            self._real_positions = {
                k: v for k, v in positions.items() if v["net"] > 0.01
            }
            self._positions_loaded = True
            log.info("clob_client.positions_loaded", count=len(self._real_positions))
        except Exception as e:
            log.error("clob_client.positions_load_error", error=str(e))

    def get_sellable_token_id(self, market_id: str) -> str | None:
        """Get the real CLOB asset_id for a position we hold, by market ID.

        Matches by checking the CLOB token IDs known for this market
        against our actual on-chain positions.
        """
        self._load_real_positions()
        if not self._real_positions:
            return None

        # Direct match on all asset_ids — check if any of our held tokens
        # correspond to the YES or NO token for this market
        for asset_id, pos in self._real_positions.items():
            if pos["net"] > 0.01:
                # The asset_id IS the token_id we need for selling
                # We need to match it to the market — check via the stored
                # clob_token mapping
                if asset_id in self._market_token_map.get(market_id, set()):
                    return asset_id
        return None

    def register_market_tokens(self, market_id: str, clob_yes: str, clob_no: str) -> None:
        """Register the CLOB token IDs for a market so sells can be matched."""
        if not hasattr(self, '_market_token_map'):
            self._market_token_map = {}
        tokens = set()
        if clob_yes:
            tokens.add(clob_yes)
        if clob_no:
            tokens.add(clob_no)
        if tokens:
            self._market_token_map[market_id] = tokens

    def _is_live_enabled(self) -> bool:
        """Check the two global gates (env var + config)."""
        return self._settings.is_live

    def _init_clob_client(self):
        """Initialize the real CLOB client. Only called for live trading."""
        if self._clob_client is not None:
            return

        from py_clob_client_v2 import ClobClient, ApiCreds

        host = "https://clob.polymarket.com"
        chain_id = 137  # Polygon mainnet

        proxy = self._settings.polymarket_proxy_address

        creds = None
        if self._settings.polymarket_api_key:
            creds = ApiCreds(
                api_key=self._settings.polymarket_api_key,
                api_secret=self._settings.polymarket_api_secret,
                api_passphrase=self._settings.polymarket_passphrase,
            )

        self._clob_client = ClobClient(
            host,
            chain_id=chain_id,
            key=self._settings.polygon_private_key,
            creds=creds,
            signature_type=2,  # POLY_GNOSIS_SAFE (Polymarket proxy wallet)
            funder=proxy if proxy else None,
        )

        # Approve pUSD collateral for buys (V2 uses pUSD, not USDC.e)
        try:
            from py_clob_client_v2 import BalanceAllowanceParams, AssetType
            self._clob_client.update_balance_allowance(
                BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=2)
            )
            log.info("clob_client.collateral_approved")
        except Exception as e:
            log.warning("clob_client.collateral_approval_error", error=str(e))

        self._approved_tokens: set[str] = set()

        log.warning("clob_client.initialized", host=host, chain_id=chain_id, version="v2")

    def prepare_order(
        self, signal: Signal, market: Market, position_size: float, is_live: bool,
    ) -> Order | None:
        """Build a Polymarket order from a signal.

        Polymarket semantics: always BUY — if the signal says SELL YES, we BUY
        the NO token instead.
        """
        # Determine token
        if signal.recommended_side == OrderSide.SELL:
            token = TokenType.NO
        else:
            token = TokenType.YES
        side = OrderSide.BUY  # Always BUY on Polymarket CLOB

        # Resolve CLOB token ID
        token_id = market.clob_token_yes if token == TokenType.YES else market.clob_token_no

        # Price for the token we're buying
        if token == TokenType.YES:
            exec_price = market.outcome_yes_price
        else:
            exec_price = (
                market.outcome_no_price
                if market.outcome_no_price > 0.01
                else (1.0 - market.outcome_yes_price)
            )

        # Round to valid Polymarket tick (1 cent increments, 0.01 - 0.99)
        exec_price = max(0.01, min(0.99, round(exec_price, 2)))

        log.info(
            "engine.order_price",
            token=token.value,
            exec_price=exec_price,
            yes_price=market.outcome_yes_price,
            no_price=market.outcome_no_price,
        )

        # Convert dollar size to token quantity. Polymarket CLOB enforces TWO
        # minimums and we have to satisfy both:
        #   * 5 tokens per order (clob_minimum_size)
        #   * $1 notional (min_size in marketable-order rejection)
        # The 5-token check alone is insufficient at low prices: 5 tokens at
        # $0.07 is only $0.35 notional, which Polymarket then rejects with
        # "invalid amount for a marketable BUY order, min size: $1".
        token_size = position_size / exec_price if exec_price > 0 else 0
        token_size = round(token_size, 2)
        notional = token_size * exec_price

        # A risk-approved trade that lands below the CLOB floor is bumped UP to
        # the minimum viable order rather than dropped — discarding edge the
        # risk manager already signed off on (and benching the market) is worse
        # than a few dollars of over-sizing. The floor is tiny: 5 tokens (or
        # enough tokens for $1 notional at low prices), so the bump is at most
        # ~$5 and always well under max_stake. ceil-to-2-decimals guarantees we
        # clear both the token and notional minimums.
        min_tokens, min_notional = 5.0, 1.0
        if token_size < min_tokens or notional < min_notional:
            bumped = max(min_tokens, math.ceil(min_notional / exec_price * 100) / 100)
            log.info(
                "prepare_order.bumped_to_min",
                market_id=market.id,
                original_tokens=token_size,
                original_notional=round(notional, 2),
                bumped_tokens=bumped,
                bumped_notional=round(bumped * exec_price, 2),
                position_size=position_size,
                exec_price=exec_price,
            )
            token_size = bumped
            notional = token_size * exec_price

        return Order(
            market_id=market.id,
            exchange="polymarket",
            token_id=token_id,
            side=side,
            token=token,
            size=token_size,
            price=exec_price,
            dry_run=not is_live,
        )

    async def place_order(self, order: Order) -> OrderResult:
        """
        Place an order. Paper trades by default.

        All three gates must be open for a real order:
        1. AURAMAUR_LIVE=true
        2. execution.live=true
        3. order.dry_run=False
        """
        # Check kill switch first
        if kill_switch_present():
            log.critical("kill_switch.active", action="order_blocked")
            return OrderResult(
                order_id="BLOCKED",
                market_id=order.market_id,
                status="rejected",
                is_paper=True,
            )

        # Geoblock: once detected, route all orders to paper for this session
        if self._geoblocked:
            order.dry_run = True
            result = await self._paper.execute(order)
            log.debug("order.geoblocked_paper", market_id=order.market_id)
            return result

        # Paper trade if ANY gate is closed
        if order.dry_run or not self._is_live_enabled():
            result = await self._paper.execute(order)
            log.info(
                "order.paper",
                market_id=order.market_id,
                side=order.side.value,
                size=order.size,
                price=order.price,
            )
            return result

        # === LIVE ORDER PATH ===
        log.warning(
            "order.live",
            market_id=order.market_id,
            side=order.side.value,
            size=order.size,
            price=order.price,
        )

        await self.clob_call(self._init_clob_client)

        # Pre-submit collateral guard for BUYs. An unfilled limit order locks its
        # notional as collateral; once prior resting orders consume the balance,
        # the CLOB rejects new BUYs with "not enough balance / allowance". Skip
        # here instead of hammering the API with doomed orders. Fail-open: if the
        # balance probe errors, fall through and let the CLOB decide.
        if order.side == OrderSide.BUY:
            free = await self._free_collateral_usd()
            if free is not None and order.size * order.price > free + 1e-6:
                log.info(
                    "order.skipped_insufficient_balance",
                    market_id=order.market_id,
                    cost=round(order.size * order.price, 2),
                    free=round(free, 2),
                )
                return OrderResult(
                    order_id="INSUFFICIENT_BALANCE",
                    market_id=order.market_id,
                    status="rejected",
                    is_paper=False,
                    error_message="insufficient free collateral (open orders lock balance)",
                )

        try:
            from py_clob_client_v2.order_builder.constants import BUY, SELL
            from py_clob_client_v2 import OrderArgs, BalanceAllowanceParams, AssetType, OrderType as ClobOrderType

            clob_side = BUY if order.side == OrderSide.BUY else SELL

            if not order.token_id:
                raise ValueError(f"No CLOB token_id for market {order.market_id}")

            # For SELL orders: approve the specific conditional token if not yet approved
            if order.side == OrderSide.SELL and order.token_id not in self._approved_tokens:
                try:
                    await self.clob_call(
                        self._clob_client.update_balance_allowance,
                        BalanceAllowanceParams(
                            asset_type=AssetType.CONDITIONAL,
                            token_id=order.token_id,
                            signature_type=2,
                        ),
                    )
                    self._approved_tokens.add(order.token_id)
                    log.info("clob_client.token_approved", token_id=order.token_id[:20])
                except Exception as e:
                    log.warning("clob_client.token_approval_failed",
                                token_id=order.token_id[:20], error=str(e))

            want_post_only = order.post_only and order.order_type == OrderType.LIMIT
            ord_args = OrderArgs(
                token_id=order.token_id,
                price=order.price,
                size=order.size,
                side=clob_side,
            )

            # Longer deadline for the actual submit: a timeout here is
            # ambiguous (the order may have landed) — the startup
            # reconciler picks up any such stray on the next session.
            signed_order = await self.clob_call(
                self._submit_clob_order,
                ord_args, want_post_only, ClobOrderType, order,
                timeout=30.0,
            )

            order_id = str(signed_order.get("orderID", signed_order.get("id", "unknown")))

            if isinstance(signed_order, dict) and signed_order.get("success") is False:
                err = str(signed_order.get("errorMsg") or signed_order.get("error") or "post-only rejected")
                log.info(
                    "order.post_only_rejected",
                    market_id=order.market_id,
                    price=order.price,
                    reason=err[:200],
                )
                return OrderResult(
                    order_id=order_id or "POST_ONLY_REJECTED",
                    market_id=order.market_id,
                    status="rejected",
                    is_paper=False,
                    error_message=err[:200],
                )

            self._live_pending[order_id] = order

            return OrderResult(
                order_id=order_id,
                market_id=order.market_id,
                status="pending",
                filled_size=0,
                filled_price=order.price,
                is_paper=False,
            )

        except Exception as e:
            err_str = str(e)
            if "restricted in your region" in err_str or "geoblock" in err_str.lower():
                self._geoblocked = True
                log.warning(
                    "order.geoblocked",
                    market_id=order.market_id,
                    msg="Polymarket trading geoblocked — routing to paper for this session",
                )
                order.dry_run = True
                return await self._paper.execute(order)
            # "not enough balance / allowance" is an expected capital constraint
            # (the live book is fully deployed), not a fault — log it at warning
            # like paper.insufficient_balance, reserving error for real failures.
            if "not enough balance" in err_str.lower() or "allowance" in err_str.lower():
                log.warning("order.insufficient_balance", error=err_str, market_id=order.market_id)
            else:
                log.error("order.live_error", error=err_str, market_id=order.market_id)
            return OrderResult(
                order_id="ERROR",
                market_id=order.market_id,
                status="rejected",
                is_paper=False,
                error_message=err_str[:200],
            )

    async def _free_collateral_usd(self) -> float | None:
        """Spendable pUSD = on-chain collateral minus collateral reserved by
        resting BUY orders. Returns None if the probe fails (caller fails open).

        ``get_balance_allowance`` reports *gross* collateral — open orders are not
        netted out (the CLOB rejection reports ``balance`` and ``sum of active
        orders`` separately) — so we subtract collateral reserved by open BUYs.
        We prefer the CLOB's authoritative open-order list so orphaned orders
        from a prior session are counted too, and fall back to our in-memory
        ``_live_pending`` only when that query is unavailable.
        """
        try:
            from py_clob_client_v2 import AssetType, BalanceAllowanceParams
            resp = await self.clob_call(
                self._clob_client.get_balance_allowance,
                BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=2),
            )
            # A non-dict / balance-less response means "unknown" — fail open rather
            # than treat it as zero collateral (which would block every BUY).
            if not isinstance(resp, dict) or "balance" not in resp:
                return None
            gross = int(resp.get("balance", 0)) / 1e6
        except Exception as e:
            log.debug("order.balance_probe_failed", error=str(e))
            return None

        try:
            reserved = await self.clob_call(self._reserved_buy_collateral_from_open_orders)
        except Exception:
            reserved = None
        if reserved is None:
            # Authoritative view unavailable — fall back to orders we placed this
            # session. Less complete (misses orphans) but better than no guard.
            reserved = sum(
                o.notional
                for o in self._live_pending.values()
                if o is not None and o.side == OrderSide.BUY
            )
        return gross - reserved

    def _reserved_buy_collateral_from_open_orders(self) -> float | None:
        """USDC reserved by *all* resting BUY orders on the CLOB, including
        orphans from prior sessions. Returns None if the query or any row fails
        to parse, so the caller can fall back to the ``_live_pending`` estimate.

        SELL orders lock conditional tokens (not USDC), so they are excluded.
        Reservation per BUY = price * (original_size - size_matched).
        """
        try:
            orders = self._clob_client.get_open_orders()
        except Exception as e:
            log.debug("order.open_orders_probe_failed", error=str(e))
            return None
        if not isinstance(orders, list):
            return None
        try:
            reserved = 0.0
            for o in orders:
                if not isinstance(o, dict):
                    return None
                if str(o.get("side", "")).upper() != "BUY":
                    continue
                price = float(o.get("price", 0) or 0)
                original = float(o.get("original_size", 0) or 0)
                matched = float(o.get("size_matched", 0) or 0)
                reserved += price * max(0.0, original - matched)
            return reserved
        except (TypeError, ValueError) as e:
            log.debug("order.open_orders_parse_failed", error=str(e))
            return None

    async def reconcile_open_orders(self) -> int:
        """Pull existing CLOB open orders into ``_live_pending`` at startup.

        Live GTC orders left resting by a prior session ("orphans") are not
        otherwise tracked, so the order monitor can neither TTL-cancel them
        (freeing locked collateral) nor record their fills. Reconstruct
        lightweight Order records stamped with each order's real ``created_at``
        so stale ones are reaped on the first monitor pass. Best-effort: returns
        the number reconciled, 0 if live trading is off or the query fails.
        """
        if not self._is_live_enabled():
            return 0
        try:
            await self.clob_call(self._init_clob_client)
            orders = await self.clob_call(self._clob_client.get_open_orders)
        except Exception as e:
            log.warning("order.reconcile_failed", error=str(e))
            return 0
        if not isinstance(orders, list):
            return 0

        from datetime import datetime, timezone

        count = 0
        for o in orders:
            if not isinstance(o, dict):
                continue
            oid = str(o.get("id") or o.get("orderID") or "")
            if not oid or oid in self._live_pending:
                continue
            try:
                side = OrderSide.BUY if str(o.get("side", "")).upper() == "BUY" else OrderSide.SELL
                created_raw = o.get("created_at")
                created = (
                    datetime.fromtimestamp(int(created_raw), tz=timezone.utc)
                    if created_raw not in (None, "")
                    else datetime.now(timezone.utc)
                )
                self._live_pending[oid] = Order(
                    market_id=str(o.get("market") or o.get("asset_id") or ""),
                    exchange="polymarket",
                    token_id=str(o.get("asset_id") or ""),
                    side=side,
                    size=float(o.get("original_size", 0) or 0),
                    price=float(o.get("price", 0) or 0),
                    order_type=OrderType.LIMIT,
                    dry_run=False,
                    created_at=created,
                    # The CLOB API doesn't echo our `source`, so a prior-session
                    # orphan would otherwise reach the monitor's fallback INSERT
                    # sourceless and get tagged 'order_monitor', masking the real
                    # book. A live *resting GTC limit* can only come from the
                    # market maker in the current regime — directional books are
                    # paper-forced, arb is both-or-nothing immediate, exits cross
                    # the spread (taker). So attribute reconciled orphans to the
                    # MM. NOTE: revisit if any other live book ever rests limits.
                    source="market_maker",
                )
                count += 1
            except (TypeError, ValueError) as e:
                log.debug("order.reconcile_parse_skip", order_id=oid, error=str(e))
                continue
        if count:
            log.info("order.reconciled_open_orders", count=count)
        return count

    def _submit_clob_order(self, ord_args, want_post_only, ClobOrderType, order):
        """Submit order to CLOB via V2 client (has built-in version mismatch retry)."""
        return self._clob_client.create_and_post_order(
            ord_args,
            order_type=ClobOrderType.GTC,
            post_only=want_post_only,
        )

    async def get_order_status(self, order_id: str) -> OrderResult:
        """Query the CLOB API for the current status of an order.

        When the CLOB returns None for an order, it means the order no longer
        exists in the active-order index — either filled-and-settled long ago,
        or cancelled. We return a terminal "cancelled" result so callers can
        drop it from their pending-order tracking instead of re-polling forever.
        """
        await self.clob_call(self._init_clob_client)
        try:
            raw = await self.clob_call(self._clob_client.get_order, order_id)
        except Exception as e:
            log.error("order_status.error", order_id=order_id, error=str(e))
            raise

        if raw is None:
            log.debug("order_status.not_found", order_id=order_id)
            return OrderResult(
                order_id=order_id,
                market_id="",
                status="cancelled",
                filled_size=0,
                filled_price=0,
                is_paper=False,
            )

        try:
            status_str = str(raw.get("status", "pending")).lower()
            status = _CLOB_STATUS_MAP.get(status_str, "pending")
            filled = float(raw.get("size_matched", 0))
            price = float(raw.get("price", 0))
            market_id = raw.get("market", raw.get("asset_id", ""))

            return OrderResult(
                order_id=order_id,
                market_id=market_id,
                status=status,  # type: ignore[arg-type]
                filled_size=filled,
                filled_price=price,
                is_paper=False,
            )
        except Exception as e:
            log.error("order_status.parse_error", order_id=order_id, error=str(e))
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a live order on the CLOB. Returns True on success.

        The v2 client has no ``cancel`` method — use ``cancel_orders([hash])``,
        whose response splits into ``canceled`` / ``not_canceled``. (The old
        ``self._clob_client.cancel(order_id)`` raised AttributeError on every
        call, so live cancels silently never happened.)
        """
        await self.clob_call(self._init_clob_client)
        try:
            resp = await self.clob_call(self._clob_client.cancel_orders, [order_id])
            not_canceled = resp.get("not_canceled", {}) if isinstance(resp, dict) else {}
            if order_id in not_canceled:
                log.warning("order_cancel.rejected", order_id=order_id,
                            reason=str(not_canceled.get(order_id))[:120])
                return False
            self._live_pending.pop(order_id, None)
            log.info("order.cancelled", order_id=order_id)
            return True
        except Exception as e:
            log.error("order_cancel.error", order_id=order_id, error=str(e))
            return False

    async def cancel_open_orders_for_token(self, token_id: str) -> int:
        """Cancel all open CLOB orders for a specific conditional token.

        Used by the exit path: stale sell orders from a previous run lock the
        token balance, so a fresh profit-target sell would be rejected with
        "not enough balance / allowance". Cancelling first releases the balance
        before we re-post at the current price.

        Returns the number of orders cancelled (0 if none were open).
        """
        if not token_id:
            return 0
        await self.clob_call(self._init_clob_client)
        try:
            from py_clob_client_v2 import OrderMarketCancelParams
            resp = await self.clob_call(
                self._clob_client.cancel_market_orders,
                OrderMarketCancelParams(asset_id=token_id),
            )
            cancelled = resp.get("canceled", []) if isinstance(resp, dict) else []
            count = len(cancelled) if isinstance(cancelled, list) else 0
            if count:
                log.info(
                    "order.stale_cancelled",
                    token_id=token_id[:20],
                    count=count,
                )
                for oid in cancelled if isinstance(cancelled, list) else []:
                    self._live_pending.pop(str(oid), None)
            return count
        except Exception as e:
            log.warning(
                "order.stale_cancel_error",
                token_id=token_id[:20],
                error=str(e)[:200],
            )
            return 0

    async def poll_until_terminal(
        self,
        order_id: str,
        timeout: float = 300,
        poll_interval: float = 10,
    ) -> OrderResult:
        """Poll order status until it reaches a terminal state or timeout.

        On timeout, attempts to cancel the order.
        """
        import asyncio
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = await self.get_order_status(order_id)
            if result.status in ("filled", "cancelled", "expired", "rejected"):
                self._live_pending.pop(order_id, None)
                return result
            await asyncio.sleep(poll_interval)

        # Timeout — cancel the order
        log.warning("order.poll_timeout", order_id=order_id, timeout=timeout)
        await self.cancel_order(order_id)
        return OrderResult(
            order_id=order_id,
            market_id=self._live_pending.pop(order_id, Order(market_id="", side="BUY", size=0, price=0)).market_id,
            status="cancelled",
            is_paper=False,
        )

    async def get_order_book(self, token_id: str) -> OrderBook:
        """Get order book for a token (public, no auth needed)."""
        from auramaur.exchange.models import OrderBook, OrderBookLevel
        try:
            await self.clob_call(self._init_clob_client)
            raw = await self.clob_call(self._clob_client.get_order_book, token_id)
            # py_clob_client_v2 returns a plain dict ({"bids": [...], "asks": [...]}),
            # not a dataclass — read by key, falling back to attribute access for
            # any client version that returns an OrderBookSummary object.
            raw_bids = (raw.get("bids") if isinstance(raw, dict) else getattr(raw, "bids", [])) or []
            raw_asks = (raw.get("asks") if isinstance(raw, dict) else getattr(raw, "asks", [])) or []
            bids = [
                OrderBookLevel(
                    price=float(getattr(b, "price", b.get("price", 0)) if isinstance(b, dict) else b.price),
                    size=float(getattr(b, "size", b.get("size", 0)) if isinstance(b, dict) else b.size),
                )
                for b in raw_bids
            ]
            asks = [
                OrderBookLevel(
                    price=float(getattr(a, "price", a.get("price", 0)) if isinstance(a, dict) else a.price),
                    size=float(getattr(a, "size", a.get("size", 0)) if isinstance(a, dict) else a.size),
                )
                for a in raw_asks
            ]
            return OrderBook(bids=bids, asks=asks)
        except Exception as e:
            log.debug("orderbook.unavailable", token_id=token_id[:20], error=str(e))
            return OrderBook()
