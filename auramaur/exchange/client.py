"""Polymarket CLOB client wrapper — ALL real money flows through this module."""

from __future__ import annotations

from pathlib import Path

import structlog

from auramaur.exchange.models import (
    Market, Order, OrderBook, OrderBookLevel, OrderResult, OrderSide, Signal, TokenType,
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
            self._market_token_map: dict[str, set[str]] = {}
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

        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds

        host = "https://clob.polymarket.com"
        chain_id = 137  # Polygon mainnet

        # Polymarket proxy wallet address (Gnosis Safe)
        proxy = self._settings.polymarket_proxy_address

        self._clob_client = ClobClient(
            host,
            key=self._settings.polygon_private_key,
            chain_id=chain_id,
            signature_type=2,  # POLY_GNOSIS_SAFE (Polymarket proxy wallet)
            funder=proxy if proxy else None,
        )

        # Derive API creds if we have them
        if self._settings.polymarket_api_key:
            self._clob_client.set_api_creds(ApiCreds(
                api_key=self._settings.polymarket_api_key,
                api_secret=self._settings.polymarket_api_secret,
                api_passphrase=self._settings.polymarket_passphrase,
            ))

        # Approve USDC collateral for buys
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
            self._clob_client.update_balance_allowance(
                BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=2)
            )
            log.info("clob_client.collateral_approved")
        except Exception as e:
            log.warning("clob_client.collateral_approval_error", error=str(e))

        # Track which token_ids we've approved for selling
        self._approved_tokens: set[str] = set()

        log.warning("clob_client.initialized", host=host, chain_id=chain_id)

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

        if token_size < 5 or notional < 1.0:
            reason = "below_token_min" if token_size < 5 else "below_notional_min"
            log.warning(
                "prepare_order.too_small",
                market_id=market.id,
                reason=reason,
                token_size=token_size,
                notional=round(notional, 2),
                min_tokens=5,
                min_notional=1.0,
                position_size=position_size,
                exec_price=exec_price,
            )
            return None

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
        if Path("KILL_SWITCH").exists():
            log.critical("kill_switch.active", action="order_blocked")
            return OrderResult(
                order_id="BLOCKED",
                market_id=order.market_id,
                status="rejected",
                is_paper=True,
            )

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

        self._init_clob_client()

        try:
            from py_clob_client.order_builder.constants import BUY, SELL
            from py_clob_client.clob_types import OrderArgs, BalanceAllowanceParams, AssetType

            clob_side = BUY if order.side == OrderSide.BUY else SELL

            if not order.token_id:
                raise ValueError(f"No CLOB token_id for market {order.market_id}")

            # For SELL orders: approve the specific conditional token if not yet approved
            if order.side == OrderSide.SELL and order.token_id not in self._approved_tokens:
                try:
                    self._clob_client.update_balance_allowance(
                        BalanceAllowanceParams(
                            asset_type=AssetType.CONDITIONAL,
                            token_id=order.token_id,
                            signature_type=2,
                        )
                    )
                    self._approved_tokens.add(order.token_id)
                    log.info("clob_client.token_approved", token_id=order.token_id[:20])
                except Exception as e:
                    log.warning("clob_client.token_approval_failed",
                                token_id=order.token_id[:20], error=str(e))

            signed_order = self._clob_client.create_and_post_order(
                OrderArgs(
                    token_id=order.token_id,
                    price=order.price,
                    size=order.size,
                    side=clob_side,
                )
            )

            order_id = str(signed_order.get("orderID", signed_order.get("id", "unknown")))

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
            log.error("order.live_error", error=str(e), market_id=order.market_id)
            return OrderResult(
                order_id="ERROR",
                market_id=order.market_id,
                status="rejected",
                is_paper=False,
                error_message=str(e)[:200],
            )

    async def get_order_status(self, order_id: str) -> OrderResult:
        """Query the CLOB API for the current status of an order.

        When the CLOB returns None for an order, it means the order no longer
        exists in the active-order index — either filled-and-settled long ago,
        or cancelled. We return a terminal "cancelled" result so callers can
        drop it from their pending-order tracking instead of re-polling forever.
        """
        self._init_clob_client()
        try:
            raw = self._clob_client.get_order(order_id)
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
        """Cancel a live order on the CLOB. Returns True on success."""
        self._init_clob_client()
        try:
            self._clob_client.cancel(order_id)
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
        self._init_clob_client()
        try:
            resp = self._clob_client.cancel_market_orders(asset_id=token_id)
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
        self._init_clob_client()
        try:
            raw = self._clob_client.get_order_book(token_id)
            # OrderBookSummary is a dataclass with .bids/.asks as lists of OrderSummary
            raw_bids = getattr(raw, "bids", []) or []
            raw_asks = getattr(raw, "asks", []) or []
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
            log.error("orderbook.error", token_id=token_id[:20], error=str(e))
            return OrderBook()
