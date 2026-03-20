"""Market making strategy — earn spread by posting two-sided quotes.

Posts limit orders on both sides of prediction markets:
- Bid: BUY YES at bid_price
- Ask: BUY NO at (1 - ask_price)

If both legs fill, we hold YES + NO which resolves to exactly $1.00
regardless of outcome. Profit = (ask_price - bid_price) * size.

Key Polymarket mechanic: you cannot "sell YES" directly. Instead you
BUY NO. So a two-sided quote means two BUY orders on different tokens.

Risk: inventory accumulation (holding directional exposure that can go
to 0 or 1). Managed by skewing quotes away from accumulated side and
capping max inventory per market.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import structlog

from auramaur.exchange.client import PolymarketClient
from auramaur.exchange.models import (
    Market,
    Order,
    OrderBook,
    OrderResult,
    OrderSide,
    OrderType,
    TokenType,
)

log = structlog.get_logger()


@dataclass
class MMQuote:
    """A two-sided quote for market making."""

    market_id: str
    token_yes_id: str
    token_no_id: str
    bid_price: float  # our buy price for YES
    ask_price: float  # our sell price for YES (= 1 - buy_no_price)
    size: float  # tokens per side
    spread_bps: int  # our spread in basis points

    # Tracking order IDs for cancellation
    bid_order_id: str = ""
    ask_order_id: str = ""
    placed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def expected_profit(self) -> float:
        """Profit per token if both legs fill."""
        return (self.ask_price - self.bid_price) * self.size

    @property
    def no_leg_price(self) -> float:
        """Price for the NO token (ask leg)."""
        return round(1.0 - self.ask_price, 2)


class MarketMaker:
    """Provides liquidity on prediction markets for spread capture.

    Strategy:
    1. Select liquid markets with reasonable spreads
    2. Post bid (buy YES) slightly above best bid
    3. Post ask (sell YES, i.e. buy NO) slightly below best ask
    4. If both legs fill, profit = spread * size
    5. Manage inventory: don't accumulate directional exposure

    The user has 0% maker fees on Polymarket, so all spread captured
    is pure profit when both legs fill.
    """

    def __init__(
        self,
        settings,
        exchange: PolymarketClient,
        db,
    ):
        self._settings = settings
        self._exchange = exchange
        self._db = db

        # MM configuration from settings
        mm_cfg = settings.market_maker
        self._min_spread_bps: int = mm_cfg.min_spread_bps
        self._quote_size: float = mm_cfg.quote_size
        self._max_inventory: float = mm_cfg.max_inventory
        self._max_markets: int = mm_cfg.max_markets
        self._refresh_seconds: int = mm_cfg.refresh_seconds

        # State
        self._active_quotes: dict[str, MMQuote] = {}  # market_id -> active quote
        self._inventory: dict[str, float] = {}  # market_id -> net YES tokens (+ = long YES)
        self._pending_orders: dict[str, dict] = {}  # order_id -> {market_id, side, size}

    async def run_cycle(self, markets: list[Market]) -> list[dict]:
        """Run one market making cycle.

        1. Select suitable markets (liquid, wide spread, active)
        2. Cancel stale quotes
        3. Fetch order books and compute fair quotes
        4. Post new two-sided quotes
        5. Track inventory from fills

        Returns list of cycle results (one per market acted on).
        """
        # Kill switch check
        if Path("KILL_SWITCH").exists():
            log.critical("market_maker.kill_switch_active")
            return []

        results: list[dict] = []

        # Step 1: Select suitable markets
        candidates = self._select_mm_markets(markets)
        if not candidates:
            log.debug("market_maker.no_suitable_markets")
            return results

        # Step 2: Cancel stale quotes
        cancelled = await self._cancel_stale_quotes()
        if cancelled:
            log.info("market_maker.cancelled_stale", count=cancelled)

        # Step 3 & 4: Compute and place quotes for each market
        for market in candidates[: self._max_markets]:
            try:
                result = await self._quote_market(market)
                if result:
                    results.append(result)
            except Exception as e:
                log.error(
                    "market_maker.quote_error",
                    market_id=market.id,
                    error=str(e),
                )

        log.info(
            "market_maker.cycle_complete",
            candidates=len(candidates),
            quoted=len(results),
            active_quotes=len(self._active_quotes),
            inventory_markets=len([v for v in self._inventory.values() if abs(v) > 0.01]),
        )

        return results

    def _select_mm_markets(self, markets: list[Market]) -> list[Market]:
        """Select markets suitable for market making.

        Criteria:
        - Has both YES and NO CLOB token IDs
        - Liquidity > $5,000
        - Spread > min_spread_bps (wide enough to profit)
        - Price between 0.15 and 0.85 (not near resolution)
        - Active and not about to expire (>48h to resolution)
        """
        suitable: list[Market] = []
        now = datetime.now(timezone.utc)

        for market in markets:
            # Must have both token IDs for two-sided quoting
            if not market.clob_token_yes or not market.clob_token_no:
                continue

            # Must be active
            if not market.active:
                continue

            # Liquidity floor
            if market.liquidity < 5000:
                continue

            # Price not too extreme (near resolution = high risk)
            if market.outcome_yes_price < 0.15 or market.outcome_yes_price > 0.85:
                continue

            # Time to resolution: at least 48 hours
            if market.end_date:
                hours_remaining = (market.end_date - now).total_seconds() / 3600
                if hours_remaining < 48:
                    continue

            # Spread must be wide enough to profit
            spread_bps = int(market.spread * 10000) if market.spread else 0
            if spread_bps < self._min_spread_bps:
                continue

            # Don't exceed max inventory on this market
            current_inv = abs(self._inventory.get(market.id, 0))
            if current_inv >= self._max_inventory:
                continue

            suitable.append(market)

        # Sort by spread (widest first = most profitable)
        suitable.sort(key=lambda m: m.spread or 0, reverse=True)

        return suitable

    async def _quote_market(self, market: Market) -> dict | None:
        """Fetch order book, compute quotes, and place two-sided order for a market."""
        # Fetch YES order book to determine BBO
        book = await self._exchange.get_order_book(market.clob_token_yes)

        if not book.bids or not book.asks:
            log.debug("market_maker.empty_book", market_id=market.id)
            return None

        # Compute our quote
        quote = self._compute_quotes(market, book)
        if quote is None:
            return None

        # Place the two-sided quote
        is_live = self._settings.is_live
        result = await self._place_two_sided(quote, is_live)

        if result.get("success"):
            self._active_quotes[market.id] = quote

        return result

    def _compute_quotes(self, market: Market, book: OrderBook) -> MMQuote | None:
        """Compute bid/ask prices for a two-sided quote.

        Strategy: post inside the current spread.
        - bid = best_bid + 0.01 (step ahead of resting bids)
        - ask = best_ask - 0.01 (step ahead of resting asks)
        - Ensure our spread >= min_spread_bps
        - Adjust for inventory: if we're long YES, lower bid and raise ask
          (discourage more YES accumulation, encourage selling YES / buying NO)

        Polymarket uses 1-cent tick sizes (0.01 increments), prices 0.01-0.99.
        """
        best_bid = book.best_bid
        best_ask = book.best_ask

        if best_bid is None or best_ask is None:
            return None

        # Start with 1-tick price improvement on both sides
        our_bid = round(best_bid + 0.01, 2)
        our_ask = round(best_ask - 0.01, 2)

        # Inventory skew: adjust quotes based on current inventory
        inventory = self._inventory.get(market.id, 0)
        if abs(inventory) > 0.01:
            # Skew factor: 0.01 per 10 tokens of inventory
            skew = round(inventory / self._max_inventory * 0.03, 2)
            # If long YES (inventory > 0): lower bid (less eager to buy YES),
            # lower ask (more eager to sell YES / buy NO)
            our_bid = round(our_bid - skew, 2)
            our_ask = round(our_ask - skew, 2)

        # Clamp to valid Polymarket price range
        our_bid = max(0.01, min(0.99, our_bid))
        our_ask = max(0.01, min(0.99, our_ask))

        # Sanity: bid must be strictly less than ask
        if our_bid >= our_ask:
            return None

        # Check our spread meets minimum
        our_spread = our_ask - our_bid
        our_spread_bps = int(our_spread * 10000)
        if our_spread_bps < self._min_spread_bps:
            return None

        # Check the NO leg price is valid
        no_price = round(1.0 - our_ask, 2)
        if no_price < 0.01 or no_price > 0.99:
            return None

        # Determine size — scale down if we're near inventory limit
        remaining_capacity = self._max_inventory - abs(inventory)
        size = min(self._quote_size, remaining_capacity)
        if size < 1.0:
            return None

        # Round size to Polymarket precision
        size = round(size, 2)

        # Verify notional is >= $1 on both legs (Polymarket minimum)
        if size * our_bid < 1.0 or size * no_price < 1.0:
            return None

        return MMQuote(
            market_id=market.id,
            token_yes_id=market.clob_token_yes,
            token_no_id=market.clob_token_no,
            bid_price=our_bid,
            ask_price=our_ask,
            size=size,
            spread_bps=our_spread_bps,
        )

    async def _place_two_sided(self, quote: MMQuote, is_live: bool) -> dict:
        """Place both legs of a two-sided quote.

        Bid leg: BUY YES at bid_price
        Ask leg: BUY NO at (1 - ask_price)

        On Polymarket you cannot "sell YES" directly. To sell YES exposure
        you BUY the NO token. If both legs fill at our prices:
        - We pay bid_price per YES token
        - We pay (1 - ask_price) per NO token
        - Total cost = bid_price + (1 - ask_price) < 1.0
        - At resolution, YES + NO = $1.00
        - Profit = 1.0 - bid_price - (1 - ask_price) = ask_price - bid_price

        Both orders are limit orders (GTC) to ensure maker execution.
        """
        no_price = quote.no_leg_price

        # Build bid leg (BUY YES)
        bid_order = Order(
            market_id=quote.market_id,
            exchange="polymarket",
            token_id=quote.token_yes_id,
            side=OrderSide.BUY,
            token=TokenType.YES,
            size=quote.size,
            price=quote.bid_price,
            order_type=OrderType.LIMIT,
            dry_run=not is_live,
        )

        # Build ask leg (BUY NO = effectively selling YES)
        ask_order = Order(
            market_id=quote.market_id,
            exchange="polymarket",
            token_id=quote.token_no_id,
            side=OrderSide.BUY,
            token=TokenType.NO,
            size=quote.size,
            price=no_price,
            order_type=OrderType.LIMIT,
            dry_run=not is_live,
        )

        # Place both legs
        bid_result = await self._exchange.place_order(bid_order)
        ask_result = await self._exchange.place_order(ask_order)

        # Track order IDs
        quote.bid_order_id = bid_result.order_id
        quote.ask_order_id = ask_result.order_id

        # Track pending orders for inventory updates on fill
        if bid_result.status in ("pending", "paper", "filled"):
            self._pending_orders[bid_result.order_id] = {
                "market_id": quote.market_id,
                "side": "bid",
                "size": quote.size,
            }
        if ask_result.status in ("pending", "paper", "filled"):
            self._pending_orders[ask_result.order_id] = {
                "market_id": quote.market_id,
                "side": "ask",
                "size": quote.size,
            }

        # If paper trading, fills are instant — update inventory
        if bid_result.status in ("paper", "filled"):
            self._update_inventory(quote.market_id, "bid", bid_result.filled_size)
        if ask_result.status in ("paper", "filled"):
            self._update_inventory(quote.market_id, "ask", ask_result.filled_size)

        success = bid_result.status not in ("rejected",) and ask_result.status not in ("rejected",)

        log.info(
            "market_maker.quote_placed",
            market_id=quote.market_id,
            bid_price=quote.bid_price,
            ask_price=quote.ask_price,
            no_price=no_price,
            spread_bps=quote.spread_bps,
            size=quote.size,
            bid_status=bid_result.status,
            ask_status=ask_result.status,
            expected_profit=round(quote.expected_profit, 4),
            is_paper=bid_order.dry_run,
        )

        return {
            "success": success,
            "market_id": quote.market_id,
            "bid_price": quote.bid_price,
            "ask_price": quote.ask_price,
            "no_price": no_price,
            "spread_bps": quote.spread_bps,
            "size": quote.size,
            "bid_order_id": bid_result.order_id,
            "ask_order_id": ask_result.order_id,
            "bid_status": bid_result.status,
            "ask_status": ask_result.status,
            "expected_profit": round(quote.expected_profit, 4),
        }

    async def _cancel_stale_quotes(self) -> int:
        """Cancel quotes that are too old (older than 2x refresh interval).

        In live mode, sends cancel requests to the CLOB.
        In paper mode, just clears internal tracking.
        """
        stale_threshold = self._refresh_seconds * 2
        now = datetime.now(timezone.utc)
        cancelled = 0

        stale_market_ids = []
        for market_id, quote in self._active_quotes.items():
            age_seconds = (now - quote.placed_at).total_seconds()
            if age_seconds > stale_threshold:
                stale_market_ids.append(market_id)

        for market_id in stale_market_ids:
            quote = self._active_quotes.pop(market_id)

            # Cancel bid leg
            if quote.bid_order_id and not quote.bid_order_id.startswith("PAPER"):
                try:
                    await self._exchange.cancel_order(quote.bid_order_id)
                except Exception as e:
                    log.debug("market_maker.cancel_bid_error", order_id=quote.bid_order_id, error=str(e))

            # Cancel ask leg
            if quote.ask_order_id and not quote.ask_order_id.startswith("PAPER"):
                try:
                    await self._exchange.cancel_order(quote.ask_order_id)
                except Exception as e:
                    log.debug("market_maker.cancel_ask_error", order_id=quote.ask_order_id, error=str(e))

            # Clean up pending order tracking
            self._pending_orders.pop(quote.bid_order_id, None)
            self._pending_orders.pop(quote.ask_order_id, None)

            cancelled += 1

        return cancelled

    def _update_inventory(self, market_id: str, side: str, size: float) -> None:
        """Track net inventory position.

        Bid fill (bought YES) -> inventory increases (+)
        Ask fill (bought NO = sold YES) -> inventory decreases (-)
        """
        if size <= 0:
            return

        current = self._inventory.get(market_id, 0)
        if side == "bid":
            self._inventory[market_id] = current + size
        elif side == "ask":
            self._inventory[market_id] = current - size

        log.debug(
            "market_maker.inventory_updated",
            market_id=market_id,
            side=side,
            size=size,
            net_inventory=self._inventory[market_id],
        )

    async def check_fills(self) -> list[dict]:
        """Check pending live orders for fills and update inventory.

        Called periodically by the bot task loop.
        """
        filled: list[dict] = []
        completed_ids: list[str] = []

        for order_id, info in self._pending_orders.items():
            # Skip paper orders (already handled at placement)
            if order_id.startswith("PAPER"):
                completed_ids.append(order_id)
                continue

            try:
                result = await self._exchange.get_order_status(order_id)
                if result.status == "filled":
                    self._update_inventory(info["market_id"], info["side"], result.filled_size)
                    completed_ids.append(order_id)
                    filled.append({
                        "order_id": order_id,
                        "market_id": info["market_id"],
                        "side": info["side"],
                        "filled_size": result.filled_size,
                    })
                    log.info(
                        "market_maker.fill",
                        order_id=order_id,
                        market_id=info["market_id"],
                        side=info["side"],
                        filled_size=result.filled_size,
                    )
                elif result.status in ("cancelled", "expired", "rejected"):
                    completed_ids.append(order_id)
            except Exception as e:
                log.debug("market_maker.fill_check_error", order_id=order_id, error=str(e))

        for oid in completed_ids:
            self._pending_orders.pop(oid, None)

        return filled

    def get_inventory_summary(self) -> dict[str, float]:
        """Return current inventory positions for monitoring."""
        return {k: v for k, v in self._inventory.items() if abs(v) > 0.01}
