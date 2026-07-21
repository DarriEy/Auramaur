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

from auramaur.strategy.protocols import ExecutionMode

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from auramaur.killswitch import kill_switch_present

import structlog

from auramaur.exchange.client import PolymarketClient
from auramaur.exchange.models import (
    Market,
    Order,
    OrderBook,
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

    # Uniform Strategy contract (see strategy/protocols.py).
    name = "market_maker"
    execution_mode = ExecutionMode.DIRECT_QUOTING

    def __init__(
        self,
        settings,
        exchange: PolymarketClient,
        db,
        gateway=None,
    ):
        self._settings = settings
        self._exchange = exchange
        self._db = db
        # Placement runs through the ExecutionGateway (the single choke point);
        # lazily built if not injected so existing callers/tests keep working.
        self._gateway = gateway

        # MM configuration from settings
        mm_cfg = settings.market_maker
        self._min_spread_bps: int = mm_cfg.min_spread_bps
        self._max_spread_bps: int = mm_cfg.max_spread_bps
        self._quote_size: float = mm_cfg.quote_size
        self._max_inventory: float = mm_cfg.max_inventory
        self._max_markets: int = mm_cfg.max_markets
        self._refresh_seconds: int = mm_cfg.refresh_seconds
        self._op_timeout: float = mm_cfg.op_timeout_seconds

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
        if kill_switch_present():
            log.critical("market_maker.kill_switch_active")
            return []

        results: list[dict] = []

        # Step 1: Select suitable markets
        candidates = self._select_mm_markets(markets)
        if not candidates:
            log.debug("market_maker.no_suitable_markets")
            return results

        # Per-operation watchdog: a stuck Polymarket call (no timeout) would hang
        # the whole loop indefinitely. Bound each op so a stalled request is
        # abandoned and the cycle continues.
        timeout = self._op_timeout

        # Step 2: Cancel stale quotes
        try:
            cancelled = await asyncio.wait_for(
                self._cancel_stale_quotes(), timeout=timeout)
            if cancelled:
                log.info("market_maker.cancelled_stale", count=cancelled)
        except asyncio.TimeoutError:
            log.warning("market_maker.op_timeout", op="cancel_stale", timeout=timeout)

        # Step 3 & 4: Compute and place quotes for each market
        skip_reasons: dict[str, int] = {}
        for market in candidates[: self._max_markets]:
            try:
                result, skip_reason = await asyncio.wait_for(
                    self._quote_market(market), timeout=timeout)
                if result:
                    results.append(result)
                elif skip_reason:
                    skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1
            except asyncio.TimeoutError:
                # A stuck call on THIS market must not stall the rest of the cycle.
                log.warning("market_maker.op_timeout", op="quote_market",
                            market_id=market.id, timeout=timeout)
                skip_reasons["timeout"] = skip_reasons.get("timeout", 0) + 1
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
            skip_reasons=skip_reasons or None,
        )
        if candidates and not results:
            log.info(
                "market_maker.no_quotes",
                candidates=len(candidates),
                checked=min(len(candidates), self._max_markets),
                skip_reasons=skip_reasons,
                min_spread_bps=self._min_spread_bps,
                quote_size=self._quote_size,
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

        risk = self._settings.risk
        blocked = set(risk.blocked_categories)
        allowed_live = (set(risk.allowed_categories_live)
                        if self._settings.is_live else None)

        for market in markets:
            # Category gate. The MM is graduation-exempt and places orders
            # outside the risk gateway, so until 2026-06-12 it was the last
            # path with NO category policy at all — it filled $6.30 of a
            # tennis match live (Stuttgart Open) hours after the allowlist
            # shipped. "Structural two-sided" only holds while quotes stay
            # flat; a one-sided fill IS directional inventory, so the MM
            # respects the same blocklist (always) and live allowlist as
            # every directional book. Classify on the spot when the
            # discovery payload carries no category.
            from auramaur.strategy.classifier import ensure_category
            category = ensure_category(
                market.question, market.description, market.category)
            if category in blocked:
                continue
            if allowed_live is not None and category not in allowed_live:
                continue

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

            # Spread must be wide enough to profit...
            spread_bps = int(market.spread * 10000) if market.spread else 0
            if spread_bps < self._min_spread_bps:
                continue

            # ...but not SO wide that it's a dead/empty book. A near-100% nominal
            # spread (e.g. bid 0.02 / ask 0.98 = 9600 bps) is not an opportunity —
            # it's a market with no real two-sided liquidity, where neither leg
            # ever fills and we churn cancel/replace every cycle. Reject it.
            if spread_bps > self._max_spread_bps:
                continue

            # Don't exceed max inventory on this market
            current_inv = abs(self._inventory.get(market.id, 0))
            if current_inv >= self._max_inventory:
                continue

            suitable.append(market)

        # Sort by spread, widest first — but now bounded above by max_spread_bps,
        # so dead books no longer dominate the top of the list. Among legitimate
        # spreads, wider = more profit per round-trip.
        suitable.sort(key=lambda m: m.spread or 0, reverse=True)

        return suitable

    async def _quote_market(self, market: Market) -> tuple[dict | None, str | None]:
        """Fetch order book, compute quotes, and place two-sided order for a market."""
        # Cash reserve floor: MM is graduation-exempt and quotes continuously,
        # so by default it absorbs EVERY free live dollar as inventory — a
        # $200 float deposited for directional entries was fully claimed by MM
        # fills within a week (2026-07). New quote pairs are skipped while
        # spendable cash sits at/below the reserve; exits and cancels are
        # never gated, so inventory still unwinds back to cash.
        reserve = float(getattr(self._settings.market_maker, "cash_reserve_usd", 0.0))
        if reserve > 0 and self._settings.is_live:
            probe = getattr(self._exchange, "_free_collateral_usd", None)
            free = await probe() if probe is not None else None
            if free is not None and free <= reserve:
                log.info("market_maker.reserve_floor", free=round(free, 2),
                         reserve=reserve, market_id=market.id)
                return None, "cash_reserve_floor"

        # Fetch YES order book to determine BBO
        book = await self._exchange.get_order_book(market.clob_token_yes)

        if not book.bids or not book.asks:
            log.debug("market_maker.empty_book", market_id=market.id)
            return None, "empty_book"

        # Compute our quote
        quote, skip_reason = self._compute_quotes(market, book)
        if quote is None:
            return None, skip_reason

        # Cancel the previous quote BEFORE posting its replacement. The
        # overwrite below used to drop the old legs' order ids on the floor,
        # leaking a resting GTC pair onto the book every refresh cycle —
        # _cancel_stale_quotes never saw them again, only the silent TTL
        # reaper (and, after a freeze, nobody) cleaned them up.
        prior = self._active_quotes.pop(market.id, None)
        if prior is not None:
            # Identical-requote guard: when the freshly computed quote matches
            # the resting pair and BOTH legs are still tracked as pending
            # (check_fills prunes filled/cancelled/rejected ids, and paper ids
            # are pruned immediately — so this path is effectively live-only),
            # KEEP the resting orders. Cancel+replace of an unchanged quote
            # forfeits book time-priority every refresh cycle, which is why
            # the live MM could quote for hours without a fill (2026-07-20:
            # identical 0.46/0.48 requoted every ~34s on three markets).
            # placed_at is refreshed so _cancel_stale_quotes doesn't reap a
            # deliberately-kept quote; if the book moves or the market drops
            # out of the candidate set, this guard fails and the normal
            # cancel/stale paths run — the orphaned-GTC leak cannot return.
            if (
                abs(prior.bid_price - quote.bid_price) < 1e-9
                and abs(prior.ask_price - quote.ask_price) < 1e-9
                and prior.size == quote.size
                and prior.bid_order_id in self._pending_orders
                and prior.ask_order_id in self._pending_orders
            ):
                prior.placed_at = datetime.now(timezone.utc)
                self._active_quotes[market.id] = prior
                log.debug(
                    "market_maker.quote_kept",
                    market_id=market.id,
                    bid=prior.bid_price,
                    ask=prior.ask_price,
                )
                return None, "quote_unchanged"
            await self._cancel_quote(prior)

        # Place the two-sided quote
        is_live = self._settings.is_live
        result = await self._place_two_sided(quote, is_live)

        if result.get("success"):
            self._active_quotes[market.id] = quote

        return result, None

    def _compute_quotes(self, market: Market, book: OrderBook) -> tuple[MMQuote | None, str | None]:
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
            return None, "no_bbo"

        # Dead-book guard on the LIVE book (selection used market.spread, a summary
        # field that can be stale or differ from the CLOB BBO). If the actual best
        # bid/ask are this far apart, there's no genuine two-sided market — quoting
        # into it just churns and risks one-sided fills at an extreme price.
        book_spread_bps = int((best_ask - best_bid) * 10000)
        if book_spread_bps > self._max_spread_bps:
            log.debug(
                "market_maker.dead_book",
                market_id=market.id,
                best_bid=best_bid,
                best_ask=best_ask,
                book_spread_bps=book_spread_bps,
                max_spread_bps=self._max_spread_bps,
            )
            return None, "dead_book"

        # Polymarket has 1-cent ticks. Try to price-improve by one tick on each
        # side; if the underlying spread is too tight for that (which is the
        # common case — most liquid markets sit at 2-3 cent spreads), quote at
        # the BBO instead (join the queue) so we still capture spread as maker.
        step = 0.01
        raw_bid = best_bid + step
        raw_ask = best_ask - step
        raw_spread_bps = int((raw_ask - raw_bid) * 10000)
        if raw_spread_bps < self._min_spread_bps:
            our_bid = round(best_bid, 2)
            our_ask = round(best_ask, 2)
        else:
            our_bid = round(raw_bid, 2)
            our_ask = round(raw_ask, 2)

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
            log.debug(
                "market_maker.quote_rejected",
                market_id=market.id,
                reason="bid_gte_ask",
                our_bid=our_bid,
                our_ask=our_ask,
                best_bid=best_bid,
                best_ask=best_ask,
            )
            return None, "bid_gte_ask"

        # Check our spread meets minimum
        our_spread = our_ask - our_bid
        our_spread_bps = int(our_spread * 10000)
        if our_spread_bps < self._min_spread_bps:
            log.debug(
                "market_maker.quote_rejected",
                market_id=market.id,
                reason="spread_too_narrow",
                our_spread_bps=our_spread_bps,
                min_spread_bps=self._min_spread_bps,
            )
            return None, "spread_too_narrow"

        # Check the NO leg price is valid
        no_price = round(1.0 - our_ask, 2)
        if no_price < 0.01 or no_price > 0.99:
            return None, "invalid_no_price"

        # Determine size — scale down if we're near inventory limit, but bump
        # above Polymarket's $1 minimum when the configured quote size would
        # otherwise make both-sided quoting impossible on low-price markets.
        remaining_capacity = self._max_inventory - abs(inventory)
        size = min(self._quote_size, remaining_capacity)
        if size < 1.0:
            return None, "inventory_capacity"

        min_size = max(1.0 / our_bid, 1.0 / no_price)
        if size * our_bid < 1.0 or size * no_price < 1.0:
            bumped = math.ceil(min_size * 100) / 100
            if bumped > remaining_capacity:
                return None, "min_notional"
            size = bumped

        # Round size to Polymarket precision
        size = round(size, 2)

        return MMQuote(
            market_id=market.id,
            token_yes_id=market.clob_token_yes,
            token_no_id=market.clob_token_no,
            bid_price=our_bid,
            ask_price=our_ask,
            size=size,
            spread_bps=our_spread_bps,
        ), None

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

        # Build bid leg (BUY YES). post_only=True: MM is only profitable as
        # maker — if our quote somehow crosses (stale book), reject rather
        # than pay taker and lose the spread.
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
            post_only=True,
            source="market_maker",
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
            post_only=True,
            source="market_maker",
        )

        # Place both legs through the gateway's two-sided adapter (bid then ask,
        # not both-or-nothing — the MM owns one-legged cleanup below). The gateway
        # is the single placement choke point; no direct exchange.place_order here.
        if self._gateway is None:
            from auramaur.broker.execution_gateway import ExecutionGateway
            self._gateway = ExecutionGateway(
                router=None, exchange=self._exchange, exchange_name="polymarket",
                settings=self._settings, db=self._db, pnl_tracker=None)
        bid_result, ask_result = await self._gateway.place_quote_pair(
            bid_order, ask_order, exchange=self._exchange)

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

        bid_live = bid_result.status not in ("rejected",)
        ask_live = ask_result.status not in ("rejected",)

        # Partial placement: one leg rested, the other was rejected — usually a
        # post-only leg that would cross a moved book. A one-legged quote is
        # naked directional exposure the MM never wants, and leaving it here is
        # how duplicates stack: success would be False below, so _quote_market
        # never records the quote in _active_quotes, so the resting leg is never
        # cancelled on refresh and a fresh pair stacks on top every cycle (the
        # orphaned-BUY cash-lock). Cancel the survivor now for a clean slate and
        # retry next cycle. The periodic order-monitor reconcile backstops a
        # cancel that itself fails.
        if bid_live != ask_live:
            survivor_id = bid_result.order_id if bid_live else ask_result.order_id
            survivor_status = bid_result.status if bid_live else ask_result.status
            if (
                survivor_status not in ("paper", "filled")
                and survivor_id
                and not str(survivor_id).startswith("PAPER")
            ):
                try:
                    await self._exchange.cancel_order(survivor_id)
                except Exception as e:
                    log.debug(
                        "market_maker.partial_leg_cancel_error",
                        order_id=survivor_id, error=str(e),
                    )
            self._pending_orders.pop(bid_result.order_id, None)
            self._pending_orders.pop(ask_result.order_id, None)
            log.warning(
                "market_maker.partial_quote_cancelled",
                market_id=quote.market_id,
                bid_status=bid_result.status,
                ask_status=ask_result.status,
            )

        success = bid_live and ask_live

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
            await self._cancel_quote(quote)
            cancelled += 1

        return cancelled

    async def _cancel_quote(self, quote) -> None:
        """Cancel both legs of a quote and drop its pending-order tracking."""
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
