"""Smart order router — decides limit vs market and computes optimal price."""

from __future__ import annotations

import structlog

from auramaur.exchange.models import (
    Market,
    Order,
    OrderBook,
    OrderSide,
    OrderType,
    Signal,
    TokenType,
)

log = structlog.get_logger()


class SmartOrderRouter:
    """Builds optimally-priced orders by inspecting the order book.

    The router delegates to ``exchange.prepare_order()`` for the base order
    (which handles Polymarket's always-BUY / token-swap semantics), then
    refines the price and order type based on current book conditions.
    """

    def __init__(self, settings, exchange):
        self._settings = settings
        self._exchange = exchange

    async def route(
        self,
        signal: Signal,
        market: Market,
        size_dollars: float,
        is_live: bool,
    ) -> Order | None:
        """Build an optimally-priced order.

        1. Call ``exchange.prepare_order()`` to get the base order (handles
           Polymarket token swap logic).
        2. Fetch the order book for the relevant token.
        3. Decision matrix:
           - spread >= 0.03 AND edge > 10%  -> LIMIT at best_bid + 0.01
             (capture spread, patient fill)
           - spread < 0.03 OR edge > 20%    -> MARKET (cross spread, fast
             execution)
        4. Adjust price based on order book depth.
        5. Return ``Order`` with optimal price and type.
        """
        base_order = self._exchange.prepare_order(
            signal=signal,
            market=market,
            position_size=size_dollars,
            is_live=is_live,
        )
        if base_order is None:
            log.info(
                "router.no_base_order",
                market_id=market.id,
                size_dollars=size_dollars,
            )
            return None

        # Fetch live order book for the token we're trading
        try:
            book = await self._exchange.get_order_book(base_order.token_id)
        except Exception:
            book = OrderBook()  # Empty book — fall back to market order

        edge_pct = abs(signal.edge)
        spread = book.spread

        # Determine order type via decision matrix
        # On 0% fee exchanges (Polymarket reward tier), limit orders are almost
        # always better — you get price improvement for free.
        #   - spread < $0.01: limit order (tight market, capture the spread)
        #   - edge > 40%:     market order (urgency overrides price)
        #   - otherwise:      limit order with price improvement (+1 tick)
        has_book = spread is not None and (book.best_bid is not None or book.best_ask is not None)

        if not has_book:
            # No book data — fall back to market order
            order_type = OrderType.MARKET
            limit_price = base_order.price
            log.info(
                "router.market_order",
                market_id=market.id,
                edge_pct=round(edge_pct, 2),
                reason="no_book",
            )
        elif edge_pct > 40.0:
            # Very high urgency — cross immediately
            order_type = OrderType.MARKET
            limit_price = base_order.price
            log.info(
                "router.market_order",
                market_id=market.id,
                edge_pct=round(edge_pct, 2),
                reason="high_urgency",
            )
        else:
            # Limit order — prefer price improvement on 0% fee exchange
            candidate_price = self._compute_limit_price(
                book, base_order.side, base_order.token
            )

            # Sanity check: limit price shouldn't deviate >30% from fair price
            fair_price = base_order.price
            if fair_price > 0 and abs(candidate_price - fair_price) / fair_price > 0.30:
                # Book too thin — use market order at fair price
                order_type = OrderType.MARKET
                limit_price = fair_price
                log.info(
                    "router.market_order",
                    market_id=market.id,
                    edge_pct=round(edge_pct, 2),
                    reason="thin_book",
                    candidate=candidate_price,
                    fair=fair_price,
                )
            else:
                order_type = OrderType.LIMIT
                limit_price = candidate_price
                reason = "tight_spread" if (spread is not None and spread < 0.01) else "price_improvement"
                log.info(
                    "router.limit_order",
                    market_id=market.id,
                    spread=round(spread, 4) if spread else None,
                    edge_pct=round(edge_pct, 2),
                    limit_price=limit_price,
                    reason=reason,
                )

        # Post-only on LIMIT orders: we use `best_bid + 1 tick` / `best_ask - 1
        # tick` as the price, which shouldn't cross — but the book can move in
        # the gap between fetch and submit. post_only guarantees we don't
        # accidentally take, which keeps us on the maker-reward side of the
        # fee schedule and flushes out any router/pricing mistakes loudly
        # (rejection) instead of quietly (taker fill at a worse price).
        want_post_only = order_type == OrderType.LIMIT

        # Build the final order with refined price / type
        routed_order = base_order.model_copy(
            update={
                "price": limit_price,
                "order_type": order_type,
                "post_only": want_post_only,
            }
        )

        log.info(
            "router.order_ready",
            market_id=market.id,
            token=routed_order.token.value,
            side=routed_order.side.value,
            order_type=routed_order.order_type.value,
            price=routed_order.price,
            size=routed_order.size,
        )

        return routed_order

    # ------------------------------------------------------------------
    # Price helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_limit_price(
        book: OrderBook,
        side: OrderSide,
        token: TokenType,
    ) -> float:
        """Compute a limit price one tick inside the book.

        For BUY orders:  best_bid + 0.01 (step ahead of resting bids).
        For SELL orders: best_ask - 0.01 (step ahead of resting asks).

        When 1-tick improvement would cross (1-cent spread markets), fall
        back to joining the BBO instead — post_only would otherwise reject.

        Result is clamped to the valid Polymarket tick range [0.01, 0.99].
        """
        if side == OrderSide.BUY:
            if book.best_bid is not None and book.best_ask is not None:
                improved = book.best_bid + 0.01
                # If improvement would touch or cross best_ask, join the bid
                # queue instead (still a maker post, at the same price as
                # everyone resting at best_bid).
                price = book.best_bid if improved >= book.best_ask else improved
            elif book.best_bid is not None:
                price = book.best_bid + 0.01
            elif book.best_ask is not None:
                price = book.best_ask - 0.01
            else:
                price = 0.50
        else:
            # SELL side
            if book.best_ask is not None and book.best_bid is not None:
                improved = book.best_ask - 0.01
                price = book.best_ask if improved <= book.best_bid else improved
            elif book.best_ask is not None:
                price = book.best_ask - 0.01
            elif book.best_bid is not None:
                price = book.best_bid + 0.01
            else:
                price = 0.50

        # Clamp to valid Polymarket price range and round to tick
        price = max(0.01, min(0.99, round(price, 2)))
        return price
