"""Execution strategy — limit order placement and spread capture."""

from __future__ import annotations

import structlog

from auramaur.exchange.models import OrderBook, OrderSide, OrderType

log = structlog.get_logger()

# Minimum spread in basis points to justify a limit order
DEFAULT_MIN_SPREAD_BPS = 50


class ExecutionStrategy:
    """Determines whether to use MARKET or LIMIT orders and computes optimal prices."""

    def __init__(self, min_spread_bps: int = DEFAULT_MIN_SPREAD_BPS):
        self.min_spread_bps = min_spread_bps

    def compute_order_params(
        self,
        side: OrderSide,
        order_book: OrderBook,
    ) -> tuple[OrderType, float]:
        """Decide order type and price based on the order book.

        If spread > min threshold: LIMIT order at one tick inside best bid/ask
        (biased toward our side for better fill probability).
        If spread <= threshold: MARKET order (tight spread = no benefit).

        Args:
            side: BUY or SELL.
            order_book: Current order book with bids and asks.

        Returns:
            (order_type, price) tuple.
        """
        best_bid = order_book.best_bid
        best_ask = order_book.best_ask

        if best_bid is None or best_ask is None:
            # No order book depth — use market order at midpoint
            mid = order_book.midpoint or 0.5
            return (OrderType.MARKET, mid)

        spread = best_ask - best_bid
        midpoint = (best_bid + best_ask) / 2.0
        spread_bps = (spread / midpoint) * 10_000 if midpoint > 0 else 0

        if spread_bps <= self.min_spread_bps:
            # Tight spread — market order
            price = best_ask if side == OrderSide.BUY else best_bid
            log.debug(
                "execution.market_order",
                spread_bps=round(spread_bps, 1),
                price=price,
            )
            return (OrderType.MARKET, price)

        # Wide spread — limit order one tick inside
        tick = 0.001  # Polymarket minimum tick
        if side == OrderSide.BUY:
            # Place bid one tick above best bid (inside the spread)
            price = best_bid + tick
        else:
            # Place ask one tick below best ask (inside the spread)
            price = best_ask - tick

        # Clamp to midpoint — don't cross the spread
        if side == OrderSide.BUY:
            price = min(price, midpoint)
        else:
            price = max(price, midpoint)

        log.debug(
            "execution.limit_order",
            spread_bps=round(spread_bps, 1),
            price=price,
            best_bid=best_bid,
            best_ask=best_ask,
        )
        return (OrderType.LIMIT, round(price, 3))
