"""Tests for execution strategy."""

from __future__ import annotations

import pytest

from auramaur.exchange.models import OrderBook, OrderBookLevel, OrderSide, OrderType
from auramaur.strategy.execution import ExecutionStrategy


def _make_book(best_bid: float, best_ask: float) -> OrderBook:
    return OrderBook(
        bids=[OrderBookLevel(price=best_bid, size=100.0)],
        asks=[OrderBookLevel(price=best_ask, size=100.0)],
    )


class TestExecutionStrategy:
    def test_tight_spread_uses_market(self):
        """Spread below threshold should produce MARKET order."""
        strategy = ExecutionStrategy(min_spread_bps=50)
        book = _make_book(0.500, 0.502)  # 4 bps spread
        order_type, price = strategy.compute_order_params(OrderSide.BUY, book)
        assert order_type == OrderType.MARKET

    def test_wide_spread_uses_limit_buy(self):
        """Wide spread BUY should place limit at bid + tick."""
        strategy = ExecutionStrategy(min_spread_bps=50)
        book = _make_book(0.450, 0.500)  # ~1000 bps spread
        order_type, price = strategy.compute_order_params(OrderSide.BUY, book)
        assert order_type == OrderType.LIMIT
        assert price > 0.450  # Above best bid
        assert price <= 0.475  # At or below midpoint

    def test_wide_spread_uses_limit_sell(self):
        """Wide spread SELL should place limit at ask - tick."""
        strategy = ExecutionStrategy(min_spread_bps=50)
        book = _make_book(0.450, 0.500)
        order_type, price = strategy.compute_order_params(OrderSide.SELL, book)
        assert order_type == OrderType.LIMIT
        assert price < 0.500  # Below best ask
        assert price >= 0.475  # At or above midpoint

    def test_empty_book_uses_market(self):
        """Empty order book should use market order."""
        strategy = ExecutionStrategy(min_spread_bps=50)
        book = OrderBook(bids=[], asks=[])
        order_type, price = strategy.compute_order_params(OrderSide.BUY, book)
        assert order_type == OrderType.MARKET

    def test_custom_min_spread(self):
        """Custom spread threshold should be respected."""
        # Use very high threshold — should always use limit
        strategy = ExecutionStrategy(min_spread_bps=1)
        book = _make_book(0.498, 0.502)  # ~8 bps
        order_type, _ = strategy.compute_order_params(OrderSide.BUY, book)
        assert order_type == OrderType.LIMIT
