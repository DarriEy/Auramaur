"""Depth-aware order-book primitives: fill_to_size (walk-the-book VWAP +
marginal sweep price) and depth_within (in-budget capacity)."""

from __future__ import annotations

from auramaur.exchange.models import OrderBook, OrderBookLevel


def _book() -> OrderBook:
    # asks deliberately worst-first to prove ordering-agnosticism
    return OrderBook(
        bids=[OrderBookLevel(price=0.40, size=100.0),
              OrderBookLevel(price=0.45, size=50.0)],
        asks=[OrderBookLevel(price=0.60, size=100.0),
              OrderBookLevel(price=0.50, size=30.0),
              OrderBookLevel(price=0.55, size=20.0)],
    )


def test_fill_within_best_level_prices_at_best():
    filled, vwap, marginal = _book().fill_to_size(10.0, is_buy=True)
    assert filled == 10.0
    assert abs(vwap - 0.50) < 1e-9
    assert abs(marginal - 0.50) < 1e-9  # fits in the 0.50 level → limit is the ask


def test_fill_sweeps_multiple_levels_vwap_and_marginal():
    # 40 shares: 30@0.50 + 10@0.55 -> vwap, marginal = 0.55
    filled, vwap, marginal = _book().fill_to_size(40.0, is_buy=True)
    assert filled == 40.0
    assert abs(vwap - (30 * 0.50 + 10 * 0.55) / 40) < 1e-9
    assert abs(marginal - 0.55) < 1e-9


def test_fill_capped_by_total_depth():
    # only 150 shares on the asks; requesting more returns what's available
    filled, vwap, marginal = _book().fill_to_size(1000.0, is_buy=True)
    assert filled == 150.0
    assert abs(marginal - 0.60) < 1e-9


def test_fill_empty_side_returns_zeros():
    assert OrderBook(asks=[]).fill_to_size(10.0, is_buy=True) == (0.0, 0.0, 0.0)
    assert _book().fill_to_size(0.0, is_buy=True) == (0.0, 0.0, 0.0)


def test_depth_within_buy_sums_asks_under_cap():
    b = _book()
    assert b.depth_within(0.50, is_buy=True) == 30.0          # only the 0.50 level
    assert b.depth_within(0.55, is_buy=True) == 50.0          # 0.50 + 0.55
    assert b.depth_within(0.60, is_buy=True) == 150.0         # all


def test_depth_within_sell_sums_bids_over_cap():
    b = _book()
    assert b.depth_within(0.45, is_buy=False) == 50.0         # only the 0.45 bid
    assert b.depth_within(0.40, is_buy=False) == 150.0        # both bids


def test_sell_walks_bids_highest_first():
    # selling 60: 50@0.45 + 10@0.40
    filled, vwap, marginal = _book().fill_to_size(60.0, is_buy=False)
    assert filled == 60.0
    assert abs(vwap - (50 * 0.45 + 10 * 0.40) / 60) < 1e-9
    assert abs(marginal - 0.40) < 1e-9
