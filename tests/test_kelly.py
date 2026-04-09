"""Tests for Kelly position sizing."""

import pytest
from auramaur.risk.kelly import KellySizer


def test_basic_kelly():
    sizer = KellySizer(fraction=0.25)
    size = sizer.calculate(
        claude_prob=0.7, market_prob=0.5, bankroll=1000,
        max_stake=25.0,
    )
    # kelly = (0.7 - 0.5) / (1 - 0.5) = 0.4
    # size = 0.4 * 0.25 * 1.0 * 1.0 * (min(1.0, default_liquidity)) * 1000
    # Should be capped at 25.0
    assert 0 < size <= 25.0


def test_no_edge():
    sizer = KellySizer(fraction=0.25)
    size = sizer.calculate(
        claude_prob=0.5, market_prob=0.5, bankroll=1000,
        max_stake=25.0,
    )
    assert size == 0


def test_sell_side_edge():
    """Claude=0.3 vs Market=0.5: YES is overpriced, SELL side has positive edge."""
    sizer = KellySizer(fraction=0.25)
    size = sizer.calculate(
        claude_prob=0.3, market_prob=0.5, bankroll=1000,
        max_stake=25.0,
    )
    assert size > 0  # Positive SELL-side edge


def test_no_edge_equal():
    """Exactly equal probabilities = no edge either direction."""
    sizer = KellySizer(fraction=0.25)
    size = sizer.calculate(
        claude_prob=0.5, market_prob=0.5, bankroll=1000,
        max_stake=25.0,
    )
    assert size == 0


def test_heat_multiplier_reduces_size():
    sizer = KellySizer(fraction=0.25)
    full = sizer.calculate(claude_prob=0.7, market_prob=0.5, bankroll=1000, heat_mult=1.0, max_stake=100)
    reduced = sizer.calculate(claude_prob=0.7, market_prob=0.5, bankroll=1000, heat_mult=0.5, max_stake=100)
    assert reduced < full


def test_red_heat_zero():
    sizer = KellySizer(fraction=0.25)
    size = sizer.calculate(claude_prob=0.7, market_prob=0.5, bankroll=1000, heat_mult=0.0, max_stake=25)
    assert size == 0


def test_capped_at_max_stake():
    sizer = KellySizer(fraction=1.0)  # Full Kelly
    size = sizer.calculate(claude_prob=0.9, market_prob=0.1, bankroll=100000, max_stake=25.0)
    assert size == 25.0


# --- Volatility multiplier tests ---


def test_volatility_multiplier_low_vol():
    """Vol <= 0.05 should return 1.0 (no reduction)."""
    # Stable prices: changes are tiny
    prices = [0.50, 0.51, 0.50, 0.51, 0.50]
    mult = KellySizer.volatility_multiplier(prices)
    assert mult == 1.0


def test_volatility_multiplier_high_vol():
    """Vol > 0.10 should reduce the multiplier significantly."""
    # Wild swings: +0.15, -0.15, +0.15, -0.15
    prices = [0.50, 0.65, 0.50, 0.65, 0.50]
    mult = KellySizer.volatility_multiplier(prices)
    assert mult < 1.0
    assert mult >= 0.3  # Floor


def test_volatility_multiplier_extreme_vol():
    """Extreme volatility should hit the 0.3 floor."""
    prices = [0.20, 0.80, 0.20, 0.80, 0.20]
    mult = KellySizer.volatility_multiplier(prices)
    assert mult == 0.3


def test_volatility_multiplier_insufficient_data():
    """Fewer than 2 price points should return 1.0."""
    assert KellySizer.volatility_multiplier([0.5]) == 1.0
    assert KellySizer.volatility_multiplier([]) == 1.0


def test_volatility_mult_reduces_kelly_size():
    """Passing volatility_mult < 1.0 should reduce position size."""
    sizer = KellySizer(fraction=0.10)  # Smaller fraction so we don't hit max_stake cap
    full = sizer.calculate(
        claude_prob=0.6, market_prob=0.5, bankroll=1000,
        volatility_mult=1.0, max_stake=500,
    )
    reduced = sizer.calculate(
        claude_prob=0.6, market_prob=0.5, bankroll=1000,
        volatility_mult=0.5, max_stake=500,
    )
    assert reduced < full
    assert reduced == pytest.approx(full * 0.5, rel=0.01)
