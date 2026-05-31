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


def test_confidence_multiplier_reduces_size():
    """Confidence multiplier applies to the edge, reducing final size."""
    sizer = KellySizer(fraction=0.25)
    full = sizer.calculate(claude_prob=0.7, market_prob=0.5, bankroll=1000, confidence_mult=1.0, max_stake=500)
    reduced = sizer.calculate(claude_prob=0.7, market_prob=0.5, bankroll=1000, confidence_mult=0.5, max_stake=500)
    # raw_edge = 0.2
    # reduced_edge = 0.1
    # full_kelly = 0.2 / (0.5 * 0.5) = 0.8 -> size = 0.8 * 0.25 * 1000 = 200
    # reduced_kelly = 0.1 / (0.5 * 0.5) = 0.4 -> size = 0.4 * 0.25 * 1000 = 100
    assert reduced < full
    assert reduced == pytest.approx(full * 0.5, rel=0.01)


def test_category_multiplier_reduces_size():
    """Category multiplier applies to the edge, reducing final size."""
    sizer = KellySizer(fraction=0.25)
    full = sizer.calculate(claude_prob=0.7, market_prob=0.5, bankroll=1000, category_mult=1.0, max_stake=500)
    reduced = sizer.calculate(claude_prob=0.7, market_prob=0.5, bankroll=1000, category_mult=0.5, max_stake=500)
    assert reduced < full
    assert reduced == pytest.approx(full * 0.5, rel=0.01)


def test_legacy_multipliers_ignored():
    """Risk-based multipliers passed via kwargs should not affect sizing."""
    sizer = KellySizer(fraction=0.25)
    full = sizer.calculate(claude_prob=0.7, market_prob=0.5, bankroll=1000, max_stake=100)
    ignored = sizer.calculate(
        claude_prob=0.7, market_prob=0.5, bankroll=1000, max_stake=100,
        heat_mult=0.5, volatility_mult=0.5, liquidity_mult=0.5
    )
    assert ignored == full


def test_capped_at_max_stake():
    sizer = KellySizer(fraction=1.0)  # Full Kelly
    size = sizer.calculate(claude_prob=0.9, market_prob=0.1, bankroll=100000, max_stake=25.0)
    assert size == 25.0

