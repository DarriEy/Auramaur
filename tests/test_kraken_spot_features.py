"""Spot-market math for the Kraken desk call.

The number that matters most is `edge_needed_prob`: the old design never told
the model what a trade costs, so it produced views worth ~44bps against ~62bps
of round-trip fees. These tests pin the cost arithmetic hardest.
"""

import math

import pytest

from auramaur.strategy.kraken_spot_features import (
    book_imbalance, breakeven_move_pct, build_features, desk_prompt,
    drawdown_from_high, edge_needed_prob, expected_move_pct,
    normalized_momentum_sigma, range_position, realized_vol_annual, vol_regime,
)


def _series(n, drift=0.0, wobble=0.01, freq=0.7, start=100.0):
    return [start * (1 + drift) ** i * (1 + wobble * math.sin(i * freq))
            for i in range(n)]


def test_breakeven_includes_both_sides_of_the_fee():
    """Kraken charges 0.26% PER SIDE. A round trip starts 52bps underwater
    before the spread is crossed twice."""
    assert breakeven_move_pct(0.0, 0.26) == pytest.approx(0.52)
    # 20bps spread adds 0.20%, and slippage is charged on both legs.
    assert breakeven_move_pct(20.0, 0.26) == pytest.approx(0.72)
    assert breakeven_move_pct(20.0, 0.26, slippage_bps=5.0) == pytest.approx(0.82)


def test_edge_needed_is_the_number_the_old_design_hid():
    """At a 3-day horizon Kraken's fee needs a probability the model never
    reached; lengthening the horizon is what makes the trade payable."""
    vol = 60.0                                  # 60% annualised, typical crypto
    breakeven = breakeven_move_pct(10.0, 0.26)  # ~0.62%

    short = expected_move_pct(vol, 72)          # 3 days
    long = expected_move_pct(vol, 72 * 7)       # 3 weeks
    assert long > short

    p_short = edge_needed_prob(breakeven, short)
    p_long = edge_needed_prob(breakeven, long)
    assert p_short > p_long, "a longer horizon must lower the bar"
    assert p_short > 0.55, "3 days needs conviction the model does not have"

    # A horizon so short that no probability pays returns None, which is the
    # honest answer rather than a number above 1.
    assert edge_needed_prob(breakeven, expected_move_pct(vol, 0.5)) is None
    assert edge_needed_prob(breakeven, None) is None


def test_momentum_is_volatility_normalised_so_pairs_compare():
    """A 3% move in a calm pair is not the same event as 3% in a wild one."""
    calm = _series(400, drift=0.0005, wobble=0.002)
    wild = _series(400, drift=0.0005, wobble=0.05)
    calm_m = normalized_momentum_sigma(calm, 72)
    wild_m = normalized_momentum_sigma(wild, 72)
    assert calm_m is not None and wild_m is not None
    assert abs(calm_m) > abs(wild_m), "same drift, more vol -> weaker signal"
    # Too little history says nothing rather than guessing.
    assert normalized_momentum_sigma(_series(10), 72) is None


def test_vol_regime_names_the_state_against_the_pairs_own_history():
    steady = _series(600, wobble=0.01)
    assert vol_regime(steady) in {"normal", "quiet", "elevated"}
    # A calm history that turns violent in the last day reads as stressed.
    spiky = _series(600, wobble=0.004)
    spiky = spiky[:-24] + [spiky[-24] * (1 + 0.06 * (-1) ** i) for i in range(24)]
    assert vol_regime(spiky) in {"stressed", "elevated"}
    assert vol_regime(_series(30)) == "unknown"


def test_range_position_and_drawdown_locate_price_in_its_range():
    rising = [100 + i for i in range(200)]
    assert range_position(rising) == pytest.approx(100.0)
    assert drawdown_from_high(rising) == pytest.approx(0.0)
    falling = [300 - i for i in range(200)]
    assert range_position(falling) == pytest.approx(0.0)
    assert drawdown_from_high(falling) < 0
    assert range_position([100.0] * 200) is None      # no range, no answer


def test_book_imbalance_is_signed_and_bounded():
    assert book_imbalance([(1, 10)], [(2, 10)]) == pytest.approx(0.0)
    assert book_imbalance([(1, 30)], [(2, 10)]) == pytest.approx(0.5)
    assert book_imbalance([(1, 0)], [(2, 10)]) == pytest.approx(-1.0)
    assert book_imbalance([], []) is None


def test_features_degrade_to_none_rather_than_guess():
    thin = build_features("XBTUSDC", [100.0, 101.0], bid=100.0, ask=100.5,
                          fee_pct_per_side=0.26, horizon_hours=72)
    assert thin.realized_vol_annual_pct is None
    assert thin.momentum_24h_sigma is None
    assert thin.vol_regime == "unknown"
    # Cost arithmetic still works — it needs no history.
    assert thin.breakeven_move_pct > 0.5


def test_the_prompt_block_states_the_cost_in_words():
    closes = _series(500, drift=0.0004, wobble=0.02)
    f = build_features("ETHUSDC", closes, bid=1880.0, ask=1881.0,
                       fee_pct_per_side=0.26, horizon_hours=72,
                       bids=[(1879, 5)], asks=[(1882, 15)])
    block = f.as_prompt_block()
    assert "COST TO TRADE" in block and "break even" in block
    assert "ETHUSDC" in block
    assert f.book_imbalance is not None and f.book_imbalance < 0   # ask-heavy


def test_desk_prompt_shows_the_book_not_just_the_market():
    """A desk decides differently when already long. The per-pair call could
    see neither the position nor the P&L."""
    closes = _series(400, wobble=0.02)
    f = build_features("XBTUSDC", closes, bid=64000.0, ask=64010.0,
                       fee_pct_per_side=0.26, horizon_hours=72)
    flat = desk_prompt([f], horizon_hours=72, budget_usd=60.0,
                       open_positions=[], recent_pnl_usd=-43.98)
    assert "the book is flat" in flat
    assert "-43.98" in flat

    held = desk_prompt([f], horizon_hours=72, budget_usd=60.0,
                       open_positions=[{"pair": "XBTUSDC", "quantity": 0.001,
                                        "entry": 63000.0,
                                        "unrealized_usd": 1.0}],
                       recent_pnl_usd=-43.98)
    assert "Open positions:" in held and "XBTUSDC" in held
