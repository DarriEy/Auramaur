"""Regime switching: growth mode at small equity, preservation at large."""

from __future__ import annotations

import pytest

from auramaur.risk.regime import (
    GROWTH_EQUITY_MAX,
    GROWTH_KELLY_FRACTION,
    GROWTH_MAX_STAKE_PCT,
    GROWTH_MIN_EDGE_PCT,
    PRESERVATION_EQUITY_MIN,
    resolve_regime,
)


# Preservation defaults that mirror config/defaults.yaml values.
BASE_KELLY = 0.30
BASE_MAX_STAKE = 25.0
BASE_MIN_EDGE = 3.5


def test_growth_regime_at_capital_starved_equity():
    # Bot at $450 (the current reality) should use growth params.
    r = resolve_regime(450.0, BASE_KELLY, BASE_MAX_STAKE, BASE_MIN_EDGE)
    assert r.name == "growth"
    assert r.kelly_fraction == GROWTH_KELLY_FRACTION
    assert r.min_edge_pct == GROWTH_MIN_EDGE_PCT
    # 10% of $450 = $45 stake cap — scales with book, unlike flat $25.
    assert r.max_stake == pytest.approx(450.0 * GROWTH_MAX_STAKE_PCT / 100.0)


def test_growth_cap_still_beats_preservation_at_tiny_book():
    # At $200, 10% is only $20 — less than the $25 preservation flat cap.
    # That's intentional: can't risk >10% of book on one market in growth.
    r = resolve_regime(200.0, BASE_KELLY, BASE_MAX_STAKE, BASE_MIN_EDGE)
    assert r.name == "growth"
    assert r.max_stake == pytest.approx(20.0)


def test_preservation_regime_at_large_equity():
    r = resolve_regime(10_000.0, BASE_KELLY, BASE_MAX_STAKE, BASE_MIN_EDGE)
    assert r.name == "preservation"
    assert r.kelly_fraction == BASE_KELLY
    assert r.max_stake == BASE_MAX_STAKE
    assert r.min_edge_pct == BASE_MIN_EDGE


def test_transition_regime_interpolates():
    # Halfway through the transition band should give halfway between
    # growth and preservation values for every param.
    mid = (GROWTH_EQUITY_MAX + PRESERVATION_EQUITY_MIN) / 2.0
    r = resolve_regime(mid, BASE_KELLY, BASE_MAX_STAKE, BASE_MIN_EDGE)
    assert r.name == "transition"
    expected_kelly = (GROWTH_KELLY_FRACTION + BASE_KELLY) / 2.0
    expected_min_edge = (GROWTH_MIN_EDGE_PCT + BASE_MIN_EDGE) / 2.0
    assert r.kelly_fraction == pytest.approx(expected_kelly)
    assert r.min_edge_pct == pytest.approx(expected_min_edge)


def test_transition_is_continuous_at_boundaries():
    # Just-below boundary should give growth; just-at-boundary should
    # start the transition from the same params (no discontinuity).
    just_below = GROWTH_EQUITY_MAX - 0.01
    at = GROWTH_EQUITY_MAX
    just_above = GROWTH_EQUITY_MAX + 0.01
    g = resolve_regime(just_below, BASE_KELLY, BASE_MAX_STAKE, BASE_MIN_EDGE)
    t_start = resolve_regime(at, BASE_KELLY, BASE_MAX_STAKE, BASE_MIN_EDGE)
    t_tiny = resolve_regime(just_above, BASE_KELLY, BASE_MAX_STAKE, BASE_MIN_EDGE)
    # Kelly fraction is continuous
    assert abs(g.kelly_fraction - t_start.kelly_fraction) < 1e-6
    assert abs(t_start.kelly_fraction - t_tiny.kelly_fraction) < 1e-3


def test_unknown_equity_falls_back_to_preservation():
    r = resolve_regime(0.0, BASE_KELLY, BASE_MAX_STAKE, BASE_MIN_EDGE)
    assert r.name == "unknown"
    assert r.kelly_fraction == BASE_KELLY
    assert r.max_stake == BASE_MAX_STAKE
    assert r.min_edge_pct == BASE_MIN_EDGE


def test_negative_equity_falls_back_to_preservation():
    r = resolve_regime(-100.0, BASE_KELLY, BASE_MAX_STAKE, BASE_MIN_EDGE)
    assert r.name == "unknown"


def test_growth_applied_uniformly_below_threshold():
    for equity in (50.0, 250.0, 500.0, 999.0):
        r = resolve_regime(equity, BASE_KELLY, BASE_MAX_STAKE, BASE_MIN_EDGE)
        assert r.name == "growth"
        assert r.kelly_fraction == GROWTH_KELLY_FRACTION
        assert r.max_stake == pytest.approx(equity * GROWTH_MAX_STAKE_PCT / 100.0)
