import math

import pytest

from auramaur.risk.ibkr_math import (
    MIN_VOL_CLOSES, MOMENTUM_HORIZONS, adverse_fill, annualized_volatility,
    min_closes_for_momentum, normalized_momentum, risk_quantity, stop_distance,
)


def test_short_history_degrades_the_momentum_signal_silently():
    """The failure mode that cost the ETF arms four days.

    normalized_momentum never raises on a short series: it drops the horizons
    that do not fit and returns None below the volatility floor. Nothing
    downstream can tell a flat market from a starved one, so the fetch window
    has to be sized from min_closes_for_momentum().
    """
    trend = [100 * 1.001 ** i for i in range(400)]
    assert min_closes_for_momentum() == max(MOMENTUM_HORIZONS) + 1 == 121

    # A "1 M" IBKR window is 20 sessions, 19 once the in-progress one is
    # dropped -- one short of the volatility floor, so the signal vanishes.
    assert annualized_volatility(trend[:MIN_VOL_CLOSES - 1]) is None
    assert normalized_momentum(trend[:19]) is None
    assert normalized_momentum(trend[:MIN_VOL_CLOSES]) is not None

    # Between the floor and the full horizon it returns a *different* signal
    # rather than an error -- the silent part.
    assert normalized_momentum(trend[:60]) != normalized_momentum(
        trend[:min_closes_for_momentum()])


def test_normalized_momentum_is_scale_invariant():
    closes = [100 * math.exp(0.001 * i + 0.002 * math.sin(i)) for i in range(121)]
    assert normalized_momentum(closes) == pytest.approx(
        normalized_momentum([x * 100 for x in closes]))
    assert annualized_volatility(closes) > 0


def test_loss_at_stop_sizing_respects_multiplier_and_fx():
    qty = risk_quantity(25, 0.5, 5, 1, fractional=False)
    assert qty == 10
    assert qty * 0.5 * 5 == 25
    assert risk_quantity(25, 0.5, 100, 1, fractional=False) == 0


def test_stop_and_fill_are_conservative():
    distance = stop_distance(100, 0.20, 2, 0.5)
    assert distance > 0.5
    assert adverse_fill(99.9, 100, "BUY", 2) > 100
    assert adverse_fill(99.9, 100, "SELL", 2) < 99.9
