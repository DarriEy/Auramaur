import math

import pytest

from auramaur.risk.ibkr_math import (
    adverse_fill, annualized_volatility, normalized_momentum, risk_quantity,
    stop_distance,
)


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
