"""Pure pricing math for Kalshi econ-indicator bins (#2 Phase B core)."""

from __future__ import annotations

import math

from auramaur.strategy.econ_pricing import (
    ECON_SERIES,
    EconSpec,
    bin_edge,
    estimate_distribution,
    indicator_series,
    normal_cdf,
    prob_above,
    spec_for_series,
)


def test_normal_cdf_basics():
    assert abs(normal_cdf(0.0) - 0.5) < 1e-9
    assert abs(normal_cdf(0.0, mean=0.0, sigma=1.0) - 0.5) < 1e-9
    # ~84.13% within +1 sigma
    assert abs(normal_cdf(1.0) - 0.8413) < 1e-3
    # degenerate sigma -> step
    assert normal_cdf(5.0, mean=0.0, sigma=0.0) == 1.0
    assert normal_cdf(-5.0, mean=0.0, sigma=0.0) == 0.0


def test_prob_above_monotone_and_symmetry():
    mean, sigma = 4.0, 1.0
    # higher threshold -> lower P(above) (the ladder monotonicity the bins need)
    ps = [prob_above(t, mean, sigma) for t in (3.0, 4.0, 5.0)]
    assert ps[0] > ps[1] > ps[2]
    assert abs(ps[1] - 0.5) < 1e-9          # at the mean, P(above)=0.5
    # symmetric ±1 sigma
    assert abs(prob_above(3.0, mean, sigma) - 0.8413) < 1e-3
    assert abs(prob_above(5.0, mean, sigma) - 0.1587) < 1e-3


def test_indicator_level_passthrough():
    spec = EconSpec("UNRATE", "level")
    assert indicator_series([4.1, 4.2, 4.3], spec) == [4.1, 4.2, 4.3]


def test_indicator_yoy():
    spec = EconSpec("CPIAUCSL", "yoy", periods_per_year=12)
    # 13 monthly index points; YoY of the 13th vs the 1st
    vals = [100.0] * 12 + [104.0]
    out = indicator_series(vals, spec)
    assert len(out) == 1
    assert abs(out[0] - 4.0) < 1e-9         # +4% YoY


def test_indicator_mom_change_scaled():
    spec = EconSpec("PAYEMS", "mom_change", scale=1000.0)
    # PAYEMS in thousands; +90 (thousand) -> 90,000 jobs
    out = indicator_series([100000.0, 100090.0], spec)
    assert out == [90000.0]


def test_estimate_distribution_floor_and_horizon():
    # flat-ish series: tiny change-vol, but sigma floored to 25% of |mean|
    ind = [4.0, 4.0, 4.01, 3.99, 4.0]
    mean, sigma = estimate_distribution(ind, horizon_periods=1)
    assert abs(mean - 4.0) < 1e-9
    assert sigma >= 0.25 * 4.0 - 1e-9       # floor bites
    # horizon scales sigma up by ~sqrt(h) when above the floor
    noisy = [0.0, 1.0, -1.0, 1.5, -1.2, 0.8, -0.9, 1.1]
    _, s1 = estimate_distribution(noisy, horizon_periods=1, sigma_floor_frac=0.0)
    _, s4 = estimate_distribution(noisy, horizon_periods=4, sigma_floor_frac=0.0)
    assert abs(s4 - s1 * 2.0) < 1e-6        # sqrt(4) = 2


def test_estimate_distribution_insufficient_history():
    assert estimate_distribution([4.0]) is None
    assert estimate_distribution([]) is None


def test_registry_lookup():
    assert spec_for_series("KXCPIYOY").transform == "yoy"
    assert spec_for_series("kxu3").transform == "level"        # case-insensitive
    assert spec_for_series("KXFEDDECISION") is None            # not registered
    assert "KXPAYROLLS" in ECON_SERIES


def test_bin_edge_sign():
    assert bin_edge(0.60, 0.45) > 0     # model richer than market -> buy YES
    assert bin_edge(0.30, 0.45) < 0     # model cheaper -> buy NO
