"""Pure pricing math for Kalshi econ-indicator bins (#2 Phase B core)."""

from __future__ import annotations


from dataclasses import dataclass

from auramaur.strategy.econ_pricing import (
    ECON_SERIES,
    EconSpec,
    bin_edge,
    estimate_distribution,
    indicator_series,
    kalshi_macro_predicate,
    normal_cdf,
    parse_kalshi_period,
    prob_above,
    spec_for_series,
)


@dataclass
class _FakeKalshiMarket:
    ticker: str = ""
    exchange: str = "kalshi"
    strike_type: str = ""
    floor_strike: float | None = None
    cap_strike: float | None = None


def test_parse_kalshi_period():
    assert parse_kalshi_period("26NOV") == "2026-11"
    assert parse_kalshi_period("26jun") == "2026-06"
    assert parse_kalshi_period("26Q4") is None      # not a monthly code
    assert parse_kalshi_period("") is None


def test_kalshi_macro_predicate_greater_threshold():
    m = _FakeKalshiMarket(ticker="KXCPIYOY-26NOV-T4.5", strike_type="greater",
                          floor_strike=4.5)
    assert kalshi_macro_predicate(m) == {
        "indicator": "KXCPIYOY", "operator": ">",
        "threshold": 4.5, "reference_period": "2026-11"}


def test_kalshi_macro_predicate_payrolls_absolute_threshold():
    m = _FakeKalshiMarket(ticker="KXPAYROLLS-26JUN-T175000",
                          strike_type="greater", floor_strike=175000.0)
    p = kalshi_macro_predicate(m)
    assert p["indicator"] == "KXPAYROLLS" and p["threshold"] == 175000.0
    assert p["reference_period"] == "2026-06" and p["operator"] == ">"


def test_kalshi_macro_predicate_rejects_unknown_or_compound():
    # unknown series prefix
    assert kalshi_macro_predicate(
        _FakeKalshiMarket(ticker="KXNFL-26NOV-T4.5", strike_type="greater",
                          floor_strike=4.5)) is None
    # 'between' / structured bin -> no single-threshold predicate
    assert kalshi_macro_predicate(
        _FakeKalshiMarket(ticker="KXCPIYOY-26NOV-B4.5", strike_type="between",
                          floor_strike=4.4, cap_strike=4.5)) is None
    # non-monthly period
    assert kalshi_macro_predicate(
        _FakeKalshiMarket(ticker="KXU3-26Q4-T4.5", strike_type="greater",
                          floor_strike=4.5)) is None


def test_cpi_series_carry_report_rounding():
    assert ECON_SERIES["KXCPIYOY"].report_round_to == 0.1
    assert ECON_SERIES["KXPAYROLLS"].report_round_to == 1000.0


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
