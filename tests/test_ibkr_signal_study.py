"""The signal study, pinned against the ways it could invent an edge.

The dangerous failures here are lookahead, mistaking market drift for skill,
and an overconfident t-statistic from overlapping windows. Each gets a test
that fails loudly if the guard is removed.
"""

import math
from datetime import date, timedelta

import pytest

from auramaur.backtest.ibkr_signal_study import (
    decile_buckets, gate_contrast, information_coefficient, observations,
)


def _days(n):
    out, day = [], date(2020, 1, 1)
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day.isoformat())
        day += timedelta(days=1)
    return out


def _bars(prices):
    days = _days(len(prices))
    return [(d, p) for d, p in zip(days, prices)]


def _trend(n, rate, wobble=0.01, phase=0.0):
    """A trend with realistic variance.

    A perfectly constant growth rate has ~zero log-return variance, so
    annualized_volatility returns ~1e-16 and every momentum z-score explodes to
    ~1e14. That is a degenerate fixture, not a signal — real instruments wobble.
    """
    return [100.0 * (1 + rate) ** i * (1 + wobble * math.sin(i * 0.7 + phase))
            for i in range(n)]


def test_a_common_uptrend_is_not_mistaken_for_signal():
    """Every instrument rising identically: excess return is zero by
    construction, so no bucket may show an edge. Without the cross-sectional
    demeaning this is where a momentum backtest fools itself."""
    rise = _trend(300, 0.002)
    bars = {f"S{k}": _bars(rise) for k in range(6)}
    obs = observations(bars, horizon=10)
    assert obs, "no observations produced"
    assert all(abs(o.excess_return) < 1e-9 for o in obs)
    for bucket in decile_buckets(obs):
        assert abs(bucket.mean_excess) < 1e-9


def test_a_planted_signal_is_recovered():
    """Sanity in the other direction: if high momentum genuinely predicts high
    forward return, the study must SEE it. A test suite that only proves the
    tool finds nothing would pass on a broken tool."""
    bars, n = {}, 300
    for k in range(6):
        # Alternating regime: some names trend up, others down, and the
        # trend persists so trailing momentum predicts forward return.
        rate = 0.004 if k % 2 == 0 else -0.004
        bars[f"S{k}"] = _bars(_trend(n, rate, phase=k))
    obs = observations(bars, horizon=10)
    buckets = decile_buckets(obs, buckets=2)
    assert len(buckets) == 2
    assert buckets[-1].mean_excess > buckets[0].mean_excess
    mean_ic, _, _, _ = information_coefficient(obs, horizon=10)
    assert mean_ic > 0.5


def test_forward_return_never_uses_the_signal_bar():
    """One instrument spikes on a single session. Momentum measured AT that
    session must not include it — the observation's momentum is computed from
    strictly earlier closes."""
    flat = [100.0] * 200
    spiked = list(flat)
    spiked[150] = 10_000.0
    obs = observations({"A": _bars(spiked), "B": _bars(flat)}, horizon=5)
    at_spike = [o for o in obs if o.key == "A" and o.date == _days(200)[150]]
    # Momentum on a flat trailing window is 0 or undefined, never explosive.
    assert all(abs(o.momentum) < 50 for o in at_spike)


def test_t_stat_uses_non_overlapping_sessions_only():
    """Overlapping forward windows inflate significance h-fold. The reported
    sample must be the subsampled one, not every session."""
    bars = {f"S{k}": _bars(_trend(300, 0.001 * (k + 1), phase=k))
            for k in range(5)}
    obs = observations(bars, horizon=20)
    _, _, used, total = information_coefficient(obs, horizon=20)
    assert total > used, "t-stat sampled every session; windows overlap"
    assert used == pytest.approx(total / 20, rel=0.3)


def test_gate_contrast_separates_what_the_book_buys_from_what_it_skips():
    bars, n = {}, 300
    for k in range(6):
        rate = 0.004 if k % 2 == 0 else -0.004
        bars[f"S{k}"] = _bars(_trend(n, rate, phase=k))
    obs = observations(bars, horizon=10)
    passing, failing = gate_contrast(obs, 0.25)
    assert passing.n and failing.n
    assert passing.mean_excess > failing.mean_excess
    # And an unreachable threshold selects nothing rather than erroring.
    empty, everything = gate_contrast(obs, 1e9)
    assert empty.n == 0 and everything.n == len(obs)


def test_too_little_history_produces_nothing():
    assert observations({"A": _bars([100.0] * 30)}, horizon=5) == []
