"""The economic entry gate, pinned to the measurements that produced it.

The old book's gate was a bare probability threshold, and it cost $926 in
commission to act on ~6bps of edge. These tests hold the arithmetic that
replaces it, using the real volatilities and the real $1,100 cost structure
measured on 2026-07-27.
"""

import math

import pytest

from auramaur.strategy.ibkr_edge_economics import (
    clears_costs, expected_absolute_move, expected_edge_bps,
    required_conviction, round_trip_cost_bps, viable_universe,
)

# Measured 2026-07-27 from 1Y of adjusted closes.
VOLS = {"TLT": 0.093, "SPY": 0.127, "VNQ": 0.140, "QQQ": 0.190,
        "DBC": 0.194, "XLE": 0.210, "GLD": 0.284, "SLV": 0.644}
HORIZON = 5
CAPITAL = 1_100.0


def _cost():
    # IBKR equity schedule at $1,100: max(1.0, min(1100*0.001, 10.0)) = $1.10.
    commission = max(1.0, min(CAPITAL * 0.001, 10.0))
    return round_trip_cost_bps(CAPITAL, commission_usd=commission,
                               spread_bps=3.0, slippage_bps=2.0)


def test_the_commission_floor_is_the_dominant_cost_at_this_size():
    """At $1,100 the $1 minimum binds; ten times the size does not cost ten
    times as much. This is why the book could not size out of the problem."""
    small = _cost()
    assert small == pytest.approx(20.0 + 3.0 + 4.0, abs=0.1)   # ~27bps

    big_commission = max(1.0, min(50_000 * 0.001, 10.0))       # capped at $10
    big = round_trip_cost_bps(50_000, commission_usd=big_commission,
                              spread_bps=3.0, slippage_bps=2.0)
    # Compare the COMMISSION component, which is the claim: spread and
    # slippage are 7bps at any size, so they floor the total.
    small_commission = small - 7.0
    big_commission_bps = big - 7.0
    assert small_commission == pytest.approx(20.0, abs=0.1)
    assert big_commission_bps == pytest.approx(4.0, abs=0.1)
    assert small_commission > 4 * big_commission_bps
    assert round_trip_cost_bps(0, commission_usd=1.0, spread_bps=3.0,
                               slippage_bps=2.0) == float("inf")


def test_expected_edge_matches_the_measured_table():
    """The numbers that killed the old design, reproduced from first
    principles: typical conviction of 0.02 earns single-digit bps on SPY."""
    for symbol, expected in (("SPY", 5.7), ("QQQ", 8.6), ("SLV", 29.0),
                             ("TLT", 4.2)):
        edge = expected_edge_bps(0.52, 0.50, VOLS[symbol], HORIZON)
        assert edge == pytest.approx(expected, abs=0.3), symbol


def test_a_forecast_below_the_benchmark_is_a_skip_not_a_short():
    """The book is long-only; a view weaker than the instrument's own drift
    must produce a negative edge and never clear the gate."""
    edge = expected_edge_bps(0.48, 0.53, VOLS["SLV"], HORIZON)
    assert edge < 0
    assert not clears_costs(0.48, 0.53, VOLS["SLV"], HORIZON, _cost())


def test_the_gate_rejects_spy_and_admits_slv_at_the_same_conviction():
    """The whole point: identical conviction, opposite verdicts, because the
    instruments are not equally worth trading."""
    cost = _cost()
    assert not clears_costs(0.56, 0.50, VOLS["SPY"], HORIZON, cost)
    assert clears_costs(0.56, 0.50, VOLS["SLV"], HORIZON, cost)


def test_required_conviction_is_out_of_reach_for_low_vol_names():
    """SPY needs ~21pp of conviction at this size. The arms have never
    produced more than 6pp, so no threshold makes SPY tradeable — it belongs
    out of the universe, not behind a tighter gate."""
    cost = _cost()
    assert required_conviction(VOLS["SPY"], HORIZON, cost) > 0.15
    assert required_conviction(VOLS["SLV"], HORIZON, cost) < 0.06
    # Monotone: more volatility always lowers the conviction required.
    needed = [required_conviction(v, HORIZON, cost)
              for v in sorted(VOLS.values())]
    assert needed == sorted(needed, reverse=True)
    assert required_conviction(0.0, HORIZON, cost) == float("inf")


def test_viable_universe_admits_only_what_the_model_can_actually_reach():
    """max_conviction is the model's MEASURED maximum (0.06), not a hope."""
    universe = viable_universe(VOLS, horizon_sessions=HORIZON,
                               cost_bps=_cost(), max_conviction=0.06)
    symbols = [s for s, _ in universe]
    assert "SLV" in symbols
    assert "SPY" not in symbols and "TLT" not in symbols
    # Sorted most-reachable first.
    assert universe == sorted(universe, key=lambda item: item[1])
    # A model with no conviction can trade nothing.
    assert viable_universe(VOLS, horizon_sessions=HORIZON, cost_bps=_cost(),
                           max_conviction=0.0) == []


def test_margin_is_load_bearing():
    """At 1.0x the gate trades a coin flip between edge and fees; the default
    2.0x survives a 50% overestimate of the edge."""
    cost = _cost()
    # Nudged off the exact boundary: clears_costs compares floats.
    marginal = 0.5 + required_conviction(VOLS["GLD"], HORIZON, cost,
                                         margin=1.0) + 1e-9
    assert clears_costs(marginal, 0.50, VOLS["GLD"], HORIZON, cost, margin=1.0)
    assert not clears_costs(marginal, 0.50, VOLS["GLD"], HORIZON, cost,
                            margin=2.0)


def test_horizon_scales_the_move_as_square_root_of_time():
    quad = expected_absolute_move(0.20, 20) / expected_absolute_move(0.20, 5)
    assert quad == pytest.approx(2.0, abs=1e-9)
    assert expected_absolute_move(0.20, 0) == 0.0
    # Sanity against the closed form for a half-normal.
    assert expected_absolute_move(0.20, 252) == pytest.approx(
        0.20 * math.sqrt(2 / math.pi))
