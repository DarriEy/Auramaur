"""Contract tests for the paper-only ETF experiment boundary."""

from dataclasses import FrozenInstanceError

import pytest

from auramaur.experiments.strategies.ibkr_etf_paper import (
    ETFEntryProposal, entry_proposal, exit_proposal,
)
from auramaur.risk.ibkr_math import adverse_fill, risk_quantity, stop_distance


def _entry(**overrides):
    values = dict(
        symbol="SPY", bid=499.9, ask=500.1, probability=0.67,
        confidence="HIGH", confidence_rank=4, min_confidence_rank=2,
        min_probability=0.6, annual_volatility=0.2, momentum=0.03,
        remaining_deployment_usd=1_000, remaining_class_usd=800,
        max_entry_usd=500, fee_usd=1, slippage_bps=5,
        stop_vol_multiple=1.5, min_stop_pct=1.0, risk_budget_usd=50,
        open_risk_usd=10, max_portfolio_risk_usd=100,
        controls_allow_entry=True)
    values.update(overrides)
    return entry_proposal(**values)


def test_entry_matches_legacy_sizing_and_cost_controls():
    proposal = _entry()
    assert proposal is not None
    expected_price = adverse_fill(499.9, 500.1, "BUY", 5)
    distance = stop_distance(expected_price, 0.2, 1.5, 1.0)
    expected_qty = min(499 / expected_price,
                       risk_quantity(50, distance, 1, 1, fractional=True))
    assert proposal.price == pytest.approx(expected_price)
    assert proposal.quantity == pytest.approx(expected_qty)
    assert proposal.stop_price == pytest.approx(expected_price - distance)
    assert proposal.initial_risk_usd == pytest.approx(expected_qty * distance)


@pytest.mark.parametrize("override", [
    {"controls_allow_entry": False}, {"probability": None},
    {"confidence_rank": 1}, {"momentum": 0},
    {"remaining_class_usd": 1}, {"annual_volatility": None},
    {"open_risk_usd": 100},
])
def test_entry_fails_closed(override):
    assert _entry(**override) is None


@pytest.mark.parametrize(("kwargs", "reason"), [
    ({"bid": 94.0, "ask": 94.1}, "stop_loss"),
    ({"bid": 108.0, "ask": 108.1}, "take_profit"),
    ({"bid": 102.0, "ask": 102.1, "peak_gain_pct": 7.0}, "trailing_stop"),
    ({"bid": 101.0, "ask": 101.1, "probability": 0.3}, "llm_bearish"),
])
def test_exit_precedence_and_reasons(kwargs, reason):
    values = dict(symbol="SPY", quantity=2, entry_price=100, bid=101, ask=101.1,
                  stored_stop=0, peak_gain_pct=1, probability=0.7,
                  stop_loss_pct=5, take_profit_pct=7,
                  trailing_stop_pct=4, exit_probability=0.4,
                  slippage_bps=5)
    values.update(kwargs)
    proposal = exit_proposal(**values)
    assert proposal is not None
    assert proposal.reason == reason
    assert proposal.price == pytest.approx(
        adverse_fill(values["bid"], values["ask"], "SELL", 5))


def test_hold_has_no_order_proposal():
    assert exit_proposal(
        symbol="SPY", quantity=2, entry_price=100, bid=101, ask=101.1,
        stored_stop=0, peak_gain_pct=1, probability=0.7,
        stop_loss_pct=5, take_profit_pct=7, trailing_stop_pct=4,
        exit_probability=0.4, slippage_bps=5) is None


def test_proposals_are_immutable():
    proposal = _entry()
    assert isinstance(proposal, ETFEntryProposal)
    with pytest.raises(FrozenInstanceError):
        proposal.quantity = 99


def test_experiment_module_has_no_broker_or_live_execution_imports():
    import auramaur.experiments.strategies.ibkr_etf_paper as module
    source = module.__loader__.get_source(module.__name__)
    assert "auramaur.exchange" not in source
    assert "auramaur.strategy" not in source
    assert "place_order" not in source
