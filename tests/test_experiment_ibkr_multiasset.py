"""Portable IBKR multi-asset proposal coverage."""
from __future__ import annotations
from dataclasses import FrozenInstanceError
import importlib
import sys
import pytest
from auramaur.experiments.strategies.ibkr_multiasset import (
    MultiAssetEntryInputs, MultiAssetEntryRules, propose_multiasset_entry,
)
from auramaur.risk.ibkr_math import adverse_fill, risk_quantity, stop_distance

def _rules(**changes):
    values = dict(budget_usd=10_000, max_position_pct=20,
                  max_deployment_pct=80, risk_per_position_pct=1,
                  max_asset_class_risk_pct=3, stop_vol_multiple=2,
                  min_stop_pct=2, slippage_bps=5)
    values.update(changes)
    return MultiAssetEntryRules(**values)

def _inputs(**changes):
    values = dict(instrument_key="SPY", asset_class="equity", bid=499.8,
                  ask=500.0, fx_to_usd=1.0, multiplier=1.0,
                  annual_volatility=.20, deployed_usd=1_000,
                  asset_class_risk_usd=25, capital_per_unit_usd=500,
                  fractional=True)
    values.update(changes)
    return MultiAssetEntryInputs(**values)

def test_proposal_preserves_legacy_sizing_and_stop_formula():
    inputs, rules = _inputs(), _rules()
    proposal = propose_multiasset_entry(inputs, rules)
    assert proposal is not None
    entry = adverse_fill(inputs.bid, inputs.ask, "BUY", rules.slippage_bps)
    distance = stop_distance(entry, inputs.annual_volatility,
                             rules.stop_vol_multiple, rules.min_stop_pct)
    expected = min(4.0, risk_quantity(100, distance, 1, 1, fractional=True))
    assert proposal.quantity == pytest.approx(expected)
    assert proposal.reference_price == pytest.approx(entry)
    assert proposal.stop_price == pytest.approx(entry - distance)
    assert proposal.initial_risk_usd == pytest.approx(expected * distance)
    assert proposal.paper_by_default is True

def test_whole_contract_assets_do_not_propose_fractional_orders():
    proposal = propose_multiasset_entry(
        _inputs(instrument_key="ES", asset_class="future", multiplier=1,
                capital_per_unit_usd=1_200, fractional=False),
        _rules(max_position_pct=50))
    assert proposal is not None
    assert proposal.quantity == float(int(proposal.quantity))

@pytest.mark.parametrize("changes", [
    {"fx_to_usd": 0}, {"ask": float("nan")}, {"ask": 499, "bid": 500},
    {"annual_volatility": 0}, {"deployed_usd": -1},
    {"capital_per_unit_usd": 0},
])
def test_invalid_market_or_portfolio_inputs_fail_closed(changes):
    assert propose_multiasset_entry(_inputs(**changes), _rules()) is None

def test_exhausted_budgets_fail_closed():
    assert propose_multiasset_entry(_inputs(deployed_usd=8_000), _rules()) is None
    assert propose_multiasset_entry(
        _inputs(asset_class_risk_usd=300), _rules()) is None

def test_contracts_are_immutable():
    proposal = propose_multiasset_entry(_inputs(), _rules())
    assert proposal is not None
    with pytest.raises(FrozenInstanceError):
        proposal.quantity = 1

def test_experiment_module_has_no_broker_or_strategy_imports():
    name = "auramaur.experiments.strategies.ibkr_multiasset"
    sys.modules.pop(name, None)
    before = set(sys.modules)
    importlib.import_module(name)
    loaded = set(sys.modules) - before
    assert not any(module.startswith(
        ("auramaur.exchange.ibkr_multiasset_execution",
         "auramaur.strategy.ibkr_multiasset_paper")) for module in loaded)
