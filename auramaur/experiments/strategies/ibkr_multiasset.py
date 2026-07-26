"""Pure order proposals for the specialized IBKR multi-asset experiment."""
from __future__ import annotations
from dataclasses import dataclass
import math
from auramaur.risk.ibkr_math import adverse_fill, risk_quantity, stop_distance

@dataclass(frozen=True)
class MultiAssetEntryRules:
    budget_usd: float
    max_position_pct: float
    max_deployment_pct: float
    risk_per_position_pct: float
    max_asset_class_risk_pct: float
    stop_vol_multiple: float
    min_stop_pct: float
    slippage_bps: float

@dataclass(frozen=True)
class MultiAssetEntryInputs:
    instrument_key: str
    asset_class: str
    ask: float
    bid: float
    fx_to_usd: float
    multiplier: float
    annual_volatility: float
    deployed_usd: float
    asset_class_risk_usd: float
    capital_per_unit_usd: float
    fractional: bool

@dataclass(frozen=True)
class MultiAssetOrderProposal:
    """An execution-independent, long-only IBKR order instruction."""
    instrument_key: str
    asset_class: str
    side: str
    quantity: float
    reference_price: float
    stop_price: float
    initial_risk_usd: float
    target_notional_usd: float
    paper_by_default: bool = True

def propose_multiasset_entry(inputs: MultiAssetEntryInputs,
                             rules: MultiAssetEntryRules) -> MultiAssetOrderProposal | None:
    """Size one candidate, failing closed on invalid or exhausted budgets."""
    values = (inputs.ask, inputs.bid, inputs.fx_to_usd, inputs.multiplier,
              inputs.annual_volatility, inputs.deployed_usd,
              inputs.asset_class_risk_usd, inputs.capital_per_unit_usd,
              rules.budget_usd, rules.max_position_pct,
              rules.max_deployment_pct, rules.risk_per_position_pct,
              rules.max_asset_class_risk_pct, rules.stop_vol_multiple,
              rules.min_stop_pct, rules.slippage_bps)
    if (not inputs.instrument_key or not inputs.asset_class
            or not all(math.isfinite(value) for value in values)
            or inputs.ask <= 0 or inputs.bid <= 0 or inputs.ask < inputs.bid
            or inputs.fx_to_usd <= 0 or inputs.multiplier <= 0
            or inputs.annual_volatility <= 0 or inputs.deployed_usd < 0
            or inputs.asset_class_risk_usd < 0
            or inputs.capital_per_unit_usd <= 0 or rules.budget_usd <= 0):
        return None
    deployment_cap = rules.budget_usd * rules.max_deployment_pct / 100
    target = min(rules.budget_usd * rules.max_position_pct / 100,
                 deployment_cap - inputs.deployed_usd)
    entry_price = adverse_fill(inputs.bid, inputs.ask, "BUY", rules.slippage_bps)
    distance = stop_distance(entry_price, inputs.annual_volatility,
                             rules.stop_vol_multiple, rules.min_stop_pct)
    risk_budget = min(
        rules.budget_usd * rules.risk_per_position_pct / 100,
        rules.budget_usd * rules.max_asset_class_risk_pct / 100
        - inputs.asset_class_risk_usd,
    )
    if target <= 0 or risk_budget <= 0 or distance <= 0:
        return None
    raw_quantity = target / inputs.capital_per_unit_usd
    capital_quantity = (math.floor(raw_quantity * 10_000) / 10_000
                        if inputs.fractional else float(math.floor(raw_quantity)))
    quantity = min(capital_quantity, risk_quantity(
        risk_budget, distance, inputs.multiplier, inputs.fx_to_usd,
        fractional=inputs.fractional))
    if not math.isfinite(quantity) or quantity <= 0:
        return None
    initial_risk = quantity * distance * inputs.multiplier * inputs.fx_to_usd
    return MultiAssetOrderProposal(
        instrument_key=inputs.instrument_key, asset_class=inputs.asset_class,
        side="BUY", quantity=quantity, reference_price=entry_price,
        stop_price=entry_price - distance, initial_risk_usd=initial_risk,
        target_notional_usd=quantity * inputs.capital_per_unit_usd)
