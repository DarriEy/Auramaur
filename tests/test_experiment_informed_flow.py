"""Parity and isolation tests for the portable informed-flow proposal."""
from __future__ import annotations
import ast
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import pytest
from auramaur.experiments.models import FeatureValue, MarketSnapshot, PortfolioSnapshot, PricePoint
from auramaur.experiments.strategies.informed_flow import (
    InformedFlowExperiment, InformedFlowInputs, InformedFlowRejection,
    InformedFlowRules, assess_informed_flow,
)

RULES = InformedFlowRules(0.1, 0.9, 100, 2, 7, 0.04, 10)

def _inputs(**changes) -> InformedFlowInputs:
    values = dict(market_id="KX-1", venue="kalshi", ticker="KXTEST", active=True,
        market_probability=0.4, liquidity=500, blocked_category=False,
        hours_to_resolution=24, already_entered_or_held=False, has_signal=True,
        informed_side="no", abnormal_count=3, baseline_size=10.0,
        signal_volume=100.0, sample=40)
    values.update(changes)
    return InformedFlowInputs(**values)

def test_assessment_matches_production_no_side_signal():
    proposal = assess_informed_flow(_inputs(), RULES).proposal
    assert proposal is not None
    assert proposal.buy_yes is False
    assert proposal.model_probability == pytest.approx(0.36)
    assert proposal.edge_percent == pytest.approx(4)
    assert proposal.reference_price == pytest.approx(0.6)
    assert proposal.max_notional == 10
    assert proposal.evidence_summary == (
        "Informed-flow follow: 3 abnormal trades (100 contracts) on the NO side "
        "vs a 10-size baseline (n=40).")

@pytest.mark.parametrize(("changes", "reason"), [
    ({"active": False}, InformedFlowRejection.INACTIVE),
    ({"venue": "polymarket"}, InformedFlowRejection.WRONG_VENUE),
    ({"ticker": None}, InformedFlowRejection.MISSING_TICKER),
    ({"market_probability": 0.95}, InformedFlowRejection.OUTSIDE_PRICE_BAND),
    ({"liquidity": 10}, InformedFlowRejection.INSUFFICIENT_LIQUIDITY),
    ({"blocked_category": True}, InformedFlowRejection.BLOCKED_CATEGORY),
    ({"hours_to_resolution": None}, InformedFlowRejection.MISSING_RESOLUTION_TIME),
    ({"hours_to_resolution": 1}, InformedFlowRejection.OUTSIDE_RESOLUTION_WINDOW),
    ({"already_entered_or_held": True}, InformedFlowRejection.ALREADY_ENTERED_OR_HELD),
    ({"has_signal": False}, InformedFlowRejection.NO_INFORMED_FLOW),
])
def test_rejection_parity(changes, reason):
    result = assess_informed_flow(_inputs(**changes), RULES)
    assert result.proposal is None
    assert result.rejection == reason

def test_adapter_emits_target_without_live_dependencies():
    now = datetime(2026, 7, 25, tzinfo=timezone.utc)
    raw = _inputs()
    snapshot = MarketSnapshot(observed_at=now, sequence=1, market_id=raw.market_id,
        venue="kalshi", prices=(PricePoint(instrument_id="KX-1:NO", price=0.6),),
        features=(FeatureValue(name="informed_flow_inputs", payload=asdict(raw),
        source="test", observed_at=now, available_at=now),), data_version="test-v1")
    targets = asyncio.run(InformedFlowExperiment(RULES).evaluate(
        snapshot, PortfolioSnapshot(observed_at=now, cash=100, equity=100)))
    assert len(targets) == 1
    assert targets[0].instrument_id == "KX-1:NO"
    assert targets[0].target_quantity == pytest.approx(10 / 0.6)
    source = Path("auramaur/experiments/strategies/informed_flow.py").read_text()
    tree = ast.walk(ast.parse(source))
    imports = {node.module or "" for node in tree if isinstance(node, ast.ImportFrom)}
    forbidden = ("auramaur.broker", "auramaur.db", "auramaur.exchange")
    assert not any(name.startswith(forbidden) for name in imports)
