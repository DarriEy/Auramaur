"""Parity and isolation tests for the portable platform-consensus decision."""

from __future__ import annotations

import ast
import asyncio
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from auramaur.experiments.models import (
    FeatureValue,
    MarketSnapshot,
    PortfolioSnapshot,
    PricePoint,
)
from auramaur.experiments.strategies.platform_consensus import (
    ConsensusForecast,
    ConsensusInputs,
    ConsensusRules,
    PlatformConsensusExperiment,
    assess_platform_consensus,
)


RULES = ConsensusRules(100, 2, 30, 0.5, 0.03, 20, 100, 25, 10)
NOW = datetime(2026, 7, 25, tzinfo=timezone.utc)


def _inputs(**changes) -> ConsensusInputs:
    values = dict(
        market_id="pm-1",
        question="Will the mission launch before September?",
        market_probability=0.4,
        active=True,
        liquidity_or_volume=1_000,
        category_blocked=False,
        end_at=NOW + timedelta(days=5),
        as_of=NOW,
        fee=0.005,
        forecasts=(ConsensusForecast(
            "[Manifold: 60%] Will the mission launch before September?",
            "Unique bettors: 40\nLiquidity: $500",
            "Manifold",
        ),),
    )
    values.update(changes)
    return ConsensusInputs(**values)


def test_assessment_matches_production_signal_fields():
    proposal = assess_platform_consensus(_inputs(), RULES)
    assert proposal is not None
    assert proposal.buy_yes is True
    assert proposal.consensus_probability == pytest.approx(0.6)
    assert proposal.edge_percent == pytest.approx(20)
    assert proposal.reference_price == pytest.approx(0.4)
    assert proposal.target_quantity == pytest.approx(25)
    assert proposal.confidence == "high"


@pytest.mark.parametrize("changes", [
    {"active": False},
    {"liquidity_or_volume": 10},
    {"category_blocked": True},
    {"end_at": None},
    {"fee": 0.2},
    {"forecasts": (ConsensusForecast(
        "[Manifold: 60%] Will the mission launch before September?",
        "Unique bettors: 2\nLiquidity: $10",
        "Manifold",
    ),)},
])
def test_assessment_fails_closed(changes):
    assert assess_platform_consensus(_inputs(**changes), RULES) is None


def test_adapter_emits_target_without_live_dependencies():
    raw = _inputs()
    payload = asdict(raw)
    payload["end_at"] = raw.end_at.isoformat() if raw.end_at else None
    payload["as_of"] = raw.as_of.isoformat()
    payload["forecasts"] = [asdict(item) for item in raw.forecasts]
    snapshot = MarketSnapshot(
        observed_at=NOW,
        sequence=1,
        market_id=raw.market_id,
        venue="polymarket",
        prices=(PricePoint(instrument_id="pm-1:YES", price=0.4),),
        features=(FeatureValue(
            name="platform_consensus_inputs",
            payload=payload,
            source="test",
            observed_at=NOW,
            available_at=NOW,
        ),),
        data_version="test-v1",
    )
    targets = asyncio.run(PlatformConsensusExperiment(RULES).evaluate(
        snapshot, PortfolioSnapshot(observed_at=NOW, cash=100, equity=100)
    ))
    assert len(targets) == 1
    assert targets[0].instrument_id == "pm-1:YES"

    source = Path("auramaur/experiments/strategies/platform_consensus.py").read_text()
    imports = {
        node.module or "" for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    }
    forbidden = (
        "auramaur.broker",
        "auramaur.db",
        "auramaur.exchange",
        "auramaur.strategy",
    )
    assert not any(name.startswith(forbidden) for name in imports)
