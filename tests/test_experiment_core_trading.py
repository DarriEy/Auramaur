from __future__ import annotations

import ast
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pytest

from auramaur.exchange.models import Market
from auramaur.experiments.models import (
    FeatureValue,
    MarketSnapshot,
    PortfolioSnapshot,
    PricePoint,
)
from auramaur.experiments.strategies.core_trading import (
    CoreTradingExperiment,
    CoreTradingInputs,
    CoreTradingRejection,
    CoreTradingRules,
    assess_core_trading,
)
from auramaur.nlp.analyzer import AnalysisResult
from auramaur.strategy.signals import detect_edge, taker_fee_rate


NOW = datetime(2026, 7, 25, tzinfo=timezone.utc)
LEGACY_RULES = CoreTradingRules(True, True, True, 1.0)


def _inputs(**changes) -> CoreTradingInputs:
    values = dict(
        market_id="m1",
        question="Will X happen?",
        market_probability=0.4,
        model_probability=0.7,
        confidence="HIGH",
        second_opinion_probability=0.65,
        divergence=0.05,
        reasoning="two independent sources support the forecast",
        evidence_count=2,
        hours_to_resolution=48.0,
        fee_rate=0.05,
    )
    values.update(changes)
    return CoreTradingInputs(**values)


@pytest.mark.parametrize("market_probability, model_probability", [(0.4, 0.7), (0.7, 0.4)])
def test_pure_proposal_matches_production_signal(market_probability, model_probability):
    market = Market(
        id="m1",
        question="Will X happen?",
        outcome_yes_price=market_probability,
        outcome_no_price=1 - market_probability,
    )
    analysis = AnalysisResult(
        probability=model_probability,
        confidence="HIGH",
        reasoning="two independent sources support the forecast",
        key_factors=["one", "two"],
        second_opinion_prob=0.65,
        divergence=0.05,
    )
    signal = detect_edge(market, analysis)
    assessment = assess_core_trading(_inputs(
        market_probability=market_probability,
        model_probability=model_probability,
        fee_rate=taker_fee_rate(market.exchange, market.category),
        hours_to_resolution=None,
    ), LEGACY_RULES)
    assert signal is not None and assessment.proposal is not None
    proposal = assessment.proposal
    assert proposal.side == signal.recommended_side.value
    assert proposal.model_probability == pytest.approx(signal.claude_prob)
    assert proposal.edge_percent == pytest.approx(signal.edge)
    assert proposal.quality == signal.recommended_size


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"model_probability": float("nan")}, CoreTradingRejection.INVALID_PROBABILITY),
        ({"model_probability": 0.4, "second_opinion_probability": None}, CoreTradingRejection.NO_EDGE),
        ({"question": "Will X not happen?", "model_probability": 0.9,
          "second_opinion_probability": None}, CoreTradingRejection.INVERTED_SEMANTICS),
        ({"model_probability": 0.405, "second_opinion_probability": None,
          "fee_rate": 0.1}, CoreTradingRejection.FEES_CONSUME_EDGE),
    ],
)
def test_fail_closed_rejections_are_explicit(changes, reason):
    assessment = assess_core_trading(_inputs(**changes), LEGACY_RULES)
    assert assessment.proposal is None
    assert assessment.rejection is reason


def test_strategic_rules_preserve_direct_estimate_and_historic_fee_semantics():
    assessment = assess_core_trading(
        _inputs(model_probability=0.405, second_opinion_probability=0.9, fee_rate=0.1),
        CoreTradingRules(False, False, False, 1.0, False),
    )
    assert assessment.proposal is not None
    assert assessment.proposal.model_probability == pytest.approx(0.405)
    assert assessment.proposal.edge_percent < 0
    assert assessment.proposal.quality is None


def test_experiment_adapter_is_isolated_from_live_execution():
    payload = asdict(_inputs())
    snapshot = MarketSnapshot(
        observed_at=NOW,
        sequence=1,
        market_id="m1",
        venue="polymarket",
        prices=(PricePoint(instrument_id="m1:YES", price=0.4),),
        features=(FeatureValue(
            name="core_trading_inputs",
            payload=payload,
            source="prospective-test",
            observed_at=NOW,
            available_at=NOW,
        ),),
        data_version="test-v1",
    )
    targets = asyncio.run(CoreTradingExperiment(LEGACY_RULES).evaluate(
        snapshot, PortfolioSnapshot(observed_at=NOW, cash=100, equity=100)
    ))
    assert len(targets) == 1
    assert targets[0].instrument_id == "m1:YES"

    source = Path("auramaur/experiments/strategies/core_trading.py").read_text()
    imports = {
        node.module or "" for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    }
    forbidden = ("auramaur.broker", "auramaur.db", "auramaur.exchange")
    assert not any(name.startswith(forbidden) for name in imports)
