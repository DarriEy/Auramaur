"""Portable proposal parity for the shared Polymarket/Kalshi agent trader."""

from __future__ import annotations

import ast
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from auramaur.exchange.models import Market
from auramaur.experiments.models import (
    FeatureValue,
    MarketSnapshot,
    PortfolioSnapshot,
    PricePoint,
)
from auramaur.experiments.strategies.agent_trader import (
    AgentTraderAssessment,
    AgentTraderExperiment,
    AgentTraderInputs,
    AgentTraderRejection,
    AgentTraderRules,
    assess_agent_trader,
    parse_decisions,
)
from auramaur.strategy.agent_trader import AgentTraderPillar


@pytest.mark.parametrize(
    ("venue", "source"),
    [("polymarket", "agent_trader_opus"),
     ("kalshi", "agent_trader_opus_kalshi")],
)
def test_shared_venue_proposal_parity(venue: str, source: str) -> None:
    rules = AgentTraderRules(min_edge_pts=5.0, stake_usd=10.0, source_tag=source)
    proposal = assess_agent_trader(AgentTraderInputs(
        market_id="m1", question="Will it happen?", market_yes=0.30,
        prob_yes=0.55, thesis="The announced mechanism is underpriced.",
    ), rules).proposal
    assert proposal is not None
    assert proposal.buy_yes
    assert proposal.edge_percent == pytest.approx(25.0)
    assert proposal.source_tag == source

    now = datetime(2026, 7, 25, tzinfo=timezone.utc)
    snapshot = MarketSnapshot(
        observed_at=now, sequence=1, market_id="m1", venue=venue,
        prices=(PricePoint(instrument_id="YES", price=0.30),),
        features=(FeatureValue(
            name="agent_trader_opinion",
            payload={"question": "Will it happen?", "prob_yes": 0.55,
                     "thesis": "The announced mechanism is underpriced."},
            source="model-opinion", observed_at=now, available_at=now,
        ),), data_version="test-v1",
    )
    targets = asyncio.run(AgentTraderExperiment(rules).evaluate(
        snapshot,
        PortfolioSnapshot(observed_at=now, cash=100.0, equity=100.0),
    ))
    assert len(targets) == 1
    assert targets[0].instrument_id == "m1:YES"
    assert targets[0].target_quantity == pytest.approx(10.0 / 0.30)


@pytest.mark.parametrize("field,value", [
    ("market_yes", float("nan")),
    ("market_yes", 0.0),
    ("prob_yes", float("inf")),
    ("prob_yes", 1.0),
    ("thesis", ""),
])
def test_invalid_opinions_fail_closed(field: str, value: object) -> None:
    values = dict(market_id="m1", question="Q?", market_yes=0.3,
                  prob_yes=0.6, thesis="concrete evidence")
    values[field] = value
    result = assess_agent_trader(
        AgentTraderInputs(**values),
        AgentTraderRules(5.0, 10.0, "agent_trader_opus"),
    )
    assert result.proposal is None
    assert result.rejection == AgentTraderRejection.INVALID_INPUT


def test_parser_rejects_nonfinite_model_probability() -> None:
    raw = '{"decisions":[{"market_id":"m1","prob_yes":NaN,"thesis":"x"}]}'
    assert parse_decisions(raw, 1) == []


@pytest.mark.asyncio
async def test_production_entry_delegates_to_pure_assessment(monkeypatch) -> None:
    pillar = object.__new__(AgentTraderPillar)
    pillar._cell_suffix = "_kalshi"
    pillar._risk = SimpleNamespace(evaluate=None)
    monkeypatch.setattr(
        "auramaur.strategy.agent_trader.assess_agent_trader",
        lambda *_: AgentTraderAssessment(
            None, AgentTraderRejection.INSUFFICIENT_EDGE),
    )
    entered = await pillar._try_enter(
        "opus",
        Market(id="m1", question="Q?", exchange="kalshi",
               outcome_yes_price=0.3, outcome_no_price=0.7),
        {"prob_yes": 0.9, "thesis": "x"},
        SimpleNamespace(min_edge_pts=5.0, stake_usd=10.0),
    )
    assert entered is False


def test_pure_module_has_no_live_runtime_imports() -> None:
    source = Path("auramaur/experiments/strategies/agent_trader.py").read_text()
    imports = {
        node.module or ""
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    }
    forbidden = ("auramaur.broker", "auramaur.db", "auramaur.exchange",
                 "auramaur.strategy")
    assert not any(name.startswith(forbidden) for name in imports)
