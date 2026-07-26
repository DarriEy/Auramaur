from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from auramaur.experiments.models import (
    ExperimentDefinition, FeatureValue, MarketSnapshot, PortfolioSnapshot, PricePoint,
)
from auramaur.experiments.registry import ExperimentRegistry
from auramaur.experiments.runtimes import (
    InMemoryShadowSink, LinearExecutionModel, ReplayRuntime, ResearchRuntime, ShadowRuntime,
)
from auramaur.experiments.strategies.settlement_arb import SettlementArbExperiment


@pytest.mark.asyncio
async def test_settlement_proposal_is_identical_across_non_live_runtimes():
    now = datetime(2026, 7, 25, tzinfo=timezone.utc)
    snapshot = MarketSnapshot(
        observed_at=now, sequence=0, market_id="cpi", venue="kalshi",
        prices=(PricePoint(instrument_id="cpi:YES", price=.80),
                PricePoint(instrument_id="cpi:NO", price=.20)),
        features=(FeatureValue(
            name="settlement_predicate",
            payload={"published_value": 3.0, "indicator": "KXCPIYOY",
                     "reference_period": "2026-06", "operator": ">=",
                     "threshold": 3.0},
            source="fred", observed_at=now, available_at=now,
        ),), data_version="fred-fixture-v1",
    )
    account = PortfolioSnapshot(
        observed_at=now - timedelta(seconds=1), cash=1000, equity=1000)
    implementation = SettlementArbExperiment(min_edge=.05, stake_usd=10)
    definition = ExperimentDefinition(
        key="settlement-poc", strategy_source="settlement_arb",
        hypothesis="Published outcomes converge after official release.",
        mechanism="settlement lag", implementation_version="proposal-v1",
        parameters={"min_edge": .05, "stake_usd": 10},
        venues=frozenset({"kalshi", "polymarket"}),
        primary_metric="net_pnl_after_costs", baseline="no_position",
        min_observations=30, holdout_days=14, max_drawdown=.15,
        cost_model="linear-v1", rejection_criteria=("holdout_pnl_lte_zero",),
    )
    bound = ExperimentRegistry().register(definition, implementation)
    research = await ResearchRuntime().run(bound, snapshot, account)
    sink = InMemoryShadowSink()
    shadow = await ShadowRuntime(sink).run(bound, snapshot, account)
    replay = await ReplayRuntime(LinearExecutionModel(version="linear-v1")).run(
        bound, [snapshot], account)
    assert research.targets == shadow.targets == replay.results[0].targets
    assert research.targets[0].instrument_id == "cpi:YES"
    assert research.targets[0].max_notional == pytest.approx(10)
