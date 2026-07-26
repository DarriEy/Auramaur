from __future__ import annotations
import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
from auramaur.experiments.models import ExperimentDefinition, FeatureValue, MarketSnapshot, PortfolioSnapshot, PricePoint
from auramaur.experiments.registry import ExperimentRegistry
from auramaur.experiments.runtimes import InMemoryShadowSink, LinearExecutionModel, ReplayRuntime, ResearchRuntime, ShadowRuntime
from auramaur.experiments.strategies.cross_venue_arb import CrossVenueArbExperiment, paired_arb_proposal

BASE = dict(market_a_id="poly-1", venue_a="polymarket", yes_price_a=.35,
            market_b_id="kalshi-1", venue_b="kalshi", yes_price_b=.60,
            orientation="same", confidence=.98, min_confidence=.95,
            required_gap=.04, stake_usd=10)

def test_paired_proposal_is_atomic_and_immutable():
    proposal = paired_arb_proposal(**BASE)
    assert proposal is not None
    assert tuple(leg.side for leg in proposal.legs) == ("BUY", "SELL")
    assert tuple(t.instrument_id for t in proposal.targets) == (
        "polymarket:poly-1:YES", "kalshi:kalshi-1:NO")
    with pytest.raises(AttributeError):
        proposal.edge = .1

@pytest.mark.parametrize("change", [{"orientation": "none"}, {"confidence": .94},
    {"required_gap": .26}, {"yes_price_a": 0}, {"venue_b": "polymarket"}])
def test_paired_proposal_fails_closed(change):
    values = BASE | change
    assert paired_arb_proposal(**values) is None

@pytest.mark.asyncio
async def test_pair_is_identical_across_non_live_runtimes():
    now = datetime(2026, 7, 25, tzinfo=timezone.utc)
    snapshot = MarketSnapshot(observed_at=now, sequence=0,
        market_id="poly-1|kalshi-1", venue="cross_venue",
        prices=(PricePoint(instrument_id="polymarket:poly-1:YES", price=.35),
                PricePoint(instrument_id="kalshi:kalshi-1:NO", price=.40)),
        features=(FeatureValue(name="cross_venue_pair", payload={
            "market_a_id": "poly-1", "venue_a": "polymarket", "yes_price_a": .35,
            "market_b_id": "kalshi-1", "venue_b": "kalshi", "yes_price_b": .60,
            "orientation": "same", "confidence": .98}, source="fixture",
            observed_at=now, available_at=now),), data_version="paired-fixture-v1")
    portfolio = PortfolioSnapshot(observed_at=now - timedelta(seconds=1), cash=1000, equity=1000)
    implementation = CrossVenueArbExperiment(min_confidence=.95, required_gap=.04, stake_usd=10)
    definition = ExperimentDefinition(key="cross-venue-poc", strategy_source="cross_venue_arb",
        hypothesis="Equivalent claims converge across venues.", mechanism="paired semantic arbitrage",
        implementation_version="paired-v1", parameters={}, venues=frozenset({"polymarket", "kalshi"}),
        primary_metric="package_net_pnl_after_costs", baseline="no_package", min_observations=30,
        holdout_days=14, max_drawdown=.15, cost_model="paired-linear-v1",
        rejection_criteria=("unhedged_leg",))
    bound = ExperimentRegistry().register(definition, implementation)
    research = await ResearchRuntime().run(bound, snapshot, portfolio)
    sink = InMemoryShadowSink()
    shadow = await ShadowRuntime(sink).run(bound, snapshot, portfolio)
    replay = await ReplayRuntime(LinearExecutionModel(version="paired-linear-v1")).run(bound, [snapshot], portfolio)
    assert len(research.targets) == 2
    assert research.targets == shadow.targets == replay.results[0].targets

def test_pure_module_has_no_live_execution_imports():
    tree = ast.parse(Path("auramaur/experiments/strategies/cross_venue_arb.py").read_text(encoding="utf-8"))
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imports.update(node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
    assert not any(name.startswith(("auramaur.broker", "auramaur.exchange", "auramaur.risk")) for name in imports)
