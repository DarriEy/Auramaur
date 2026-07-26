from __future__ import annotations

import ast
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from auramaur.experiments.models import (
    ExperimentDefinition,
    FeatureValue,
    MarketSnapshot,
    PortfolioSnapshot,
    PricePoint,
    TargetPosition,
)
from auramaur.experiments.migration import MIGRATION_INVENTORY, MigrationStatus
from auramaur.experiments.registry import ExperimentRegistry
from auramaur.experiments.runtimes import (
    InMemoryShadowSink,
    LinearExecutionModel,
    ReplayRuntime,
    ResearchRuntime,
    ShadowRuntime,
)
from auramaur.experiments.strategies.bias_harvest import (
    BiasHarvestExperiment,
    BiasHarvestRules,
    select_bias_band,
)
from auramaur.exchange.models import Market
from auramaur.strategy.bias_harvest import BiasHarvestPillar
from auramaur.strategy.registry import STRATEGY_SPECS


def test_strategy_experiment_package_has_no_transitive_live_imports():
    script = r'''import importlib, pkgutil, sys
import auramaur.experiments.strategies as strategies
for item in pkgutil.iter_modules(strategies.__path__):
    importlib.import_module(f"{strategies.__name__}.{item.name}")
forbidden = ("auramaur.broker", "auramaur.db", "auramaur.exchange")
loaded = sorted(name for name in sys.modules if name.startswith(forbidden))
if loaded:
    raise SystemExit("transitive live imports: " + ", ".join(loaded))
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout

NOW = datetime(2026, 7, 25, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parent.parent


def definition(**updates) -> ExperimentDefinition:
    values = {
        "key": "price-gap",
        "strategy_source": "price_gap",
        "hypothesis": "Large price gaps mean revert after costs.",
        "mechanism": "temporary liquidity imbalance",
        "implementation_version": "1",
        "parameters": {"threshold": 0.05},
        "venues": frozenset({"paper"}),
        "primary_metric": "net_pnl_after_costs",
        "baseline": "no_position",
        "min_observations": 30,
        "holdout_days": 14,
        "max_drawdown": 0.10,
        "cost_model": "linear-v1",
        "rejection_criteria": ("holdout_pnl_lte_zero",),
    }
    values.update(updates)
    return ExperimentDefinition(**values)


def snapshot(*, price: float = 0.40, seconds: int = 0, sequence: int = 0):
    observed = NOW + timedelta(seconds=seconds)
    return MarketSnapshot(
        observed_at=observed,
        sequence=sequence,
        market_id="m1",
        venue="paper",
        prices=(PricePoint(instrument_id="m1:YES", price=price),),
        features=(FeatureValue(
            name="fair",
            payload=0.55,
            source="fixture",
            observed_at=observed - timedelta(seconds=2),
            available_at=observed - timedelta(seconds=1),
        ),),
        data_version="fixture-v1",
    )


def portfolio():
    return PortfolioSnapshot(
        observed_at=NOW - timedelta(seconds=1), cash=1000, equity=1000
    )


class GapExperiment:
    async def evaluate(self, market, account):
        del account
        return [TargetPosition(
            instrument_id="m1:YES",
            target_quantity=10,
            reference_price=market.price("m1:YES"),
            rationale="fair value exceeds price",
            max_notional=10,
        )]


def registered(implementation=None):
    implementation = implementation or GapExperiment()
    return ExperimentRegistry().register(definition(), implementation)


def test_lineage_hashes_the_complete_preregistration():
    original = definition()
    assert original.lineage_id == definition(parameters={"threshold": 0.05}).lineage_id
    for change in (
        {"parameters": {"threshold": 0.06}},
        {"hypothesis": "A different hypothesis."},
        {"primary_metric": "brier_score"},
        {"holdout_days": 21},
        {"rejection_criteria": ("different_rule",)},
    ):
        assert original.lineage_id != definition(**change).lineage_id


def test_definition_payload_is_deeply_immutable():
    source = {"nested": {"threshold": 0.05}}
    item = definition(parameters=source)
    lineage = item.lineage_id
    source["nested"]["threshold"] = 0.99
    decoded = item.parameters.decoded()
    decoded["nested"]["threshold"] = 0.80
    assert item.lineage_id == lineage
    assert item.parameters.decoded() == {"nested": {"threshold": 0.05}}


def test_registry_binds_definition_to_one_implementation():
    registry = ExperimentRegistry()
    implementation = GapExperiment()
    bound = registry.register(definition(), implementation)
    assert registry.get_by_key("price-gap") is bound
    assert bound.implementation is implementation
    with pytest.raises(ValueError, match="another implementation"):
        registry.register(definition(), GapExperiment())
    with pytest.raises(ValueError, match="already frozen"):
        registry.register(definition(parameters={"threshold": 0.08}), implementation)


@pytest.mark.parametrize("field", ["target_quantity", "reference_price", "max_notional"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_target_rejects_non_finite_values(field, value):
    values = {
        "instrument_id": "m1:YES",
        "target_quantity": 1,
        "reference_price": 0.4,
        "rationale": "test",
        "max_notional": 1,
    }
    values[field] = value
    with pytest.raises(ValidationError):
        TargetPosition(**values)


def test_feature_must_be_available_at_snapshot_time():
    with pytest.raises(ValidationError, match="not yet available"):
        MarketSnapshot(
            observed_at=NOW,
            sequence=0,
            market_id="m1",
            venue="paper",
            prices=(PricePoint(instrument_id="m1:YES", price=0.4),),
            features=(FeatureValue(
                name="future",
                payload=1,
                source="fixture",
                observed_at=NOW,
                available_at=NOW + timedelta(seconds=1),
            ),),
            data_version="fixture-v1",
        )


@pytest.mark.asyncio
async def test_research_and_shadow_use_bound_implementation():
    bound = registered()
    research = await ResearchRuntime().run(bound, snapshot(), portfolio())
    sink = InMemoryShadowSink()
    shadow = await ShadowRuntime(sink).run(bound, snapshot(), portfolio())
    assert research.runtime == "research"
    assert shadow.runtime == "shadow"
    assert research.lineage_id == shadow.lineage_id == bound.lineage_id
    assert sink.results == [shadow]


@pytest.mark.asyncio
async def test_replay_evolves_portfolio_chronologically_after_costs():
    runtime = ReplayRuntime(LinearExecutionModel(
        version="linear-v1", fee_bps=100, slippage_bps=100
    ))
    report = await runtime.run(
        registered(),
        [snapshot(price=0.40, sequence=0), snapshot(price=0.50, seconds=1, sequence=1)],
        portfolio(),
    )
    assert report.execution_model_version == "linear-v1"
    assert len(report.results) == 2
    assert report.results[0].costs == pytest.approx(0.0804)
    assert report.results[1].net_pnl == pytest.approx(1.0)
    assert report.final_equity == pytest.approx(1000.9196)


@pytest.mark.asyncio
async def test_replay_rejects_out_of_order_and_duplicate_events():
    runtime = ReplayRuntime(LinearExecutionModel(version="linear-v1"))
    later = snapshot(seconds=1, sequence=1)
    earlier = snapshot(sequence=0)
    with pytest.raises(ValueError, match="chronological"):
        await runtime.run(registered(), [later, earlier], portfolio())
    with pytest.raises(ValueError, match="duplicate"):
        await runtime.run(registered(), [earlier, earlier], portfolio())


@pytest.mark.asyncio
async def test_replay_rejects_unpriced_holdings():
    initial = portfolio().model_copy(update={"positions": (("other", 1.0),)})
    runtime = ReplayRuntime(LinearExecutionModel(version="linear-v1"))
    with pytest.raises(ValueError, match="cannot value"):
        await runtime.run(registered(), [snapshot()], initial)


@pytest.mark.asyncio
async def test_replay_rejects_false_target_reference_price():
    class FalseReference:
        async def evaluate(self, market, account):
            del market, account
            return [TargetPosition(
                instrument_id="m1:YES",
                target_quantity=1,
                reference_price=0.01,
                rationale="misstated reference",
                max_notional=1,
            )]

    runtime = ReplayRuntime(LinearExecutionModel(version="linear-v1"))
    with pytest.raises(ValueError, match="reference price"):
        await runtime.run(registered(FalseReference()), [snapshot()], portfolio())


class _BiasSettings:
    class bias_harvest:
        band_lo = 0.60
        band_hi = 0.97


@pytest.mark.parametrize("yes_price", [0, 0.03, 0.40, 0.65, 0.75, 0.93, 0.98, 1])
def test_migrated_bias_band_matches_production_adapter(yes_price):
    pillar = object.__new__(BiasHarvestPillar)
    pillar._settings = _BiasSettings()
    market = Market(
        id="m1",
        question="Will this resolve yes?",
        outcome_yes_price=yes_price,
        outcome_no_price=1 - yes_price,
    )
    old_shape = pillar._band_check(market)
    decision = select_bias_band(yes_price, band_lo=0.60, band_hi=0.97)
    new_shape = None if decision is None else (
        decision.favored_price,
        decision.buy_yes,
        decision.action_description,
        decision.source_tag,
    )
    assert old_shape == new_shape


def test_experiment_package_has_no_production_side_effect_imports():
    forbidden = {"auramaur.broker", "auramaur.db", "auramaur.exchange"}
    for path in (ROOT / "auramaur" / "experiments").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        offenders = {
            name for name in imports
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in forbidden)
        }
        assert not offenders, (path, offenders)


def test_experiment_package_exposes_no_live_runtime():
    import auramaur.experiments.runtimes as runtimes
    assert not hasattr(runtimes, "LiveRuntime")


def _bias_snapshot(*, observed_price=0.92, maker_bid=0.90, seconds=0, sequence=0):
    observed = NOW + timedelta(seconds=seconds)
    return MarketSnapshot(
        observed_at=observed,
        sequence=sequence,
        market_id="bias-1",
        venue="polymarket",
        prices=(
            PricePoint(instrument_id="bias-1:YES", price=maker_bid),
            PricePoint(instrument_id="bias-1:NO", price=1 - observed_price),
        ),
        features=(FeatureValue(
            name="bias_harvest_inputs",
            payload={
                "yes_price": observed_price, "no_price": 1 - observed_price,
                "active": True, "dispute_risk": "READY", "liquidity": 5000,
                "category_blocked": False,
                "end_at": (observed + timedelta(days=7)).isoformat(),
                "already_entered_or_held": False, "paper_admitted": True,
                "selected_token_available": True,
                "best_bid": maker_bid, "best_ask": maker_bid + 0.03,
            },
            source="production-compatible-fixture",
            observed_at=observed,
            available_at=observed,
        ),),
        data_version="bias-poc-v1",
    )


def _bias_registered():
    rules = BiasHarvestRules(
        band_lo=0.90, band_hi=0.97, edge_uplift=0.04, stake_usd=10,
        min_liquidity=1000, min_hours_to_resolution=6,
        max_days_to_resolution=45, skip_disputed=True, maker_entry=True,
        maker_min_spread=0.02, paper=True,
    )
    spec = ExperimentDefinition(
        key="bias-harvest-poc", strategy_source="bias_harvest",
        hypothesis="Deep favorites are underpriced after maker costs.",
        mechanism="favorite-longshot bias captured passively",
        implementation_version="proposal-v1", parameters=rules.__dict__,
        venues=frozenset({"polymarket"}), primary_metric="net_pnl_after_costs",
        baseline="no_position", min_observations=100, holdout_days=14,
        max_drawdown=0.15, cost_model="maker-linear-v1",
        rejection_criteria=("holdout_pnl_lte_zero",),
    )
    return ExperimentRegistry().register(spec, BiasHarvestExperiment(rules))


@pytest.mark.asyncio
async def test_bias_poc_uses_same_bound_strategy_in_research_replay_and_shadow():
    bound = _bias_registered()
    market = _bias_snapshot()
    account = PortfolioSnapshot(
        observed_at=NOW - timedelta(seconds=1), cash=1000, equity=1000
    )
    research = await ResearchRuntime().run(bound, market, account)
    sink = InMemoryShadowSink()
    shadow = await ShadowRuntime(sink).run(bound, market, account)
    replay = await ReplayRuntime(LinearExecutionModel(
        version="maker-linear-v1"
    )).run(bound, [market], account)
    assert research.targets == shadow.targets == replay.results[0].targets
    target = research.targets[0]
    assert target.instrument_id == "bias-1:YES"
    assert target.reference_price == pytest.approx(0.90)
    assert target.max_notional == pytest.approx(10)


def test_every_production_strategy_is_in_the_migration_inventory():
    assert set(MIGRATION_INVENTORY) == {spec.key for spec in STRATEGY_SPECS}
    proof_of_concepts = {
        key for key, (_, status) in MIGRATION_INVENTORY.items()
        if status is MigrationStatus.PROOF_OF_CONCEPT
    }
    assert proof_of_concepts == set()
    migrated = {
        key for key, (_, status) in MIGRATION_INVENTORY.items()
        if status is MigrationStatus.MIGRATED
    }
    assert migrated == {
        "bias_harvest",
        "core_trading", "news_reactor", "platform_consensus", "long_horizon",
        "agent_trader", "agent_trader_kalshi", "term_structure", "vol_anchor",
        "informed_flow", "econ_indicator", "interim_manager", "settlement_arb",
        "weather_temp", "resolution_lens", "resolution_lens_kalshi",
        "momentum_coupling",
        "arbitrage", "entailment_arb", "cross_venue_arb", "market_maker",
    }
