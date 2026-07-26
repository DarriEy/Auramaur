"""Portable vol-anchor proposal parity and non-live runtime coverage."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from auramaur.experiments.models import (
    CapitalEligibility,
    ExperimentDefinition,
    FeatureValue,
    MarketSnapshot,
    PortfolioSnapshot,
    PricePoint,
)
from auramaur.experiments.registry import ExperimentRegistry
from auramaur.experiments.runtimes import InMemoryShadowSink, ResearchRuntime, ShadowRuntime
from auramaur.experiments.strategies.vol_anchor import (
    VolAnchorExperiment,
    VolAnchorInputs,
    VolAnchorRules,
    assess_vol_anchor,
    blended_sigma,
    terminal_above_prob,
    touch_prob,
)

NOW = datetime(2026, 7, 9, tzinfo=timezone.utc)


def _rules() -> VolAnchorRules:
    return VolAnchorRules(
        min_liquidity=1_000.0,
        min_edge_pts=8.0,
        stake_usd=10.0,
        tau_years=0.25,
        long_run_vol={"ethereum": 0.70},
    )


def _inputs(**changes) -> VolAnchorInputs:
    values = {
        "market_id": "eth-touch",
        "question": "Will Ethereum reach $3,000 by December 31, 2026?",
        "yes_price": 0.09,
        "no_price": 0.91,
        "active": True,
        "venue": "polymarket",
        "liquidity": 5_000.0,
        "category_blocked": False,
        "as_of": NOW,
        "spot": 1_734.54,
        "realized_vol": 0.52,
        "already_held": False,
        "term_sigma": None,
    }
    values.update(changes)
    return VolAnchorInputs(**values)


def test_pure_assessment_matches_legacy_formula_path():
    assessment = assess_vol_anchor(_inputs(), _rules())
    proposal = assessment.proposal
    assert proposal is not None
    years = (proposal.deadline - NOW).total_seconds() / (365 * 86400)
    sigma = blended_sigma(0.52, 0.70, years, 0.25)
    assert proposal.sigma == pytest.approx(sigma)
    assert proposal.fair_probability == pytest.approx(
        touch_prob(1_734.54, 3_000.0, sigma, years)
    )
    assert proposal.buy_yes is True
    assert proposal.edge_percent >= 8.0
    assert proposal.max_notional == 10.0


@pytest.mark.parametrize(
    ("question", "market_price", "expected"),
    [
        (
            "Will Ethereum be above $1,600 on December 31, 2026?",
            0.10,
            lambda sigma, years: terminal_above_prob(1_734.54, 1_600, sigma, years),
        ),
        (
            "Will Ethereum be below $1,600 on December 31, 2026?",
            0.90,
            lambda sigma, years: 1.0
            - terminal_above_prob(1_734.54, 1_600, sigma, years),
        ),
    ],
)
def test_terminal_kinds_preserve_pricing_parity(question, market_price, expected):
    assessment = assess_vol_anchor(
        _inputs(question=question, yes_price=market_price), _rules()
    )
    proposal = assessment.proposal
    assert proposal is not None
    years = (proposal.deadline - NOW).total_seconds() / (365 * 86400)
    assert proposal.fair_probability == pytest.approx(expected(proposal.sigma, years))


def _registered():
    rules = _rules()
    definition = ExperimentDefinition(
        key="vol_anchor_poc",
        strategy_source="vol_anchor",
        hypothesis="Long-horizon threshold prices underweight volatility reversion.",
        mechanism="Mean-reverting volatility anchor with GBM threshold pricing.",
        implementation_version="proposal-v1",
        parameters={
            "min_liquidity": rules.min_liquidity,
            "min_edge_pts": rules.min_edge_pts,
            "stake_usd": rules.stake_usd,
            "tau_years": rules.tau_years,
            "long_run_vol": rules.long_run_vol,
        },
        venues=frozenset({"polymarket"}),
        primary_metric="net_pnl",
        baseline="market_probability",
        min_observations=30,
        holdout_days=14,
        max_drawdown=0.15,
        cost_model="linear-v1",
        rejection_criteria=("non_positive_net_pnl",),
        capital_eligibility=CapitalEligibility.PAPER_ONLY,
    )
    return ExperimentRegistry().register(definition, VolAnchorExperiment(rules))


def _snapshot() -> MarketSnapshot:
    details = _inputs().__dict__.copy()
    for key in ("market_id", "venue", "as_of"):
        details.pop(key)
    details["term_sigma"] = None
    return MarketSnapshot(
        observed_at=NOW,
        sequence=1,
        market_id="eth-touch",
        venue="polymarket",
        prices=(PricePoint(instrument_id="eth-touch:YES", price=0.09),),
        features=(FeatureValue(
            name="vol_anchor_inputs",
            payload=details,
            source="test-fixture",
            observed_at=NOW,
            available_at=NOW,
        ),),
        data_version="fixture-v1",
    )


@pytest.mark.asyncio
async def test_research_and_shadow_emit_identical_targets_without_live_orders():
    registered = _registered()
    snapshot = _snapshot()
    portfolio = PortfolioSnapshot(
        observed_at=NOW, cash=100.0, equity=100.0, positions=()
    )
    research = await ResearchRuntime().run(registered, snapshot, portfolio)
    sink = InMemoryShadowSink()
    shadow = await ShadowRuntime(sink).run(registered, snapshot, portfolio)

    assert research.targets == shadow.targets
    assert research.targets[0].instrument_id == "eth-touch:YES"
    assert research.targets[0].max_notional == pytest.approx(10.0)
    assert sink.results == [shadow]
