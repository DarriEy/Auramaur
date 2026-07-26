from datetime import datetime, timezone

import pytest

from auramaur.experiments.models import (
    FeatureValue,
    MarketSnapshot,
    PortfolioSnapshot,
    PricePoint,
)
from auramaur.experiments.strategies.term_structure import (
    TermStructureCandidate,
    TermStructureExperiment,
    TermStructureRules,
    select_term_structure_proposals,
)


RULES = TermStructureRules(8.0, 1_000.0, 2, 10.0)


def test_pure_selector_matches_production_ranking_and_direction():
    proposals = select_term_structure_proposals([
        TermStructureCandidate("a", 0.10, 0.40, 5_000),
        TermStructureCandidate("b", 0.30, 0.70, 5_000),
        TermStructureCandidate("c", 0.50, 0.72, 5_000),
        TermStructureCandidate("thin", 0.20, 0.90, 100),
        TermStructureCandidate("claimed", 0.90, 0.10, 5_000, claimed=True),
    ], RULES)
    assert [proposal.market_id for proposal in proposals] == ["b", "a"]
    assert all(proposal.buy_yes for proposal in proposals)
    assert proposals[0].edge_percent == pytest.approx(40.0)
    assert proposals[0].target_quantity == pytest.approx(10.0 / 0.30)


def test_pure_selector_has_no_live_capability():
    assert not hasattr(TermStructureExperiment, "submit")
    assert not hasattr(TermStructureExperiment, "execute")
    assert select_term_structure_proposals([
        TermStructureCandidate("small", 0.30, 0.35, 5_000),
    ], RULES) == []


@pytest.mark.asyncio
async def test_portable_experiment_matches_selector_targets():
    now = datetime(2026, 7, 25, tzinfo=timezone.utc)
    values = [
        {"market_id": "a", "market_probability": 0.10,
         "model_probability": 0.40, "liquidity": 5_000},
        {"market_id": "b", "market_probability": 0.30,
         "model_probability": 0.70, "liquidity": 5_000},
        {"market_id": "c", "market_probability": 0.50,
         "model_probability": 0.72, "liquidity": 5_000},
    ]
    snapshot = MarketSnapshot(
        market_id="family:event", venue="polymarket", observed_at=now,
        sequence=0,
        features=(FeatureValue(
            name="term_structure_candidates", payload=values,
            observed_at=now, available_at=now, source="curve",
        ),),
        prices=(PricePoint(instrument_id="family:event", price=0.5),),
        data_version="test-v1",
    )
    targets = await TermStructureExperiment(RULES).evaluate(
        snapshot,
        PortfolioSnapshot(observed_at=now, cash=100.0, equity=100.0, positions=()),
    )
    assert [target.instrument_id for target in targets] == ["b:YES", "a:YES"]
    assert [target.max_notional for target in targets] == [10.0, 10.0]
