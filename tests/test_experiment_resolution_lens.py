from datetime import datetime, timezone

import pytest

from auramaur.experiments.models import (
    FeatureValue,
    MarketSnapshot,
    PortfolioSnapshot,
    PricePoint,
)
from auramaur.experiments.strategies.resolution_lens import (
    ResolutionLensCandidate,
    ResolutionLensExperiment,
    ResolutionLensRules,
    form_resolution_lens_proposal,
)


RULES = ResolutionLensRules(
    min_entry_price=0.55,
    high_conf_gap_score=0.8,
    stake_usd=12.0,
)


def test_proposal_preserves_production_signal_semantics():
    proposal = form_resolution_lens_proposal(
        ResolutionLensCandidate("market-1", 0.60, 0.82, 0.85, "literal deadline"),
        RULES,
    )
    assert proposal is not None
    assert proposal.buy_yes
    assert proposal.high_confidence
    assert proposal.edge_percent == pytest.approx(22.0)
    assert proposal.evidence_summary == "Resolution lens (gap 0.85): literal deadline"
    assert proposal.mispricing_reason == "behavioral: literal deadline"
    assert proposal.instrument_id == "market-1:YES"
    assert proposal.target_quantity == pytest.approx(20.0)

    sell = form_resolution_lens_proposal(
        ResolutionLensCandidate("market-2", 0.30, 0.10, 0.5, "permanence clause"),
        RULES,
    )
    assert sell is not None
    assert not sell.buy_yes
    assert not sell.high_confidence
    assert sell.instrument_id == "market-2:NO"
    assert sell.reference_price == pytest.approx(0.70)


@pytest.mark.parametrize("candidate", [
    ResolutionLensCandidate("", 0.6, 0.8, 0.9, "deadline"),
    ResolutionLensCandidate("m", float("nan"), 0.8, 0.9, "deadline"),
    ResolutionLensCandidate("m", 0.6, 1.2, 0.9, "deadline"),
    ResolutionLensCandidate("m", 0.6, 0.8, -0.1, "deadline"),
    ResolutionLensCandidate("m", 0.6, 0.8, 0.9, "  "),
    ResolutionLensCandidate("m", 0.60, 0.60, 0.9, "deadline"),
    ResolutionLensCandidate("m", 0.40, 0.80, 0.9, "deadline"),
])
def test_proposal_fails_closed(candidate):
    assert form_resolution_lens_proposal(candidate, RULES) is None


@pytest.mark.asyncio
async def test_portable_experiment_matches_pure_proposal_without_execution_capability():
    now = datetime(2026, 7, 25, tzinfo=timezone.utc)
    snapshot = MarketSnapshot(
        observed_at=now,
        sequence=4,
        market_id="market-1",
        venue="polymarket",
        prices=(PricePoint(instrument_id="market-1:YES", price=0.60),),
        features=(FeatureValue(
            name="resolution_lens_candidate",
            payload={
                "market_probability": 0.60,
                "fair_probability": 0.82,
                "gap_score": 0.85,
                "mechanism": "literal deadline",
            },
            source="grounded-lens",
            observed_at=now,
            available_at=now,
        ),),
        data_version="test-v1",
    )
    experiment = ResolutionLensExperiment(RULES)
    targets = await experiment.evaluate(
        snapshot,
        PortfolioSnapshot(observed_at=now, cash=100, equity=100),
    )
    assert len(targets) == 1
    assert targets[0].instrument_id == "market-1:YES"
    assert targets[0].reference_price == pytest.approx(0.60)
    assert targets[0].max_notional == 12.0
    assert not hasattr(experiment, "submit")
    assert not hasattr(experiment, "execute")
