"""Portable economic-indicator decision and runtime contract."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from auramaur.experiments.models import (
    FeatureValue,
    MarketSnapshot,
    PortfolioSnapshot,
    PricePoint,
)
from auramaur.experiments.strategies.econ_indicator import (
    EconIndicatorExperiment,
    EconIndicatorInputs,
    EconIndicatorRejection,
    assess_econ_indicator,
)


def test_assessment_matches_existing_entry_rules():
    sell = assess_econ_indicator(EconIndicatorInputs(
        "m1", 0.50, 0.31, 0.07, 0.30, False
    ))
    assert sell.proposal is not None
    assert not sell.proposal.buy_yes
    assert sell.proposal.instrument_id == "m1:NO"
    assert sell.proposal.edge_percent == 19.0

    assert assess_econ_indicator(EconIndicatorInputs(
        "m1", 0.50, 0.44, 0.07, 0.30, False
    )).rejection == EconIndicatorRejection.INSUFFICIENT_EDGE
    assert assess_econ_indicator(EconIndicatorInputs(
        "m1", 0.03, 0.50, 0.07, 0.30, False
    )).rejection == EconIndicatorRejection.IMPLAUSIBLE_DIVERGENCE
    assert assess_econ_indicator(EconIndicatorInputs(
        "m1", 0.50, 0.31, 0.07, 0.30, True
    )).rejection == EconIndicatorRejection.ALREADY_ENTERED_OR_HELD


def test_experiment_emits_target_without_execution_dependencies():
    async def run():
        now = datetime.now(timezone.utc)
        snapshot = MarketSnapshot(
            market_id="m1",
            venue="kalshi",
            observed_at=now,
            sequence=0,
            prices=(PricePoint(instrument_id="m1:NO", price=0.50),),
            features=(FeatureValue(
                name="econ_indicator_inputs",
                payload={
                    "market_probability": 0.50,
                    "model_probability": 0.31,
                    "required_edge": 0.07,
                    "max_divergence": 0.30,
                    "already_entered_or_held": False,
                },
                observed_at=now,
                available_at=now,
                source="test",
            ),),
            data_version="test-v1",
        )
        portfolio = PortfolioSnapshot(
            observed_at=now, cash=100.0, equity=100.0, positions=()
        )
        targets = await EconIndicatorExperiment(10.0).evaluate(snapshot, portfolio)
        assert len(targets) == 1
        assert targets[0].instrument_id == "m1:NO"
        assert targets[0].target_quantity == 20.0
        assert targets[0].max_notional == 10.0

    asyncio.run(run())
