from datetime import datetime, timedelta, timezone

import pytest

from auramaur.experiments.strategies.long_horizon import (
    LongHorizonInputs,
    LongHorizonRejection,
    LongHorizonRules,
    assess_long_horizon,
)


NOW = datetime(2026, 7, 25, tzinfo=timezone.utc)


def _rules(**changes) -> LongHorizonRules:
    values = dict(
        venue="polymarket",
        band_lo=0.58,
        band_hi=0.85,
        slope=1.32,
        min_edge=0.03,
        stake_usd=8.0,
        min_liquidity=1000.0,
        min_days_to_resolution=30,
        max_days_to_resolution=365,
        source_tag="long_horizon",
    )
    values.update(changes)
    return LongHorizonRules(**values)


def _inputs(**changes) -> LongHorizonInputs:
    values = dict(
        market_id="m1",
        yes_price=0.70,
        active=True,
        venue="polymarket",
        liquidity=5000.0,
        category_blocked=False,
        end_at=NOW + timedelta(days=60),
        as_of=NOW,
        already_entered_or_held=False,
    )
    values.update(changes)
    return LongHorizonInputs(**values)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"yes_price": 0.50}, LongHorizonRejection.OUT_OF_BAND),
        ({"active": False}, LongHorizonRejection.INACTIVE),
        ({"venue": "kalshi"}, LongHorizonRejection.WRONG_VENUE),
        ({"liquidity": 100.0}, LongHorizonRejection.INSUFFICIENT_LIQUIDITY),
        ({"category_blocked": True}, LongHorizonRejection.BLOCKED_CATEGORY),
        ({"end_at": None}, LongHorizonRejection.MISSING_END),
        ({"end_at": NOW + timedelta(days=10)}, LongHorizonRejection.TOO_SOON),
        ({"end_at": NOW + timedelta(days=400)}, LongHorizonRejection.TOO_FAR),
        (
            {"already_entered_or_held": True},
            LongHorizonRejection.ALREADY_ENTERED_OR_HELD,
        ),
    ],
)
def test_long_horizon_rejections_are_explicit(changes, reason):
    assessment = assess_long_horizon(_inputs(**changes), _rules())
    assert assessment.proposal is None
    assert assessment.rejection is reason


@pytest.mark.parametrize(
    ("yes_price", "buy_yes", "instrument", "fair_side"),
    [
        (0.70, True, "m1:YES", "above"),
        (0.30, False, "m1:NO", "below"),
    ],
)
def test_long_horizon_proposal_preserves_direction_and_fair(
    yes_price, buy_yes, instrument, fair_side
):
    proposal = assess_long_horizon(
        _inputs(yes_price=yes_price), _rules()
    ).proposal
    assert proposal is not None
    assert proposal.buy_yes is buy_yes
    assert proposal.instrument_id == instrument
    assert proposal.max_notional == 8.0
    assert proposal.target_quantity * proposal.entry_price == pytest.approx(8.0)
    if fair_side == "above":
        assert proposal.fair_probability > yes_price
    else:
        assert proposal.fair_probability < yes_price


def test_long_horizon_pure_module_has_no_live_execution_imports():
    import auramaur.experiments.strategies.long_horizon as module

    source = module.__file__
    assert source is not None
    text = open(source, encoding="utf-8").read()
    assert "ExecutionGateway" not in text
    assert "TradeIntent" not in text
