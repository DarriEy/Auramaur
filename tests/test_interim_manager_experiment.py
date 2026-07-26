"""Portable interim-manager proposal assessment and production boundary."""

from datetime import datetime, timedelta, timezone

import pytest

from auramaur.experiments.strategies.interim_manager import (
    InterimManagerInputs,
    InterimManagerRejection,
    InterimManagerRules,
    assess_interim_manager,
)


NOW = datetime(2026, 7, 25, tzinfo=timezone.utc)
THESIS = (
    "Resolution fine print prices NO events as YES while the market ignores "
    "the settlement source's published revision schedule."
)


def _rules(**changes):
    values = dict(
        proposal_ttl_hours=24.0,
        max_open_positions=3,
        min_robust_edge=0.05,
        default_uncertainty_buffer=0.02,
        slippage_buffer=0.005,
        liquidity_penalty=0.01,
        thin_liquidity_usd=100.0,
        correlation_penalty_per_position=0.005,
        stake_usd=10.0,
        paper=True,
    )
    values.update(changes)
    return InterimManagerRules(**values)


def _inputs(**changes):
    values = dict(
        proposal_id=7,
        market_id="M1",
        thesis=THESIS,
        side="BUY",
        fair_probability=0.72,
        requested_stake_usd=25.0,
        market_yes_price=0.40,
        market_liquidity=500.0,
        category="economics",
        created_at=NOW - timedelta(hours=1),
        sunset_at=NOW + timedelta(hours=6),
        as_of=NOW,
        delegated_to=None,
        open_positions=0,
        open_positions_in_category=0,
        fee_rate=0.0,
        confidence_lo=0.70,
        confidence_hi=0.74,
    )
    values.update(changes)
    return InterimManagerInputs(**values)


def test_valid_candidate_becomes_bounded_paper_proposal():
    result = assess_interim_manager(_inputs(), _rules())

    assert result.rejection is None
    assert result.proposal is not None
    assert result.proposal.buy_yes is True
    assert result.proposal.entry_price == pytest.approx(0.40)
    assert result.proposal.requested_stake_usd == 10.0
    assert result.proposal.force_paper is True
    assert result.proposal.robust_edge == pytest.approx(0.295)


@pytest.mark.parametrize(
    ("changes", "rejection"),
    [
        ({"created_at": NOW - timedelta(hours=25)}, InterimManagerRejection.EXPIRED),
        ({"thesis": "just vibes"}, InterimManagerRejection.THESIS_TOO_SHORT),
        ({"delegated_to": "econ_indicator"}, InterimManagerRejection.DELEGATED),
        ({"sunset_at": NOW}, InterimManagerRejection.SUNSET_REACHED),
        ({"open_positions": 3}, InterimManagerRejection.POSITION_CAP),
        ({"max_entry_price": 0.39}, InterimManagerRejection.ENTRY_LIMIT),
        ({"fair_probability": 0.47}, InterimManagerRejection.INSUFFICIENT_EDGE),
    ],
)
def test_charter_rejections_are_deterministic(changes, rejection):
    result = assess_interim_manager(_inputs(**changes), _rules())
    assert result.proposal is None
    assert result.rejection == rejection


def test_sell_uses_no_entry_price_and_directional_edge():
    result = assess_interim_manager(
        _inputs(side="SELL", fair_probability=0.15, market_yes_price=0.40),
        _rules(),
    )
    assert result.proposal is not None
    assert result.proposal.buy_yes is False
    assert result.proposal.entry_price == pytest.approx(0.60)


def test_pure_module_has_no_live_execution_dependencies():
    import ast

    source = __import__(
        "auramaur.experiments.strategies.interim_manager", fromlist=["dummy"]
    ).__file__
    with open(source, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    imports = {
        node.module or "" for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    forbidden = ("auramaur.broker", "auramaur.db", "auramaur.exchange")
    assert not any(name.startswith(forbidden) for name in imports)
