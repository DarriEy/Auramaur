"""Contract and production-parity tests for entailment pair formation."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from auramaur.exchange.models import Market, OrderSide
from auramaur.experiments.strategies.entailment_arb import (
    EntailmentMarket,
    form_entailment_pair,
)
from auramaur.strategy.entailment_arb import EntailmentArbPillar


def _market(mid: str, probability: float) -> EntailmentMarket:
    return EntailmentMarket(mid, f"{mid}?", probability)


def test_positive_implication_pair_is_atomic_and_immutable():
    pair = form_entailment_pair(
        _market("above-70", 0.62), _market("above-60", 0.50),
        rationale="above 70 => above 60", confidence=1.0, minimum_gap=0.05,
    )
    assert pair is not None
    assert (pair.leg_a.side, pair.leg_b.side) == ("SELL", "BUY")
    assert pair.gap == pytest.approx(0.12)
    assert pair.leg_a.fair_probability == pytest.approx(0.50)
    assert pair.leg_b.fair_probability == pytest.approx(0.62)
    with pytest.raises(FrozenInstanceError):
        pair.gap = 0.2  # type: ignore[misc]


def test_negative_implication_forms_two_no_legs():
    pair = form_entailment_pair(
        _market("a", 0.64), _market("b", 0.48), rationale="mutually exclusive",
        confidence=0.99, minimum_gap=0.10, negative_implication=True,
    )
    assert pair is not None
    assert (pair.leg_a.side, pair.leg_b.side) == ("SELL", "SELL")
    assert pair.gap == pytest.approx(0.12)
    assert pair.leg_a.fair_probability == pytest.approx(0.52)


@pytest.mark.parametrize(
    ("a", "b", "rationale", "confidence", "minimum_gap"),
    [
        (_market("a", .5), _market("a", .4), "same", 1.0, .01),
        (_market("a", float("nan")), _market("b", .4), "bad", 1.0, .01),
        (_market("a", .51), _market("b", .50), "thin", 1.0, .02),
        (_market("a", .7), _market("b", .5), "", 1.0, .02),
        (_market("a", .7), _market("b", .5), "bad confidence", 1.1, .02),
    ],
)
def test_invalid_or_ineligible_pair_fails_closed(a, b, rationale, confidence, minimum_gap):
    assert form_entailment_pair(
        a, b, rationale=rationale, confidence=confidence, minimum_gap=minimum_gap,
    ) is None


def test_production_adapter_preserves_pair_signal_semantics():
    implier = Market(id="a", question="a?", outcome_yes_price=.7)
    implied = Market(id="b", question="b?", outcome_yes_price=.5)
    pair = form_entailment_pair(
        EntailmentArbPillar._experiment_market(implier),
        EntailmentArbPillar._experiment_market(implied),
        rationale="a => b", confidence=1.0, minimum_gap=.05,
    )
    assert pair is not None
    pillar = EntailmentArbPillar.__new__(EntailmentArbPillar)
    sig_a = pillar._leg_signal(
        implier, OrderSide(pair.leg_a.side), pair.leg_a.fair_probability,
        pair.gap, pair.rationale,
    )
    sig_b = pillar._leg_signal(
        implied, OrderSide(pair.leg_b.side), pair.leg_b.fair_probability,
        pair.gap, pair.rationale,
    )
    assert (sig_a.recommended_side, sig_b.recommended_side) == (
        OrderSide.SELL, OrderSide.BUY,
    )
    assert sig_a.claude_prob == pytest.approx(pair.leg_a.fair_probability)
    assert sig_b.edge == pytest.approx(pair.leg_b.edge_percent)


def test_pure_pair_module_has_no_live_or_production_imports():
    source = Path("auramaur/experiments/strategies/entailment_arb.py").read_text(
        encoding="utf-8"
    )
    assert "auramaur.strategy" not in source
    assert "auramaur.exchange" not in source
    assert "ExecutionGateway" not in source
    assert "place_order" not in source
