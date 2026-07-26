"""Contract and production-parity tests for paired arbitrage experiments."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from auramaur.exchange.models import Market, OrderSide
from auramaur.experiments.strategies.arbitrage import (
    ArbitrageMarket,
    form_arbitrage_pair,
)
from auramaur.strategy.arbitrage import ArbitrageExecutor


def _market(mid: str, price: float) -> ArbitrageMarket:
    return ArbitrageMarket(mid, f"{mid}?", price, yes_token=f"{mid}-y", no_token=f"{mid}-n")


def test_conditional_pair_is_atomic_and_immutable():
    opp = {"type": "conditional_violation", "market_a": "a", "market_b": "b",
           "price_a": 0.7, "price_b": 0.6}
    pair = form_arbitrage_pair(opp, _market("a", 0.7), _market("b", 0.6))
    assert pair is not None
    assert (pair.buy.market_id, pair.sell.market_id) == ("b", "a")
    assert pair.buy.edge_percent == pytest.approx(5.0)
    with pytest.raises(FrozenInstanceError):
        pair.edge = 0.2  # type: ignore[misc]


def test_divergence_and_production_conversion_have_parity():
    opp = {"type": "price_divergence", "market_a": "a", "market_b": "b",
           "price_a": 0.42, "price_b": 0.54, "divergence": 0.12}
    pair = form_arbitrage_pair(opp, _market("a", 0.42), _market("b", 0.54))
    assert pair is not None
    buy, sell, returned = ArbitrageExecutor._proposal_to_signals(pair, opp)
    assert returned is opp
    assert (buy.market_id, buy.recommended_side) == ("a", OrderSide.BUY)
    assert (sell.market_id, sell.recommended_side) == ("b", OrderSide.SELL)
    assert buy.claude_prob == pytest.approx(pair.buy.fair_probability)
    assert sell.edge == pytest.approx(pair.sell.edge_percent)


@pytest.mark.parametrize("opp", [
    {},
    {"type": "unknown", "market_a": "a", "market_b": "b", "price_a": .7, "price_b": .6},
    {"type": "price_divergence", "market_a": "a", "market_b": "b", "price_a": .5, "price_b": .5},
    {"type": "price_divergence", "market_a": "a", "market_b": "b", "price_a": float("nan"), "price_b": .5},
])
def test_malformed_or_edgeless_candidates_fail_closed(opp):
    assert form_arbitrage_pair(opp, _market("a", .7), _market("b", .6)) is None


def test_missing_required_leg_token_fails_the_whole_pair():
    a = ArbitrageMarket("a", "a?", .4, yes_token="a-y", no_token="a-n")
    b = ArbitrageMarket("b", "b?", .6, yes_token="b-y", no_token="")
    opp = {"type": "price_divergence", "market_a": "a", "market_b": "b",
           "price_a": .4, "price_b": .6}
    assert form_arbitrage_pair(opp, a, b) is None


def test_production_adapter_delegates_to_pure_pair(monkeypatch):
    seen = {}
    def fake(opp, a, b):
        seen.update(opp=opp, a=a, b=b)
        return None
    monkeypatch.setattr("auramaur.strategy.arbitrage.form_arbitrage_pair", fake)
    executor = ArbitrageExecutor.__new__(ArbitrageExecutor)
    ma = Market(id="a", question="a?", outcome_yes_price=.6)
    mb = Market(id="b", question="b?", outcome_yes_price=.4)
    assert executor._experiment_market(ma).market_id == "a"
    assert fake({}, executor._experiment_market(ma), executor._experiment_market(mb)) is None
    assert seen["a"].yes_price == .6


def test_pure_module_has_no_live_or_production_imports():
    source = Path("auramaur/experiments/strategies/arbitrage.py").read_text(encoding="utf-8")
    assert "auramaur.strategy" not in source
    assert "auramaur.exchange" not in source
    assert "place_order" not in source
