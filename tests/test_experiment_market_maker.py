"""Parity and safety tests for the specialized market-making experiment."""
from types import SimpleNamespace

import pytest

from auramaur.exchange.models import Market, OrderBook, OrderBookLevel
from auramaur.experiments.strategies.market_maker import (
    QuoteReconciliation,
    form_coupled_quote,
    reconcile_quote,
)
from auramaur.strategy.market_maker import MarketMaker


def _maker() -> MarketMaker:
    settings = SimpleNamespace(
        is_live=False,
        market_maker=SimpleNamespace(
            min_spread_bps=40,
            max_spread_bps=1500,
            quote_size=10.0,
            max_inventory=50.0,
            max_markets=5,
            refresh_seconds=30,
            op_timeout_seconds=15.0,
            paper=True,
        ),
        risk=SimpleNamespace(blocked_categories=[], allowed_categories_live=[]),
    )
    return MarketMaker(settings, SimpleNamespace(), SimpleNamespace())


def _pure(**overrides):
    values = dict(
        market_id="m1",
        token_yes_id="yes",
        token_no_id="no",
        best_bid=0.40,
        best_ask=0.46,
        inventory=0.0,
        quote_size=10.0,
        max_inventory=50.0,
        min_spread_bps=40,
        max_spread_bps=1500,
    )
    values.update(overrides)
    return form_coupled_quote(**values)


def test_production_quote_formation_matches_pure_contract():
    maker = _maker()
    maker._inventory["m1"] = 20.0
    market = Market(id="m1", question="Will it happen?", clob_token_yes="yes", clob_token_no="no")
    book = OrderBook(
        bids=[OrderBookLevel(price=0.40, size=100)],
        asks=[OrderBookLevel(price=0.46, size=100)],
    )

    production, reason = maker._compute_quotes(market, book)
    pure = _pure(inventory=20.0)

    assert reason is None
    assert production is not None and pure.proposal is not None
    assert (production.bid_price, production.ask_price, production.size, production.spread_bps) == (
        pure.proposal.bid_price,
        pure.proposal.ask_price,
        pure.proposal.size,
        pure.proposal.spread_bps,
    )
    assert pure.proposal.inventory.net_yes == 20.0
    assert pure.proposal.inventory.remaining_capacity == 30.0
    assert pure.proposal.post_only is True


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"best_bid": None}, "no_bbo"),
        ({"best_bid": 0.02, "best_ask": 0.98}, "dead_book"),
        ({"inventory": 50.0}, "inventory_capacity"),
        ({"token_no_id": ""}, "invalid_input"),
        ({"max_inventory": 0.0}, "invalid_input"),
        ({"best_bid": float("nan")}, "invalid_input"),
    ],
)
def test_quote_formation_fails_closed(changes, reason):
    formed = _pure(**changes)
    assert formed.proposal is None
    assert formed.rejection_reason == reason


def test_coupled_quote_retains_both_legs_and_notional_floor():
    formed = _pure(best_bid=0.04, best_ask=0.10)
    assert formed.proposal is not None
    quote = formed.proposal
    assert quote.token_yes_id == "yes"
    assert quote.token_no_id == "no"
    assert quote.size * quote.bid_price >= 1.0
    assert quote.size * quote.no_leg_price >= 1.0


def test_reconciliation_preserves_keep_and_cancel_replace_semantics():
    proposal = _pure().proposal
    assert proposal is not None
    assert reconcile_quote(
        proposal,
        active_bid_price=None,
        active_ask_price=None,
        active_size=None,
        both_legs_pending=False,
    ) is QuoteReconciliation.PLACE
    assert reconcile_quote(
        proposal,
        active_bid_price=proposal.bid_price,
        active_ask_price=proposal.ask_price,
        active_size=proposal.size,
        both_legs_pending=True,
    ) is QuoteReconciliation.KEEP
    assert reconcile_quote(
        proposal,
        active_bid_price=proposal.bid_price,
        active_ask_price=proposal.ask_price,
        active_size=proposal.size,
        both_legs_pending=False,
    ) is QuoteReconciliation.CANCEL_REPLACE


def test_experiment_module_has_no_live_execution_imports():
    import auramaur.experiments.strategies.market_maker as module

    source = open(module.__file__, encoding="utf-8").read()
    assert "auramaur.exchange" not in source
    assert "ExecutionGateway" not in source
    assert "PolymarketClient" not in source
    assert "place_order" not in source
