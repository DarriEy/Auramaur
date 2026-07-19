from datetime import datetime, timedelta

import pytest

from auramaur.backtest.walk_forward import expanding_walk_forward
from auramaur.research.polymarket_strategies import (
    DecisionTracker,
    OutcomeQuote,
    Relation,
    complete_set_arbitrage,
    flow_confirmation_multiplier,
    logical_constraint_edge,
    maker_quote_economics,
    normal_bin_probability,
    source_latency_edge,
)


def test_complete_set_uses_executable_prices_fees_and_depth():
    quotes = [
        OutcomeQuote("a", "A", .40, .42, 100, 50, .005),
        OutcomeQuote("b", "B", .50, .52, 100, 20, .005),
    ]
    opportunities = complete_set_arbitrage("event", quotes, safety_buffer=.01)
    assert len(opportunities) == 1
    assert opportunities[0].mechanism == "buy_all_below_one"
    assert opportunities[0].expected_edge == pytest.approx(.04)
    assert opportunities[0].capacity_usd == pytest.approx(19.0)


def test_constraint_requires_strict_semantic_verification():
    assert logical_constraint_edge("r", Relation.A_IMPLIES_B, .7, .5,
                                   verification_confidence=.90) is None
    result = logical_constraint_edge("r", Relation.A_IMPLIES_B, .7, .5,
                                     verification_confidence=.99)
    assert result is not None and result.expected_edge == pytest.approx(.18)


def test_source_latency_is_authoritative_fresh_and_same_direction():
    assert source_latency_edge(
        "m", source_age_seconds=10, observed_market_move=.01,
        expected_move=.08, source_is_authoritative=True,
    ) is not None
    assert source_latency_edge(
        "m", source_age_seconds=10, observed_market_move=-.01,
        expected_move=.08, source_is_authoritative=True,
    ) is None


def test_release_nowcast_and_maker_economics():
    assert normal_bin_probability(0, 1, None, 0) == pytest.approx(.5)
    assert maker_quote_economics(
        "m", spread_capture=.02, fill_probability=.5, expected_rebate=.005,
        adverse_selection=.005, inventory_cost=.002, cancel_failure_cost=.001,
    ) is not None


def test_flow_never_originates_signal_and_is_bounded():
    assert flow_confirmation_multiplier(base_signal_edge=0, signed_flow_z=5,
                                        same_direction=True) == 0
    assert flow_confirmation_multiplier(base_signal_edge=.1, signed_flow_z=10,
                                        same_direction=True) == 1.25
    assert flow_confirmation_multiplier(base_signal_edge=.1, signed_flow_z=10,
                                        same_direction=False) == .25


def test_walk_forward_is_chronological_and_event_unique():
    start = datetime(2026, 1, 1)
    rows = [{"event": str(i // 2), "at": start + timedelta(days=i)}
            for i in range(12)]
    folds = expanding_walk_forward(
        rows, timestamp=lambda x: x["at"], event_key=lambda x: x["event"],
        min_train=3, test_size=2,
    )
    assert len(folds) == 2
    assert len(folds[0].train) == 3
    assert {x["event"] for x in folds[0].train}.isdisjoint(
        {x["event"] for x in folds[0].test})


@pytest.mark.asyncio
async def test_decision_tracker_persists_immutable_snapshot(tmp_path):
    from auramaur.db.database import Database

    db = Database(str(tmp_path / "research.db"))
    await db.connect()
    tracker = DecisionTracker(db)
    kwargs = dict(market_id="m", strategy_source="llm", signal_id=7,
                  side="BUY", fair_probability=.6, reference_price=.5,
                  executable_price=.51, best_bid=.49, best_ask=.51,
                  requested_size=10, fee_estimate=.1)
    await tracker.capture(**kwargs)
    await tracker.capture(**{**kwargs, "executable_price": .99})
    row = await db.fetchone("SELECT executable_price FROM decision_snapshots")
    assert row["executable_price"] == pytest.approx(.51)
    await db.close()
