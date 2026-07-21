from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import pytest
from auramaur.evaluation.portfolio import MarketSnapshot, SimulationPolicy, VirtualPortfolio

NOW = datetime(2026, 7, 21, tzinfo=timezone.utc)
@dataclass(frozen=True)
class Forecast:
    action: str

def snap(**kw):
    data = dict(market_id="m1", category="politics", timestamp=NOW,
                yes_bid=.39, yes_ask=.40, no_bid=.59, no_ask=.60,
                yes_bid_depth=1000, yes_ask_depth=1000,
                no_bid_depth=1000, no_ask_depth=1000)
    data.update(kw)
    return MarketSnapshot(**data)

def test_independent_arms_identical_fills_and_same_market_allowed():
    a, b = VirtualPortfolio("a", SimulationPolicy()), VirtualPortfolio("b", SimulationPolicy())
    assert a.enter(snap(), Forecast("yes")) == b.enter(snap(), Forecast("yes"))
    a.settle("m1", True, NOW + timedelta(days=1))
    assert "m1" in b.positions

def test_abstain_invalid_price_and_friction_depth():
    p = SimulationPolicy(fee_rate=.01, slippage_bps=100)
    book = VirtualPortfolio("a", p)
    assert book.enter(snap(), Forecast("abstain")) is None
    assert book.enter(snap(yes_ask=0), Forecast("yes")) is None
    fill = book.enter(snap(yes_ask_depth=10), Forecast("yes"))
    assert fill and fill.price == pytest.approx(.404) and fill.quantity == 10
    assert fill.fee == pytest.approx(.0404)

def test_settlement_capital_time_and_limits():
    p = SimulationPolicy(initial_cash=50, order_notional=40, max_position_notional=30,
                         max_total_exposure=35, max_category_exposure=30, max_positions=2)
    book = VirtualPortfolio("a", p)
    first = book.enter(snap(), Forecast("yes"))
    assert first and first.quantity * first.price == pytest.approx(30)
    assert book.enter(snap(market_id="m2", timestamp=NOW+timedelta(seconds=1)), Forecast("yes")) is None
    third = book.enter(snap(market_id="m3", category="sports", timestamp=NOW+timedelta(seconds=2)), Forecast("yes"))
    assert third and third.quantity * third.price == pytest.approx(5)
    pnl = book.settle("m1", True, NOW + timedelta(hours=24))
    assert pnl == pytest.approx(45)
    assert book.metrics().capital_hours > 0

def test_partial_exit_uses_executable_bid_and_metrics():
    book = VirtualPortfolio("a", SimulationPolicy(order_notional=40, fee_rate=.01))
    book.enter(snap(), Forecast("yes"))
    fill = book.exit(snap(timestamp=NOW+timedelta(hours=1), yes_bid=.5, yes_bid_depth=25))
    assert fill and fill.quantity == 25 and fill.price == pytest.approx(.5)
    assert book.positions["m1"].quantity == pytest.approx(75)
    assert book.metrics().turnover == pytest.approx(52.5)
