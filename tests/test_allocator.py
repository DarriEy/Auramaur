from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from auramaur.broker.allocator import CandidateTrade, CapitalAllocator
from tests.test_risk_manager import _make_market, _make_signal


def test_expected_value_is_ranked_after_cash_hurdle():
    settings = SimpleNamespace(
        benchmark=SimpleNamespace(risk_free_annual_rate=0.05))
    allocator = CapitalAllocator(settings)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    market = _make_market()
    market.end_date = now + timedelta(days=365)
    signal = _make_signal(edge=10.0)

    # High-confidence forecast EV is $10; a one-year 5% cash hurdle costs $5.
    assert allocator.compute_expected_value(
        signal, 100.0, market, now=now) == pytest.approx(5.0, abs=0.02)


def test_expected_value_preserves_legacy_result_without_resolution_date():
    settings = SimpleNamespace(
        benchmark=SimpleNamespace(risk_free_annual_rate=0.05))
    allocator = CapitalAllocator(settings)
    market = _make_market()
    market.end_date = None
    signal = _make_signal(edge=10.0)

    assert allocator.compute_expected_value(signal, 100.0, market) == 10.0


def test_allocator_does_not_fund_nonpositive_excess_value():
    settings = SimpleNamespace(
        benchmark=SimpleNamespace(risk_free_annual_rate=0.05),
        risk=SimpleNamespace(
            max_open_positions=10, category_exposure_cap_pct=100))
    allocator = CapitalAllocator(settings)
    market = _make_market(category="tech")
    signal = _make_signal()
    candidate = CandidateTrade(
        market=market,
        signal=signal,
        risk_decision=SimpleNamespace(),
        kelly_size=10.0,
        expected_value=0.0,
    )

    assert allocator.allocate([candidate], 100.0, []) == []
    assert candidate.allocated_size == 0.0
