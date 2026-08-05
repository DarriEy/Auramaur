from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from auramaur.broker.allocator import CandidateTrade, CapitalAllocator
from tests.test_risk_manager import _make_market, _make_signal


def _settings(hurdle: bool, **risk):
    return SimpleNamespace(
        benchmark=SimpleNamespace(
            risk_free_annual_rate=0.05,
            allocator_cash_hurdle_enabled=hurdle),
        **({"risk": SimpleNamespace(**risk)} if risk else {}),
    )


def test_expected_value_is_ranked_after_cash_hurdle_when_armed():
    allocator = CapitalAllocator(_settings(hurdle=True))
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    market = _make_market()
    market.end_date = now + timedelta(days=365)
    signal = _make_signal(edge=10.0)

    # High-confidence forecast EV is $10; a one-year 5% cash hurdle costs $5.
    assert allocator.compute_expected_value(
        signal, 100.0, market, now=now) == pytest.approx(5.0, abs=0.02)


def test_expected_value_ignores_hurdle_while_disarmed():
    """Tracked default: the ENTRY veto stays off until the 2026-10-31
    pre-registered cell review adjudicates the long-dated book — the same
    market that costs $5 of hurdle when armed ranks at its full forecast
    EV while disarmed."""
    allocator = CapitalAllocator(_settings(hurdle=False))
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    market = _make_market()
    market.end_date = now + timedelta(days=365)
    signal = _make_signal(edge=10.0)

    assert allocator.compute_expected_value(
        signal, 100.0, market, now=now) == 10.0


def test_expected_value_preserves_legacy_result_without_resolution_date():
    allocator = CapitalAllocator(_settings(hurdle=True))
    market = _make_market()
    market.end_date = None
    signal = _make_signal(edge=10.0)

    assert allocator.compute_expected_value(signal, 100.0, market) == 10.0


def test_allocator_does_not_fund_nonpositive_excess_value():
    allocator = CapitalAllocator(_settings(
        hurdle=True, max_open_positions=10, category_exposure_cap_pct=100))
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


def test_tracked_default_keeps_the_entry_veto_disarmed():
    """The committed YAML must not arm the hurdle silently — arming is the
    2026-10-31 review's decision, not a config drift."""
    from config.settings import Settings

    assert Settings().benchmark.allocator_cash_hurdle_enabled is False
