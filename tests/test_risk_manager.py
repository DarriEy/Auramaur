"""Integration tests for RiskManager.evaluate() — exercises the full flow
including pre/post sizing checks, Kelly bankroll=cash, and edge capping."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from auramaur.exchange.models import Confidence, Market, OrderSide, Signal
from auramaur.risk.manager import RiskManager


def _make_settings(
    *,
    max_drawdown_pct=15.0,
    daily_loss_limit=200.0,
    max_open_positions=500,
    min_edge_pct=5.0,
    min_liquidity=1000.0,
    max_spread_pct=5.0,
    confidence_floor="LOW",
    implied_prob_min=0.05,
    implied_prob_max=0.95,
    category_exposure_cap_pct=60.0,
    max_correlated_positions=10,
    time_to_resolution_min_hours=24,
    time_to_resolution_max_days=0,
    second_opinion_divergence_max=0.25,
    max_stake_per_market=25.0,
    kelly_fraction=0.40,
    paper_initial_balance=500.0,
    blocked_categories=None,
):
    from config.settings import RiskConfig
    s = MagicMock()
    rc = RiskConfig(
        max_drawdown_pct=max_drawdown_pct,
        daily_loss_limit=daily_loss_limit,
        max_open_positions=max_open_positions,
        min_edge_pct=min_edge_pct,
        min_liquidity=min_liquidity,
        max_spread_pct=max_spread_pct,
        confidence_floor=confidence_floor,
        implied_prob_min=implied_prob_min,
        implied_prob_max=implied_prob_max,
        category_exposure_cap_pct=category_exposure_cap_pct,
        max_correlated_positions=max_correlated_positions,
        time_to_resolution_min_hours=time_to_resolution_min_hours,
        time_to_resolution_max_days=time_to_resolution_max_days,
        second_opinion_divergence_max=second_opinion_divergence_max,
        max_stake_per_market=max_stake_per_market,
        blocked_categories=blocked_categories or [],
    )
    s.risk = rc
    s.risk_tolerance = 50.0  # neutral -> scaling is a no-op in tests
    s.kelly = MagicMock()
    s.kelly.fraction = kelly_fraction
    s.execution = MagicMock()
    s.execution.paper_initial_balance = paper_initial_balance
    return s


def _make_signal(
    *,
    edge=10.0,
    claude_prob=0.60,
    market_prob=0.50,
    confidence=Confidence.HIGH,
    divergence=0.05,
):
    return Signal(
        market_id="test-market-1",
        claude_prob=claude_prob,
        claude_confidence=confidence,
        market_prob=market_prob,
        edge=edge,
        divergence=divergence,
        recommended_side=OrderSide.BUY,
    )


def _make_market(
    *,
    yes_price=0.50,
    liquidity=10000.0,
    spread=1.0,
    category="politics",
    days_until_resolution=7,
):
    end = datetime.now(timezone.utc) + timedelta(days=days_until_resolution)
    return Market(
        id="test-market-1",
        exchange="polymarket",
        ticker="test-market-1",
        question="Will test event happen?",
        outcome_yes_price=yes_price,
        outcome_no_price=1.0 - yes_price,
        liquidity=liquidity,
        volume=liquidity,
        spread=spread,
        category=category,
        end_date=end,
        active=True,
    )


def _mock_portfolio(*, drawdown=3.0, daily_pnl=0.0, positions=None, category_exposure=None, correlated=0):
    tracker = MagicMock()
    tracker.get_drawdown = AsyncMock(return_value=drawdown)
    tracker.get_daily_pnl = AsyncMock(return_value=daily_pnl)
    tracker.get_positions = AsyncMock(return_value=positions or [])
    tracker.get_category_exposure = AsyncMock(return_value=category_exposure or {})
    tracker.get_correlated_markets = AsyncMock(return_value=correlated)
    return tracker


@pytest.mark.asyncio
@patch("auramaur.risk.manager.check_kill_switch")
async def test_evaluate_passing_case(mock_kill):
    from auramaur.risk.checks import CheckResult
    mock_kill.return_value = CheckResult(name="kill_switch", passed=True, reason="", value=False)

    settings = _make_settings()
    db = MagicMock()
    db.fetchone = AsyncMock(return_value=None)

    manager = RiskManager(settings, db)
    manager.portfolio = _mock_portfolio()

    signal = _make_signal(edge=10.0, claude_prob=0.60, market_prob=0.50)
    market = _make_market()

    decision = await manager.evaluate(signal, market, available_cash=500.0)

    assert decision.approved is True
    assert decision.position_size > 0
    assert len(decision.checks) == 16
    assert all(c.passed for c in decision.checks)


@pytest.mark.asyncio
@patch("auramaur.risk.manager.check_kill_switch")
async def test_evaluate_fails_on_low_edge(mock_kill):
    from auramaur.risk.checks import CheckResult
    mock_kill.return_value = CheckResult(name="kill_switch", passed=True, reason="", value=False)

    settings = _make_settings(min_edge_pct=5.0)
    db = MagicMock()
    db.fetchone = AsyncMock(return_value=None)

    manager = RiskManager(settings, db)
    manager.portfolio = _mock_portfolio()

    signal = _make_signal(edge=2.0, claude_prob=0.52, market_prob=0.50)
    market = _make_market()

    decision = await manager.evaluate(signal, market, available_cash=500.0)

    assert decision.approved is False
    assert decision.position_size == 0.0
    failed_names = [c.name for c in decision.checks if not c.passed]
    assert "min_edge" in failed_names


@pytest.mark.asyncio
@patch("auramaur.risk.manager.check_kill_switch")
async def test_bankroll_uses_cash_not_equity(mock_kill):
    """Kelly bankroll should be based on cash, not equity (cash + position notional)."""
    from auramaur.risk.checks import CheckResult
    mock_kill.return_value = CheckResult(name="kill_switch", passed=True, reason="", value=False)

    settings = _make_settings(max_stake_per_market=25.0)
    db = MagicMock()
    db.fetchone = AsyncMock(return_value=None)

    manager = RiskManager(settings, db)
    manager.portfolio = _mock_portfolio()

    signal = _make_signal(edge=10.0, claude_prob=0.60, market_prob=0.50)
    market = _make_market()

    small_cash = await manager.evaluate(signal, market, available_cash=100.0)
    big_cash = await manager.evaluate(signal, market, available_cash=500.0)

    assert small_cash.approved is True
    assert big_cash.approved is True
    assert small_cash.position_size < big_cash.position_size


@pytest.mark.asyncio
@patch("auramaur.risk.manager.check_kill_switch")
async def test_edge_cap_limits_outsized_bets(mock_kill):
    """A 45% claimed edge should be capped at 20% inside Kelly, preventing
    outsized bets relative to a more moderate 15% edge signal."""
    from auramaur.risk.checks import CheckResult
    mock_kill.return_value = CheckResult(name="kill_switch", passed=True, reason="", value=False)

    settings = _make_settings()
    db = MagicMock()
    db.fetchone = AsyncMock(return_value=None)

    manager = RiskManager(settings, db)
    manager.portfolio = _mock_portfolio()
    market = _make_market()

    huge_edge = _make_signal(edge=45.0, claude_prob=0.95, market_prob=0.50)
    moderate_edge = _make_signal(edge=15.0, claude_prob=0.65, market_prob=0.50)

    huge_decision = await manager.evaluate(huge_edge, market, available_cash=500.0)
    moderate_decision = await manager.evaluate(moderate_edge, market, available_cash=500.0)

    assert huge_decision.approved is True
    assert moderate_decision.approved is True
    # With the 20% edge cap, the "huge" edge signal should NOT produce
    # a dramatically larger position than the moderate one
    ratio = huge_decision.position_size / moderate_decision.position_size if moderate_decision.position_size > 0 else float("inf")
    assert ratio < 3.0, f"Edge cap not working: huge/moderate ratio = {ratio:.1f}"


@pytest.mark.asyncio
@patch("auramaur.risk.manager.check_kill_switch")
async def test_max_stake_check_runs_after_sizing(mock_kill):
    """The 15th check (max_stake) should validate the actual computed size,
    not the signal's recommended_size."""
    from auramaur.risk.checks import CheckResult
    mock_kill.return_value = CheckResult(name="kill_switch", passed=True, reason="", value=False)

    settings = _make_settings()
    db = MagicMock()
    db.fetchone = AsyncMock(return_value=None)

    manager = RiskManager(settings, db)
    manager.portfolio = _mock_portfolio()

    signal = _make_signal(edge=10.0, claude_prob=0.60, market_prob=0.50)
    market = _make_market()

    decision = await manager.evaluate(signal, market, available_cash=500.0)

    stake_check = next(c for c in decision.checks if c.name == "max_stake")
    assert stake_check.passed is True
    # The stake check's value should be the actual position_size, not 0 or the signal's recommended_size
    assert stake_check.value == decision.position_size
