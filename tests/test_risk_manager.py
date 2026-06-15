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
    allowed_categories_live=None,
    is_live=False,
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
        **({"allowed_categories_live": allowed_categories_live}
           if allowed_categories_live is not None else {}),
    )
    s.risk = rc
    # MagicMock attributes are truthy — pin the live flag so the allowlist
    # check only arms when a test asks for it.
    s.is_live = is_live
    s.risk_tolerance = 50.0  # neutral -> scaling is a no-op in tests
    from config.settings import GraduationConfig
    s.graduation = GraduationConfig(mode="off")  # ladder exercised in test_graduation.py
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
    spread=0.01,  # price-unit fraction: 0.01 == 1% spread (within the 5% cap)
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
    # 15 pre + mispricing_named + blocked_category + max_stake
    assert len(decision.checks) == 18
    assert all(c.passed for c in decision.checks)


@pytest.mark.asyncio
@patch("auramaur.risk.manager.check_kill_switch")
async def test_per_market_stake_clamped_to_absolute_ceiling(mock_kill):
    """2%-of-equity * regime can exceed the documented cap as the book grows;
    the absolute ceiling must bind so per-market stake never drifts past it."""
    from auramaur.risk.checks import CheckResult
    mock_kill.return_value = CheckResult(name="kill_switch", passed=True, reason="", value=False)

    # max_stake as a FRACTION (0.50 = 50% of equity) — huge against a big
    # bankroll. The $25 absolute ceiling must clamp it.
    settings = _make_settings(max_stake_per_market=0.50, min_edge_pct=2.5)
    settings.risk.max_stake_abs_ceiling = 25.0
    db = MagicMock()
    db.fetchone = AsyncMock(return_value=None)

    manager = RiskManager(settings, db)
    manager.portfolio = _mock_portfolio()

    # Strong signal + large cash: Kelly wants far more than $25.
    signal = _make_signal(edge=20.0, claude_prob=0.70, market_prob=0.50)
    market = _make_market()

    decision = await manager.evaluate(signal, market, available_cash=5000.0)

    assert decision.approved is True
    assert decision.position_size > 0
    # Without the ceiling this would size to thousands (50% of $5000 equity);
    # the clamp holds it at the documented $25.
    assert decision.position_size <= 25.0 + 1e-9


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


@pytest.mark.asyncio
@patch("auramaur.risk.manager.check_kill_switch")
async def test_blocked_category_enforced_at_gateway_for_all_entry_paths(mock_kill):
    """Regression (2026-06-10): news_speed bought $42 of politics_us live —
    blocked_categories was only an engine-level SELECTION filter that
    analyze_market paths never traverse. It is now risk check #17 at the
    single gateway: directional strategies blocked everywhere, structural
    two-sided strategies (graduation's exempt list) still free."""
    from auramaur.risk.checks import CheckResult
    mock_kill.return_value = CheckResult(name="kill_switch", passed=True, reason="", value=False)

    settings = _make_settings(blocked_categories=["politics_us", "sports"])
    db = MagicMock()
    db.fetchone = AsyncMock(return_value=None)
    manager = RiskManager(settings, db)
    manager.portfolio = _mock_portfolio()
    market = _make_market(category="politics_us")

    # news_speed (the live offender): blocked.
    sig = _make_signal(edge=10.0, claude_prob=0.60, market_prob=0.50)
    sig.strategy_source = "news_speed"
    d = await manager.evaluate(sig, market, available_cash=500.0)
    assert d.approved is False
    assert any(c.name == "blocked_category" and not c.passed for c in d.checks)

    # llm: blocked too.
    sig2 = _make_signal(edge=10.0, claude_prob=0.60, market_prob=0.50)
    d2 = await manager.evaluate(sig2, market, available_cash=500.0)
    assert any(c.name == "blocked_category" and not c.passed for c in d2.checks)

    # arbitrage (structural, exempt): passes the category check.
    sig3 = _make_signal(edge=10.0, claude_prob=0.60, market_prob=0.50)
    sig3.strategy_source = "arbitrage"
    d3 = await manager.evaluate(sig3, market, available_cash=500.0)
    assert not any(c.name == "blocked_category" and not c.passed for c in d3.checks)


@pytest.mark.asyncio
@patch("auramaur.risk.manager.check_kill_switch")
async def test_blocked_category_classifies_uncategorized_markets(mock_kill):
    """A discovery object with category='' is classified on the spot — empty
    categories were the original gate leak (156 politics_us markets hid as
    blank strings)."""
    from auramaur.risk.checks import CheckResult
    mock_kill.return_value = CheckResult(name="kill_switch", passed=True, reason="", value=False)

    settings = _make_settings(blocked_categories=["politics_us"])
    db = MagicMock()
    db.fetchone = AsyncMock(return_value=None)
    manager = RiskManager(settings, db)
    manager.portfolio = _mock_portfolio()

    market = _make_market(category="politics_us")
    market.category = ""
    market.question = "Will the Republican Party control the Senate after the 2026 election?"
    sig = _make_signal(edge=10.0, claude_prob=0.60, market_prob=0.50)
    sig.strategy_source = "news_speed"
    d = await manager.evaluate(sig, market, available_cash=500.0)
    blocked = next(c for c in d.checks if c.name == "blocked_category")
    assert blocked.passed is False
    assert blocked.value == "politics_us"  # classified on the spot


@pytest.mark.asyncio
@patch("auramaur.risk.manager.check_kill_switch")
async def test_blocked_category_catches_stale_stored_label(mock_kill):
    """Regression (2026-06-10): 247 active sports markets were stored as
    'other'/'politics_intl' (stale labels), dodging the block because the
    gateway trusted the stored category. The gate now blocks on a FRESH
    classification too — a sports matchup stored as 'other' is blocked."""
    from auramaur.risk.checks import CheckResult
    mock_kill.return_value = CheckResult(name="kill_switch", passed=True, reason="", value=False)

    settings = _make_settings(blocked_categories=["sports", "politics_us"])
    db = MagicMock()
    db.fetchone = AsyncMock(return_value=None)
    manager = RiskManager(settings, db)
    manager.portfolio = _mock_portfolio()

    market = _make_market(category="other")  # stale stored label
    market.question = "Warriors vs. Celtics"  # the real stored-as-other pattern
    market.description = ""
    sig = _make_signal(edge=10.0, claude_prob=0.60, market_prob=0.50)
    d = await manager.evaluate(sig, market, available_cash=500.0)
    assert d.approved is False
    blocked = next(c for c in d.checks if c.name == "blocked_category")
    assert blocked.passed is False
    assert blocked.value == "sports"  # the fresh classification, not 'other'


@pytest.mark.asyncio
@patch("auramaur.risk.manager.check_kill_switch")
async def test_live_allowlist_fails_closed_on_other_and_unknown(mock_kill):
    """2026-06-12 inversion: a blocklist fails OPEN on every classification
    gap (the mislabel leak bought tennis stored as politics_us live). In
    live mode the stored category must be ON the allowlist — 'other',
    unknown, and unlisted categories stay paper-only."""
    from auramaur.risk.checks import CheckResult
    mock_kill.return_value = CheckResult(name="kill_switch", passed=True, reason="", value=False)

    settings = _make_settings(is_live=True,
                              allowed_categories_live=["tech", "crypto"])
    db = MagicMock()
    db.fetchone = AsyncMock(return_value=None)
    manager = RiskManager(settings, db)
    manager.portfolio = _mock_portfolio()

    for cat in ("other", "politics_us", "sports", "never-seen-venue-label"):
        market = _make_market(category=cat)
        sig = _make_signal(edge=10.0, claude_prob=0.60, market_prob=0.50)
        sig.strategy_source = "news_speed"
        d = await manager.evaluate(sig, market, available_cash=500.0)
        gate = next(c for c in d.checks if c.name == "category_allowlist")
        assert gate.passed is False, cat
        assert d.approved is False, cat


@pytest.mark.asyncio
@patch("auramaur.risk.manager.check_kill_switch")
async def test_paper_forced_strategy_bypasses_live_only_gates(mock_kill):
    """A paper-forced exploration pillar (cfg.paper=True) must NOT be gated by
    the live-only category allowlist or the divergence-adverse filter, so it can
    build a complete paper record across all categories for the graduation
    ladder. Genuine live entries still hit both gates (tested above)."""
    from auramaur.risk.checks import CheckResult
    mock_kill.return_value = CheckResult(name="kill_switch", passed=True, reason="", value=False)

    settings = _make_settings(is_live=True, allowed_categories_live=["crypto"])
    # genuine bool -> _paper_forced_strategy trips (MagicMock attrs would not)
    settings.resolution_lens = MagicMock()
    settings.resolution_lens.paper = True
    settings.risk.divergence_filter_enabled = True  # would reject MEDIUM in-band
    db = MagicMock()
    db.fetchone = AsyncMock(return_value=None)
    manager = RiskManager(settings, db)
    manager.portfolio = _mock_portfolio()

    market = _make_market(category="other")   # NOT on the allowlist
    # 10% divergence, MEDIUM confidence -> the divergence filter WOULD reject a
    # live forecast here.
    sig = _make_signal(edge=10.0, claude_prob=0.60, market_prob=0.50,
                       confidence=Confidence.MEDIUM)
    sig.strategy_source = "resolution_lens"
    d = await manager.evaluate(sig, market, available_cash=500.0)

    failed = [c.name for c in d.checks if not c.passed]
    assert "category_allowlist" not in [c.name for c in d.checks]  # gate not run
    assert "divergence_band" not in failed                          # bypassed
    assert d.approved is True


@pytest.mark.asyncio
@patch("auramaur.risk.manager.check_kill_switch")
async def test_live_allowlist_passes_allowed_category(mock_kill):
    """An allowlisted stored label trades live even when the fresh keyword
    classification is inconclusive ('other') — eligibility hangs on the
    authoritative stored label, not keyword recognition; #17 stays the
    tripwire for confidently-BAD fresh labels."""
    from auramaur.risk.checks import CheckResult
    mock_kill.return_value = CheckResult(name="kill_switch", passed=True, reason="", value=False)

    settings = _make_settings(is_live=True,
                              allowed_categories_live=["tech", "crypto"])
    db = MagicMock()
    db.fetchone = AsyncMock(return_value=None)
    manager = RiskManager(settings, db)
    manager.portfolio = _mock_portfolio()

    market = _make_market(category="tech")
    market.question = "Will the gadget ship this quarter?"  # classifies 'other'
    sig = _make_signal(edge=10.0, claude_prob=0.60, market_prob=0.50)
    sig.strategy_source = "news_speed"
    d = await manager.evaluate(sig, market, available_cash=500.0)
    gate = next(c for c in d.checks if c.name == "category_allowlist")
    assert gate.passed is True


@pytest.mark.asyncio
@patch("auramaur.risk.manager.check_kill_switch")
async def test_live_allowlist_skipped_in_paper_and_for_exempt(mock_kill):
    """Paper mode keeps blocklist-gated exploration (no allowlist check at
    all); structural two-sided strategies stay exempt in live mode."""
    from auramaur.risk.checks import CheckResult
    mock_kill.return_value = CheckResult(name="kill_switch", passed=True, reason="", value=False)

    db = MagicMock()
    db.fetchone = AsyncMock(return_value=None)

    # Paper: no category_allowlist check appended.
    settings = _make_settings(is_live=False,
                              allowed_categories_live=["tech"])
    manager = RiskManager(settings, db)
    manager.portfolio = _mock_portfolio()
    market = _make_market(category="other")
    sig = _make_signal(edge=10.0, claude_prob=0.60, market_prob=0.50)
    d = await manager.evaluate(sig, market, available_cash=500.0)
    assert not any(c.name == "category_allowlist" for c in d.checks)

    # Live + exempt strategy: check passes without judging the category.
    settings2 = _make_settings(is_live=True,
                               allowed_categories_live=["tech"])
    manager2 = RiskManager(settings2, db)
    manager2.portfolio = _mock_portfolio()
    sig2 = _make_signal(edge=10.0, claude_prob=0.60, market_prob=0.50)
    sig2.strategy_source = "arbitrage"
    d2 = await manager2.evaluate(sig2, market, available_cash=500.0)
    gate = next(c for c in d2.checks if c.name == "category_allowlist")
    assert gate.passed is True
