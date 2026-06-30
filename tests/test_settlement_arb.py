"""Tests for settlement_arb — the FRED-first known-outcome / settlement-lag pillar.

Focus on the deterministic core (indicator_at_period + is_satisfied + the lag
gate), which is what makes this pillar low-variance: the LLM only extracts the
predicate, the RESOLVE step is pure.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from auramaur.strategy.econ_pricing import ECON_SERIES
from auramaur.strategy.settlement_arb import (
    SettlementArbPillar,
    has_econ_trigger,
    indicator_at_period,
    is_satisfied,
)


def _obs(*pairs):
    """(YYYY-MM, value) -> FRED-style (datetime, value), oldest-first."""
    return [(datetime(int(p[:4]), int(p[5:7]), 1), v) for p, v in pairs]


# ---------------------------------------------------------------------------
# is_satisfied — the comparison
# ---------------------------------------------------------------------------

def test_is_satisfied_operators():
    assert is_satisfied(3.0, ">=", 3.0) is True
    assert is_satisfied(3.0, ">", 3.0) is False
    assert is_satisfied(2.9, "<", 3.0) is True
    assert is_satisfied(3.1, "<=", 3.0) is False
    assert is_satisfied(3.0, "==", 3.0) is True
    assert is_satisfied(3.0, "??", 3.0) is False   # unknown operator -> not satisfied


def test_point_bin_uses_grid_tolerance_not_exact_equality():
    """A CPI YoY point-bin '== 3.8' must resolve YES for any continuous indicator
    that ROUNDS into the [3.75, 3.85) bin — exact float equality never matched,
    so every point-bin priced fair=0 and the pillar entered nothing."""
    assert is_satisfied(3.7841, "==", 3.8) is True    # rounds to 3.8 -> in bin
    assert is_satisfied(3.84, "==", 3.8) is True       # still in [3.75, 3.85)
    assert is_satisfied(3.86, "==", 3.8) is False      # rounds to 3.9 -> out
    assert is_satisfied(3.74, "==", 3.8) is False      # rounds to 3.7 -> out
    # precision follows the threshold: an integer bin is one whole unit wide
    assert is_satisfied(3.4, "==", 3.0) is True
    assert is_satisfied(3.6, "==", 3.0) is False


def test_cpi_yoy_resolves_on_non_seasonally_adjusted_series():
    """Kalshi CPI-YoY markets settle on the NSA headline index (CPIAUCNS)."""
    assert ECON_SERIES["KXCPIYOY"].fred_series == "CPIAUCNS"


# ---------------------------------------------------------------------------
# indicator_at_period — the deterministic resolve
# ---------------------------------------------------------------------------

def test_level_indicator_returns_period_value():
    spec = ECON_SERIES["KXU3"]   # UNRATE, level
    obs = _obs(("2026-04", 4.0), ("2026-05", 4.1), ("2026-06", 4.3))
    assert indicator_at_period(obs, spec, "2026-06") == 4.3


def test_level_period_not_published_yet_is_none():
    """The print for the reference period isn't out -> undetermined (skip)."""
    spec = ECON_SERIES["KXU3"]
    obs = _obs(("2026-04", 4.0), ("2026-05", 4.1))   # June not yet released
    assert indicator_at_period(obs, spec, "2026-06") is None


def test_yoy_indicator_computes_against_prior_year():
    spec = ECON_SERIES["KXCPIYOY"]   # CPIAUCNS, yoy
    # June 2025 = 100, June 2026 = 103 -> +3.0% YoY
    obs = _obs(("2025-06", 100.0), ("2026-05", 102.5), ("2026-06", 103.0))
    assert indicator_at_period(obs, spec, "2026-06") == pytest.approx(3.0)


def test_yoy_missing_base_year_is_none():
    spec = ECON_SERIES["KXCPIYOY"]
    obs = _obs(("2026-05", 102.5), ("2026-06", 103.0))   # no June 2025 base
    assert indicator_at_period(obs, spec, "2026-06") is None


def test_mom_change_scaled():
    spec = ECON_SERIES["KXPAYROLLS"]   # PAYEMS, mom_change, scale=1000 (thousands)
    # +0.15 (millions) MoM -> 150 thousand jobs
    obs = _obs(("2026-05", 159.00), ("2026-06", 159.15))
    assert indicator_at_period(obs, spec, "2026-06") == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# has_econ_trigger — candidate routing
# ---------------------------------------------------------------------------

def test_has_econ_trigger():
    assert has_econ_trigger("Will CPI inflation be above 3% in June?")
    assert has_econ_trigger("Unemployment rate under 4.2% for June?")
    assert has_econ_trigger("Nonfarm payrolls above 150k in June?")
    assert not has_econ_trigger("Will Bitcoin hit $200k?")
    assert not has_econ_trigger("Will Trump win in 2028?")


# ---------------------------------------------------------------------------
# Lag gate — only trade the un-converged distance, on the right side
# ---------------------------------------------------------------------------

def _pillar(*, fred_obs, settings_min_edge=0.05):
    settings = SimpleNamespace(settlement_arb=SimpleNamespace(
        enabled=True, paper=True, stake_usd=10.0, min_edge=settings_min_edge,
        min_liquidity=1000.0, min_extract_confidence=0.8, verify_min_confidence=0.8,
        max_entries_per_cycle=5, history_n=60, interval_seconds=1800))
    fred = SimpleNamespace(get_observations=AsyncMock(return_value=fred_obs))
    p = SettlementArbPillar(
        db=MagicMock(), settings=settings, discovery=MagicMock(), exchange=MagicMock(),
        risk_manager=MagicMock(), pnl_tracker=MagicMock(), fred_source=fred, analyzer=None)
    return p, settings


@pytest.mark.asyncio
async def test_locked_yes_underpriced_buys_yes():
    """CPI YoY 3.0% already published, criterion '>=3.0' locks YES, market still
    at 0.80 -> BUY (long YES) at the un-converged distance."""
    obs = _obs(("2025-06", 100.0), ("2026-06", 103.0))
    p, _ = _pillar(fred_obs=obs)
    market = SimpleNamespace(id="m1", outcome_yes_price=0.80, outcome_no_price=0.20)
    pred = {"indicator": "KXCPIYOY", "operator": ">=", "threshold": 3.0,
            "reference_period": "2026-06"}

    from auramaur.exchange.models import OrderSide
    captured = {}

    async def fake_eval(signal, m):
        captured["side"] = signal.recommended_side
        captured["fair"] = signal.claude_prob
        return SimpleNamespace(approved=False, position_size=0, reason="paper-test-stop")
    p._risk.evaluate = fake_eval

    await p._maybe_enter(market, pred)
    assert captured["fair"] == 1.0                       # YES locked
    assert captured["side"] == OrderSide.BUY             # long YES


@pytest.mark.asyncio
async def test_locked_no_buys_no_via_sell_side():
    """Criterion fails (YES locked NO), market still pricing YES at 0.30 ->
    SELL (short YES = long NO)."""
    obs = _obs(("2025-06", 100.0), ("2026-06", 101.0))   # YoY +1.0%, < 3.0 -> NO
    p, _ = _pillar(fred_obs=obs)
    market = SimpleNamespace(id="m2", outcome_yes_price=0.30, outcome_no_price=0.70)
    pred = {"indicator": "KXCPIYOY", "operator": ">=", "threshold": 3.0,
            "reference_period": "2026-06"}

    from auramaur.exchange.models import OrderSide
    captured = {}

    async def fake_eval(signal, m):
        captured["side"] = signal.recommended_side
        captured["fair"] = signal.claude_prob
        return SimpleNamespace(approved=False, position_size=0, reason="stop")
    p._risk.evaluate = fake_eval

    await p._maybe_enter(market, pred)
    assert captured["fair"] == 0.0                       # NO locked
    assert captured["side"] == OrderSide.SELL            # short YES = long NO


@pytest.mark.asyncio
async def test_already_converged_market_is_skipped():
    """YES locked and market already at 0.99 -> no lag, no trade."""
    obs = _obs(("2025-06", 100.0), ("2026-06", 103.0))
    p, _ = _pillar(fred_obs=obs)
    market = SimpleNamespace(id="m3", outcome_yes_price=0.99, outcome_no_price=0.01)
    pred = {"indicator": "KXCPIYOY", "operator": ">=", "threshold": 3.0,
            "reference_period": "2026-06"}
    p._risk.evaluate = AsyncMock()
    entered = await p._maybe_enter(market, pred)
    assert entered is False
    p._risk.evaluate.assert_not_awaited()                # never even evaluated


@pytest.mark.asyncio
async def test_print_not_published_never_trades():
    """The reference-period print isn't out yet -> undetermined, never forecast."""
    obs = _obs(("2026-05", 102.5))   # June not released, no 2025 base either
    p, _ = _pillar(fred_obs=obs)
    market = SimpleNamespace(id="m4", outcome_yes_price=0.50, outcome_no_price=0.50)
    pred = {"indicator": "KXCPIYOY", "operator": ">=", "threshold": 3.0,
            "reference_period": "2026-06"}
    p._risk.evaluate = AsyncMock()
    entered = await p._maybe_enter(market, pred)
    assert entered is False
    p._risk.evaluate.assert_not_awaited()


# ---------------------------------------------------------------------------
# Candidate scan — admit illiquid tail bins (NBER w34702), reject only dust
# ---------------------------------------------------------------------------

def _row(mid, question, yes=0.7, liquidity=150.0):
    return {
        "id": mid, "question": question, "description": "",
        "outcome_yes_price": yes, "outcome_no_price": round(1 - yes, 2),
        "liquidity": liquidity, "category": "economics",
        "clob_token_yes": "ty", "clob_token_no": "tn", "end_date": None,
    }


@pytest.mark.asyncio
async def test_candidates_admit_tail_bins_reject_dust_and_noneon():
    """The low liquidity floor admits an illiquid tail-bin econ market (where the
    settlement lag lives) but still rejects untradeable dust below the floor and
    non-econ markets."""
    rows = [
        _row("tail", "CPI YoY above 3.0% in June?", liquidity=150.0),  # admit
        _row("dust", "Unemployment above 4% in June?", liquidity=20.0),  # < floor
        _row("noneon", "Will Bitcoin hit $200k?", liquidity=5000.0),     # no trigger
        _row("converged", "Payrolls above 150k in June?", yes=1.0),      # price out of (0,1)
    ]
    settings = SimpleNamespace(settlement_arb=SimpleNamespace(min_liquidity=100.0))
    db = MagicMock()
    db.fetchall = AsyncMock(return_value=rows)
    p = SettlementArbPillar(
        db=db, settings=settings, discovery=MagicMock(), exchange=MagicMock(),
        risk_manager=MagicMock(), pnl_tracker=MagicMock(), fred_source=MagicMock(),
        analyzer=None)

    cands = await p._candidates()
    ids = {m.id for m in cands}
    assert ids == {"tail"}


# ---------------------------------------------------------------------------
# Kalshi monthly macro-bin path — deterministic predicate, per-series scan,
# resolve on the published (rounded) figure
# ---------------------------------------------------------------------------

def _kx(ticker, yes=0.7, strike_type="greater", floor=4.5):
    from auramaur.exchange.models import Market
    return Market(id=ticker, exchange="kalshi", ticker=ticker, question="",
                  outcome_yes_price=yes, outcome_no_price=round(1 - yes, 2),
                  strike_type=strike_type, floor_strike=floor)


@pytest.mark.asyncio
async def test_kalshi_candidates_scanned_per_series_priced_only():
    """The Kalshi branch pulls each registered econ series and keeps only bins
    with a real two-sided price (a no-bid far-dated bin has no lag to trade)."""
    kdisc = SimpleNamespace(get_markets_by_series=AsyncMock(side_effect=lambda s, limit=200: [
        _kx(f"{s}-26NOV-T4.5", yes=0.6),     # priced -> keep
        _kx(f"{s}-26NOV-T9.9", yes=0.0),     # no price -> drop
    ]))
    settings = SimpleNamespace(settlement_arb=SimpleNamespace(min_liquidity=100.0))
    db = MagicMock(); db.fetchall = AsyncMock(return_value=[])  # no Poly candidates
    p = SettlementArbPillar(
        db=db, settings=settings, discovery=MagicMock(), exchange=MagicMock(),
        risk_manager=MagicMock(), pnl_tracker=MagicMock(), fred_source=MagicMock(),
        analyzer=None, kalshi_discovery=kdisc, kalshi_exchange=MagicMock())
    cands = await p._candidates()
    assert {m.id for m in cands} == {
        "KXCPIYOY-26NOV-T4.5", "KXU3-26NOV-T4.5", "KXPAYROLLS-26NOV-T4.5"}
    assert kdisc.get_markets_by_series.await_count == 3   # one call per series


@pytest.mark.asyncio
async def test_run_once_fetches_each_fred_series_once_per_cycle():
    """A fan-out of same-series Kalshi bins must collapse to ONE FRED call per
    series per cycle — not one per candidate (which bursts past FRED's rate
    limit, the fred_observations_failed spike)."""
    bins = [_kx(f"KXCPIYOY-26NOV-T{t}", yes=0.5) for t in (4.3, 4.4, 4.5, 4.6, 4.7)]
    kdisc = SimpleNamespace(get_markets_by_series=AsyncMock(
        side_effect=lambda s, limit=200: bins if s == "KXCPIYOY" else []))
    settings = SimpleNamespace(settlement_arb=SimpleNamespace(
        enabled=True, paper=True, stake_usd=10.0, min_edge=0.05, min_liquidity=100.0,
        min_extract_confidence=0.8, verify_min_confidence=0.8, max_entries_per_cycle=5,
        history_n=60, interval_seconds=1800))
    fred = SimpleNamespace(get_observations=AsyncMock(return_value=[]))  # print not out
    db = MagicMock()
    db.fetchall = AsyncMock(return_value=[])
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    p = SettlementArbPillar(
        db=db, settings=settings, discovery=MagicMock(), exchange=MagicMock(),
        risk_manager=MagicMock(), pnl_tracker=MagicMock(), fred_source=fred,
        analyzer=None, kalshi_discovery=kdisc, kalshi_exchange=MagicMock())
    await p.run_once()
    # 5 same-series bins -> ONE get_observations call (memoized), not 5
    assert fred.get_observations.await_count == 1


@pytest.mark.asyncio
async def test_kalshi_predicate_is_deterministic_no_llm():
    """A Kalshi macro bin's predicate comes from strike fields — analyzer/db
    never touched (analyzer=None would crash the LLM path)."""
    p, _ = _pillar(fred_obs=[])
    pred = await p._predicate(_kx("KXCPIYOY-26NOV-T4.5", strike_type="greater", floor=4.5))
    assert pred == {"indicator": "KXCPIYOY", "operator": ">",
                    "threshold": 4.5, "reference_period": "2026-11"}


@pytest.mark.asyncio
async def test_resolve_uses_published_rounded_figure():
    """CPI YoY computes to 4.449% continuously, but BLS reports 4.4% — a bin
    'above 4.4' must resolve NO (4.4 is not > 4.4), not YES off the raw value."""
    obs = _obs(("2025-06", 100.0), ("2026-06", 104.449))   # raw YoY = +4.449%
    p, _ = _pillar(fred_obs=obs)
    market = SimpleNamespace(id="KXCPIYOY-26JUN-T4.4", exchange="kalshi",
                             outcome_yes_price=0.80, outcome_no_price=0.20)
    pred = {"indicator": "KXCPIYOY", "operator": ">", "threshold": 4.4,
            "reference_period": "2026-06"}
    captured = {}

    async def fake_eval(signal, m):
        captured["fair"] = signal.claude_prob
        return SimpleNamespace(approved=False, position_size=0, reason="stop")
    p._risk.evaluate = fake_eval

    await p._maybe_enter(market, pred)
    assert captured["fair"] == 0.0    # rounded 4.4 is NOT > 4.4 -> NO locked
