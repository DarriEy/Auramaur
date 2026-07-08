"""Term-structure pillar: deadline parsing, family grouping, curve parsing
with isotonic clamping, cache amortization, and the standard-rails entry path
(risk gate, market claim, per-family entry cap)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from auramaur.db.database import Database
from auramaur.exchange.models import Market
from auramaur.strategy.term_structure import (
    TermStructurePillar,
    family_key,
    parse_curve,
    parse_deadline,
)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def test_parse_deadline_variants():
    assert parse_deadline("US x Iran diplomatic meeting by July 10, 2026?") == \
        datetime(2026, 7, 10, tzinfo=timezone.utc)
    assert parse_deadline("GPT-5.6 released by August 5?").month == 8
    assert parse_deadline("Will X happen by July 32, 2026?") is None  # bad day
    assert parse_deadline("Will X happen this year?") is None
    assert parse_deadline("") is None


def test_family_key_strips_deadline_and_normalizes():
    a = family_key("US x Iran diplomatic meeting by July 10, 2026?")
    b = family_key("US x Iran diplomatic meeting by July 17, 2026?")
    assert a == b == "us x iran diplomatic meeting"
    assert family_key("No deadline here?") is None


def _strike(mid: str, day: int, yes: float) -> Market:
    return Market(id=mid, question=f"Event by July {day}, 2026?",
                  outcome_yes_price=yes, outcome_no_price=round(1 - yes, 2),
                  liquidity=5000.0, volume=10000.0, active=True,
                  exchange="polymarket")


def test_parse_curve_isotonic_clamp():
    """A noisy read with P(by T1) > P(by T2) is clamped to non-decreasing."""
    strikes = [_strike("a", 5, 0.2), _strike("b", 15, 0.4), _strike("c", 25, 0.6)]
    raw = '{"thesis": "t", "curve": [{"market_id": "a", "prob": 0.5},' \
          '{"market_id": "b", "prob": 0.3}, {"market_id": "c", "prob": 0.9}]}'
    thesis, probs = parse_curve(raw, strikes)
    assert thesis == "t"
    assert probs["a"] == pytest.approx(0.5)
    assert probs["b"] == pytest.approx(0.5)  # clamped up to running max
    assert probs["c"] == pytest.approx(0.9)


def test_parse_curve_garbage_is_empty():
    strikes = [_strike("a", 5, 0.2)]
    assert parse_curve("cannot help", strikes) == ("", {})
    assert parse_curve('{"curve": "nope"}', strikes) == ("", {})


# ---------------------------------------------------------------------------
# Pillar wiring
# ---------------------------------------------------------------------------


def _settings():
    s = MagicMock()
    cfg = s.term_structure
    cfg.enabled = True
    cfg.paper = True
    cfg.model = "claude-opus-4-8"
    cfg.effort = "medium"
    cfg.scan_limit = 100
    cfg.min_strikes = 3
    cfg.max_families = 12
    cfg.families_per_cycle = 3
    cfg.curve_ttl_hours = 24.0
    cfg.max_entries_per_family = 2
    cfg.stake_usd = 10.0
    cfg.min_liquidity = 1000.0
    cfg.min_days = 0.25
    cfg.max_days = 90.0
    cfg.min_edge_pts = 8.0
    cfg.llm_timeout_seconds = 420
    cfg.exclude_categories = []
    s.risk.blocked_categories = []
    s.nlp.daily_claude_call_budget = 0
    return s


async def _pillar(tmp_path, markets, llm_reply: str):
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    discovery = MagicMock()
    discovery.get_markets = AsyncMock(return_value=markets)
    discovery.search_markets = AsyncMock(return_value=[])
    risk = MagicMock()
    decision = MagicMock()
    decision.approved = True
    decision.position_size = 8.0
    decision.reason = ""
    decision.force_paper = False
    risk.evaluate = AsyncMock(return_value=decision)
    calibration = MagicMock()
    calibration.record_prediction = AsyncMock()
    pillar = TermStructurePillar(
        db=db, settings=_settings(), discovery=discovery, exchange=MagicMock(),
        risk_manager=risk, pnl_tracker=MagicMock(), calibration=calibration)

    result = MagicMock()
    result.status = "paper"
    result.reason = ""
    order = MagicMock()
    order.token.value = "YES"
    order.token_id = "tok"
    order.price = 0.30
    order.size = 33.3
    fill = MagicMock()
    fill.is_paper = True
    fill.filled_size = 33.3
    fill.filled_price = 0.30

    def _submit_side_effect(intent):
        order.market_id = intent.market.id
        result.order = order
        result.result = fill
        return result

    pillar._gateway = MagicMock()
    pillar._gateway.submit = AsyncMock(side_effect=_submit_side_effect)
    pillar._call_model = AsyncMock(return_value=llm_reply)
    return pillar, db, risk


def _ladder():
    return [_strike("a", 5, 0.10), _strike("b", 15, 0.30), _strike("c", 25, 0.50)]


@pytest.mark.asyncio
async def test_one_read_trades_multiple_strikes(tmp_path):
    """One curve read produces entries across the family — capped by
    max_entries_per_family, largest gaps first."""
    reply = ('{"thesis": "timeline says sooner", "curve": ['
             '{"market_id": "a", "prob": 0.40},'
             '{"market_id": "b", "prob": 0.70},'
             '{"market_id": "c", "prob": 0.72}]}')
    pillar, db, risk = await _pillar(tmp_path, _ladder(), reply)
    try:
        entered = await pillar.run_once()
        assert entered == 2                        # cap, not 3
        assert pillar._call_model.await_count == 1  # ONE read for the family
        sigs = await db.fetchall(
            "SELECT market_id, edge FROM signals ORDER BY edge DESC")
        assert {r["market_id"] for r in sigs} == {"b", "a"}  # gaps 40, 30 (c=22)
        assert risk.evaluate.await_count == 2      # full gate per entry
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_cached_curve_spends_no_call(tmp_path):
    reply = ('{"thesis": "t", "curve": [{"market_id": "a", "prob": 0.40},'
             '{"market_id": "b", "prob": 0.70}, {"market_id": "c", "prob": 0.72}]}')
    pillar, db, _ = await _pillar(tmp_path, _ladder(), reply)
    try:
        await pillar.run_once()
        assert pillar._call_model.await_count == 1
        await pillar.run_once()                    # curve cached, markets claimed
        assert pillar._call_model.await_count == 1  # no second read
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_small_gaps_enter_nothing(tmp_path):
    reply = ('{"thesis": "t", "curve": [{"market_id": "a", "prob": 0.12},'
             '{"market_id": "b", "prob": 0.33}, {"market_id": "c", "prob": 0.54}]}')
    pillar, db, _ = await _pillar(tmp_path, _ladder(), reply)
    try:
        assert await pillar.run_once() == 0        # all gaps < 8 pts
        pillar._gateway.submit.assert_not_awaited()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_claimed_market_skipped(tmp_path):
    reply = ('{"thesis": "t", "curve": [{"market_id": "a", "prob": 0.40},'
             '{"market_id": "b", "prob": 0.70}, {"market_id": "c", "prob": 0.72}]}')
    pillar, db, _ = await _pillar(tmp_path, _ladder(), reply)
    try:
        await db.execute(
            """INSERT INTO trades (market_id, timestamp, side, size, price,
               is_paper, order_id, status, exchange, strategy_source)
               VALUES ('b', datetime('now'), 'BUY', 10, 0.3, 1, 'x', 'paper',
                       'polymarket', 'llm')""")
        await db.commit()
        entered = await pillar.run_once()
        # b is claimed -> the two entries come from the remaining strikes.
        sigs = {r["market_id"] for r in await db.fetchall(
            "SELECT market_id FROM signals")}
        assert "b" not in sigs
        assert entered == 2 and sigs == {"a", "c"}
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_families_below_min_strikes_ignored(tmp_path):
    pillar, db, _ = await _pillar(
        tmp_path, [_strike("a", 5, 0.10), _strike("b", 15, 0.30)], "")
    try:
        assert await pillar.run_once() == 0
        pillar._call_model.assert_not_awaited()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_seed_and_search_completes_a_family(tmp_path):
    """A volume-ranked scan yields lone ladder members (deep strikes are
    low-volume); a single seed must trigger a live sibling search that
    completes the family."""
    reply = ('{"thesis": "t", "curve": [{"market_id": "a", "prob": 0.40},'
             '{"market_id": "b", "prob": 0.70}, {"market_id": "c", "prob": 0.72}]}')
    seed_only = [_strike("a", 5, 0.10)]  # scan sees ONE strike
    pillar, db, _ = await _pillar(tmp_path, seed_only, reply)
    pillar._settings.term_structure.min_strikes = 3
    full = _ladder()
    pillar._discovery.search_markets = AsyncMock(return_value=full + [
        # noise the merge must reject: wrong family / no deadline
        Market(id="x", question="Unrelated by July 9, 2026?",
               outcome_yes_price=0.5, outcome_no_price=0.5, liquidity=5000,
               volume=100, active=True, exchange="polymarket"),
        Market(id="y", question="Event happening soon?", outcome_yes_price=0.5,
               outcome_no_price=0.5, liquidity=5000, volume=100, active=True,
               exchange="polymarket"),
    ])
    try:
        entered = await pillar.run_once()
        assert entered == 2
        pillar._discovery.search_markets.assert_awaited_once()
        sigs = {r["market_id"] for r in await db.fetchall(
            "SELECT market_id FROM signals")}
        assert sigs == {"a", "b"}
    finally:
        await db.close()
