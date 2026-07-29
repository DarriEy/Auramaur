"""Term-structure pillar: deadline parsing, family grouping, curve parsing
with isotonic clamping, cache amortization, and the standard-rails entry path
(risk gate, market claim, per-family entry cap)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from auramaur.db.database import Database
from auramaur.exchange.models import Confidence, Market
from auramaur.strategy.term_structure import (
    TermStructurePillar,
    family_key,
    monotonicity_violations,
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


def test_parse_deadline_extended_ladder_variants():
    assert parse_deadline("Event by end of 2026?") == datetime(
        2026, 12, 31, tzinfo=timezone.utc)
    assert parse_deadline("Event by Q3 2026?") == datetime(
        2026, 9, 30, tzinfo=timezone.utc)
    assert parse_deadline("Event before September 2026?") == datetime(
        2026, 9, 1, tzinfo=timezone.utc)
    assert parse_deadline("Event by February 2028?") == datetime(
        2028, 2, 29, tzinfo=timezone.utc)
    assert family_key("Event by Q3 2026?") == "event"
    assert family_key("Event before September 2026?") == "event"


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
    cfg.context_min_liquidity = 100.0
    cfg.min_days = 0.25
    cfg.max_days = 90.0
    cfg.min_edge_pts = 8.0
    cfg.high_conf_min_strikes = 4
    cfg.llm_timeout_seconds = 420
    cfg.gemini_fallback = True
    cfg.gemini_daily_call_limit = 30
    cfg.gemini_price_per_mtok = [2.0, 12.0]
    cfg.openai_fallback = True
    cfg.openai_model = "gpt-5.6-sol"
    cfg.openai_effort = "high"
    cfg.openai_primary_on_claude_block = True
    cfg.openai_grounded = True
    cfg.openai_daily_call_limit = 16
    cfg.openai_max_output_tokens = 8000
    cfg.openai_price_per_mtok = [5.0, 30.0]
    cfg.exclude_categories = []
    s.openai_api_key = "test-openai-key"
    s.risk.blocked_categories = []
    s.nlp.daily_claude_call_budget = 0
    s.gemini.enabled = True
    s.gemini.model = "gemini-test"
    s.gemini_api_key = "test-key"
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
        # A market is claimed while the position is still HELD. The trade row
        # alone is append-only and used to block the market forever.
        await db.execute(
            """INSERT INTO cost_basis (market_id, token, token_id, size,
               avg_cost, total_cost, realized_pnl, is_paper, updated_at)
               VALUES ('b', 'YES', 'x', 10, 0.3, 3.0, 0, 1, datetime('now'))""")
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


@pytest.mark.asyncio
async def test_thin_strike_informs_curve_but_is_not_traded(tmp_path):
    strikes = _ladder()
    strikes[1].liquidity = 200.0  # context floor passes; execution floor fails
    reply = ('{"thesis": "t", "curve": [{"market_id": "a", "prob": 0.40},'
             '{"market_id": "b", "prob": 0.90}, '
             '{"market_id": "c", "prob": 0.72}]}')
    pillar, db, _ = await _pillar(tmp_path, strikes, reply)
    try:
        assert await pillar.run_once() == 2
        traded = {r["market_id"] for r in await db.fetchall(
            "SELECT market_id FROM signals")}
        assert traded == {"a", "c"}
        row = await db.fetchone(
            """SELECT disposition FROM term_structure_observations
                WHERE market_id='b' ORDER BY id DESC LIMIT 1""")
        assert row["disposition"] == "context_only"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_changed_strike_set_invalidates_cached_curve(tmp_path):
    reply = ('{"thesis": "t", "curve": [{"market_id": "a", "prob": 0.40},'
             '{"market_id": "b", "prob": 0.70}, '
             '{"market_id": "c", "prob": 0.72}]}')
    pillar, db, _ = await _pillar(tmp_path, _ladder(), reply)
    try:
        await pillar._ensure_schema()
        await pillar._read_family("event", _ladder(), pillar._settings.term_structure)
        expanded = _ladder() + [_strike("d", 28, 0.60)]
        cached = await pillar._cached_curve(
            "event", expanded, pillar._settings.term_structure)
        assert cached is None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_grounded_gemini_fallback_records_cost(tmp_path):
    pillar, db, _ = await _pillar(tmp_path, _ladder(), "")
    pillar._gemini_request = AsyncMock(return_value={
        "candidates": [{"content": {"parts": [{"text": '{"curve": []}'}]}}],
        "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 500},
    })
    try:
        text = await pillar._call_gemini(
            "prompt", pillar._settings.term_structure, "test")
        assert text == '{"curve": []}'
        assert pillar._last_reader == ("gemini", "gemini-test")
        row = await db.fetchone(
            """SELECT calls, usd FROM agent_trader_costs
                WHERE model_alias='term_structure_gemini'""")
        assert row["calls"] == 1
        assert row["usd"] == pytest.approx(0.008)
        body = pillar._gemini_request.await_args.args[1]
        assert body["tools"] == [{"google_search": {}}]
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Confidence escalation — gates the live-only adverse-divergence band
# ---------------------------------------------------------------------------


def test_monotonicity_violations_tolerates_spread_noise():
    """A 2pt dip is quote noise; a 10pt one breaks P(by T1) <= P(by T2)."""
    assert monotonicity_violations(
        [_strike("a", 5, 0.30), _strike("b", 15, 0.28)]) == []
    bad = monotonicity_violations(
        [_strike("a", 5, 0.40), _strike("b", 15, 0.30)])
    assert [m.id for m, _ in bad] == ["b"]


def test_monotonicity_violations_compares_against_running_max():
    """One bad strike must not re-baseline the ladder and mask the next."""
    bad = monotonicity_violations(
        [_strike("a", 5, 0.50), _strike("b", 15, 0.10), _strike("c", 25, 0.20)])
    assert [m.id for m, _ in bad] == ["b", "c"]
    assert [prev for _, prev in bad] == [0.50, 0.50]


def _reply(ids_probs):
    entries = ", ".join(
        '{"market_id": "%s", "prob": %s}' % (i, p) for i, p in ids_probs)
    return '{"thesis": "t", "curve": [%s]}' % entries


@pytest.mark.asyncio
async def test_wide_monotone_ladder_earns_high_confidence(tmp_path):
    """4+ strikes, monotone: the curve is well-formed, so it says HIGH.

    Guards the fix for the 2026-07-24 promotion silence — a hardcoded MEDIUM
    put every entry inside the live-only adverse-divergence band [5%,20%),
    which requires HIGH, so 100% of live candidates were rejected.
    """
    ladder = [_strike("a", 5, 0.10), _strike("b", 15, 0.20),
              _strike("c", 25, 0.30), _strike("d", 30, 0.40)]
    pillar, db, risk = await _pillar(tmp_path, ladder, _reply(
        [("a", 0.30), ("b", 0.40), ("c", 0.50), ("d", 0.60)]))
    try:
        assert await pillar.run_once() > 0
        confidences = {
            call.args[0].claude_confidence for call in risk.evaluate.await_args_list}
        assert confidences == {Confidence.HIGH}
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_thin_ladder_stays_medium(tmp_path):
    """A 3-strike family is the bare minimum curve — it has not earned HIGH."""
    pillar, db, risk = await _pillar(tmp_path, _ladder(), _reply(
        [("a", 0.30), ("b", 0.50), ("c", 0.70)]))
    try:
        assert await pillar.run_once() > 0
        confidences = {
            call.args[0].claude_confidence for call in risk.evaluate.await_args_list}
        assert confidences == {Confidence.MEDIUM}
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_self_contradicting_ladder_stays_medium(tmp_path):
    """Width alone is not enough: a ladder that breaks its own ordering is
    withheld from HIGH even at 4 strikes."""
    ladder = [_strike("a", 5, 0.60), _strike("b", 15, 0.20),
              _strike("c", 25, 0.30), _strike("d", 30, 0.40)]
    pillar, db, risk = await _pillar(tmp_path, ladder, _reply(
        [("a", 0.80), ("b", 0.85), ("c", 0.90), ("d", 0.95)]))
    try:
        assert await pillar.run_once() > 0
        confidences = {
            call.args[0].claude_confidence for call in risk.evaluate.await_args_list}
        assert confidences == {Confidence.MEDIUM}
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Reader fallback chain: Claude -> Gemini -> OpenAI
# ---------------------------------------------------------------------------


def _openai_reply(text: str, in_tok: int = 1000, out_tok: int = 200) -> dict:
    return {
        "output": [{"type": "message",
                    "content": [{"type": "output_text", "text": text}]}],
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }


@pytest.mark.asyncio
async def test_openai_takes_over_when_gemini_cap_is_spent(tmp_path):
    """The 2026-07-28 outage: Claude weekly-limited AND the shared Gemini cap
    exhausted meant zero curve reads. OpenAI must carry the read instead."""
    pillar, db, _ = await _pillar(tmp_path, _ladder(), "")
    cfg = pillar._settings.term_structure
    pillar._call_gemini = AsyncMock(
        side_effect=RuntimeError("shared Gemini daily call limit exhausted"))
    pillar._openai_request = AsyncMock(
        return_value=_openai_reply('{"thesis": "t", "curve": []}'))
    try:
        text = await pillar._fallback("prompt", cfg, "claude_call_failed")
        assert text == '{"thesis": "t", "curve": []}'
        assert pillar._last_reader == ("openai", "gpt-5.6-sol")
        row = await db.fetchone(
            """SELECT calls, usd FROM agent_trader_costs
                WHERE model_alias='term_structure_openai'""")
        assert row["calls"] == 1
        assert row["usd"] == pytest.approx(1000 * 5.0 / 1e6 + 200 * 30.0 / 1e6)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_gemini_still_preferred_over_openai(tmp_path):
    """Cost order holds: OpenAI is not called while Gemini can still answer."""
    pillar, db, _ = await _pillar(tmp_path, _ladder(), "")
    pillar._call_gemini = AsyncMock(return_value="gemini text")
    pillar._openai_request = AsyncMock()
    try:
        text = await pillar._fallback(
            "prompt", pillar._settings.term_structure, "claude_daily_budget")
        assert text == "gemini text"
        pillar._openai_request.assert_not_awaited()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_all_arms_spent_reports_every_failure(tmp_path):
    """A family errors out only when every arm is spent, and the error names
    each one — silence about which reader died is what cost a day of reads."""
    pillar, db, _ = await _pillar(tmp_path, _ladder(), "")
    pillar._call_gemini = AsyncMock(side_effect=RuntimeError("gemini cap"))
    pillar._openai_request = AsyncMock(return_value={"error": "insufficient_quota"})
    try:
        with pytest.raises(RuntimeError) as excinfo:
            await pillar._fallback(
                "prompt", pillar._settings.term_structure, "claude_call_failed")
        detail = str(excinfo.value)
        assert "gemini cap" in detail and "insufficient_quota" in detail
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_openai_respects_its_own_daily_cap(tmp_path):
    """The cap is scoped to this alias — a drained shared pool must not also
    disable the arm that exists to relieve it."""
    pillar, db, _ = await _pillar(tmp_path, _ladder(), "")
    cfg = pillar._settings.term_structure
    cfg.openai_daily_call_limit = 1
    await pillar._ensure_schema()
    # A different arm burning the shared day total must not block this one.
    await db.execute(
        """INSERT INTO agent_trader_costs (day, model_alias, calls, usd)
           VALUES (date('now'), 'some_other_arm', 99, 1.0)""")
    await db.commit()
    pillar._openai_request = AsyncMock(
        return_value=_openai_reply('{"thesis": "t", "curve": []}'))
    try:
        assert await pillar._call_openai("p", cfg, "r")
        with pytest.raises(RuntimeError, match="daily call limit exhausted"):
            await pillar._call_openai("p", cfg, "r")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_openai_grounding_toggle(tmp_path):
    """Grounded by default (parity with the other two arms), off on request."""
    pillar, db, _ = await _pillar(tmp_path, _ladder(), "")
    cfg = pillar._settings.term_structure
    pillar._openai_request = AsyncMock(
        return_value=_openai_reply('{"thesis": "t", "curve": []}'))
    try:
        await pillar._call_openai("p", cfg, "r")
        assert pillar._openai_request.await_args.args[0]["tools"] == [
            {"type": "web_search"}]
        cfg.openai_grounded = False
        await pillar._call_openai("p", cfg, "r")
        assert "tools" not in pillar._openai_request.await_args.args[0]
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Executable-price edge, and OpenAI leading during a Claude outage
# ---------------------------------------------------------------------------


def _wide(mid: float, spread: float, mid_day: int = 15) -> Market:
    m = _strike("w", mid_day, mid)
    m.spread = spread
    return m


@pytest.mark.asyncio
async def test_spread_edge_is_not_traded(tmp_path):
    """A 15pt mid gap on a 16pt spread is ~7pt reachable — below the floor.

    Market 2245064 on 2026-07-29: model 0.68 vs mid 0.525 looked like 15.5pts
    while bid/ask were 0.46/0.62. Paying the ask leaves ~6pts. It must not
    become a candidate.
    """
    wide = _wide(0.525, 0.16)
    ladder = [_strike("a", 5, 0.10), wide, _strike("c", 25, 0.70)]
    reply = ('{"thesis": "t", "curve": [{"market_id": "a", "prob": 0.11},'
             '{"market_id": "w", "prob": 0.68},'
             '{"market_id": "c", "prob": 0.70}]}')
    pillar, db, risk = await _pillar(tmp_path, ladder, reply)
    try:
        assert await pillar.run_once() == 0
        risk.evaluate.assert_not_awaited()
        row = await db.fetchone(
            "SELECT disposition, gap_pts FROM term_structure_observations "
            "WHERE market_id='w'")
        assert row["disposition"] == "below_edge"
        assert row["gap_pts"] == pytest.approx(7.5)   # 68 - 60.5, not 15.5
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_tight_book_keeps_its_edge(tmp_path):
    """The same gap on a 1pt spread still trades — the haircut is the spread,
    not a blanket penalty."""
    tight = _wide(0.525, 0.01)
    ladder = [_strike("a", 5, 0.10), tight, _strike("c", 25, 0.70)]
    reply = ('{"thesis": "t", "curve": [{"market_id": "a", "prob": 0.11},'
             '{"market_id": "w", "prob": 0.68},'
             '{"market_id": "c", "prob": 0.70}]}')
    pillar, db, risk = await _pillar(tmp_path, ladder, reply)
    try:
        assert await pillar.run_once() == 1
        row = await db.fetchone(
            "SELECT disposition, gap_pts FROM term_structure_observations "
            "WHERE market_id='w'")
        assert row["disposition"] == "candidate"
        assert row["gap_pts"] == pytest.approx(15.0)  # 68 - 53
        # The risk gate must see the reachable edge, not the mid figure.
        signal = risk.evaluate.await_args.args[0]
        assert signal.edge == pytest.approx(15.0)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_model_inside_the_spread_never_flips_direction(tmp_path):
    """Fair value between bid and ask is NO trade. An abs() edge would turn it
    into a sell at a price also worse than fair."""
    wide = _wide(0.50, 0.20)          # bid 0.40 / ask 0.60
    ladder = [_strike("a", 5, 0.10), wide, _strike("c", 25, 0.70)]
    reply = ('{"thesis": "t", "curve": [{"market_id": "a", "prob": 0.11},'
             '{"market_id": "w", "prob": 0.55},'
             '{"market_id": "c", "prob": 0.70}]}')
    pillar, db, risk = await _pillar(tmp_path, ladder, reply)
    try:
        assert await pillar.run_once() == 0
        risk.evaluate.assert_not_awaited()
        row = await db.fetchone(
            "SELECT disposition, gap_pts FROM term_structure_observations "
            "WHERE market_id='w'")
        assert row["disposition"] == "below_edge"
        assert row["gap_pts"] < 0     # signed, so it cannot resurface
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_openai_leads_while_claude_is_weekly_blocked(tmp_path):
    """A weekly limit is a multi-day outage; Gemini's shared cap cannot cover
    it, so the arm with its own budget goes first."""
    from datetime import timedelta

    pillar, db, _ = await _pillar(tmp_path, _ladder(), "")
    cfg = pillar._settings.term_structure
    pillar._call_gemini = AsyncMock(return_value="gemini text")
    pillar._openai_request = AsyncMock(
        return_value=_openai_reply('{"thesis": "t", "curve": []}'))
    pillar._claude_blocked_until = (
        datetime.now(timezone.utc) + timedelta(hours=6))
    try:
        text = await pillar._fallback("p", cfg, "claude_quota_circuit")
        assert text == '{"thesis": "t", "curve": []}'
        pillar._call_gemini.assert_not_awaited()
        assert pillar._last_reader == ("openai", "gpt-5.6-sol")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_gemini_leads_again_once_the_block_expires(tmp_path):
    """Cost order returns the moment the circuit closes."""
    from datetime import timedelta

    pillar, db, _ = await _pillar(tmp_path, _ladder(), "")
    pillar._call_gemini = AsyncMock(return_value="gemini text")
    pillar._openai_request = AsyncMock()
    pillar._claude_blocked_until = (
        datetime.now(timezone.utc) - timedelta(minutes=1))   # expired
    try:
        text = await pillar._fallback(
            "p", pillar._settings.term_structure, "claude_call_failed")
        assert text == "gemini text"
        pillar._openai_request.assert_not_awaited()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_openai_lead_still_falls_back_to_gemini(tmp_path):
    """Leading is not exclusive — if OpenAI is also spent, Gemini still runs."""
    from datetime import timedelta

    pillar, db, _ = await _pillar(tmp_path, _ladder(), "")
    pillar._call_gemini = AsyncMock(return_value="gemini text")
    pillar._openai_request = AsyncMock(
        return_value={"error": {"message": "insufficient_quota"}})
    pillar._claude_blocked_until = (
        datetime.now(timezone.utc) + timedelta(hours=6))
    try:
        assert await pillar._fallback(
            "p", pillar._settings.term_structure, "claude_quota_circuit"
        ) == "gemini text"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_truncated_read_is_still_billed_and_capped(tmp_path):
    """Reasoning tokens are billed even when the reply comes back incomplete.

    Accounting only successes made a failing arm burn budget invisibly — the
    cap never saw it. Observed live 2026-07-29 01:51 with max_output_tokens
    2048 against sol at effort=high.
    """
    pillar, db, _ = await _pillar(tmp_path, _ladder(), "")
    cfg = pillar._settings.term_structure
    pillar._openai_request = AsyncMock(return_value={
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [],
        "usage": {"input_tokens": 40000, "output_tokens": 2048},
    })
    try:
        with pytest.raises(RuntimeError, match="truncated"):
            await pillar._call_openai("p", cfg, "r")
        row = await db.fetchone(
            """SELECT calls, usd FROM agent_trader_costs
                WHERE model_alias='term_structure_openai'""")
        assert row["calls"] == 1          # counted against the cap
        assert row["usd"] == pytest.approx(40000 * 5.0 / 1e6 + 2048 * 30.0 / 1e6)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_max_output_tokens_is_configurable(tmp_path):
    pillar, db, _ = await _pillar(tmp_path, _ladder(), "")
    cfg = pillar._settings.term_structure
    pillar._openai_request = AsyncMock(
        return_value=_openai_reply('{"thesis": "t", "curve": []}'))
    try:
        await pillar._call_openai("p", cfg, "r")
        assert pillar._openai_request.await_args.args[0][
            "max_output_tokens"] == 8000
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_thin_ladder_trades_paper_instead_of_being_refused(tmp_path):
    """A MEDIUM ladder must reach the book as a PAPER fill, not a rejection.

    The adverse-divergence band rejects; it does not demote. Judged as live, a
    3-strike family was refused outright (observed 2026-07-29 01:52, market
    2734400) and its graduation cell accumulated nothing — the original dead
    end, narrowed to thin ladders. The pillar now declares paper intent before
    the gate runs, which is also what skips the live-only band.
    """
    pillar, db, risk = await _pillar(tmp_path, _ladder(), _reply(
        [("a", 0.30), ("b", 0.50), ("c", 0.70)]))
    try:
        assert await pillar.run_once() > 0
        # Declared paper to the gate...
        assert risk.evaluate.await_args.kwargs["force_paper"] is True
        # ...and submitted paper.
        intent = pillar._gateway.submit.await_args.args[0]
        assert intent.force_paper is True
        assert intent.signal.claude_confidence is Confidence.MEDIUM
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_strong_ladder_is_not_paper_forced(tmp_path):
    """The escalation must still buy something: a HIGH curve goes to the gate
    as live, or the whole confidence mechanism is decorative."""
    ladder = [_strike("a", 5, 0.10), _strike("b", 15, 0.20),
              _strike("c", 25, 0.30), _strike("d", 30, 0.40)]
    pillar, db, risk = await _pillar(tmp_path, ladder, _reply(
        [("a", 0.30), ("b", 0.40), ("c", 0.50), ("d", 0.60)]))
    pillar._settings.term_structure.paper = False
    try:
        assert await pillar.run_once() > 0
        assert risk.evaluate.await_args.kwargs["force_paper"] is False
        intent = pillar._gateway.submit.await_args.args[0]
        assert intent.force_paper is False
        assert intent.signal.claude_confidence is Confidence.HIGH
    finally:
        await db.close()
