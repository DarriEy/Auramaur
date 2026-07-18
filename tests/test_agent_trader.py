"""Agent day-trader pillar: the intelligence-cap A/B on the bot's rails.

Covers the tested core: tolerant decision parsing, per-model attribution
cells, the full risk gate (never bypassed), hallucinated-market rejection,
edge-floor enforcement, thesis memory persistence, and budget/LLM failure
isolation between model arms.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from auramaur.db.database import Database
from auramaur.exchange.models import Market, OrderSide
from auramaur.strategy.agent_trader import AgentTraderPillar, parse_decisions


# ---------------------------------------------------------------------------
# parse_decisions — a bad LLM reply must never crash a cycle
# ---------------------------------------------------------------------------


def test_parse_accepts_fenced_and_prefixed_json():
    raw = 'Sure! Here you go:\n```json\n{"decisions": [{"market_id": "m1", ' \
          '"prob_yes": 0.7, "thesis": "capture done, timeline is the bar"}]}\n```'
    out = parse_decisions(raw, max_entries=2)
    assert len(out) == 1
    assert out[0]["market_id"] == "m1"
    assert out[0]["prob_yes"] == pytest.approx(0.7)


def test_parse_drops_malformed_and_caps_count():
    raw = ('{"decisions": ['
           '{"market_id": "a", "prob_yes": 0.6, "thesis": "t1"},'
           '{"market_id": "", "prob_yes": 0.6, "thesis": "no id"},'
           '{"market_id": "b", "prob_yes": 1.7, "thesis": "bad prob"},'
           '{"market_id": "c", "prob_yes": 0.4, "thesis": ""},'
           '{"market_id": "d", "prob_yes": 0.55, "thesis": "t2"},'
           '{"market_id": "e", "prob_yes": 0.45, "thesis": "t3"}]}')
    out = parse_decisions(raw, max_entries=2)
    assert [d["market_id"] for d in out] == ["a", "d"]


def test_parse_garbage_returns_empty():
    assert parse_decisions("I cannot help with that.", 2) == []
    assert parse_decisions('{"decisions": "nope"}', 2) == []
    assert parse_decisions("", 2) == []


# ---------------------------------------------------------------------------
# Pillar wiring
# ---------------------------------------------------------------------------


def _market(mid="m1", yes=0.30, liquidity=5000.0, volume=10000.0) -> Market:
    return Market(id=mid, question=f"Q {mid}?", outcome_yes_price=yes,
                  outcome_no_price=round(1 - yes, 2), liquidity=liquidity,
                  volume=volume, active=True, exchange="polymarket")


def _model_spec(alias="haiku", model="claude-haiku-4-5"):
    spec = MagicMock()
    spec.alias = alias
    spec.model = model
    spec.effort = "medium"
    return spec


def _settings(models):
    s = MagicMock()
    cfg = s.agent_trader
    cfg.enabled = True
    cfg.paper = True
    cfg.models = models
    cfg.scan_limit = 100
    cfg.markets_per_cycle = 10
    cfg.max_entries_per_cycle = 2
    cfg.max_open_per_model = 10
    cfg.stake_usd = 10.0
    cfg.min_liquidity = 1000.0
    cfg.min_days_to_resolution = 0.25
    cfg.max_days_to_resolution = 30.0
    cfg.min_edge_pts = 5.0
    cfg.memory_events = 12
    cfg.exclude_categories = []
    cfg.llm_timeout_seconds = 240
    cfg.decline_ttl_hours = 24.0
    s.risk.blocked_categories = []
    s.nlp.daily_claude_call_budget = 0  # unlimited in tests
    return s


async def _pillar(tmp_path, models, markets, llm_reply: str):
    db = Database(str(tmp_path / "test.db"))
    await db.connect()
    discovery = MagicMock()
    discovery.get_markets = AsyncMock(return_value=markets)
    risk = MagicMock()
    decision = MagicMock()
    decision.approved = True
    decision.position_size = 8.0
    decision.reason = ""
    decision.force_paper = False
    risk.evaluate = AsyncMock(return_value=decision)
    calibration = MagicMock()
    calibration.record_prediction = AsyncMock()

    pillar = AgentTraderPillar(
        db=db, settings=_settings(models), discovery=discovery,
        exchange=MagicMock(), risk_manager=risk,
        pnl_tracker=MagicMock(), calibration=calibration,
    )
    # Paper order result through a stubbed gateway (the gateway itself is
    # covered by its own tests).
    result = MagicMock()
    result.status = "paper"
    result.reason = ""
    order = MagicMock()
    order.market_id = markets[0].id if markets else "m1"
    order.token.value = "YES"
    order.token_id = "tok"
    order.price = 0.30
    order.size = 26.7
    fill = MagicMock()
    fill.is_paper = True
    fill.filled_size = 26.7
    fill.filled_price = 0.30
    result.order = order
    result.result = fill
    pillar._gateway = MagicMock()
    pillar._gateway.submit = AsyncMock(return_value=result)
    pillar._call_model = AsyncMock(return_value=llm_reply)
    return pillar, db, risk


@pytest.mark.asyncio
async def test_enters_through_risk_gate_with_per_model_cell(tmp_path):
    reply = '{"decisions": [{"market_id": "m1", "prob_yes": 0.55, "thesis": "t"}]}'
    pillar, db, risk = await _pillar(
        tmp_path, [_model_spec("haiku")], [_market("m1", yes=0.30)], reply)
    try:
        entered = await pillar.run_once()
        assert entered == 1
        risk.evaluate.assert_awaited()  # full gate, never bypassed
        sig = await db.fetchone(
            "SELECT strategy_source, edge, claude_prob FROM signals")
        assert sig["strategy_source"] == "agent_trader_haiku"
        assert sig["edge"] == pytest.approx(25.0)  # absolute points contract
        thesis = await db.fetchone(
            "SELECT model_alias, market_id, prob FROM agent_trader_theses")
        assert thesis["model_alias"] == "haiku"
        assert thesis["prob"] == pytest.approx(0.55)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_risk_rejection_blocks_entry(tmp_path):
    reply = '{"decisions": [{"market_id": "m1", "prob_yes": 0.55, "thesis": "t"}]}'
    pillar, db, risk = await _pillar(
        tmp_path, [_model_spec()], [_market("m1", yes=0.30)], reply)
    risk.evaluate.return_value.approved = False
    try:
        assert await pillar.run_once() == 0
        pillar._gateway.submit.assert_not_awaited()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_hallucinated_market_id_is_ignored(tmp_path):
    reply = '{"decisions": [{"market_id": "not-offered", "prob_yes": 0.9, "thesis": "t"}]}'
    pillar, db, _ = await _pillar(
        tmp_path, [_model_spec()], [_market("m1", yes=0.30)], reply)
    try:
        assert await pillar.run_once() == 0
        pillar._gateway.submit.assert_not_awaited()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_edge_below_floor_skipped(tmp_path):
    # Model claims 0.33 vs market 0.30 — 3 points, floor is 5.
    reply = '{"decisions": [{"market_id": "m1", "prob_yes": 0.33, "thesis": "t"}]}'
    pillar, db, _ = await _pillar(
        tmp_path, [_model_spec()], [_market("m1", yes=0.30)], reply)
    try:
        assert await pillar.run_once() == 0
        pillar._gateway.submit.assert_not_awaited()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_one_model_failure_does_not_stop_other_arms(tmp_path):
    reply = '{"decisions": [{"market_id": "m1", "prob_yes": 0.55, "thesis": "t"}]}'
    pillar, db, _ = await _pillar(
        tmp_path, [_model_spec("haiku"), _model_spec("opus", "claude-opus-4-8")],
        [_market("m1", yes=0.30)], reply)
    calls = {"n": 0}

    async def flaky(prompt, model, effort, cfg):
        calls["n"] += 1
        if model == "claude-haiku-4-5":
            raise RuntimeError("model call timed out")
        return reply

    pillar._call_model = flaky
    try:
        entered = await pillar.run_once()
        assert calls["n"] == 2       # both arms attempted
        assert entered == 1          # the healthy arm still entered
        sig = await db.fetchone("SELECT strategy_source FROM signals")
        assert sig["strategy_source"] == "agent_trader_opus"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_declined_market_not_reoffered_within_ttl(tmp_path):
    """A market the arm passed on is remembered for decline_ttl_hours — the
    old behavior re-offered the same unseen markets every cycle and burned a
    call re-declining them."""
    pillar, db, _ = await _pillar(
        tmp_path, [_model_spec()], [_market("m1")], '{"decisions": []}')
    calls = {"n": 0}

    async def counting(prompt, model, effort, cfg):
        calls["n"] += 1
        return '{"decisions": []}'

    pillar._call_model = counting
    try:
        await pillar.run_once()
        assert calls["n"] == 1
        row = await db.fetchone(
            "SELECT model_alias FROM agent_trader_declines WHERE market_id='m1'")
        assert row["model_alias"] == "haiku"
        # Same slate next cycle -> nothing to offer -> no call burned.
        await pillar.run_once()
        assert calls["n"] == 1

        # After the TTL the market is offered again.
        await db.execute(
            "UPDATE agent_trader_declines SET declined_at = datetime('now', '-25 hours')")
        await db.commit()
        await pillar.run_once()
        assert calls["n"] == 2
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_decided_market_is_not_declined(tmp_path):
    """A market the arm traded goes to theses, not to the decline table."""
    reply = '{"decisions": [{"market_id": "m1", "prob_yes": 0.55, "thesis": "t"}]}'
    pillar, db, _ = await _pillar(
        tmp_path, [_model_spec()], [_market("m1", yes=0.30), _market("m2", yes=0.50)],
        reply)
    try:
        assert await pillar.run_once() == 1
        declined = await db.fetchall(
            "SELECT market_id FROM agent_trader_declines")
        assert [r["market_id"] for r in declined] == ["m2"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_arm_order_rotates_per_cycle(tmp_path):
    """Arms run sequentially and first entrant claims the market, so a fixed
    order hands every contested market to the same arm. The starting arm must
    rotate each cycle."""
    reply = '{"decisions": []}'
    pillar, db, _ = await _pillar(
        tmp_path,
        [_model_spec("haiku"), _model_spec("sonnet", "claude-sonnet-5"),
         _model_spec("opus", "claude-opus-4-8")],
        [_market("m1")], reply)
    # Disable decline memory (#266): with it on, m1 is off every arm's slate
    # after cycle 1 and later cycles never call — this test is about order.
    pillar._settings.agent_trader.decline_ttl_hours = 0.0
    order: list[str] = []

    async def record(prompt, model, effort, cfg):
        order.append(model)
        return reply

    pillar._call_model = record
    try:
        await pillar.run_once()
        await pillar.run_once()
        await pillar.run_once()
        assert order[0] == "claude-haiku-4-5"
        assert order[3] == "claude-sonnet-5"   # cycle 2 starts one later
        assert order[6] == "claude-opus-4-8"   # cycle 3 starts two later
        assert len(set(order[:3])) == 3        # every arm still runs each cycle
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_second_arm_blocked_on_claimed_market_records_prediction(tmp_path):
    """Settlement attribution is market-level earliest-entrant-wins, so once
    one arm (or any strategy) has a trade on a market, another arm must NOT
    position there — even on the opposite token — or its P&L lands in the
    first cell. The blocked arm's prediction is still recorded (stake 0,
    entered=0) and never shows up in its open book."""
    reply = '{"decisions": [{"market_id": "m1", "prob_yes": 0.55, "thesis": "t"}]}'
    pillar, db, _ = await _pillar(
        tmp_path, [_model_spec("haiku")], [_market("m1", yes=0.30)], reply)
    try:
        # Another strategy already traded this market.
        await db.execute(
            """INSERT INTO trades (market_id, timestamp, side, size, price,
               is_paper, order_id, status, exchange, strategy_source)
               VALUES ('m1', datetime('now'), 'BUY', 10, 0.3, 1, 'x', 'paper',
                       'polymarket', 'agent_trader_sonnet')""")
        await db.commit()

        assert await pillar.run_once() == 0
        pillar._gateway.submit.assert_not_awaited()
        thesis = await db.fetchone(
            "SELECT model_alias, stake, entered, prob FROM agent_trader_theses")
        assert thesis["model_alias"] == "haiku"
        assert thesis["entered"] == 0
        assert thesis["stake"] == 0
        assert thesis["prob"] == pytest.approx(0.55)
        # Prediction-only rows are not open positions.
        assert await pillar._open_theses("haiku") == []
        # And the market is not re-offered to this alias next cycle.
        assert await pillar.run_once() == 0
        n = await db.fetchone("SELECT COUNT(*) c FROM agent_trader_theses")
        assert n["c"] == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_cli_call_isolated_and_tool_enabled(tmp_path, monkeypatch):
    """The subprocess must run from a NEUTRAL cwd (claude -p loads CLAUDE.md
    and project memory from its working directory — run from the repo root the
    arms cited the operator's own analyses, contaminating the A/B) and with
    the research tools enabled."""
    import tempfile as _tempfile

    import auramaur.strategy.agent_trader as at

    captured = {}

    class FakeProc:
        returncode = 0

        async def communicate(self):
            return b'{"decisions": []}', b""

    async def fake_exec(*cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(at.asyncio, "create_subprocess_exec", fake_exec)
    pillar, db, _ = await _pillar(
        tmp_path, [_model_spec()], [_market("m1")], "")
    pillar._call_model = AgentTraderPillar._call_model.__get__(pillar)
    try:
        cfg = pillar._settings.agent_trader
        out = await pillar._call_model("prompt", "claude-haiku-4-5", "medium", cfg)
        assert out == '{"decisions": []}'
        cmd = captured["cmd"]
        assert "--allowedTools" in cmd
        assert cmd[cmd.index("--allowedTools") + 1] == "WebSearch,WebFetch"
        assert "--model" in cmd and "claude-haiku-4-5" in cmd
        assert captured["kwargs"]["cwd"] == _tempfile.gettempdir()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_sell_side_derived_from_prob_vs_market(tmp_path):
    # Model says 0.10 vs market 0.30 -> SELL YES (buy NO downstream).
    reply = '{"decisions": [{"market_id": "m1", "prob_yes": 0.10, "thesis": "t"}]}'
    pillar, db, _ = await _pillar(
        tmp_path, [_model_spec()], [_market("m1", yes=0.30)], reply)
    try:
        assert await pillar.run_once() == 1
        intent = pillar._gateway.submit.await_args.args[0]
        assert intent.signal.recommended_side == OrderSide.SELL
        assert intent.force_paper is True
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Gemini arms — paid provider, metered cost, own daily ceiling
# ---------------------------------------------------------------------------


def _gemini_spec(alias="gpro", model="gemini-3.1-pro-preview"):
    spec = MagicMock()
    spec.alias = alias
    spec.model = model
    spec.effort = "medium"
    spec.provider = "gemini"
    return spec


def _gemini_cfg(pillar):
    cfg = pillar._settings.agent_trader
    cfg.gemini_daily_call_limit = 30
    cfg.gemini_price_per_mtok = {"gemini-3.1-pro-preview": [2.0, 12.0]}
    pillar._settings.gemini_api_key = "k"
    return cfg


@pytest.mark.asyncio
async def test_gemini_arm_routes_meters_cost_and_enters(tmp_path):
    """A gemini-provider arm uses the REST path (not the Claude CLI), its
    token usage is metered into agent_trader_costs at configured prices, and
    its decisions trade through the same rails/attribution."""
    reply = {"candidates": [{"content": {"parts": [
                {"text": '{"decisions": [{"market_id": "m1", "prob_yes": 0.55, "thesis": "t"}]}'}]}}],
             "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 500}}
    pillar, db, _ = await _pillar(
        tmp_path, [_gemini_spec()], [_market("m1", yes=0.30)], "")
    _gemini_cfg(pillar)
    pillar._call_model = AsyncMock(side_effect=AssertionError("claude path used for gemini arm"))
    pillar._gemini_request = AsyncMock(return_value=reply)
    try:
        assert await pillar.run_once() == 1
        sig = await db.fetchone("SELECT strategy_source FROM signals")
        assert sig["strategy_source"] == "agent_trader_gpro"
        cost = await db.fetchone(
            "SELECT calls, usd FROM agent_trader_costs WHERE model_alias='gpro'")
        # 1000*2.0/1e6 + 500*12.0/1e6 = 0.002 + 0.006 = 0.008
        assert cost["calls"] == 1
        assert cost["usd"] == pytest.approx(0.008)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_gemini_daily_ceiling_blocks_without_stopping_claude_arms(tmp_path):
    """The gemini ceiling is independent of the Claude pool: an exhausted
    gemini budget errors that arm's cycle (isolated) while claude arms run."""
    reply = '{"decisions": []}'
    pillar, db, _ = await _pillar(
        tmp_path, [_gemini_spec(), _model_spec("haiku")],
        [_market("m1", yes=0.30)], reply)
    cfg = _gemini_cfg(pillar)
    cfg.gemini_daily_call_limit = 1
    # Pre-burn the day's single gemini call.
    await pillar._ensure_schema()
    await pillar._record_gemini_cost("gpro", 0.001)
    claude_calls = {"n": 0}

    async def claude_ok(prompt, model, effort, c):
        claude_calls["n"] += 1
        return reply

    pillar._call_model = claude_ok
    pillar._gemini_request = AsyncMock(
        side_effect=AssertionError("request made past the ceiling"))
    try:
        await pillar.run_once()
        assert claude_calls["n"] == 1          # claude arm unaffected
        row = await db.fetchone(
            "SELECT SUM(calls) AS n FROM agent_trader_costs")
        assert row["n"] == 1                   # no new gemini call
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_claude_arms_unaffected_by_provider_field_default(tmp_path):
    """Back-compat: arms without an explicit provider run the Claude path."""
    reply = '{"decisions": []}'
    pillar, db, _ = await _pillar(
        tmp_path, [_model_spec("haiku")], [_market("m1")], reply)
    pillar._gemini_request = AsyncMock(
        side_effect=AssertionError("gemini path used for claude arm"))
    try:
        await pillar.run_once()
    finally:
        await db.close()
