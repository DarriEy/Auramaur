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
