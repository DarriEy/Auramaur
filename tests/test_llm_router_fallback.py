"""Router fallback: a Claude failure must have somewhere to go.

The fallback used to run one way only — Gemini failing fell back to Claude,
but Claude failing propagated and the caller produced nothing. That is not a
corner case: should_use_gemini keys off the DAILY budget while record_call()
counts only SUCCESSES, so a WEEKLY-limit outage leaves the counter low, never
trips the threshold, and every analysis outside off-hours dies. On 2026-07-29
it silenced the llm pillar for a day across 1189 rc=1 CLI failures.
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from auramaur.nlp import call_budget
from auramaur.nlp.llm_router import route


def _settings(*, openai_fallback=True, gemini_enabled=True, off_hours=None):
    return SimpleNamespace(
        gemini=SimpleNamespace(
            enabled=gemini_enabled, model="gemini-test",
            off_hours_utc=off_hours if off_hours is not None else [],
            claude_budget_threshold=0.8),
        gemini_api_key="gem-key",
        openai_api_key="oai-key",
        nlp=SimpleNamespace(
            daily_claude_call_budget=175, max_tokens=4096,
            openai_fallback=openai_fallback, openai_model="gpt-5.6-sol",
            openai_effort="medium", openai_daily_call_limit=100,
            openai_price_per_mtok=[5.0, 30.0], openai_max_output_tokens=4000),
    )


@pytest.fixture(autouse=True)
def _isolated_counter():
    d = tempfile.mkdtemp()
    call_budget.set_db_path(os.path.join(d, "t.db"))
    yield
    call_budget.set_db_path("auramaur.db")


@pytest.mark.asyncio
async def test_claude_failure_falls_back_to_gemini():
    """Outside off-hours the router goes straight to Claude. When Claude dies
    it must still produce an answer rather than propagating."""
    claude = AsyncMock(side_effect=RuntimeError("Claude CLI failed (rc=1): "))
    with patch("auramaur.nlp.llm_router.call_gemini",
               AsyncMock(return_value="gemini answer")) as gem:
        out = await route(_settings(), 0, "prompt", claude)
    assert out == "gemini answer"
    gem.assert_awaited_once()


@pytest.mark.asyncio
async def test_openai_covers_when_claude_and_gemini_are_both_down():
    claude = AsyncMock(side_effect=RuntimeError("Claude CLI failed (rc=1): "))
    with patch("auramaur.nlp.llm_router.call_gemini",
               AsyncMock(side_effect=RuntimeError("gemini down"))), \
         patch("auramaur.nlp.llm_router.call_openai",
               AsyncMock(return_value="openai answer")) as oai:
        out = await route(_settings(), 0, "prompt", claude)
    assert out == "openai answer"
    oai.assert_awaited_once()


@pytest.mark.asyncio
async def test_gemini_is_not_retried_after_it_already_failed():
    """When routing already tried Gemini and it failed, the Claude-failure
    path must skip straight to OpenAI rather than paying for Gemini twice."""
    settings = _settings(off_hours=list(range(24)))   # always prefer Gemini
    claude = AsyncMock(side_effect=RuntimeError("Claude CLI failed (rc=1): "))
    gem = AsyncMock(side_effect=RuntimeError("gemini down"))
    with patch("auramaur.nlp.llm_router.call_gemini", gem), \
         patch("auramaur.nlp.llm_router.call_openai",
               AsyncMock(return_value="openai answer")):
        out = await route(settings, 0, "prompt", claude)
    assert out == "openai answer"
    assert gem.await_count == 1


@pytest.mark.asyncio
async def test_original_claude_error_is_raised_when_every_arm_fails():
    """The caller should see why CLAUDE failed — that is the actionable
    error — not whichever fallback happened to be tried last."""
    claude = AsyncMock(side_effect=RuntimeError("Claude CLI failed (rc=1): "))
    with patch("auramaur.nlp.llm_router.call_gemini",
               AsyncMock(side_effect=RuntimeError("gemini down"))), \
         patch("auramaur.nlp.llm_router.call_openai",
               AsyncMock(side_effect=RuntimeError("openai down"))):
        with pytest.raises(RuntimeError, match="Claude CLI failed"):
            await route(_settings(), 0, "prompt", claude)


@pytest.mark.asyncio
async def test_healthy_claude_is_never_diverted():
    """The fallback must not change which model normally answers."""
    with patch("auramaur.nlp.llm_router.call_gemini", AsyncMock()) as gem, \
         patch("auramaur.nlp.llm_router.call_openai", AsyncMock()) as oai:
        out = await route(_settings(), 0, "prompt",
                          AsyncMock(return_value="claude answer"))
    assert out == "claude answer"
    gem.assert_not_awaited()
    oai.assert_not_awaited()


@pytest.mark.asyncio
async def test_openai_arm_can_be_disabled():
    claude = AsyncMock(side_effect=RuntimeError("Claude CLI failed (rc=1): "))
    with patch("auramaur.nlp.llm_router.call_gemini",
               AsyncMock(side_effect=RuntimeError("gemini down"))), \
         patch("auramaur.nlp.llm_router.call_openai", AsyncMock()) as oai:
        with pytest.raises(RuntimeError):
            await route(_settings(openai_fallback=False), 0, "prompt", claude)
    oai.assert_not_awaited()


@pytest.mark.asyncio
async def test_openai_daily_cap_blocks_further_spend():
    """A long Claude outage must not run up an unbounded bill."""
    from auramaur.nlp.llm_router import call_openai

    settings = _settings()
    settings.nlp.openai_daily_call_limit = 2
    for _ in range(2):
        call_budget.record_openai_call()
    with pytest.raises(RuntimeError, match="daily cap"):
        await call_openai(settings, "prompt")


@pytest.mark.asyncio
async def test_failed_openai_call_still_consumes_the_cap():
    """OpenAI bills reasoning tokens on truncated replies too, so counting
    only successes would let a failing arm spend with the cap blind to it."""
    from auramaur.nlp.llm_router import call_openai

    settings = _settings()
    before = call_budget.openai_calls_today()

    class _Resp:
        status = 200

        async def json(self):
            return {"status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "output": [], "usage": {"input_tokens": 3000,
                                            "output_tokens": 4000}}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _Session:
        def post(self, *a, **kw):
            return _Resp()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    with patch("aiohttp.ClientSession", lambda *a, **kw: _Session()):
        with pytest.raises(RuntimeError, match="truncated"):
            await call_openai(settings, "prompt")
    assert call_budget.openai_calls_today() == before + 1


@pytest.mark.asyncio
async def test_call_gemini_is_capped_and_counted(tmp_path):
    """The analyzer Gemini route must refuse past its daily ceiling.

    Regression (2026-08-01): this route had no cap, no counter and no log
    line, while its `call_openai` sibling had all three. It carries the
    analyzer's full volume (150-293 calls/day) through a premium model for the
    entire off-hours window plus every budget-threshold switch, and quietly
    ran to roughly $1000 — an audit that measured only the *instrumented*
    Gemini arms put the total at $0.49 and was wrong by three orders of
    magnitude.
    """
    from auramaur.nlp.llm_router import call_gemini

    call_budget.set_db_path(str(tmp_path / "budget.db"))
    settings = SimpleNamespace(
        gemini=SimpleNamespace(
            enabled=True, model="gemini-test", off_hours_utc=[],
            claude_budget_threshold=0.8, daily_call_limit=2,
            price_per_mtok=[2.0, 12.0]),
        gemini_api_key="gem-key",
        nlp=SimpleNamespace(max_tokens=256),
    )

    payload = {
        "candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}],
        "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 100},
    }

    class _Resp:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def json(self): return payload

    class _Session:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def post(self, *a, **k): return _Resp()

    with patch("aiohttp.ClientSession", lambda *a, **k: _Session()):
        assert await call_gemini(settings, "p") == '{"ok": true}'
        assert call_budget.gemini_calls_today() == 1
        await call_gemini(settings, "p")
        assert call_budget.gemini_calls_today() == 2

        # Third call is over the ceiling and must not reach the API.
        with pytest.raises(RuntimeError, match="daily cap"):
            await call_gemini(settings, "p")
        assert call_budget.gemini_calls_today() == 2
