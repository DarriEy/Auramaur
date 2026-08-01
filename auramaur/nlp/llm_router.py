"""Shared LLM routing — send analysis to Gemini during off-hours or when Claude's
daily budget is near-exhausted, with automatic fallback to Claude.

Used by both ClaudeAnalyzer and StrategicAnalyzer so routing behaves identically
regardless of analysis mode.
"""

from __future__ import annotations

from typing import Awaitable, Callable

import structlog

log = structlog.get_logger()


def should_use_gemini(settings, daily_calls: int) -> bool:
    """True when we should prefer Gemini: off-hours OR Claude budget near-exhausted."""
    from datetime import datetime, timezone

    g = settings.gemini
    if not (g.enabled and settings.gemini_api_key):
        return False
    budget = settings.nlp.daily_claude_call_budget
    if budget > 0 and daily_calls >= int(budget * g.claude_budget_threshold):
        return True  # Claude budget near-exhausted
    return datetime.now(timezone.utc).hour in g.off_hours_utc  # off-hours


async def call_gemini(settings, prompt: str) -> str:
    """Call Gemini via REST (JSON mode). No SDK dependency.

    Capped, counted and priced — none of which it was until 2026-08-01.

    This route is not the agent_trader/term_structure Gemini arms. Those log
    `*.gemini_call` with a usd figure and share a 30/day ceiling, and over the
    retained window they totalled 89 calls and $0.49. THIS one carries the
    analyzer's entire volume (150-293 calls/day) for the whole `off_hours_utc`
    window plus every `claude_budget_threshold` switch, through a premium
    model, and it had no cap, no counter, and emitted no log line at all. It
    ran to roughly $1000 while the instrumented arms showed under a dollar,
    which is precisely how an audit of "Gemini spend" reached the wrong answer
    by three orders of magnitude.

    The sibling `call_openai` had all three guards from the start; this is the
    same treatment applied to the arm that actually carries the volume.
    """
    import aiohttp

    from auramaur.nlp import call_budget

    g = settings.gemini
    limit = int(getattr(g, "daily_call_limit", 0) or 0)
    if limit > 0 and call_budget.gemini_calls_today() >= limit:
        raise RuntimeError(f"Gemini analysis daily cap ({limit}) reached")
    # Counted BEFORE the reply is judged: the prompt is billed whether or not
    # the response parses, so counting only successes leaves the cap blind to
    # a failing arm.
    call_budget.record_gemini_call()

    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{g.model}:generateContent?key={settings.gemini_api_key}")
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": settings.nlp.max_tokens,
            "temperature": 0.3,
            "responseMimeType": "application/json",
        },
    }
    async with aiohttp.ClientSession() as s:
        async with s.post(url, json=body, timeout=aiohttp.ClientTimeout(total=120)) as r:
            data = await r.json()

    usage = data.get("usageMetadata") or {}
    prices = list(getattr(g, "price_per_mtok", None) or [0.0, 0.0])
    usd = (float(usage.get("promptTokenCount", 0) or 0) * float(prices[0])
           + float(usage.get("candidatesTokenCount", 0) or 0) * float(prices[1])) / 1e6
    log.info("llm.gemini_call", model=g.model,
             calls_today=call_budget.gemini_calls_today(), cap=limit,
             in_tok=usage.get("promptTokenCount"),
             out_tok=usage.get("candidatesTokenCount"), usd=round(usd, 5))

    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


async def call_openai(settings, prompt: str) -> str:
    """Last-resort analysis arm. Ungrounded, capped, counted.

    Deliberately no web_search: this carries the analyzer's full volume and
    the grounded configuration measured ~$0.249/call, which is $37-73/day at
    the observed 150-293 calls/day. The analyzer prompt already carries
    gathered evidence.

    The cap rides call_budget's sidecar rather than the trading DB — same
    reason the Claude counter does: no analyzer constructor has to grow an
    async Database dependency (see that module's docstring).
    """
    import aiohttp

    from auramaur.nlp import call_budget

    cfg = settings.nlp
    key = getattr(settings, "openai_api_key", "")
    if not key:
        raise RuntimeError("OpenAI analysis fallback unavailable: no API key")

    if (cfg.openai_daily_call_limit > 0
            and call_budget.openai_calls_today() >= cfg.openai_daily_call_limit):
        raise RuntimeError(
            f"OpenAI analysis daily cap ({cfg.openai_daily_call_limit}) reached")
    # Counted BEFORE the reply is judged: OpenAI bills reasoning tokens even
    # on a truncated or errored response, so counting only successes would let
    # a failing arm spend with the cap blind to it.
    call_budget.record_openai_call()

    body = {
        "model": str(cfg.openai_model),
        "input": prompt,
        "reasoning": {"effort": str(cfg.openai_effort)},
        "store": False,
        "max_output_tokens": int(cfg.openai_max_output_tokens),
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/responses", json=body,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=180),
        ) as response:
            data = await response.json()

    usage = data.get("usage") or {}
    prices = cfg.openai_price_per_mtok
    usd = (float(usage.get("input_tokens", 0) or 0) * float(prices[0])
           + float(usage.get("output_tokens", 0) or 0) * float(prices[1])) / 1e6
    log.info("llm.openai_call", model=cfg.openai_model,
             calls_today=call_budget.openai_calls_today(),
             cap=cfg.openai_daily_call_limit, usd=round(usd, 5))

    if data.get("error"):
        raise RuntimeError(f"OpenAI analysis error: {str(data['error'])[:200]}")
    if str(data.get("status") or "") == "incomplete":
        raise RuntimeError(
            "OpenAI analysis truncated "
            f"({str(data.get('incomplete_details'))[:120]}) — raise "
            f"nlp.openai_max_output_tokens (now {cfg.openai_max_output_tokens})")
    text = ""
    for output in data.get("output", []) or []:
        if output.get("type") != "message":
            continue
        for content in output.get("content", []) or []:
            if content.get("type") == "refusal":
                raise RuntimeError("OpenAI analysis refused")
            if content.get("type") == "output_text":
                text += content.get("text", "")
    text = text.strip()
    if not text:
        raise RuntimeError(f"OpenAI analysis empty reply: {str(data)[:200]}")
    return text


async def route(settings, daily_calls: int, prompt: str,
                claude_fn: Callable[[str], Awaitable[str]]) -> str:
    """Route analysis to a working model.

    Preference order is unchanged: Gemini when off-hours or the Claude daily
    budget is near-exhausted, otherwise Claude.

    What IS new (2026-07-29) is that a CLAUDE FAILURE now has somewhere to go.
    The fallback used to run one way only — Gemini failing fell back to Claude,
    but Claude failing propagated and the caller produced nothing. That is not
    a rare corner: `should_use_gemini` keys off the DAILY budget, and
    `call_budget.record_call()` only counts SUCCESSES, so a weekly-limit
    outage leaves the counter low, never trips the threshold, and every
    analysis outside the off-hours window dies. On 2026-07-29 that silenced
    the llm pillar — the only book with live authority — for a full day while
    the CLI logged 1189 consecutive rc=1 failures.

    pin_claude callers do NOT come through here. They bypass routing entirely
    because for them the model identity IS the edge (resolution_lens), and
    that stays true.
    """
    tried_gemini = False
    if should_use_gemini(settings, daily_calls):
        tried_gemini = True
        try:
            out = await call_gemini(settings, prompt)
            log.info("llm.routed", provider="gemini", model=settings.gemini.model)
            return out
        except Exception as e:  # noqa: BLE001 — fall back to Claude
            log.warning("gemini.failed_fallback_claude", error=str(e)[:120])

    try:
        return await claude_fn(prompt)
    except Exception as claude_error:  # noqa: BLE001 — try the remaining arms
        errors = [f"claude: {str(claude_error)[:120]}"]
        arms = []
        if not tried_gemini:
            arms.append(("gemini", lambda: call_gemini(settings, prompt)))
        if getattr(settings.nlp, "openai_fallback", False):
            arms.append(("openai", lambda: call_openai(settings, prompt)))
        for name, fn in arms:
            try:
                out = await fn()
                log.warning("llm.claude_failed_fell_back", provider=name,
                            claude_error=str(claude_error)[:160])
                return out
            except Exception as e:  # noqa: BLE001 — try the next arm
                errors.append(f"{name}: {str(e)[:120]}")
        log.error("llm.all_arms_failed", errors=errors)
        raise claude_error
