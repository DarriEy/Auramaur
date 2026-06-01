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
    """Call Gemini via REST (JSON mode). No SDK dependency."""
    import aiohttp

    g = settings.gemini
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
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


async def route(settings, daily_calls: int, prompt: str,
                claude_fn: Callable[[str], Awaitable[str]]) -> str:
    """Route to Gemini when appropriate; else call `claude_fn`. Falls back to
    Claude if Gemini errors."""
    if should_use_gemini(settings, daily_calls):
        try:
            out = await call_gemini(settings, prompt)
            log.info("llm.routed", provider="gemini", model=settings.gemini.model)
            return out
        except Exception as e:  # noqa: BLE001 — fall back to Claude
            log.warning("gemini.failed_fallback_claude", error=str(e)[:120])
    return await claude_fn(prompt)
