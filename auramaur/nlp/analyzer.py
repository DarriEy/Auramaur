"""Claude-based probability estimation using Claude Code CLI (Max+ subscription)."""

from __future__ import annotations

import asyncio
import json
import re

import structlog
from pydantic import BaseModel, Field

from auramaur.data_sources.base import NewsItem
from auramaur.exchange.models import Market
from auramaur.nlp.cache import NLPCache, make_cache_key
from auramaur.nlp.prompts import (
    ADVERSARIAL_PROMPT,
    PROBABILITY_ESTIMATION_PROMPT,
    format_evidence,
)
from config.settings import Settings

log = structlog.get_logger()


class AnalysisResult(BaseModel):
    """Result of a Claude probability analysis."""

    probability: float = Field(ge=0, le=1)
    calibrated_probability: float | None = None
    confidence: str = "MEDIUM"  # LOW / MEDIUM / HIGH
    reasoning: str = ""
    key_factors: list[str] = Field(default_factory=list)
    time_sensitivity: str = "MEDIUM"  # LOW / MEDIUM / HIGH
    second_opinion_prob: float | None = None
    divergence: float | None = None
    skipped_reason: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json_span(text: str) -> str | None:
    """Return the first COMPLETE JSON array or object in ``text`` — the leading
    value followed by balanced brackets — or None. Tracks nesting and quoted
    strings so brackets inside string literals don't fool it. This recovers the
    common LLM shape of a valid JSON value followed by explanatory prose, e.g.
    ``[]\\n\\nNo matches found...`` (which the old object-only regex missed
    because it never looked for arrays)."""
    start = next((i for i, ch in enumerate(text) if ch in "[{"), None)
    if start is None:
        return None
    depth = 0
    in_str = esc = False
    for j in range(start, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                return text[start:j + 1]
    return None


def _parse_claude_json(text: str) -> dict | list:
    """Robustly parse JSON (object OR array) from Claude's response."""
    # Strip markdown code fences if present
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1)

    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # The model often emits a valid JSON value then trails off into prose —
    # extract the leading balanced array/object and parse that.
    span = _extract_json_span(text)
    if span is not None:
        try:
            return json.loads(span)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from Claude response: {text[:200]}")


def _evidence_digest(evidence: list[NewsItem]) -> str:
    """Create a short digest of evidence for cache keying."""
    import hashlib

    parts = [f"{e.source}:{e.title}" for e in evidence]
    raw = "|".join(sorted(parts))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class ClaudeAnalyzer:
    """Calls Claude via the CLI (Max+ subscription) to estimate probabilities."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = settings.nlp.model

    async def _call_claude_cli(self, prompt: str, *, effort: str | None = None,
                               reserved: bool = False) -> str:
        """Call Claude via the CLI using the Max+ subscription.

        Uses `claude -p <prompt> --output-format text` which authenticates
        through the user's Max+ plan — no API credits needed.

        ``effort`` is the CLI effort level (low|medium|high|max); defaults to
        ``nlp.effort_primary``.

        ``reserved`` callers (pin_claude paths — the proven edges) may spend up
        to the full daily budget; everyone else stops ``claude_reserve_for_pinned``
        calls early, so bulk consumers can't starve the money-making calls
        (2026-07: the arb matcher burned the whole budget by 07:00 UTC daily
        while the lens never got a Claude call).

        Retries up to 3 times on transient failures (timeout, non-zero exit).
        """
        from auramaur.nlp import call_budget

        use_effort = effort or self._settings.nlp.effort_primary

        budget = self._settings.nlp.daily_claude_call_budget
        if budget > 0:
            limit = budget if reserved else max(
                0, budget - self._settings.nlp.claude_reserve_for_pinned)
            if call_budget.calls_today() >= limit:
                from auramaur.nlp.errors import BudgetExhausted
                raise BudgetExhausted(
                    f"Daily Claude call budget ({limit}/{budget}"
                    f"{' reserved' if reserved else ''}) exhausted"
                )

        max_attempts = 3
        backoff_seconds = [5, 10, 20]
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "claude", "-p", prompt,
                    "--output-format", "text",
                    "--model", self._model,
                    "--effort", use_effort,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)

                if proc.returncode != 0:
                    err_msg = stderr.decode().strip()
                    log.error("claude_cli.error", returncode=proc.returncode, stderr=err_msg, attempt=attempt)
                    raise RuntimeError(f"Claude CLI failed (rc={proc.returncode}): {err_msg}")

                log.info(
                    "claude_cli.call",
                    daily_calls=call_budget.record_call(),
                    budget=budget,
                )
                return stdout.decode().strip()

            except (TimeoutError, asyncio.TimeoutError, RuntimeError) as e:
                last_error = e
                if attempt < max_attempts:
                    delay = backoff_seconds[attempt - 1]
                    log.warning(
                        "claude_cli.retry",
                        attempt=attempt,
                        max_attempts=max_attempts,
                        delay=delay,
                        error=str(e),
                    )
                    await asyncio.sleep(delay)

        raise last_error  # type: ignore[misc]

    async def _call_llm(self, prompt: str, *, effort: str | None = None,
                        pin_claude: bool = False) -> str:
        """Route to Gemini (off-hours / Claude budget low) else Claude.

        ``pin_claude=True`` bypasses routing entirely and always calls Claude
        (against the reserved budget slice). For calls where the model identity
        IS the edge — the resolution lens's fine-print reads collapsed when
        conservation routing silently handed them to Gemini (2026-06-25 →
        2026-07-02: zero qualifying gaps, zero entries)."""
        from functools import partial

        from auramaur.nlp.llm_router import route
        from auramaur.nlp import call_budget
        if pin_claude:
            return await self._call_claude_cli(prompt, effort=effort, reserved=True)
        claude_fn = partial(self._call_claude_cli, effort=effort)
        return await route(self._settings, call_budget.calls_today(), prompt, claude_fn)

    # ------------------------------------------------------------------
    # Core estimation methods
    # ------------------------------------------------------------------

    async def estimate_probability(
        self,
        market: Market,
        evidence: list[NewsItem],
    ) -> AnalysisResult:
        """Call Claude (primary) to estimate the probability."""
        evidence_text = format_evidence(evidence)
        prompt = PROBABILITY_ESTIMATION_PROMPT.format(
            question=market.question,
            description=market.description,
            market_price=market.outcome_yes_price,
            evidence=evidence_text,
        )

        raw = await self._call_llm(prompt, effort=self._settings.nlp.effort_primary)
        parsed = _parse_claude_json(raw)
        return AnalysisResult(**parsed)

    async def get_second_opinion(
        self,
        market: Market,
        evidence: list[NewsItem],
        first_estimate: float | None = None,
    ) -> AnalysisResult:
        """Call Claude (adversarial second opinion)."""
        evidence_text = format_evidence(evidence)
        prompt = ADVERSARIAL_PROMPT.format(
            question=market.question,
            description=market.description,
            market_price=market.outcome_yes_price,
            first_estimate=first_estimate if first_estimate is not None else 0.5,
            evidence=evidence_text,
        )

        raw = await self._call_llm(prompt, effort=self._settings.nlp.effort_adversarial)
        parsed = _parse_claude_json(raw)
        return AnalysisResult(**parsed)

    async def analyze(
        self,
        market: Market,
        evidence: list[NewsItem],
        cache: NLPCache,
    ) -> AnalysisResult:
        """Full analysis pipeline: cache -> primary -> second opinion -> cache."""
        digest = _evidence_digest(evidence)
        cache_key = make_cache_key(market.question, digest)

        from auramaur.monitoring.display import show_cache_hit, show_claude_thinking

        # 1. Check cache (with price-move invalidation)
        cached = await cache.get(cache_key, current_price=market.outcome_yes_price)
        if cached is not None:
            show_cache_hit()
            return AnalysisResult(**cached)

        # 2. Primary estimate
        show_claude_thinking(market.id, "primary")
        try:
            primary = await self.estimate_probability(market, evidence)
        except Exception as e:
            log.warning("analyze.primary_failed", market_id=market.id, error=str(e))
            return AnalysisResult(
                probability=0.5,
                confidence="LOW",
                skipped_reason=f"Primary estimation failed: {e}",
            )

        # 3. Second opinion (skip if configured, or smart-skip on slam dunks)
        edge_pct = abs(primary.probability - market.outcome_yes_price) * 100.0
        smart_skip = (
            primary.confidence == "HIGH"
            and edge_pct > 15.0
        )
        if smart_skip:
            log.info(
                "analyzer.skip_second_opinion_smart",
                market_id=market.id,
                confidence=primary.confidence,
                edge_pct=round(edge_pct, 1),
            )
        if not self._settings.nlp.skip_second_opinion and not smart_skip:
            show_claude_thinking(market.id, "second")
            try:
                second = await self.get_second_opinion(market, evidence, first_estimate=primary.probability)
                primary.second_opinion_prob = second.probability
                primary.divergence = abs(primary.probability - second.probability)
            except Exception as e:
                log.warning("analyze.second_opinion_failed", market_id=market.id, error=str(e))

        log.info(
            "analyzer.complete",
            market_id=market.id,
            primary_prob=primary.probability,
            second_prob=primary.second_opinion_prob,
            divergence=primary.divergence,
        )

        # 5. Cache the result (with market price for price-move invalidation)
        ttl = self._settings.nlp.cache_ttl_breaking_seconds
        await cache.put(
            cache_key, market.id, primary.model_dump(), ttl,
            market_price=market.outcome_yes_price,
        )

        return primary
