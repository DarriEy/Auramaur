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

def _parse_claude_json(text: str) -> dict:
    """Robustly parse JSON from Claude's response."""
    # Strip markdown code fences if present
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1)

    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract first JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
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
        self._daily_calls = 0
        self._daily_calls_date = ""

    async def _call_claude_cli(self, prompt: str) -> str:
        """Call Claude via the CLI using the Max+ subscription.

        Uses `claude -p <prompt> --output-format text` which authenticates
        through the user's Max+ plan — no API credits needed.

        Retries up to 3 times on transient failures (timeout, non-zero exit).
        """
        from datetime import date

        today = date.today().isoformat()
        if self._daily_calls_date != today:
            self._daily_calls = 0
            self._daily_calls_date = today
        budget = self._settings.nlp.daily_claude_call_budget
        if budget > 0 and self._daily_calls >= budget:
            raise RuntimeError(
                f"Daily Claude call budget ({budget}) exhausted"
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
                    "--effort", "max",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)

                if proc.returncode != 0:
                    err_msg = stderr.decode().strip()
                    log.error("claude_cli.error", returncode=proc.returncode, stderr=err_msg, attempt=attempt)
                    raise RuntimeError(f"Claude CLI failed (rc={proc.returncode}): {err_msg}")

                self._daily_calls += 1
                log.info(
                    "claude_cli.call",
                    daily_calls=self._daily_calls,
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

        raw = await self._call_claude_cli(prompt)
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

        raw = await self._call_claude_cli(prompt)
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
