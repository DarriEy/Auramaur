"""Tool-use analyzer — Claude Code drives its own web_search / web_fetch.

Used as a refinement step after the strategic batch: for markets where the
batch surfaced a strong edge, we re-ask Claude with ``WebSearch`` and
``WebFetch`` enabled so it can look up specifics (resolution criteria,
recent developments, primary sources) rather than relying only on the
pre-aggregated evidence.

Works by shelling out to ``claude -p`` with ``--allowedTools
WebSearch,WebFetch`` and ``--json-schema`` for structured output, matching
the existing ``ClaudeAnalyzer._call_claude_cli`` pattern.

Fails open: on timeout / parse error / budget exhaustion, returns ``None``
so the caller falls back to the batch result.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

from auramaur.exchange.models import Market
from auramaur.nlp.strategic import BatchAnalysisResult
from config.settings import Settings

log = structlog.get_logger()


_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "probability": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Probability the market resolves YES.",
        },
        "confidence": {
            "type": "string",
            "enum": ["LOW", "MEDIUM_LOW", "MEDIUM", "MEDIUM_HIGH", "HIGH"],
        },
        "reasoning": {
            "type": "string",
            "description": "2-5 sentences citing what was found via web search.",
        },
        "key_factors": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 6,
        },
        "sources_used": {
            "type": "array",
            "items": {"type": "string"},
            "description": "URLs Claude actually fetched, for audit.",
            "maxItems": 10,
        },
    },
    "required": ["probability", "confidence", "reasoning"],
    "additionalProperties": False,
}


def _build_prompt(market: Market, batch_result: BatchAnalysisResult) -> str:
    """Construct the refinement prompt.

    Includes the batch's initial estimate so Claude knows what it's asked
    to refine and has a starting point — then invites it to use the tools
    to find evidence that either confirms or revises the estimate.
    """
    batch_prob_pct = int(batch_result.probability * 100)
    market_prob_pct = int((market.outcome_yes_price or 0.5) * 100)
    edge_pct = batch_prob_pct - market_prob_pct
    end_date = market.end_date.isoformat() if market.end_date else "unknown"

    return f"""You are refining a prediction for the following market:

QUESTION: {market.question}
DESCRIPTION: {(market.description or '')[:1200]}
RESOLUTION DATE: {end_date}
CURRENT MARKET PRICE (YES): {market_prob_pct}%

A prior batch analysis produced this estimate:
  Estimated probability: {batch_prob_pct}%
  Confidence: {batch_result.confidence}
  Reasoning: {batch_result.reasoning[:500]}
  Key factors: {batch_result.key_factors[:6]}

  Implied edge vs market price: {edge_pct:+d}% (YES)

Your task: use WebSearch and WebFetch to verify or refine this estimate.
Focus on:
  1. What SPECIFIC information is needed to decide this? (resolution criteria, named entities, dates)
  2. Has anything material happened recently that the batch analysis might have missed?
  3. Does the prior estimate hold up when you look at primary sources, or should it be revised?

Output STRICT JSON matching this schema:
{json.dumps(_OUTPUT_SCHEMA, indent=2)}

Be parsimonious with tool calls — aim for 2-4 searches total. If after
searching you don't find evidence that materially changes the prior estimate,
return a probability close to it and say so in the reasoning.
"""


class ToolUseAnalyzer:
    """Refine a batch result by letting Claude Code drive its own searches."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = settings.nlp.tool_use_model or settings.nlp.model

    async def refine(
        self,
        market: Market,
        batch_result: BatchAnalysisResult,
    ) -> BatchAnalysisResult | None:
        """Return a refined ``BatchAnalysisResult`` or ``None`` on failure.

        ``None`` is the signal to fall back to ``batch_result``.
        """
        prompt = _build_prompt(market, batch_result)

        cmd = [
            "claude", "-p", prompt,
            "--allowedTools", "WebSearch,WebFetch",
            "--output-format", "json",
            "--json-schema", json.dumps(_OUTPUT_SCHEMA),
            "--model", self._model,
            "--effort", "high",
            "--no-session-persistence",
        ]
        # --max-budget-usd is only meaningful for API-key users; Max+
        # subscriptions are flat-rate. The setting is retained for
        # operators who later switch to an API key.

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
        except asyncio.TimeoutError:
            log.warning("tool_use.timeout", market_id=market.id)
            return None
        except Exception as e:
            log.warning("tool_use.subprocess_error", market_id=market.id, error=str(e))
            return None

        if proc.returncode != 0:
            log.warning(
                "tool_use.nonzero_exit",
                market_id=market.id,
                returncode=proc.returncode,
                stderr=stderr.decode()[:300],
            )
            return None

        raw_stdout = stdout.decode()
        parsed = self._parse_output(raw_stdout)
        if parsed is None:
            log.warning(
                "tool_use.parse_failed",
                market_id=market.id,
                stdout_head=raw_stdout[:400],
                stderr_head=stderr.decode()[:200],
            )
            return None

        try:
            prob = float(parsed["probability"])
            if not 0.0 <= prob <= 1.0:
                log.warning("tool_use.invalid_probability", market_id=market.id, prob=prob)
                return None
            return BatchAnalysisResult(
                market_id=market.id,
                probability=prob,
                confidence=str(parsed.get("confidence", batch_result.confidence)),
                reasoning=str(parsed.get("reasoning", ""))[:1000],
                key_factors=list(parsed.get("key_factors", []) or [])[:6],
                cross_market_notes=batch_result.cross_market_notes,
            )
        except Exception as e:
            log.warning(
                "tool_use.shape_error",
                market_id=market.id,
                error=str(e),
                parsed_keys=list(parsed.keys()) if isinstance(parsed, dict) else None,
            )
            return None

    @staticmethod
    def _parse_output(raw: str) -> dict | None:
        """Parse the JSON envelope Claude CLI emits with ``--output-format json``.

        When ``--json-schema`` is supplied, the CLI populates
        ``structured_output`` with the parsed object and leaves ``result``
        empty. Without a schema, the model's JSON is in ``result`` as a
        string. We check both paths.
        """
        raw = raw.strip()
        if not raw:
            return None
        try:
            env = json.loads(raw)
        except Exception:
            env = None

        if isinstance(env, dict):
            # Preferred: ``structured_output`` when --json-schema was used.
            so = env.get("structured_output")
            if isinstance(so, dict) and "probability" in so:
                return so

            # Fallback: ``result`` string with embedded JSON (no-schema mode).
            if "result" in env and isinstance(env["result"], str):
                inner = env["result"].strip()
                if inner:
                    try:
                        return json.loads(inner)
                    except Exception:
                        start = inner.find("{")
                        end = inner.rfind("}")
                        if start != -1 and end > start:
                            try:
                                return json.loads(inner[start:end + 1])
                            except Exception:
                                return None
                        return None
            # Some envelopes put the model JSON at the top level directly.
            if "probability" in env:
                return env

        # Last-ditch: scan raw for a JSON object with a "probability" key.
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except Exception:
                return None
        return None
