"""OpenAI Responses API adapter for the IBKR ETF intelligence experiment."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time

import aiohttp
import structlog

log = structlog.get_logger()


@dataclass(frozen=True)
class ETFAnalysis:
    probability: float
    confidence: str
    thesis: str
    key_risks: tuple[str, ...]
    skipped_reason: str | None = None
    intelligence_cost_usd: float = 0.0


_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "probability": {"type": "number", "minimum": 0, "maximum": 1},
        "confidence": {"type": "string", "enum": [
            "LOW", "MEDIUM_LOW", "MEDIUM", "MEDIUM_HIGH", "HIGH"]},
        "thesis": {"type": "string"},
        "key_risks": {"type": "array", "items": {"type": "string"},
                      "maxItems": 5},
    },
    "required": ["probability", "confidence", "thesis", "key_risks"],
}

_INSTRUCTIONS = """You are an institutional cross-asset ETF forecaster in a
strict paper-trading experiment. Estimate whether the named ETF will be higher
at the stated horizon. Use only the supplied evidence. Separate facts from
inference, account for the current macro regime, avoid recency bias, and return
a calibrated probability rather than a trade recommendation. Confidence means
confidence in forecast quality, not bullishness. A weak or conflicting evidence
set should remain near 0.50 with LOW confidence. Never infer missing market data.
"""


class OpenAIETFAnalyzer:
    """One model arm. No trading methods and no access to portfolio state."""

    _URL = "https://api.openai.com/v1/responses"

    def __init__(self, api_key: str, model: str, effort: str,
                 timeout_seconds: int = 120, db=None,
                 model_alias: str = "", input_cost_per_million: float = 0.0,
                 output_cost_per_million: float = 0.0,
                 instructions: str = _INSTRUCTIONS) -> None:
        self.model = model
        self.effort = effort
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._session: aiohttp.ClientSession | None = None
        self._warned_missing_key = False
        self._db = db
        self._model_alias = model_alias
        self._input_cost_per_million = input_cost_per_million
        self._output_cost_per_million = output_cost_per_million
        self._instructions = instructions

    async def _start_attempt(self) -> int | None:
        if self._db is None:
            return None
        cursor = await self._db.execute(
            """INSERT INTO ibkr_etf_openai_attempts
               (model_alias, model, status) VALUES (?, ?, 'started')""",
            (self._model_alias, self.model))
        await self._db.commit()
        return cursor.lastrowid

    async def _finish_attempt(self, attempt_id: int | None, status: str,
                              started: float, data=None, error: str = "") -> float:
        if self._db is None or attempt_id is None:
            return 0.0
        data = data or {}
        usage = data.get("usage") or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        cost = (
            input_tokens * self._input_cost_per_million
            + output_tokens * self._output_cost_per_million
        ) / 1_000_000
        await self._db.execute(
            """UPDATE ibkr_etf_openai_attempts SET status=?, response_id=?,
                 input_tokens=?, output_tokens=?, total_tokens=?, cost_usd=?, latency_ms=?,
                 error=?, finished_at=datetime('now') WHERE id=?""",
            (status, str(data.get("id", "")), input_tokens, output_tokens,
             int(usage.get("total_tokens", 0) or 0),
             cost, int((time.monotonic() - started) * 1000), error[:300], attempt_id))
        if cost > 0:
            await self._db.execute(
                """INSERT OR IGNORE INTO ibkr_etf_ledger
                   (model_alias, kind, pnl, source_ref)
                   VALUES (?, 'intelligence', ?, ?)""",
                (self._model_alias, -cost, f"openai:{attempt_id}"))
        await self._db.commit()
        return cost

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout),
                headers={"Authorization": f"Bearer {self._api_key}",
                         "Content-Type": "application/json",
                         "User-Agent": "auramaur-ibkr-etf/1.0"})
        return self._session

    @staticmethod
    def _evidence_text(evidence) -> str:
        parts = []
        for item in evidence[:12]:
            published = getattr(item, "published_at", "")
            parts.append(
                f"SOURCE: {getattr(item, 'source', '')}\n"
                f"DATE: {published}\nTITLE: {getattr(item, 'title', '')}\n"
                f"CONTENT: {getattr(item, 'content', '')[:1200]}"
            )
        return "\n\n".join(parts) or "No usable evidence was retrieved."

    async def analyze(self, market, evidence, cache=None) -> ETFAnalysis | None:
        del cache
        if not self._api_key:
            if not self._warned_missing_key:
                log.warning("openai_etf.missing_api_key")
                self._warned_missing_key = True
            return None
        started = time.monotonic()
        attempt_id = await self._start_attempt()
        prompt = (
            f"QUESTION: {market.question}\n"
            f"CONTEXT: {market.description}\n\n"
            f"EVIDENCE:\n{self._evidence_text(evidence)}"
        )
        payload = {
            "model": self.model,
            "instructions": self._instructions,
            "input": prompt,
            "reasoning": {"effort": self.effort},
            "text": {"format": {"type": "json_schema", "name": "etf_forecast",
                                "strict": True, "schema": _SCHEMA}},
            "store": False,
            "max_output_tokens": 700,
        }
        session = await self._get_session()
        try:
            async with session.post(self._URL, json=payload) as response:
                data = await response.json()
                if response.status >= 400:
                    await self._finish_attempt(
                        attempt_id, "api_error", started, data,
                        str(data.get("error", {})))
                    log.warning("openai_etf.api_error", model=self.model,
                                status=response.status,
                                error=str(data.get("error", {}))[:200])
                    return None
        except Exception as exc:  # noqa: BLE001
            await self._finish_attempt(
                attempt_id, "request_error", started, error=str(exc))
            log.warning("openai_etf.request_error", model=self.model,
                        error=str(exc)[:160])
            return None
        raw = None
        for output in data.get("output", []):
            if output.get("type") != "message":
                continue
            for content in output.get("content", []):
                if content.get("type") == "refusal":
                    await self._finish_attempt(attempt_id, "refusal", started, data)
                    log.warning("openai_etf.refusal", model=self.model)
                    return None
                if content.get("type") == "output_text":
                    raw = content.get("text")
        if not raw:
            await self._finish_attempt(attempt_id, "empty", started, data)
            return None
        try:
            parsed = json.loads(raw)
            result = ETFAnalysis(
                probability=float(parsed["probability"]),
                confidence=str(parsed["confidence"]),
                thesis=str(parsed["thesis"]),
                key_risks=tuple(str(x) for x in parsed["key_risks"]),
            )
            cost = await self._finish_attempt(attempt_id, "completed", started, data)
            return ETFAnalysis(
                probability=result.probability, confidence=result.confidence,
                thesis=result.thesis, key_risks=result.key_risks,
                intelligence_cost_usd=cost)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            await self._finish_attempt(
                attempt_id, "parse_error", started, data, str(exc))
            log.warning("openai_etf.parse_error", model=self.model,
                        error=str(exc)[:120])
            return None

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
