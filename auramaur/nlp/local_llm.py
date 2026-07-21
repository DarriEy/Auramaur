"""Shared client for the local Ollama LLM tier.

One module-level singleton serves every consumer (distiller, triage, ensemble
arm) so a single semaphore serializes access to the GPU. The tier is strictly
evidence-side and hard fail-open: every error path returns ``None`` and the
caller proceeds exactly as if the feature were off for that call.

A small circuit breaker keeps a dead endpoint from stacking per-call timeouts
onto latency-sensitive loops (the news reactor runs every ~60s): after three
consecutive transport failures the client returns ``None`` instantly until the
cooldown lapses.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Any

import aiohttp
import structlog

log = structlog.get_logger()

_RETRY_DELAYS = [2.0, 5.0]
_BREAKER_THRESHOLD = 3


@asynccontextmanager
async def write_txn(db: Any):
    """Write inside ``db.transaction()`` when the helper exists (the
    legacy-writer migration branch), else fall back to main's current
    execute + commit convention. Keeps this tier mergeable either side."""
    txn = getattr(db, "transaction", None)
    if txn is not None:
        async with txn():
            yield
    else:
        yield
        await db.commit()


class LocalLLMClient:
    """aiohttp client for Ollama's /api/chat with JSON-constrained output."""

    def __init__(self, settings: Any, db: Any | None = None) -> None:
        cfg = settings.local_llm
        self._cfg = cfg
        self._db = db
        self._session: aiohttp.ClientSession | None = None
        self._semaphore = asyncio.Semaphore(max(1, cfg.concurrency))
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0
        self._health_ok: bool | None = None
        self._health_checked_at = 0.0

    @property
    def available(self) -> bool:
        """Circuit-breaker state; no I/O."""
        return time.monotonic() >= self._breaker_open_until

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._cfg.timeout_seconds),
                headers={"Content-Type": "application/json"})
        return self._session

    async def _request(self, payload: dict, timeout: float) -> tuple[int, dict]:
        """Single HTTP round-trip. Separate method so tests stub it."""
        session = await self._get_session()
        url = f"{self._cfg.base_url.rstrip('/')}/api/chat"
        async with session.post(
                url, json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            data = await response.json(content_type=None)
            return response.status, data

    async def generate_json(
        self,
        prompt: str,
        *,
        system: str = "",
        schema: dict | None = None,
        purpose: str = "generic",
        max_tokens: int = 800,
        temperature: float = 0.0,
        seed: int | None = None,
        timeout: float | None = None,
    ) -> dict | None:
        """One JSON-constrained completion. None on any failure (fail-open)."""
        if not self._cfg.enabled:
            return None
        if not self.available:
            return None
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self._cfg.model,
            "messages": messages,
            "stream": False,
            # qwen3 is a thinking model; without this its <think> blocks burn
            # tokens and fight the JSON format constraint.
            "think": False,
            "format": schema if schema is not None else "json",
            "keep_alive": self._cfg.keep_alive,
            "options": {
                "num_ctx": self._cfg.num_ctx,
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if seed is not None:
            payload["options"]["seed"] = int(seed)
        effective_timeout = timeout or float(self._cfg.timeout_seconds)
        started = time.monotonic()
        status = "request_error"
        error = ""
        data: dict = {}
        for attempt in range(1 + len(_RETRY_DELAYS)):
            try:
                http_status, data = await asyncio.wait_for(
                    self._locked_request(payload, effective_timeout),
                    timeout=effective_timeout + 10)
                if http_status >= 500:
                    status, error = "api_error", f"http {http_status}"
                    if attempt < len(_RETRY_DELAYS):
                        await asyncio.sleep(_RETRY_DELAYS[attempt])
                        continue
                elif http_status >= 400:
                    status, error = "api_error", f"http {http_status}"
                else:
                    status = "ok"
                break
            except (TimeoutError, asyncio.TimeoutError):
                status, error = "timeout", "timeout"
                break  # a timed-out call already burned the budgeted wait
            except Exception as exc:  # noqa: BLE001
                status, error = "request_error", str(exc)[:120]
                if attempt < len(_RETRY_DELAYS):
                    await asyncio.sleep(_RETRY_DELAYS[attempt])
                    continue
                break

        result: dict | None = None
        if status == "ok":
            self._consecutive_failures = 0
            content = (data.get("message") or {}).get("content", "")
            try:
                result = json.loads(content)
                if not isinstance(result, dict):
                    raise ValueError("non-object JSON")
            except (ValueError, TypeError) as exc:
                status, error, result = "parse_error", str(exc)[:120], None
                log.warning("local_llm.parse_error", purpose=purpose,
                            error=error, content=str(content)[:160])
        elif status in ("request_error", "timeout"):
            self._consecutive_failures += 1
            if self._consecutive_failures >= _BREAKER_THRESHOLD:
                self._breaker_open_until = (
                    time.monotonic() + self._cfg.failure_cooldown_seconds)
                log.warning("local_llm.circuit_open",
                            cooldown_seconds=self._cfg.failure_cooldown_seconds,
                            failures=self._consecutive_failures)
            log.warning("local_llm.request_error", purpose=purpose,
                        status=status, error=error)

        await self._record_call(
            purpose=purpose, status=status, prompt_chars=len(prompt),
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
            duration_ms=int((time.monotonic() - started) * 1000),
            error=error)
        return result

    async def _locked_request(self, payload: dict, timeout: float) -> tuple[int, dict]:
        async with self._semaphore:
            return await self._request(payload, timeout)

    async def health(self) -> bool:
        """GET /api/version with a 5s timeout; result cached for 60s."""
        now = time.monotonic()
        if self._health_ok is not None and now - self._health_checked_at < 60:
            return self._health_ok
        try:
            session = await self._get_session()
            url = f"{self._cfg.base_url.rstrip('/')}/api/version"
            async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                self._health_ok = response.status == 200
        except Exception:  # noqa: BLE001
            self._health_ok = False
        self._health_checked_at = now
        return self._health_ok

    async def _record_call(self, *, purpose: str, status: str, prompt_chars: int,
                           prompt_tokens: int, output_tokens: int,
                           duration_ms: int, error: str) -> None:
        if self._db is None:
            return
        try:
            async with write_txn(self._db):
                await self._db.execute(
                    """INSERT INTO local_llm_calls
                       (purpose, model, status, prompt_chars, prompt_tokens,
                        output_tokens, duration_ms, error)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (purpose, self._cfg.model, status, prompt_chars,
                     prompt_tokens, output_tokens, duration_ms, error))
        except Exception as exc:  # noqa: BLE001 — stats must never break a caller
            log.debug("local_llm.stats_write_failed", error=str(exc)[:120])

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None


_client: LocalLLMClient | None = None


def get_client(settings: Any, db: Any | None = None) -> LocalLLMClient:
    """Process-wide singleton so one semaphore serializes the GPU."""
    global _client
    if _client is None:
        _client = LocalLLMClient(settings, db)
    return _client


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None
