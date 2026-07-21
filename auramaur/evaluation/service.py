"""Operational shadow collector for intelligence x exploration evaluation.

This module reads discovery data and calls a local model. It has deliberately
no broker, risk, exchange execution, or production portfolio dependencies.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from datetime import datetime, timezone

import structlog

from auramaur.evaluation.domain import (
    EpisodeSnapshot, EvaluationForecast, EvaluationRun, RunStatus,
)
from auramaur.evaluation.runner import (
    AdapterResponse, ArmSpec, Episode, ExplorationPolicy, GenerationRequest,
    IntelligenceEvalRunner, TreatmentSpec,
)
from auramaur.evaluation.store import EvaluationStore
from auramaur.evaluation.evidence import ForecastScoreMaterializer

log = structlog.get_logger()


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value

_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "prob_yes": {"type": "number", "minimum": 0, "maximum": 1},
        "action": {"type": "string", "enum": ["YES", "NO", "ABSTAIN"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "thesis": {"type": "string"},
    },
    "required": ["prob_yes", "action", "thesis"],
    "additionalProperties": False,
}


class LocalForecastAdapter:
    def __init__(self, client, prompt_version: str) -> None:
        self._client = client
        self._prompt_version = prompt_version

    async def generate(self, request: GenerationRequest) -> AdapterResponse:
        candidates = ""
        if request.candidate_forecasts:
            candidates = "\nINDEPENDENT FORECASTS TO CRITIQUE:\n" + json.dumps([
                {"prob_yes": item.prob_yes, "action": item.action,
                 "confidence": item.confidence, "thesis": item.thesis}
                for item in request.candidate_forecasts
            ], sort_keys=True)
        prompt = (
            f"PROMPT VERSION: {self._prompt_version}\n"
            "You are producing a prospective binary-event forecast. Use ONLY "
            "the frozen snapshot below. Do not use knowledge or events after "
            "evidence_cutoff. ABSTAIN when the snapshot is insufficient. "
            "Return strict JSON matching the supplied schema.\n"
            f"STAGE: {request.stage}\nSNAPSHOT:\n"
            f"{json.dumps(_plain(request.episode_payload), sort_keys=True)}"
            f"{candidates}"
        )
        started = time.monotonic()
        output = await self._client.generate_json(
            prompt, schema=_OUTPUT_SCHEMA, purpose="intelligence_eval",
            temperature=0.35 if request.stage == "sample" else 0.0,
            seed=request.seed,
        )
        if output is None:
            raise RuntimeError("local model returned no forecast")
        return AdapterResponse(output, {
            "duration_ms": int((time.monotonic() - started) * 1000),
            "stage": request.stage,
        })


class IntelligenceEvalService:
    """Collect paired, timestamp-locked forecasts without placing trades."""

    def __init__(self, db, settings, discovery, local_client) -> None:
        self._db = db
        self._cfg = settings.intelligence_eval
        self._local_cfg = settings.local_llm
        self._discovery = discovery
        self._store = EvaluationStore(db)
        self._score_materializer = ForecastScoreMaterializer(db)
        adapter = LocalForecastAdapter(local_client, self._cfg.prompt_version)
        arm = ArmSpec("local", self._local_cfg.model, adapter, {
            "num_ctx": self._local_cfg.num_ctx,
        })
        self._treatments = tuple(TreatmentSpec(
            item.name, arm, ExplorationPolicy(item.policy), item.samples, item.base_seed,
        ) for item in self._cfg.treatments)
        self._runner = IntelligenceEvalRunner(self._cfg.max_concurrency)

    async def run_once(self) -> int:
        # ResolutionTracker owns venue truth. Refreshing here makes newly
        # canonicalized results visible even during quiet/no-resolution cycles.
        await self._score_materializer.refresh()
        markets = await self._discovery.get_markets(
            active=True, limit=self._cfg.scan_limit)
        eligible = [market for market in markets if (
            market.active and not market.closed
            and market.liquidity >= self._cfg.min_liquidity
            and 0 < market.outcome_yes_price < 1
        )]
        eligible.sort(key=lambda market: market.volume, reverse=True)
        recorded = 0
        for market in eligible[:self._cfg.markets_per_cycle]:
            recorded += await self._evaluate_market(market)
        log.info("intelligence_eval.cycle", markets=len(eligible), forecasts=recorded)
        return recorded

    async def _evaluate_market(self, market) -> int:
        now = datetime.now(timezone.utc)
        spread = max(0.0, float(market.spread or 0))
        mid = float(market.outcome_yes_price)
        snapshot = EpisodeSnapshot(
            venue=market.exchange or "polymarket", market_id=market.id,
            event_family=market.neg_risk_market_id or market.id,
            observed_at=now, evidence_cutoff=now, market_prob_yes=mid,
            question=market.question, rules=market.description or "",
            yes_bid=max(0.0, mid - spread / 2),
            yes_ask=min(1.0, mid + spread / 2),
            context={
                "category": market.category, "volume": market.volume,
                "liquidity": market.liquidity,
                "end_date": market.end_date.isoformat() if market.end_date else None,
            },
        )
        await self._store.put_episode(snapshot)
        payload = json.loads(snapshot.canonical_json())
        episode = Episode(snapshot.episode_hash, payload)
        results = await self._runner.run(episode, self._treatments)
        written = 0
        for treatment, result in zip(self._treatments, results, strict=True):
            run_id = hashlib.sha256(
                f"{snapshot.episode_hash}:{treatment.treatment_id}".encode()).hexdigest()
            duration_ms = sum(int(attempt.telemetry.get("duration_ms", 0))
                              for attempt in result.attempts)
            status = RunStatus.SUCCEEDED if result.succeeded else RunStatus.FAILED
            errors = "; ".join(attempt.error for attempt in result.attempts
                               if attempt.error)
            await self._store.put_run(EvaluationRun(
                run_id=run_id, arm_name=treatment.treatment_id,
                model=treatment.arm.model,
                quantization=str(treatment.arm.metadata.get("quantization", "")),
                exploration_policy=treatment.policy.value, seed=treatment.base_seed,
                prompt_version=self._cfg.prompt_version,
                output_schema_version=self._cfg.output_schema_version,
                status=status, duration_ms=duration_ms,
                compute_seconds=duration_ms / 1000, error=errors,
                started_at=now, completed_at=datetime.now(timezone.utc),
            ))
            if result.final_forecast is None:
                continue
            forecast_id = hashlib.sha256(f"{run_id}:forecast".encode()).hexdigest()
            forecast = result.final_forecast
            await self._store.put_forecast(EvaluationForecast(
                forecast_id=forecast_id, run_id=run_id,
                episode_hash=snapshot.episode_hash, prob_yes=forecast.prob_yes,
                action=forecast.action, thesis=forecast.thesis or "",
                uncertainty=None if forecast.confidence is None
                else 1 - forecast.confidence,
            ))
            written += 1
        return written
