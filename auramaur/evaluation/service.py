"""Operational shadow collector for intelligence x exploration evaluation.

This module reads discovery data and calls a local model. It has deliberately
no broker, risk, exchange execution, or production portfolio dependencies.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import math
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
        self._claims_treatments = frozenset(
            item.name for item in self._cfg.treatments if item.claims_evidence)
        self._runner = IntelligenceEvalRunner(self._cfg.max_concurrency)

    async def run_once(self) -> int:
        cycle_started = datetime.now(timezone.utc)
        wall_started = time.monotonic()
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
        selected = await self._select_markets(eligible, cycle_started)
        expensive_n = math.ceil(len(selected) * self._cfg.expensive_fraction)
        informative = sorted(selected, key=self._information_priority)[:expensive_n]
        expensive_ids = {market.id for market in informative}
        semaphore = asyncio.Semaphore(self._cfg.market_concurrency)

        async def evaluate(market):
            async with semaphore:
                treatments = (self._treatments if market.id in expensive_ids
                              else self._treatments[:1])
                return await self._evaluate_market(market, treatments)

        results = await asyncio.gather(*(evaluate(market) for market in selected))
        recorded = sum(item[0] for item in results)
        attempts = sum(item[1] for item in results)
        failed = sum(item[2] for item in results)
        compute = sum(item[3] for item in results)
        completed = datetime.now(timezone.utc)
        metrics = dict(
            cycle_id=hashlib.sha256(cycle_started.isoformat().encode()).hexdigest(),
            started_at=cycle_started.isoformat(), completed_at=completed.isoformat(),
            eligible_markets=len(eligible), selected_markets=len(selected),
            unique_families=len({self._family(m) for m in selected}),
            forecasts=recorded, attempts=attempts, failed_attempts=failed,
            duration_ms=int((time.monotonic() - wall_started) * 1000),
            compute_seconds=compute,
        )
        await self._store.put_cycle(metrics)
        log.info("intelligence_eval.cycle", **metrics)
        return recorded

    async def _attach_claims(self, treatments, market_id: str):
        """Give claims_evidence arms this market's distilled claims.

        Evidence rides the REQUEST payload, never the frozen episode, so the
        episode hash (and therefore pairing) is identical across arms. An arm
        with no matched claims is dropped for that market — running it would
        duplicate the bare arm and dilute the paired comparison. Claims are
        no-lookahead by construction: only already-distilled rows exist.
        """
        if not self._claims_treatments or not any(
                t.treatment_id in self._claims_treatments for t in treatments):
            return treatments
        rows = await self._db.fetchall(
            """SELECT claim, source, event_date FROM distilled_claims
               WHERE markets_affected LIKE ?
               ORDER BY created_at DESC LIMIT 8""",
            (f'%"{market_id}"%',))
        claims = [{"claim": r["claim"], "source": r["source"],
                   "event_date": r["event_date"]} for r in rows or []]
        kept = []
        for spec in treatments:
            if spec.treatment_id not in self._claims_treatments:
                kept.append(spec)
            elif claims:
                kept.append(dataclasses.replace(
                    spec, extra_payload={"distilled_evidence": claims}))
        return tuple(kept)

    @staticmethod
    def _family(market) -> str:
        return market.neg_risk_market_id or market.id

    def _information_priority(self, market):
        end = market.end_date or datetime.max.replace(tzinfo=timezone.utc)
        days = (end - datetime.now(timezone.utc)).total_seconds() / 86400
        horizon_bucket = 0 if 0 <= days <= self._cfg.near_resolution_days else 1
        ambiguity = abs(float(market.outcome_yes_price) - 0.5)
        return (horizon_bucket, end, ambiguity, -float(market.volume or 0))

    async def _select_markets(self, eligible, now):
        """Select changed/novel markets, one per family, with category rotation."""
        latest = await self._store.latest_market_observations()
        candidates = []
        for market in eligible:
            venue = (market.exchange or "polymarket").lower()
            previous = latest.get((venue, market.id))
            if previous:
                observed = datetime.fromisoformat(previous["observed_at"])
                age_hours = (now - observed.astimezone(timezone.utc)).total_seconds() / 3600
                moved = abs(float(market.outcome_yes_price)
                            - float(previous["market_prob_yes"]))
                if (age_hours < self._cfg.reevaluate_after_hours
                        and moved < self._cfg.reprice_threshold):
                    continue
            candidates.append(market)

        candidates.sort(key=self._information_priority)
        families, categories, selected = set(), {}, []
        for market in candidates:
            family = self._family(market)
            if family in families:
                continue
            categories.setdefault(market.category or "unknown", []).append(market)
        queues = list(categories.values())
        while queues and len(selected) < self._cfg.markets_per_cycle:
            next_queues = []
            for queue in queues:
                if queue and len(selected) < self._cfg.markets_per_cycle:
                    market = queue.pop(0)
                    family = self._family(market)
                    if family not in families:
                        selected.append(market)
                        families.add(family)
                if queue:
                    next_queues.append(queue)
            queues = next_queues
        return selected

    async def _evaluate_market(self, market, treatments=None) -> tuple[int, int, int, float]:
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
        treatments = treatments or self._treatments
        treatments = await self._attach_claims(treatments, market.id)
        results = await self._runner.run(episode, treatments)
        written = 0
        attempt_count = failed_count = 0
        compute_seconds = 0.0
        for treatment, result in zip(treatments, results, strict=True):
            run_id = hashlib.sha256(
                f"{snapshot.episode_hash}:{treatment.treatment_id}".encode()).hexdigest()
            duration_ms = sum(int(attempt.telemetry.get("duration_ms", 0))
                              for attempt in result.attempts)
            compute_seconds += duration_ms / 1000
            attempt_count += len(result.attempts)
            failed_count += sum(not attempt.succeeded for attempt in result.attempts)
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
            for attempt in result.attempts:
                await self._store.put_attempt(run_id, snapshot.episode_hash, attempt)
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
        return written, attempt_count, failed_count, compute_seconds
