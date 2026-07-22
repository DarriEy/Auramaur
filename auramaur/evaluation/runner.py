"""Deterministic, provider-agnostic shadow forecast orchestration.

This module deliberately has no imports from brokers, exchanges, or persistence.
An adapter only receives frozen episode data and returns model output plus telemetry.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    raise TypeError(f"episode payload contains unsupported value {type(value).__name__}")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


@dataclass(frozen=True)
class Episode:
    episode_id: str
    payload: Mapping[str, Any]
    payload_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.episode_id.strip():
            raise ValueError("episode_id is required")
        frozen = _freeze(self.payload)
        canonical = json.dumps(_plain(frozen), sort_keys=True, separators=(",", ":"))
        object.__setattr__(self, "payload", frozen)
        object.__setattr__(self, "payload_hash", hashlib.sha256(canonical.encode()).hexdigest())


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    model: str
    adapter: "ModelAdapter | AdapterCallable"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.arm_id or not self.model:
            raise ValueError("arm_id and model are required")
        object.__setattr__(self, "metadata", _freeze(self.metadata))


class ExplorationPolicy(str, Enum):
    SINGLE = "single"
    SAMPLES = "samples"
    SAMPLES_CRITIC = "samples_critic"


@dataclass(frozen=True)
class TreatmentSpec:
    treatment_id: str
    arm: ArmSpec
    policy: ExplorationPolicy = ExplorationPolicy.SINGLE
    sample_count: int = 1
    base_seed: int = 0

    def __post_init__(self) -> None:
        if not self.treatment_id:
            raise ValueError("treatment_id is required")
        if self.sample_count < 1:
            raise ValueError("sample_count must be positive")
        if self.policy is ExplorationPolicy.SINGLE and self.sample_count != 1:
            raise ValueError("single policy requires sample_count=1")


@dataclass(frozen=True)
class GenerationRequest:
    episode_id: str
    episode_hash: str
    episode_payload: Mapping[str, Any]
    treatment_id: str
    arm_id: str
    model: str
    stage: str
    sample_index: int | None
    seed: int
    candidate_forecasts: tuple["Forecast", ...] = ()


@dataclass(frozen=True)
class AdapterResponse:
    output: Mapping[str, Any] | str
    telemetry: Mapping[str, Any] = field(default_factory=dict)


class ModelAdapter(Protocol):
    def generate(self, request: GenerationRequest) -> Awaitable[AdapterResponse] | AdapterResponse: ...


AdapterCallable = Callable[[GenerationRequest], Awaitable[AdapterResponse] | AdapterResponse]


@dataclass(frozen=True)
class Forecast:
    prob_yes: float
    action: str
    confidence: float | None = None
    thesis: str | None = None


@dataclass(frozen=True)
class ForecastAttempt:
    stage: str
    sample_index: int | None
    seed: int
    forecast: Forecast | None
    telemetry: Mapping[str, Any]
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.forecast is not None


@dataclass(frozen=True)
class TreatmentResult:
    episode_id: str
    episode_hash: str
    treatment_id: str
    arm_id: str
    attempts: tuple[ForecastAttempt, ...]
    aggregate_forecast: Forecast | None
    final_forecast: Forecast | None

    @property
    def succeeded(self) -> bool:
        return self.final_forecast is not None


def _seed(spec: TreatmentSpec, episode: Episode, stage: str, index: int | None) -> int:
    material = f"{spec.base_seed}|{episode.episode_id}|{episode.payload_hash}|{spec.treatment_id}|{stage}|{index}"
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], "big")


def _parse(output: Mapping[str, Any] | str) -> Forecast:
    if isinstance(output, str):
        try:
            value = json.loads(output)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc.msg}") from exc
    else:
        value = output
    if not isinstance(value, Mapping):
        raise ValueError("output must be a JSON object")
    allowed = {"prob_yes", "action", "confidence", "thesis"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown fields: {sorted(unknown)}")
    if "prob_yes" not in value or "action" not in value:
        raise ValueError("prob_yes and action are required")
    probability = value["prob_yes"]
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise ValueError("prob_yes must be numeric")
    probability = float(probability)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("prob_yes must be finite and in [0, 1]")
    action = value["action"]
    if not isinstance(action, str) or action.upper() not in {"YES", "NO", "ABSTAIN"}:
        raise ValueError("action must be YES, NO, or ABSTAIN")
    confidence = value.get("confidence")
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("confidence must be numeric")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be finite and in [0, 1]")
    thesis = value.get("thesis")
    if thesis is not None and not isinstance(thesis, str):
        raise ValueError("thesis must be a string")
    return Forecast(probability, action.upper(), confidence, thesis)


def _aggregate(forecasts: Sequence[Forecast]) -> Forecast | None:
    if not forecasts:
        return None
    probability = math.fsum(item.prob_yes for item in forecasts) / len(forecasts)
    action = "YES" if probability > 0.5 else "NO" if probability < 0.5 else "ABSTAIN"
    return Forecast(probability, action)


class IntelligenceEvalRunner:
    """Run paired shadow treatments with globally bounded model concurrency."""

    def __init__(self, max_concurrency: int = 4):
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def run(
        self, episode: Episode, treatments: Sequence[TreatmentSpec]
    ) -> tuple[TreatmentResult, ...]:
        ids = [item.treatment_id for item in treatments]
        if len(ids) != len(set(ids)):
            raise ValueError("treatment_id values must be unique")
        # gather preserves input order, making persistence and replays deterministic.
        return tuple(await asyncio.gather(*(self._run_treatment(episode, t) for t in treatments)))

    async def _invoke(self, spec: TreatmentSpec, request: GenerationRequest) -> ForecastAttempt:
        queued_at = time.monotonic()
        queue_ms = 0
        try:
            async with self._semaphore:
                queue_ms = int((time.monotonic() - queued_at) * 1000)
                adapter = spec.arm.adapter
                call = adapter.generate if hasattr(adapter, "generate") else adapter
                response = call(request)
                if inspect.isawaitable(response):
                    response = await response
            if not isinstance(response, AdapterResponse):
                raise TypeError("adapter must return AdapterResponse")
            telemetry_values = dict(response.telemetry)
            telemetry_values["queue_ms"] = queue_ms
            telemetry = _freeze(telemetry_values)
            forecast = _parse(response.output)
            return ForecastAttempt(request.stage, request.sample_index, request.seed, forecast, telemetry)
        except Exception as exc:  # Model, transport, and parse failures are observations.
            return ForecastAttempt(
                request.stage,
                request.sample_index,
                request.seed,
                None,
                MappingProxyType({"queue_ms": queue_ms}),
                f"{type(exc).__name__}: {exc}",
            )

    async def _run_treatment(self, episode: Episode, spec: TreatmentSpec) -> TreatmentResult:
        count = 1 if spec.policy is ExplorationPolicy.SINGLE else spec.sample_count
        requests = [
            GenerationRequest(
                episode.episode_id,
                episode.payload_hash,
                episode.payload,
                spec.treatment_id,
                spec.arm.arm_id,
                spec.arm.model,
                "sample",
                index,
                _seed(spec, episode, "sample", index),
            )
            for index in range(count)
        ]
        attempts = list(await asyncio.gather(*(self._invoke(spec, item) for item in requests)))
        successes = tuple(item.forecast for item in attempts if item.forecast is not None)
        aggregate = _aggregate(successes)
        final = aggregate

        if spec.policy is ExplorationPolicy.SAMPLES_CRITIC and successes:
            critic_request = GenerationRequest(
                episode.episode_id,
                episode.payload_hash,
                episode.payload,
                spec.treatment_id,
                spec.arm.arm_id,
                spec.arm.model,
                "critic",
                None,
                _seed(spec, episode, "critic", None),
                successes,
            )
            critic = await self._invoke(spec, critic_request)
            attempts.append(critic)
            if critic.forecast is not None:
                final = critic.forecast

        return TreatmentResult(
            episode.episode_id,
            episode.payload_hash,
            spec.treatment_id,
            spec.arm.arm_id,
            tuple(attempts),
            aggregate,
            final,
        )
