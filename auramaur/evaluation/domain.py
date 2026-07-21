"""Immutable domain records for prospective intelligence evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FrozenRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


class EpisodeSnapshot(FrozenRecord):
    venue: str
    market_id: str
    event_family: str = ""
    observed_at: datetime
    market_prob_yes: float = Field(ge=0, le=1)
    question: str
    rules: str = ""
    yes_bid: float | None = Field(default=None, ge=0, le=1)
    yes_ask: float | None = Field(default=None, ge=0, le=1)
    bid_depth: float = Field(default=0, ge=0)
    ask_depth: float = Field(default=0, ge=0)
    evidence_cutoff: datetime
    evidence_ids: tuple[str, ...] = ()
    context: dict[str, Any] = Field(default_factory=dict)

    _observed_utc = field_validator("observed_at")(_utc)
    _cutoff_utc = field_validator("evidence_cutoff")(_utc)

    def canonical_json(self) -> str:
        def normalize(value: Any) -> Any:
            if isinstance(value, datetime):
                return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [normalize(item) for item in value]
            return value
        return json.dumps(normalize(self.model_dump()), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @property
    def episode_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class RunStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class EvaluationRun(FrozenRecord):
    run_id: str
    arm_name: str
    model: str
    quantization: str = ""
    exploration_policy: str
    seed: int = 0
    prompt_version: str
    output_schema_version: str
    status: RunStatus = RunStatus.RUNNING
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    duration_ms: int = Field(default=0, ge=0)
    compute_seconds: float = Field(default=0, ge=0)
    error: str = ""
    started_at: datetime
    completed_at: datetime | None = None

    _started_utc = field_validator("started_at")(_utc)
    _completed_utc = field_validator("completed_at")(
        lambda value: None if value is None else _utc(value))


class EvaluationForecast(FrozenRecord):
    forecast_id: str
    run_id: str
    episode_hash: str
    prob_yes: float = Field(ge=0, le=1)
    action: Literal["YES", "NO", "ABSTAIN"]
    min_acceptable_price: float | None = Field(default=None, ge=0, le=1)
    max_acceptable_price: float | None = Field(default=None, ge=0, le=1)
    thesis: str = ""
    uncertainty: float | None = Field(default=None, ge=0, le=1)
    evidence_ids: tuple[str, ...] = ()


class EvaluationOutcome(FrozenRecord):
    episode_hash: str
    outcome: Literal[0, 1]
    resolved_at: datetime
    source: str = ""

    _resolved_utc = field_validator("resolved_at")(_utc)
