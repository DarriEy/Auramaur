"""Immutable, side-effect-free domain models for market experiments."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("value must contain only finite JSON data") from exc


class CanonicalPayload(BaseModel):
    """Deeply immutable JSON represented by its canonical encoding."""

    model_config = ConfigDict(frozen=True)
    encoded: str

    @model_validator(mode="before")
    @classmethod
    def encode_input(cls, value: Any) -> dict[str, str]:
        if isinstance(value, cls):
            return {"encoded": value.encoded}
        if isinstance(value, dict) and set(value) == {"encoded"}:
            return {"encoded": _canonical(json.loads(value["encoded"]))}
        return {"encoded": _canonical(value)}

    def decoded(self) -> Any:
        return json.loads(self.encoded)


class CapitalEligibility(str, Enum):
    RESEARCH_ONLY = "research_only"
    PAPER_ONLY = "paper_only"
    GRADUATABLE = "graduatable"
    STRUCTURAL_LIVE = "structural_live"


class ExperimentDefinition(BaseModel):
    """Deeply immutable, pre-registered hypothesis and evaluation contract."""

    model_config = ConfigDict(frozen=True)
    key: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    strategy_source: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    mechanism: str = Field(min_length=1)
    implementation_version: str = Field(min_length=1)
    parameters: CanonicalPayload = Field(
        default_factory=lambda: CanonicalPayload.model_validate({})
    )
    venues: frozenset[str] = Field(min_length=1)
    instruments: tuple[str, ...] = ()
    cadence_seconds: int | None = Field(default=None, gt=0)
    primary_metric: str = Field(min_length=1)
    baseline: str = Field(min_length=1)
    min_observations: int = Field(gt=0)
    holdout_days: int = Field(ge=0)
    max_drawdown: float = Field(ge=0, le=1)
    cost_model: str = Field(min_length=1)
    rejection_criteria: tuple[str, ...] = Field(min_length=1)
    capital_eligibility: CapitalEligibility = CapitalEligibility.RESEARCH_ONLY

    @field_validator("parameters", mode="before")
    @classmethod
    def canonicalize_parameters(cls, value: Any) -> CanonicalPayload:
        return CanonicalPayload.model_validate(value)

    @field_validator("max_drawdown")
    @classmethod
    def drawdown_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("max_drawdown must be finite")
        return value

    @property
    def lineage_id(self) -> str:
        """Hash the complete preregistration; any material change forks evidence."""
        return hashlib.sha256(self.registration_json.encode("utf-8")).hexdigest()

    @property
    def registration_json(self) -> str:
        """Canonical complete preregistration stored beside prospective evidence."""
        identity = {
            "key": self.key,
            "strategy_source": self.strategy_source,
            "hypothesis": self.hypothesis,
            "mechanism": self.mechanism,
            "implementation_version": self.implementation_version,
            "parameters": self.parameters.decoded(),
            "venues": sorted(self.venues),
            "instruments": list(self.instruments),
            "cadence_seconds": self.cadence_seconds,
            "primary_metric": self.primary_metric,
            "baseline": self.baseline,
            "min_observations": self.min_observations,
            "holdout_days": self.holdout_days,
            "max_drawdown": self.max_drawdown,
            "cost_model": self.cost_model,
            "rejection_criteria": list(self.rejection_criteria),
            "capital_eligibility": self.capital_eligibility.value,
        }
        return _canonical(identity)


class PricePoint(BaseModel):
    model_config = ConfigDict(frozen=True)
    instrument_id: str = Field(min_length=1)
    price: float = Field(gt=0)

    @field_validator("price")
    @classmethod
    def price_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("price must be finite")
        return value


class FeatureValue(BaseModel):
    """A value with both source observation and knowable-at timestamps."""

    model_config = ConfigDict(frozen=True)
    name: str = Field(min_length=1)
    payload: CanonicalPayload
    source: str = Field(min_length=1)
    observed_at: datetime
    available_at: datetime

    @field_validator("payload", mode="before")
    @classmethod
    def canonicalize_payload(cls, value: Any) -> CanonicalPayload:
        return CanonicalPayload.model_validate(value)

    @field_validator("observed_at", "available_at")
    @classmethod
    def timestamps_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("feature timestamps must be timezone-aware")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def availability_cannot_precede_observation(self) -> "FeatureValue":
        if self.available_at < self.observed_at:
            raise ValueError("feature available_at cannot precede observed_at")
        return self


class MarketSnapshot(BaseModel):
    """Information knowable at one event time."""

    model_config = ConfigDict(frozen=True)
    observed_at: datetime
    sequence: int = Field(ge=0)
    market_id: str = Field(min_length=1)
    venue: str = Field(min_length=1)
    prices: tuple[PricePoint, ...] = Field(min_length=1)
    features: tuple[FeatureValue, ...] = ()
    data_version: str = Field(min_length=1)

    @field_validator("observed_at")
    @classmethod
    def timestamp_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def snapshot_is_point_in_time(self) -> "MarketSnapshot":
        price_ids = [point.instrument_id for point in self.prices]
        if len(price_ids) != len(set(price_ids)):
            raise ValueError("snapshot contains duplicate instrument prices")
        feature_names = [feature.name for feature in self.features]
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("snapshot contains duplicate feature names")
        if any(feature.available_at > self.observed_at for feature in self.features):
            raise ValueError("snapshot contains a feature not yet available")
        return self

    def price(self, instrument_id: str) -> float:
        for point in self.prices:
            if point.instrument_id == instrument_id:
                return point.price
        raise KeyError(f"snapshot has no price for {instrument_id}")

    def feature(self, name: str) -> Any:
        for feature in self.features:
            if feature.name == name:
                return feature.payload.decoded()
        raise KeyError(f"snapshot has no feature {name}")


class PortfolioSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    observed_at: datetime
    cash: float = Field(ge=0)
    equity: float = Field(gt=0)
    positions: tuple[tuple[str, float], ...] = ()

    @field_validator("observed_at")
    @classmethod
    def timestamp_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        return value.astimezone(timezone.utc)

    @field_validator("cash", "equity")
    @classmethod
    def money_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("portfolio values must be finite")
        return value

    @field_validator("positions")
    @classmethod
    def positions_must_be_unique_and_finite(
        cls, value: tuple[tuple[str, float], ...]
    ) -> tuple[tuple[str, float], ...]:
        names = [name for name, _ in value]
        if len(names) != len(set(names)):
            raise ValueError("positions must have unique instruments")
        if any(not name or not math.isfinite(quantity) for name, quantity in value):
            raise ValueError("positions must have names and finite quantities")
        return value

    def quantities(self) -> dict[str, float]:
        return dict(self.positions)


class TargetPosition(BaseModel):
    """Desired exposure emitted by an experiment; never an order instruction."""

    model_config = ConfigDict(frozen=True)
    instrument_id: str = Field(min_length=1)
    target_quantity: float
    reference_price: float = Field(gt=0)
    rationale: str = Field(min_length=1)
    max_notional: float = Field(gt=0)

    @field_validator("target_quantity", "reference_price", "max_notional")
    @classmethod
    def values_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("target values must be finite")
        return value

    @model_validator(mode="after")
    def target_must_fit_declared_bound(self) -> "TargetPosition":
        if abs(self.target_quantity * self.reference_price) > self.max_notional + 1e-9:
            raise ValueError("target notional exceeds max_notional")
        return self


class ExperimentResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    lineage_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime: str
    observed_at: datetime
    event_sequence: int = Field(ge=0)
    market_id: str
    data_version: str
    execution_model_version: str | None = None
    targets: tuple[TargetPosition, ...] = ()
    net_pnl: float | None = None
    costs: float = Field(default=0.0, ge=0)
    metadata: CanonicalPayload = Field(
        default_factory=lambda: CanonicalPayload.model_validate({})
    )

    @field_validator("metadata", mode="before")
    @classmethod
    def canonicalize_metadata(cls, value: Any) -> CanonicalPayload:
        return CanonicalPayload.model_validate(value)

    @field_validator("net_pnl", "costs")
    @classmethod
    def results_must_be_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("result values must be finite")
        return value
