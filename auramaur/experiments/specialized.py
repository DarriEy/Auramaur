"""Semantics-preserving runtimes for non-TargetPosition experiment contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Iterable, Protocol

from pydantic import BaseModel, ConfigDict, Field

from auramaur.experiments.models import (
    CanonicalPayload,
    ExperimentDefinition,
    MarketSnapshot,
    PortfolioSnapshot,
)


class PackageSemantics(str, Enum):
    ALL_OR_NONE = "all_or_none"
    COUPLED_QUOTES = "coupled_quotes"
    ROUTING = "routing"
    MANUAL_ACTION = "manual_action"
    ASSET_ORDER = "asset_order"


class ProposalPackage(BaseModel):
    """One indivisible specialized proposal, never an execution instruction."""

    model_config = ConfigDict(frozen=True)
    package_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    semantics: PackageSemantics
    payload: CanonicalPayload


class SpecializedExperiment(Protocol):
    async def evaluate_package(
        self, snapshot: MarketSnapshot, portfolio: PortfolioSnapshot
    ) -> ProposalPackage | None: ...


class SpecializedResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    lineage_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime: str
    observed_at: datetime
    event_sequence: int = Field(ge=0)
    market_id: str
    data_version: str
    proposal: ProposalPackage | None = None


@dataclass(frozen=True)
class RegisteredSpecializedExperiment:
    definition: ExperimentDefinition
    implementation: SpecializedExperiment

    @property
    def lineage_id(self) -> str:
        return self.definition.lineage_id


async def _evaluate_package(
    registered: RegisteredSpecializedExperiment,
    snapshot: MarketSnapshot,
    portfolio: PortfolioSnapshot,
    runtime: str,
) -> SpecializedResult:
    if portfolio.observed_at > snapshot.observed_at:
        raise ValueError("portfolio snapshot cannot come from the future")
    proposal = await registered.implementation.evaluate_package(snapshot, portfolio)
    return SpecializedResult(
        lineage_id=registered.lineage_id,
        runtime=runtime,
        observed_at=snapshot.observed_at,
        event_sequence=snapshot.sequence,
        market_id=snapshot.market_id,
        data_version=snapshot.data_version,
        proposal=proposal,
    )


@dataclass
class InMemoryPackageSink:
    results: list[SpecializedResult] = field(default_factory=list)

    async def record(self, result: SpecializedResult) -> None:
        self.results.append(result)


class SpecializedShadowRuntime:
    def __init__(self, sink: InMemoryPackageSink) -> None:
        self.sink = sink

    async def run(
        self,
        registered: RegisteredSpecializedExperiment,
        snapshot: MarketSnapshot,
        portfolio: PortfolioSnapshot,
    ) -> SpecializedResult:
        result = await _evaluate_package(registered, snapshot, portfolio, "shadow")
        await self.sink.record(result)
        return result


@dataclass(frozen=True)
class SpecializedReplayReport:
    lineage_id: str
    results: tuple[SpecializedResult, ...]


class SpecializedReplayRuntime:
    """Chronological proposal replay with deliberately no invented fill model."""

    async def run(
        self,
        registered: RegisteredSpecializedExperiment,
        snapshots: Iterable[MarketSnapshot],
        portfolio: PortfolioSnapshot,
    ) -> SpecializedReplayReport:
        events = list(snapshots)
        ordered = sorted(events, key=lambda item: (item.observed_at, item.sequence))
        if events != ordered:
            raise ValueError("replay snapshots must be in chronological sequence order")
        identities = [(item.observed_at, item.sequence) for item in events]
        if len(identities) != len(set(identities)):
            raise ValueError("replay snapshots contain duplicate event identities")
        results = tuple([
            await _evaluate_package(registered, item, portfolio, "specialized_replay")
            for item in events
        ])
        return SpecializedReplayReport(registered.lineage_id, results)
