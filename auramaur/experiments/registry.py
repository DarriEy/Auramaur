"""Definition-to-implementation registry with immutable lineage semantics."""

from __future__ import annotations

from dataclasses import dataclass

from auramaur.experiments.models import ExperimentDefinition
from auramaur.experiments.protocols import Experiment


@dataclass(frozen=True)
class RegisteredExperiment:
    definition: ExperimentDefinition
    implementation: Experiment

    @property
    def lineage_id(self) -> str:
        return self.definition.lineage_id


class ExperimentRegistry:
    def __init__(self) -> None:
        self._by_lineage: dict[str, RegisteredExperiment] = {}
        self._lineage_by_key: dict[str, str] = {}

    def register(
        self, definition: ExperimentDefinition, implementation: Experiment
    ) -> RegisteredExperiment:
        lineage = definition.lineage_id
        prior = self._lineage_by_key.get(definition.key)
        if prior is not None and prior != lineage:
            raise ValueError(
                f"experiment key {definition.key!r} is already frozen as lineage {prior}"
            )
        existing = self._by_lineage.get(lineage)
        if existing is not None and existing.implementation is not implementation:
            raise ValueError("experiment lineage is already bound to another implementation")
        registered = RegisteredExperiment(definition, implementation)
        self._lineage_by_key[definition.key] = lineage
        self._by_lineage[lineage] = registered
        return registered

    def get(self, lineage_id: str) -> RegisteredExperiment:
        try:
            return self._by_lineage[lineage_id]
        except KeyError as exc:
            raise KeyError(f"unknown experiment lineage: {lineage_id}") from exc

    def get_by_key(self, key: str) -> RegisteredExperiment:
        try:
            return self.get(self._lineage_by_key[key])
        except KeyError as exc:
            raise KeyError(f"unknown experiment key: {key}") from exc

    def experiments(self) -> tuple[RegisteredExperiment, ...]:
        return tuple(self._by_lineage[key] for key in sorted(self._by_lineage))
