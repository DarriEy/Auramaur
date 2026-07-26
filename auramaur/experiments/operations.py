"""Manifest loading and operator-facing experiment orchestration."""

from __future__ import annotations

import importlib
import inspect
import json
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Literal, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, model_validator

from auramaur.experiments.adapters import FeatureProposalAdapter
from auramaur.experiments.models import (
    ExperimentDefinition,
    MarketSnapshot,
    PortfolioSnapshot,
)
from auramaur.experiments.registry import ExperimentRegistry, RegisteredExperiment
from auramaur.experiments.specialized import (
    PackageSemantics,
    RegisteredSpecializedExperiment,
)


class ExperimentBinding(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["portable", "specialized"]
    implementation: str | None = None
    evaluator: str | None = None
    constructor_arguments: dict[str, Any] = Field(default_factory=dict)
    feature_name: str | None = None
    proposal_kind: str | None = None
    semantics: PackageSemantics | None = None
    fixed_arguments: dict[str, Any] = Field(default_factory=dict)
    package_id_field: str = "package_id"

    @model_validator(mode="after")
    def binding_is_complete(self) -> "ExperimentBinding":
        if self.kind == "portable" and not self.implementation:
            raise ValueError("portable binding requires implementation")
        if self.kind == "specialized" and not all((
            self.evaluator, self.feature_name, self.proposal_kind, self.semantics
        )):
            raise ValueError("specialized binding requires evaluator, feature, kind, semantics")
        return self


class ExperimentManifest(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema_version: Literal[1] = 1
    definition: ExperimentDefinition
    binding: ExperimentBinding
    snapshots: tuple[MarketSnapshot, ...] = Field(min_length=1)
    initial_portfolio: PortfolioSnapshot
    execution_model: dict[str, Any] | None = None


def load_manifest(path: str | Path) -> ExperimentManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExperimentManifest.model_validate(raw)


def bind_manifest(
    manifest: ExperimentManifest,
) -> RegisteredExperiment | RegisteredSpecializedExperiment:
    binding = manifest.binding
    if binding.kind == "portable":
        implementation_type = _load_symbol(binding.implementation or "")
        arguments = _coerce_arguments(
            implementation_type.__init__, binding.constructor_arguments
        )
        implementation = implementation_type(**arguments)
        return ExperimentRegistry().register(manifest.definition, implementation)
    evaluator = _load_symbol(binding.evaluator or "")
    adapter = FeatureProposalAdapter(
        feature_name=binding.feature_name or "",
        evaluator=evaluator,
        kind=binding.proposal_kind or "",
        semantics=binding.semantics or PackageSemantics.ASSET_ORDER,
        fixed_arguments=binding.fixed_arguments,
        package_id_field=binding.package_id_field,
    )
    return RegisteredSpecializedExperiment(manifest.definition, adapter)


def _load_symbol(reference: str):
    try:
        module_name, symbol_name = reference.split(":", 1)
    except ValueError as exc:
        raise ValueError("binding references must use module:object syntax") from exc
    if not module_name.startswith("auramaur.experiments.strategies."):
        raise ValueError("bindings may only load pure experiment strategy modules")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise ValueError(f"unknown binding symbol: {reference}") from exc


def _coerce_arguments(callable_object, values: dict[str, Any]) -> dict[str, Any]:
    hints = get_type_hints(callable_object)
    result = dict(values)
    for name, value in tuple(result.items()):
        annotation = hints.get(name)
        if annotation is not None and is_dataclass(annotation) and isinstance(value, dict):
            result[name] = annotation(**value)
    signature = inspect.signature(callable_object)
    unknown = set(result) - set(signature.parameters)
    if unknown:
        raise ValueError(f"unknown constructor arguments: {sorted(unknown)}")
    return result
