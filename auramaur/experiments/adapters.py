"""Generic adapters from pure dataclass proposal functions to package runtimes."""

from __future__ import annotations

import inspect
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Callable, get_type_hints

from pydantic import BaseModel

from auramaur.experiments.models import MarketSnapshot, PortfolioSnapshot
from auramaur.experiments.specialized import PackageSemantics, ProposalPackage


class FeatureProposalAdapter:
    """Call a pure proposal function using one JSON feature as keyword arguments."""

    def __init__(
        self,
        *,
        feature_name: str,
        evaluator: Callable[..., Any],
        kind: str,
        semantics: PackageSemantics,
        fixed_arguments: dict[str, Any] | None = None,
        package_id_field: str = "package_id",
    ) -> None:
        self.feature_name = feature_name
        self.evaluator = evaluator
        self.kind = kind
        self.semantics = semantics
        self.fixed_arguments = dict(fixed_arguments or {})
        self.package_id_field = package_id_field

    async def evaluate_package(
        self, snapshot: MarketSnapshot, portfolio: PortfolioSnapshot
    ) -> ProposalPackage | None:
        del portfolio
        payload = snapshot.feature(self.feature_name)
        if not isinstance(payload, dict):
            raise TypeError(f"{self.feature_name} must contain an object")
        arguments = {**payload, **self.fixed_arguments}
        hints = get_type_hints(self.evaluator)
        for name, value in tuple(arguments.items()):
            annotation = hints.get(name)
            if annotation is not None and is_dataclass(annotation) and isinstance(value, dict):
                arguments[name] = annotation(**value)
        proposal = self.evaluator(**arguments)
        if inspect.isawaitable(proposal):
            proposal = await proposal
        proposal = getattr(proposal, "proposal", proposal)
        if proposal is None:
            return None
        encoded = _json_value(proposal)
        package_id = str(
            encoded.get(self.package_id_field)
            or encoded.get("market_id")
            or encoded.get("symbol")
            or snapshot.market_id
        )
        return ProposalPackage(
            package_id=package_id,
            kind=self.kind,
            semantics=self.semantics,
            payload=encoded,
        )


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _json_value(value.model_dump(mode="json"))
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value
