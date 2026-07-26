"""Behavioral boundaries shared by research, replay, and shadow runtimes."""

from __future__ import annotations

from typing import Protocol

from auramaur.experiments.models import MarketSnapshot, PortfolioSnapshot, TargetPosition


class Experiment(Protocol):
    async def evaluate(
        self,
        snapshot: MarketSnapshot,
        portfolio: PortfolioSnapshot,
    ) -> list[TargetPosition]: ...


class ShadowSink(Protocol):
    async def record(self, result: object) -> None: ...
