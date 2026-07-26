"""Research, chronological replay, and shadow runtimes.

There is deliberately no live runtime in this package. Production adapters
remain beside the broker, where Auramaur's risk, graduation, kill-switch, and
triple-live gates can be enforced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from auramaur.experiments.models import (
    ExperimentResult,
    MarketSnapshot,
    PortfolioSnapshot,
)
from auramaur.experiments.protocols import ShadowSink
from auramaur.experiments.registry import RegisteredExperiment


def _validate_time(snapshot: MarketSnapshot, portfolio: PortfolioSnapshot) -> None:
    if portfolio.observed_at > snapshot.observed_at:
        raise ValueError("portfolio snapshot cannot come from the future")


async def _evaluate(
    registered: RegisteredExperiment,
    snapshot: MarketSnapshot,
    portfolio: PortfolioSnapshot,
    *,
    runtime: str,
) -> ExperimentResult:
    _validate_time(snapshot, portfolio)
    targets = tuple(await registered.implementation.evaluate(snapshot, portfolio))
    target_ids = [target.instrument_id for target in targets]
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("experiment emitted duplicate targets for one instrument")
    return ExperimentResult(
        lineage_id=registered.lineage_id,
        runtime=runtime,
        observed_at=snapshot.observed_at,
        event_sequence=snapshot.sequence,
        market_id=snapshot.market_id,
        data_version=snapshot.data_version,
        targets=targets,
    )


class ResearchRuntime:
    async def run(
        self,
        registered: RegisteredExperiment,
        snapshot: MarketSnapshot,
        portfolio: PortfolioSnapshot,
    ) -> ExperimentResult:
        return await _evaluate(registered, snapshot, portfolio, runtime="research")


@dataclass(frozen=True)
class LinearExecutionModel:
    """Deterministic immediate-fill model with explicit linear costs."""

    version: str
    fee_bps: float = 0.0
    slippage_bps: float = 0.0

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("execution model version is required")
        if not all(math.isfinite(x) and x >= 0 for x in (self.fee_bps, self.slippage_bps)):
            raise ValueError("execution costs must be finite and non-negative")


@dataclass(frozen=True)
class ReplayReport:
    lineage_id: str
    execution_model_version: str
    initial_equity: float
    final_equity: float
    results: tuple[ExperimentResult, ...]


class ReplayRuntime:
    """Chronologically evolve positions and cash across market events."""

    def __init__(self, execution_model: LinearExecutionModel) -> None:
        self.execution_model = execution_model

    async def run(
        self,
        registered: RegisteredExperiment,
        snapshots: Iterable[MarketSnapshot],
        initial_portfolio: PortfolioSnapshot,
    ) -> ReplayReport:
        events = list(snapshots)
        ordered = sorted(events, key=lambda event: (event.observed_at, event.sequence))
        if events != ordered:
            raise ValueError("replay snapshots must be in chronological sequence order")
        identities = [(event.observed_at, event.sequence) for event in events]
        if len(identities) != len(set(identities)):
            raise ValueError("replay snapshots contain duplicate event identities")

        cash = initial_portfolio.cash
        positions = initial_portfolio.quantities()
        last_prices: dict[str, float] = {}
        previous_equity = initial_portfolio.equity
        results: list[ExperimentResult] = []
        for snapshot in events:
            if snapshot.observed_at < initial_portfolio.observed_at:
                raise ValueError("replay event predates the initial portfolio")
            prices = {point.instrument_id: point.price for point in snapshot.prices}
            last_prices.update(prices)
            missing = set(positions) - set(last_prices)
            if missing:
                raise ValueError(
                    f"replay cannot value held instruments without prices: {sorted(missing)}"
                )
            marked_equity = cash + sum(
                quantity * last_prices.get(instrument, 0.0)
                for instrument, quantity in positions.items()
            )
            portfolio = PortfolioSnapshot(
                observed_at=snapshot.observed_at,
                cash=max(0.0, cash),
                equity=max(marked_equity, 1e-12),
                positions=tuple(sorted(positions.items())),
            )
            evaluated = await _evaluate(registered, snapshot, portfolio, runtime="replay")
            costs = 0.0
            turnover = 0.0
            for target in evaluated.targets:
                price = snapshot.price(target.instrument_id)
                if not math.isclose(target.reference_price, price, rel_tol=0, abs_tol=1e-12):
                    raise ValueError("target reference price does not match the replay event")
                current = positions.get(target.instrument_id, 0.0)
                delta = target.target_quantity - current
                direction = 1.0 if delta > 0 else -1.0
                fill_price = price * (
                    1.0 + direction * self.execution_model.slippage_bps / 10_000.0
                )
                notional = abs(delta * fill_price)
                fee = notional * self.execution_model.fee_bps / 10_000.0
                cash -= delta * fill_price + fee
                if cash < -1e-9:
                    raise ValueError("replay target requires unavailable cash")
                cash = max(0.0, cash)
                positions[target.instrument_id] = target.target_quantity
                turnover += notional
                costs += fee + abs(delta * (fill_price - price))
            final_equity = cash + sum(
                quantity * last_prices.get(instrument, 0.0)
                for instrument, quantity in positions.items()
            )
            result = ExperimentResult(
                lineage_id=evaluated.lineage_id,
                runtime=evaluated.runtime,
                observed_at=evaluated.observed_at,
                event_sequence=evaluated.event_sequence,
                market_id=evaluated.market_id,
                data_version=evaluated.data_version,
                execution_model_version=self.execution_model.version,
                targets=evaluated.targets,
                net_pnl=final_equity - previous_equity,
                costs=costs,
                metadata={"turnover": turnover, "equity": final_equity},
            )
            results.append(result)
            previous_equity = final_equity

        return ReplayReport(
            lineage_id=registered.lineage_id,
            execution_model_version=self.execution_model.version,
            initial_equity=initial_portfolio.equity,
            final_equity=previous_equity,
            results=tuple(results),
        )


@dataclass
class InMemoryShadowSink:
    results: list[ExperimentResult] = field(default_factory=list)

    async def record(self, result: object) -> None:
        if not isinstance(result, ExperimentResult):
            raise TypeError("shadow sink only accepts ExperimentResult")
        self.results.append(result)


class ShadowRuntime:
    def __init__(self, sink: ShadowSink) -> None:
        self.sink = sink

    async def run(
        self,
        registered: RegisteredExperiment,
        snapshot: MarketSnapshot,
        portfolio: PortfolioSnapshot,
    ) -> ExperimentResult:
        result = await _evaluate(registered, snapshot, portfolio, runtime="shadow")
        await self.sink.record(result)
        return result
