"""Leakage-resistant walk-forward helpers and metrics for strategy research."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class WalkForwardFold:
    train: tuple
    test: tuple


@dataclass(frozen=True)
class StrategyObservation:
    timestamp: datetime
    equity: float
    net_pnl: float | None = None
    turnover: float = 0.0
    gross_exposure: float = 0.0


@dataclass(frozen=True)
class StrategyEvaluation:
    observations: int
    cagr: float | None
    annualized_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    turnover: float
    profit_factor: float | None
    hit_rate: float | None
    tail_loss: float
    average_gross_exposure: float


def event_unique(rows: Sequence[T], event_key: Callable[[T], str]) -> list[T]:
    """Keep the earliest observation per independent event."""
    seen: set[str] = set()
    out: list[T] = []
    for row in rows:
        key = event_key(row)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def expanding_walk_forward(
    rows: Sequence[T],
    *,
    timestamp: Callable[[T], datetime],
    event_key: Callable[[T], str],
    min_train: int = 100,
    test_size: int = 25,
    event_end: Callable[[T], datetime] | None = None,
    embargo: timedelta = timedelta(0),
) -> list[WalkForwardFold]:
    """Chronological expanding-window folds with event-disjoint test sets."""
    if min_train < 1 or test_size < 1:
        raise ValueError("min_train and test_size must be positive")
    if embargo < timedelta(0):
        raise ValueError("embargo must be non-negative")
    ordered = event_unique(sorted(rows, key=timestamp), event_key)
    folds: list[WalkForwardFold] = []
    cursor = min_train
    while cursor < len(ordered):
        test = ordered[cursor : cursor + test_size]
        if not test:
            break
        train = ordered[:cursor]
        if event_end is not None:
            cutoff = timestamp(test[0]) - embargo
            train = [x for x in train if event_end(x) < cutoff]
        train_events = {event_key(x) for x in train}
        test = [x for x in test if event_key(x) not in train_events]
        if test:
            folds.append(WalkForwardFold(tuple(train), tuple(test)))
        cursor += test_size
    return folds


def evaluate_strategy(
    observations: Sequence[StrategyObservation],
    *,
    annual_periods: int = 252,
    risk_free_rate: float = 0.0,
    tail_probability: float = 0.05,
) -> StrategyEvaluation:
    """Calculate generic portfolio metrics from chronological net observations."""
    if annual_periods < 1:
        raise ValueError("annual_periods must be positive")
    if not 0 < tail_probability <= 1:
        raise ValueError("tail_probability must be in (0, 1]")
    points = sorted(observations, key=lambda x: x.timestamp)
    if any(not math.isfinite(p.equity) or p.equity <= 0 for p in points):
        raise ValueError("equity must be finite and positive")
    if any(p.turnover < 0 or p.gross_exposure < 0 for p in points):
        raise ValueError("turnover and gross_exposure must be non-negative")
    returns = [points[i].equity / points[i - 1].equity - 1 for i in range(1, len(points))]
    cagr = None
    if len(points) >= 2:
        years = (points[-1].timestamp - points[0].timestamp).total_seconds() / (365.2425 * 86400)
        if years > 0:
            cagr = (points[-1].equity / points[0].equity) ** (1 / years) - 1
    mean = sum(returns) / len(returns) if returns else 0.0
    variance = (
        sum((x - mean) ** 2 for x in returns) / (len(returns) - 1) if len(returns) >= 2 else 0.0
    )
    std = math.sqrt(variance)
    vol = std * math.sqrt(annual_periods)
    rf = (1 + risk_free_rate) ** (1 / annual_periods) - 1
    sharpe = (mean - rf) / std * math.sqrt(annual_periods) if std else 0.0
    down = [min(0.0, x - rf) for x in returns]
    dev = math.sqrt(sum(x * x for x in down) / len(down)) if down else 0.0
    sortino = (mean - rf) / dev * math.sqrt(annual_periods) if dev else 0.0
    peak = max_dd = 0.0
    for p in points:
        peak = max(peak, p.equity)
        max_dd = max(max_dd, (peak - p.equity) / peak)
    pnls = [p.net_pnl for p in points if p.net_pnl is not None]
    gains = sum(x for x in pnls if x > 0)
    losses = -sum(x for x in pnls if x < 0)
    pf = gains / losses if losses else (math.inf if gains else None)
    hit = sum(x > 0 for x in pnls) / len(pnls) if pnls else None
    avg = sum(p.equity for p in points) / len(points) if points else 0.0
    turnover = sum(p.turnover for p in points) / avg if avg else 0.0
    exposure = sum(p.gross_exposure / p.equity for p in points) / len(points) if points else 0.0
    n = max(1, int(len(returns) * tail_probability)) if returns else 0
    tail = max(0.0, -sum(sorted(returns)[:n]) / n) if n else 0.0
    return StrategyEvaluation(
        len(points), cagr, vol, sharpe, sortino, max_dd, turnover, pf, hit, tail, exposure
    )
