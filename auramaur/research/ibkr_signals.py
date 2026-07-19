"""Pure, point-in-time signal primitives for IBKR strategy research.

These functions do not fetch data, size positions, or place orders.  Callers must
provide an explicit ``as_of`` time and already-known observations.  Invalid,
future-dated, stale, or insufficient inputs raise :class:`SignalInputError`, so
an evaluator cannot silently turn incomplete data into a tradable signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from statistics import fmean, pstdev
from typing import Sequence


class SignalInputError(ValueError):
    """Input is unsafe or insufficient for a point-in-time signal."""


@dataclass(frozen=True)
class PricePoint:
    observed_at: datetime
    value: float


@dataclass(frozen=True)
class CurvePoint:
    expiry: datetime
    price: float


@dataclass(frozen=True)
class ResearchSignal:
    """A research direction: -1 short, 0 flat, or 1 long."""

    direction: int
    score: float
    rationale: str

    def __post_init__(self) -> None:
        if self.direction not in (-1, 0, 1):
            raise ValueError("direction must be -1, 0, or 1")
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")


def _utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise SignalInputError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _positive(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise SignalInputError(f"{name} must be finite and positive")
    return float(value)


def _finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise SignalInputError(f"{name} must be finite")
    return float(value)


def _prices(
    points: Sequence[PricePoint],
    *,
    as_of: datetime,
    minimum: int,
    max_age: timedelta | None,
    name: str,
) -> list[float]:
    cutoff = _utc(as_of, "as_of")
    if len(points) < minimum:
        raise SignalInputError(f"{name} requires at least {minimum} observations")
    times = [_utc(point.observed_at, f"{name}.observed_at") for point in points]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise SignalInputError(f"{name} timestamps must be strictly increasing")
    if times[-1] > cutoff:
        raise SignalInputError(f"{name} contains data after as_of")
    if max_age is not None:
        if max_age <= timedelta(0):
            raise SignalInputError("max_age must be positive")
        if cutoff - times[-1] > max_age:
            raise SignalInputError(f"{name} latest observation is stale")
    return [_positive(point.value, f"{name}.value") for point in points]


def _zscore(current: float, history: Sequence[float], name: str) -> float:
    sigma = pstdev(history)
    if sigma <= 0 or not math.isfinite(sigma):
        raise SignalInputError(f"{name} history has zero variance")
    return (current - fmean(history)) / sigma


def short_term_mean_reversion(
    closes: Sequence[PricePoint],
    *,
    as_of: datetime,
    return_lookback: int = 20,
    trend_lookback: int = 60,
    entry_z: float = 2.0,
    max_age: timedelta | None = None,
) -> ResearchSignal:
    """Fade an abnormal one-period move only in the prevailing price trend.

    The current log return is compared with the *preceding* return window.  A
    negative shock above ``entry_z`` is long only above the trailing trend mean;
    a positive shock is short only below it.  This separation avoids including
    the decision observation in its own normalization baseline.
    """
    if return_lookback < 2 or trend_lookback < 2:
        raise SignalInputError("lookbacks must be at least 2")
    entry_z = _positive(entry_z, "entry_z")
    needed = max(return_lookback + 2, trend_lookback + 1)
    values = _prices(closes, as_of=as_of, minimum=needed, max_age=max_age, name="closes")
    returns = [math.log(b / a) for a, b in zip(values, values[1:])]
    z = _zscore(returns[-1], returns[-return_lookback - 1 : -1], "return")
    trend_mean = fmean(values[-trend_lookback - 1 : -1])
    if z <= -entry_z and values[-1] > trend_mean:
        return ResearchSignal(1, -z, "negative shock within an uptrend")
    if z >= entry_z and values[-1] < trend_mean:
        return ResearchSignal(-1, z, "positive shock within a downtrend")
    return ResearchSignal(0, abs(z), "shock or trend filter not satisfied")


def relative_value_residual_zscore(
    dependent: Sequence[PricePoint],
    independent: Sequence[PricePoint],
    *,
    as_of: datetime,
    hedge_beta: float,
    intercept: float = 0.0,
    lookback: int = 60,
    entry_z: float = 2.0,
    max_age: timedelta | None = None,
) -> ResearchSignal:
    """Return the mean-reversion direction of a pre-fitted log-price spread.

    ``hedge_beta`` and ``intercept`` must have been fitted before ``as_of``.
    Direction describes the dependent leg; the independent leg has exposure
    ``-direction * hedge_beta``.  Both series must have identical timestamps.
    """
    if lookback < 2:
        raise SignalInputError("lookback must be at least 2")
    beta = _finite(hedge_beta, "hedge_beta")
    alpha = _finite(intercept, "intercept")
    entry_z = _positive(entry_z, "entry_z")
    needed = lookback + 1
    ys = _prices(dependent, as_of=as_of, minimum=needed, max_age=max_age, name="dependent")
    xs = _prices(independent, as_of=as_of, minimum=needed, max_age=max_age, name="independent")
    y_times = [_utc(point.observed_at, "dependent.observed_at") for point in dependent]
    x_times = [_utc(point.observed_at, "independent.observed_at") for point in independent]
    if y_times != x_times:
        raise SignalInputError("pair observations must have identical timestamps")
    residuals = [math.log(y) - alpha - beta * math.log(x) for y, x in zip(ys, xs)]
    z = _zscore(residuals[-1], residuals[-lookback - 1 : -1], "residual")
    direction = -1 if z >= entry_z else 1 if z <= -entry_z else 0
    return ResearchSignal(direction, abs(z), "residual mean reversion")


def futures_carry_trend(
    curve: Sequence[CurvePoint],
    closes: Sequence[PricePoint],
    *,
    as_of: datetime,
    trend_lookback: int = 60,
    min_annualized_carry: float = 0.0,
    max_age: timedelta | None = None,
) -> ResearchSignal:
    """Combine front/back futures curve carry with trailing price trend.

    Positive carry denotes backwardation (near price above far price).  A signal
    is emitted only when carry and trailing log-price trend agree.
    """
    if trend_lookback < 1:
        raise SignalInputError("trend_lookback must be positive")
    threshold = _finite(min_annualized_carry, "min_annualized_carry")
    if threshold < 0:
        raise SignalInputError("min_annualized_carry cannot be negative")
    cutoff = _utc(as_of, "as_of")
    if len(curve) < 2:
        raise SignalInputError("curve requires at least two contracts")
    expiries = [_utc(point.expiry, "curve.expiry") for point in curve]
    if any(right <= left for left, right in zip(expiries, expiries[1:])):
        raise SignalInputError("curve expiries must be strictly increasing")
    if expiries[0] <= cutoff:
        raise SignalInputError("curve contains an expired front contract")
    near = _positive(curve[0].price, "curve.near.price")
    far = _positive(curve[-1].price, "curve.far.price")
    years = (expiries[-1] - expiries[0]).total_seconds() / (365.25 * 86400)
    if years <= 0:
        raise SignalInputError("curve tenor must be positive")
    carry = math.log(near / far) / years
    values = _prices(
        closes, as_of=as_of, minimum=trend_lookback + 1, max_age=max_age, name="closes"
    )
    trend = math.log(values[-1] / values[-trend_lookback - 1])
    carry_side = 1 if carry > threshold else -1 if carry < -threshold else 0
    trend_side = 1 if trend > 0 else -1 if trend < 0 else 0
    direction = carry_side if carry_side == trend_side else 0
    return ResearchSignal(
        direction, min(abs(carry), abs(trend)), f"annualized_carry={carry:.6f}; trend={trend:.6f}"
    )


def fx_carry_trend(
    closes: Sequence[PricePoint],
    *,
    as_of: datetime,
    base_rate: float,
    quote_rate: float,
    trend_lookback: int = 60,
    min_rate_spread: float = 0.0,
    max_age: timedelta | None = None,
) -> ResearchSignal:
    """Combine BASE/QUOTE interest carry with spot trend.

    Rates are annual decimal yields known at ``as_of``.  Long BASE/QUOTE earns
    approximately ``base_rate - quote_rate`` before broker financing costs.
    """
    if trend_lookback < 1:
        raise SignalInputError("trend_lookback must be positive")
    spread = _finite(base_rate, "base_rate") - _finite(quote_rate, "quote_rate")
    threshold = _finite(min_rate_spread, "min_rate_spread")
    if threshold < 0:
        raise SignalInputError("min_rate_spread cannot be negative")
    values = _prices(
        closes, as_of=as_of, minimum=trend_lookback + 1, max_age=max_age, name="closes"
    )
    trend = math.log(values[-1] / values[-trend_lookback - 1])
    carry_side = 1 if spread > threshold else -1 if spread < -threshold else 0
    trend_side = 1 if trend > 0 else -1 if trend < 0 else 0
    direction = carry_side if carry_side == trend_side else 0
    return ResearchSignal(
        direction, min(abs(spread), abs(trend)), f"rate_spread={spread:.6f}; trend={trend:.6f}"
    )
