"""Small, auditable risk primitives for the IBKR paper experiments."""

from __future__ import annotations

import math
from statistics import fmean, pstdev


def closes_from_bars(bars) -> list[float]:
    return [float(close) for _, close in bars if close is not None and float(close) > 0]


def log_returns(closes: list[float]) -> list[float]:
    return [math.log(b / a) for a, b in zip(closes, closes[1:]) if a > 0 and b > 0]


def annualized_volatility(closes: list[float], periods: int = 252) -> float | None:
    returns = log_returns(closes)
    if len(returns) < 20:
        return None
    vol = pstdev(returns) * math.sqrt(periods)
    return vol if math.isfinite(vol) and vol > 0 else None


def normalized_momentum(closes: list[float], horizons=(20, 60, 120),
                        periods: int = 252) -> float | None:
    """Mean horizon return divided by its forecast standard deviation.

    Missing long horizons are ignored, allowing a safe warm-up at 20 sessions.
    The result is dimensionless and therefore comparable across asset classes.
    """
    vol = annualized_volatility(closes, periods)
    if vol is None:
        return None
    scores = []
    for horizon in horizons:
        if len(closes) <= horizon:
            continue
        ret = math.log(closes[-1] / closes[-horizon - 1])
        forecast_sigma = vol * math.sqrt(horizon / periods)
        if forecast_sigma > 0:
            scores.append(ret / forecast_sigma)
    return fmean(scores) if scores else None


def stop_distance(price: float, annual_vol: float, stop_vol_multiple: float,
                  floor_pct: float) -> float:
    daily_sigma = price * annual_vol / math.sqrt(252)
    return max(price * floor_pct / 100, daily_sigma * stop_vol_multiple)


def risk_quantity(risk_budget_usd: float, stop_distance_price: float,
                  multiplier: float, fx_to_usd: float, *, fractional: bool) -> float:
    unit_risk = stop_distance_price * multiplier * fx_to_usd
    if unit_risk <= 0 or risk_budget_usd <= 0:
        return 0.0
    raw = risk_budget_usd / unit_risk
    return math.floor(raw * 10_000) / 10_000 if fractional else float(math.floor(raw))


def adverse_fill(bid: float, ask: float, side: str, slippage_bps: float) -> float:
    """Cross the spread and add a conservative, deterministic impact floor."""
    if side == "BUY":
        return ask * (1 + slippage_bps / 10_000)
    return bid * (1 - slippage_bps / 10_000)
