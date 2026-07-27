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


# Lookbacks the momentum signal is designed to blend. Exported so the data
# layer can size its fetch from the math instead of guessing: a request that
# returns fewer bars than min_closes_for_momentum() does not fail, it silently
# returns a *different* (or absent) signal.
MOMENTUM_HORIZONS = (20, 60, 120)

# annualized_volatility needs 20 log returns before it returns a number at all,
# and every momentum score is divided by it -- so 21 closes is the hard floor
# below which normalized_momentum is None regardless of horizon.
MIN_VOL_CLOSES = 21


def min_closes_for_momentum(horizons=MOMENTUM_HORIZONS) -> int:
    """Completed closes needed for the *full* signal, not the warm-up.

    The IBKR ETF arms fetched one month (20 bars) against these 120-session
    horizons for four days: 19 completed closes, one short of MIN_VOL_CLOSES,
    so every arm's momentum read was None and momentum_control -- whose whole
    view is this signal -- recorded zero forecasts (2026-07-27).
    """
    return max(MIN_VOL_CLOSES, max(horizons) + 1)


def normalized_momentum(closes: list[float], horizons=MOMENTUM_HORIZONS,
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


def horizon_up_rate(closes: list[float], horizon: int) -> float | None:
    """Unconditional P(close is higher `horizon` sessions later), from history.

    The benchmark an ETF arm's forecast has to beat. The graduation ladder's
    second bar is a Brier EDGE — strategy error minus reference error — and for
    a prediction market the reference is the market's own probability. A broad
    ETF has no market-implied probability of "up in five sessions", so the
    honest reference is the instrument's own drift: does the forecast add
    information beyond knowing that equities tend to rise?

    Deliberately not 0.5. Equities drift up, so a coin-flip reference would
    hand an arm a positive Brier edge for constantly answering 0.55 — an edge
    for knowing the direction of the last century, not for forecasting.

    Windows overlap, so this is a point estimate and not a test statistic. That
    is fine for a reference level; callers must pass only closes completed
    BEFORE the forecast date (see ``completed_closes``) or it leaks.
    """
    if horizon < 1 or len(closes) <= horizon:
        return None
    total = len(closes) - horizon
    wins = sum(1 for i in range(total) if closes[i + horizon] > closes[i])
    return wins / total


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
