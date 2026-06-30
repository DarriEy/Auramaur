"""Data-driven pricing for Kalshi economic-indicator bins (#2 Phase B).

Kalshi runs "Above X" threshold ladders on official indicators (CPI YoY,
unemployment, payrolls, ...). Each ladder is an implied CDF of the indicator
at a future release. This module prices those bins from the official history
(FRED) instead of guessing: estimate the indicator's distribution at the
target release, then P(Above X) = 1 - CDF(X). Where the model and the market
disagree by more than fees + a margin, there is a (forecast-based, not
model-free) edge.

EVERYTHING here is pure and deterministic — the FRED fetch and order placement
live in the pillar. Model risk is real (a wrong transform = confident-wrong
bets), so: transforms are explicit per series, the sigma has a floor (never
over-concentrate), and the pillar is paper-forced until the paper ledger +
calibration prove it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


# ----------------------------------------------------------------------
# Registry: Kalshi series prefix -> how to build the indicator from FRED
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class EconSpec:
    fred_series: str
    # how to turn the raw FRED level into the indicator the bins resolve on:
    #   "level"      — the series value itself (e.g. unemployment rate)
    #   "yoy"        — % change vs 12 periods ago (e.g. CPI YoY)
    #   "mom_change" — change vs previous period, times `scale` (e.g. payrolls)
    transform: str
    periods_per_year: int = 12  # 12 monthly, 4 quarterly — for YoY/horizon
    scale: float = 1.0          # mom_change unit fix (PAYEMS is in thousands)
    # Precision the official agency REPORTS the figure to (CPI YoY & U3 to 0.1pp,
    # payrolls to the nearest 1,000 jobs). A FRED-derived indicator is computed
    # continuously (e.g. CPI YoY 4.4997%), but Kalshi/Poly bins resolve on the
    # ROUNDED published figure — so round to this grid before the threshold test,
    # else a value a hair off a strike boundary resolves the wrong side.
    report_round_to: float = 0.0  # 0 => no rounding


# Conservative starter registry — only series with an unambiguous transform.
# (GDP annualization / Fed-decision categoricals are intentionally omitted
# until each is validated against a real print.)
ECON_SERIES: dict[str, EconSpec] = {
    # Kalshi CPI-YoY markets resolve on the headline BLS number, which is the
    # NOT-seasonally-adjusted index (CPIAUCNS). CPIAUCSL is the seasonally
    # adjusted series — its YoY can differ by a tenth or more, exactly the
    # margin a point-bin turns on, so the SA series would mis-resolve bins.
    "KXCPIYOY": EconSpec("CPIAUCNS", "yoy", periods_per_year=12, report_round_to=0.1),
    "KXU3": EconSpec("UNRATE", "level", periods_per_year=12, report_round_to=0.1),
    "KXPAYROLLS": EconSpec("PAYEMS", "mom_change", periods_per_year=12,
                           scale=1000.0, report_round_to=1000.0),
}

# Kalshi monthly-period date codes: "26NOV" -> ("2026", 11).
_KALSHI_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
_KALSHI_PERIOD_RE = re.compile(r"^(\d{2})([A-Z]{3})$")
# strike_type -> comparison operator for the YES outcome.
_STRIKE_OPERATORS = {
    "greater": ">", "greater_or_equal": ">=",
    "less": "<", "less_or_equal": "<=",
}


def parse_kalshi_period(code: str) -> str | None:
    """'26NOV' -> '2026-11'. None if the code isn't a YY-MON monthly period."""
    m = _KALSHI_PERIOD_RE.match((code or "").strip().upper())
    if not m:
        return None
    yy, mon = m.groups()
    month = _KALSHI_MONTHS.get(mon)
    if month is None:
        return None
    return f"20{yy}-{month:02d}"


def kalshi_macro_predicate(market) -> dict | None:
    """Build a settlement_arb predicate DETERMINISTICALLY from a Kalshi macro-bin
    market's ticker + structured strike fields — no LLM extraction.

    Returns {indicator, operator, threshold, reference_period} for a recognized
    monthly econ-series threshold bin (e.g. KXCPIYOY-26NOV-T4.5, strike_type
    'greater', floor 4.5 -> '> 4.5' on KXCPIYOY for 2026-11), else None
    (unknown series, non-monthly period, or a 'between'/structured bin we don't
    resolve with a single threshold).
    """
    ticker = (getattr(market, "ticker", "") or "")
    parts = ticker.split("-")
    if len(parts) < 3:
        return None
    series = parts[0].upper()
    if series not in ECON_SERIES:
        return None
    period = parse_kalshi_period(parts[1])
    if period is None:
        return None
    op = _STRIKE_OPERATORS.get((getattr(market, "strike_type", "") or "").lower())
    if op is None:
        return None  # 'between'/structured/unknown -> not a single-threshold bin
    floor = getattr(market, "floor_strike", None)
    cap = getattr(market, "cap_strike", None)
    threshold = floor if op in (">", ">=") else (cap if cap is not None else floor)
    if threshold is None:
        return None
    return {"indicator": series, "operator": op,
            "threshold": float(threshold), "reference_period": period}


def spec_for_series(series_ticker: str) -> EconSpec | None:
    """Look up the EconSpec for a Kalshi series prefix (e.g. 'KXCPIYOY')."""
    return ECON_SERIES.get((series_ticker or "").upper())


# ----------------------------------------------------------------------
# Distribution math (pure)
# ----------------------------------------------------------------------

def normal_cdf(x: float, mean: float = 0.0, sigma: float = 1.0) -> float:
    """Standard normal CDF via erf — no SciPy dependency."""
    if sigma <= 0:
        return 1.0 if x >= mean else 0.0
    return 0.5 * (1.0 + math.erf((x - mean) / (sigma * math.sqrt(2.0))))


def prob_above(threshold: float, mean: float, sigma: float) -> float:
    """P(indicator > threshold) under Normal(mean, sigma)."""
    return 1.0 - normal_cdf(threshold, mean, sigma)


def indicator_series(values: list[float], spec: EconSpec) -> list[float]:
    """Transform a raw FRED level series into the indicator the bins resolve on.

    Input is oldest-first raw values; output is the transformed series (also
    oldest-first), which may be shorter (YoY drops the first `periods_per_year`).
    """
    if spec.transform == "level":
        return list(values)
    if spec.transform == "yoy":
        p = spec.periods_per_year
        if len(values) <= p:
            return []
        out = []
        for i in range(p, len(values)):
            base = values[i - p]
            if base == 0:
                continue
            out.append((values[i] / base - 1.0) * 100.0)
        return out
    if spec.transform == "mom_change":
        return [(values[i] - values[i - 1]) * spec.scale
                for i in range(1, len(values))]
    raise ValueError(f"unknown transform {spec.transform!r}")


def estimate_distribution(
    indicator: list[float],
    horizon_periods: int = 1,
    sigma_floor_frac: float = 0.25,
    lookback: int = 24,
) -> tuple[float, float] | None:
    """Estimate (mean, sigma) for the indicator at a release `horizon_periods`
    ahead, from its recent history.

    mean  — the latest observed value (random-walk nowcast; no false precision).
    sigma — std of recent period-over-period changes, scaled by sqrt(horizon),
            floored at `sigma_floor_frac` of |mean| so a quiet recent window
            can't make the model overconfident. Returns None if too little
            history to estimate a change-vol.
    """
    if len(indicator) < 3:
        return None
    mean = indicator[-1]
    recent = indicator[-lookback:]
    diffs = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
    if not diffs:
        return None
    mu = sum(diffs) / len(diffs)
    var = sum((d - mu) ** 2 for d in diffs) / len(diffs)
    sigma_1 = math.sqrt(var)
    sigma = sigma_1 * math.sqrt(max(1, horizon_periods))
    floor = sigma_floor_frac * abs(mean)
    sigma = max(sigma, floor, 1e-6)
    return mean, sigma


def bin_edge(model_p: float, market_p: float) -> float:
    """Signed model edge for a YES position on an 'Above X' bin."""
    return model_p - market_p
