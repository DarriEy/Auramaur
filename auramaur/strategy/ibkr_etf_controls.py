"""Deterministic, leakage-resistant controls for the IBKR ETF experiment."""

from __future__ import annotations
from dataclasses import dataclass
from datetime import date
import hashlib
from statistics import fmean
from auramaur.nlp.openai_etf import ETFAnalysis
from auramaur.risk.ibkr_math import annualized_volatility, normalized_momentum


def completed_closes(bars, as_of: str) -> list[float]:
    return [float(c) for day, c in bars if day < as_of and c and float(c) > 0]


def dual_momentum_score(closes, cash_closes) -> float | None:
    asset, cash = normalized_momentum(closes), normalized_momentum(cash_closes)
    return None if asset is None or cash is None or asset <= 0 else asset - cash


def cross_sectional_winners(histories, top_n: int = 3) -> list[str]:
    ranked = [
        (s, k) for k, v in histories.items() if (s := normalized_momentum(v)) is not None and s > 0
    ]
    return [k for _, k in sorted(ranked, reverse=True)[:top_n]]


def regime_allows_risk(closes, *, max_annual_vol: float = 0.30) -> bool:
    vol = annualized_volatility(closes)
    return bool(
        vol is not None
        and vol <= max_annual_vol
        and len(closes) >= 201
        and closes[-1] > fmean(closes[-200:])
    )


class MomentumETFAnalyzer:
    model = "deterministic_momentum_v1"

    async def analyze_symbol(self, client, symbol: str, as_of: str | None = None):
        closes = completed_closes(
            await client.get_adjusted_daily_closes(symbol), as_of or date.today().isoformat()
        )
        score = normalized_momentum(closes)
        if score is None:
            return None
        return ETFAnalysis(
            0.70 if score > 0 else 0.30,
            "HIGH",
            f"deterministic normalized momentum={score:.4f}",
            (),
        )


@dataclass(frozen=True)
class ControlReturn:
    cash: float
    buy_hold: float
    momentum: float
    placebo: float


def compare_controls(
    closes, *, warmup: int = 120, placebo_rate: float = 0.5, seed: str = "ibkr-etf-v1"
) -> ControlReturn:
    """Walk-forward returns with decisions made before each scored return."""
    if len(closes) <= warmup or not 0 <= placebo_rate <= 1:
        raise ValueError("controls require sufficient closes and a valid placebo rate")
    buy_hold = momentum = placebo = 1.0
    for i in range(warmup, len(closes)):
        ret = closes[i] / closes[i - 1]
        buy_hold *= ret
        if (normalized_momentum(closes[:i]) or 0) > 0:
            momentum *= ret
        digest = hashlib.sha256(f"{seed}:{i}".encode()).digest()
        if int.from_bytes(digest[:8], "big") / (2**64 - 1) < placebo_rate:
            placebo *= ret
    return ControlReturn(0.0, buy_hold - 1, momentum - 1, placebo - 1)
