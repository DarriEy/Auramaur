"""Proper and market-relative scoring for binary forecasts."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ForecastScore:
    brier: float
    log_loss: float
    market_brier: float
    brier_delta: float
    brier_skill: float | None


def score_forecast(prob_yes: float, market_prob_yes: float, outcome: int,
                   epsilon: float = 1e-6) -> ForecastScore:
    if not 0 <= prob_yes <= 1 or not 0 <= market_prob_yes <= 1:
        raise ValueError("probabilities must be in [0, 1]")
    if outcome not in (0, 1):
        raise ValueError("outcome must be 0 or 1")
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be in (0, 0.5)")
    brier = (prob_yes - outcome) ** 2
    market_brier = (market_prob_yes - outcome) ** 2
    clipped = min(1 - epsilon, max(epsilon, prob_yes))
    log_loss = -(outcome * math.log(clipped) + (1 - outcome) * math.log(1 - clipped))
    return ForecastScore(
        brier=brier,
        log_loss=log_loss,
        market_brier=market_brier,
        brier_delta=market_brier - brier,
        brier_skill=None if market_brier == 0 else 1 - brier / market_brier,
    )
