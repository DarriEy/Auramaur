"""Fractional Kelly criterion position sizing."""

from __future__ import annotations

import statistics

from auramaur.exchange.models import Confidence


HEAT_MULTIPLIERS: dict[str, float] = {
    "GREEN": 1.0,
    "YELLOW": 0.75,
    "ORANGE": 0.5,
    "RED": 0.0,
}

CONFIDENCE_MULTIPLIERS: dict[Confidence, float] = {
    Confidence.HIGH: 1.0,
    Confidence.MEDIUM: 0.75,
    Confidence.LOW: 0.5,
}


class KellySizer:
    """Quarter-Kelly (by default) position sizer with drawdown / confidence /
    liquidity adjustments."""

    def __init__(self, fraction: float = 0.25):
        self.fraction = fraction

    def calculate(
        self,
        claude_prob: float,
        market_prob: float,
        bankroll: float,
        heat_mult: float = 1.0,
        confidence_mult: float = 1.0,
        liquidity_mult: float = 1.0,
        category_mult: float = 1.0,
        volatility_mult: float = 1.0,
        max_stake: float = 25.0,
    ) -> float:
        """Return the optimal stake in dollars.

        kelly = (p_claude - p_market) / (1 - p_market)
        size  = kelly * fraction * heat_mult * confidence_mult * liquidity_mult * category_mult * volatility_mult * bankroll
        Capped at *max_stake*.  Returns 0 if the edge is non-positive.
        """
        if market_prob >= 1.0 or market_prob <= 0.0:
            return 0.0

        # Determine edge direction
        edge = claude_prob - market_prob
        if edge > 0:
            # BUY YES: Claude thinks YES is underpriced
            kelly = edge / (1.0 - market_prob)
        elif edge < 0:
            # SELL YES / BUY No: Claude thinks YES is overpriced
            kelly = abs(edge) / market_prob
        else:
            return 0.0

        if kelly <= 0:
            return 0.0

        size = (
            kelly
            * self.fraction
            * heat_mult
            * confidence_mult
            * liquidity_mult
            * category_mult
            * volatility_mult
            * bankroll
        )
        return min(size, max_stake)

    # ------------------------------------------------------------------
    # Convenience helpers to derive multiplier values from raw inputs
    # ------------------------------------------------------------------

    @staticmethod
    def heat_multiplier(heat: str) -> float:
        return HEAT_MULTIPLIERS.get(heat, 0.0)

    @staticmethod
    def confidence_multiplier(confidence: Confidence) -> float:
        return CONFIDENCE_MULTIPLIERS.get(confidence, 0.5)

    @staticmethod
    def liquidity_multiplier(liquidity: float, reference: float = 10_000.0) -> float:
        return min(1.0, liquidity / reference)

    @staticmethod
    def volatility_multiplier(prices: list[float]) -> float:
        """Compute a volatility multiplier from recent price history.

        Calculates realized volatility as the standard deviation of
        price changes.  If vol > 0.10, the multiplier scales down
        aggressively (min 0.3).  If vol <= 0.05, no reduction (1.0).
        """
        if len(prices) < 2:
            return 1.0

        changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        vol = statistics.stdev(changes) if len(changes) >= 2 else abs(changes[0])

        if vol <= 0.05:
            return 1.0

        # Linear scale-down: at vol=0.05 mult=1.0, at vol=0.19 mult=0.3
        mult = 1.0 - (vol - 0.05) * 5
        return max(0.3, mult)
