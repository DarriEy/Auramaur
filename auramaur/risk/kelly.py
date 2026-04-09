"""Fractional Kelly criterion position sizing with geometric optimization."""

from __future__ import annotations

import math
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
    Confidence.MEDIUM_HIGH: 0.875,
    Confidence.MEDIUM: 0.75,
    Confidence.MEDIUM_LOW: 0.625,
    Confidence.LOW: 0.5,
}


class KellySizer:
    """Geometric Kelly position sizer — maximizes expected log-wealth
    for superior long-run compounding.

    Standard Kelly maximizes E[gain]. Geometric Kelly maximizes E[log(wealth)],
    which accounts for the fact that a 50% loss requires a 100% gain to recover.
    This is critical with 500 position slots where compounding matters.
    """

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
        book_imbalance_mult: float = 1.0,
        max_stake: float = 25.0,
    ) -> float:
        """Return the optimal stake in dollars using geometric Kelly.

        Geometric Kelly formula:
          For BUY YES at price p, with true probability q:
            kelly = q/p - (1-q)/(1-p)  (simplified: maximize E[log(wealth)])
          This is equivalent to: kelly = (q*(1-p) - (1-q)*p) / (p*(1-p))
                                       = (q - p) / (p*(1-p))

        This naturally produces SMALLER bets than linear Kelly when edge is
        slim relative to risk, which is exactly what we want for compounding.
        """
        if market_prob >= 0.99 or market_prob <= 0.01:
            return 0.0

        edge = claude_prob - market_prob
        if abs(edge) < 0.001:
            return 0.0

        if edge > 0:
            # BUY YES: geometric Kelly
            # kelly = (q - p) / (p * (1 - p))
            # where q = claude_prob, p = market_prob
            kelly = edge / (market_prob * (1.0 - market_prob))
        else:
            # BUY NO: same formula with inverted probabilities
            no_claude = 1.0 - claude_prob
            no_market = 1.0 - market_prob
            kelly = (no_claude - no_market) / (no_market * (1.0 - no_market))

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
            * book_imbalance_mult
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
    def book_imbalance_multiplier(bids_depth: float, asks_depth: float) -> float:
        """Compute a sizing multiplier from order book imbalance.

        If the book is heavily bid (more buy pressure), and we're buying,
        that's confirmation — size up. If the book disagrees with our
        direction, size down.

        Returns a multiplier 0.5 - 1.5:
          imbalance > 0 (more bids than asks) → bullish signal
          imbalance < 0 (more asks than bids) → bearish signal

        Note: the caller should invert the signal for NO-side bets.
        """
        total = bids_depth + asks_depth
        if total <= 0:
            return 1.0

        # Imbalance ratio: -1 (all asks) to +1 (all bids)
        imbalance = (bids_depth - asks_depth) / total

        # Scale: 0% imbalance → 1.0x, ±50% → 1.25x/0.75x, ±100% → 1.5x/0.5x
        return max(0.5, min(1.5, 1.0 + imbalance * 0.5))

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
