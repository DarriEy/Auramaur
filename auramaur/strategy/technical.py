"""Technical analysis strategy — mean reversion and momentum signals."""

from __future__ import annotations

import statistics
import structlog

from auramaur.exchange.models import Confidence, Market, OrderSide, Signal
from auramaur.strategy.protocols import TradeCandidate

log = structlog.get_logger()


class TechnicalAnalyzer:
    """Generates trading signals based on price action and momentum.

    This analyzer works independently of LLM analysis, providing a fast,
    data-driven "second leg" to the bot's strategy.
    """

    def __init__(self, settings):
        self.settings = settings
        # Thresholds for technical signals
        self.min_move_pct = 5.0  # 5% move for momentum
        self.mean_rev_threshold = 0.10  # 10% deviation for mean reversion
        self.min_history_points = 5

    async def analyze_markets(
        self,
        markets: list[Market],
        price_history: dict[str, list[float]] | None = None,
    ) -> list[TradeCandidate]:
        """Analyze markets for technical signals."""
        # Honor the config gate — it existed in TechnicalConfig but was never
        # checked, so the book couldn't be wound down by config. Exits are
        # engine-level and position-based, so disabling signal generation
        # strands nothing (same shape as the news_speed wind-down).
        if not getattr(self.settings.technical, "enabled", True):
            return []
        if not price_history:
            return []

        candidates: list[TradeCandidate] = []

        for market in markets:
            history = price_history.get(market.id)
            if not history or len(history) < self.min_history_points:
                continue

            # 1. Mean Reversion Signal
            mean_rev_signal = self._check_mean_reversion(market, history)
            if mean_rev_signal:
                candidates.append(TradeCandidate(market=market, signal=mean_rev_signal))
                continue

            # 2. Momentum Signal
            momentum_signal = self._check_momentum(market, history)
            if momentum_signal:
                candidates.append(TradeCandidate(market=market, signal=momentum_signal))

        if candidates:
            log.info(
                "technical_analyzer.signals_detected",
                count=len(candidates),
                markets=[c.market.id[:12] for c in candidates],
            )

        return candidates

    def _check_mean_reversion(self, market: Market, history: list[float]) -> Signal | None:
        """Detect if the price has deviated significantly from its mean."""
        current_price = market.outcome_yes_price
        avg_price = statistics.mean(history)

        if avg_price < 0.01:
            return None

        deviation = (current_price - avg_price) / avg_price

        # Oversold (Price < Mean) -> Buy YES
        if deviation < -self.mean_rev_threshold:
            # deviation (relative to the mean) is the TRIGGER; the emitted
            # edge follows the Signal.edge contract: absolute points between
            # fair value (the mean) and the current price. A relative figure
            # here overstated the edge on cheap markets by 1/price.
            edge = abs(avg_price - current_price) * 100
            return Signal(
                market_id=market.id,
                market_question=market.question,
                claude_prob=avg_price,  # Use mean as our "fair value" target
                claude_confidence=Confidence.LOW,
                market_prob=current_price,
                edge=edge,
                evidence_summary=f"Technical: Mean Reversion (Oversold). Price {current_price:.2f} is {abs(deviation):.1%} below 24h mean {avg_price:.2f}.",
                recommended_side=OrderSide.BUY,
                recommended_size=0.4,  # Lower conviction than LLM
                strategy_source="technical_mean_reversion",
            )

        # Overbought (Price > Mean) -> Buy NO (Sell YES)
        if deviation > self.mean_rev_threshold:
            edge = abs(current_price - avg_price) * 100
            return Signal(
                market_id=market.id,
                market_question=market.question,
                claude_prob=avg_price,
                claude_confidence=Confidence.LOW,
                market_prob=current_price,
                edge=edge,
                evidence_summary=f"Technical: Mean Reversion (Overbought). Price {current_price:.2f} is {deviation:.1%} above 24h mean {avg_price:.2f}.",
                recommended_side=OrderSide.SELL,
                recommended_size=0.4,
                strategy_source="technical_mean_reversion",
            )

        return None

    def _check_momentum(self, market: Market, history: list[float]) -> Signal | None:
        """Detect if the price is trending steadily in one direction."""
        current_price = market.outcome_yes_price
        start_price = history[0]

        if start_price < 0.01:
            return None

        total_move = (current_price - start_price) / start_price

        # Strong Upward Momentum -> Buy YES
        if total_move >= (self.min_move_pct / 100.0):
            # Check if trend is consistent (last 3 points are increasing)
            if history[-1] > history[-2] > history[-3]:
                # total_move (relative) is the TRIGGER; the emitted edge is
                # absolute points to the continuation target (Signal.edge
                # contract).
                fair = min(0.99, current_price + 0.05)  # Expect continuation
                edge = (fair - current_price) * 100
                return Signal(
                    market_id=market.id,
                    market_question=market.question,
                    claude_prob=fair,
                    claude_confidence=Confidence.LOW,
                    market_prob=current_price,
                    edge=edge,
                    evidence_summary=f"Technical: Momentum (Upward). Price moved {total_move:.1%} in 24h with consistent trend.",
                    recommended_side=OrderSide.BUY,
                    recommended_size=0.3,
                    strategy_source="technical_momentum",
                )

        # Strong Downward Momentum -> Buy NO (Sell YES)
        if total_move <= -(self.min_move_pct / 100.0):
            # Check if trend is consistent (last 3 points are decreasing)
            if history[-1] < history[-2] < history[-3]:
                fair = max(0.01, current_price - 0.05)
                edge = (current_price - fair) * 100
                return Signal(
                    market_id=market.id,
                    market_question=market.question,
                    claude_prob=fair,
                    claude_confidence=Confidence.LOW,
                    market_prob=current_price,
                    edge=edge,
                    evidence_summary=f"Technical: Momentum (Downward). Price moved {abs(total_move):.1%} in 24h with consistent trend.",
                    recommended_side=OrderSide.SELL,
                    recommended_size=0.3,
                    strategy_source="technical_momentum",
                )

        return None
