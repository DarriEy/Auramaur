"""Protocol definitions for swappable market analysis."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from auramaur.exchange.models import Market, Signal


class TradeCandidate(BaseModel):
    """A market paired with its analysis signal — output of any MarketAnalyzer."""

    market: Market
    signal: Signal


class MarketAnalyzer(Protocol):
    """Protocol for the swappable analysis layer.

    Both the pipeline (data → NLP → signal detection) and a future
    agentic approach implement this same interface.  Everything
    downstream — risk checks, allocation, execution — consumes
    TradeCandidate objects identically regardless of how they were
    produced.
    """

    async def analyze_markets(
        self,
        markets: list[Market],
        price_history: dict[str, list[float]] | None = None,
    ) -> list[TradeCandidate]:
        """Analyze a batch of markets and return trade candidates.

        Args:
            markets: Pre-filtered, ranked markets to analyze.
            price_history: Recent price history keyed by market ID.

        Returns:
            Trade candidates with probability estimates, confidence,
            edge, and recommended side.  The caller (TradingEngine)
            handles risk checks and execution.
        """
        ...
