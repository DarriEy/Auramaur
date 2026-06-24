"""Protocol definitions for swappable market analysis."""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from auramaur.exchange.models import Market, Signal


class ExecutionMode(str, Enum):
    """How a strategy pillar commits its orders.

    The bot's pillars share the pipeline discover → signal → risk → allocate →
    execute, but they legitimately differ at the EXECUTE stage. This enum makes
    each pillar's execution path an explicit, declared, testable contract — so a
    justified gateway-bypass (market maker, concurrent arb, equities) is
    distinguishable from an accidental one (the conformance guard asserts it).
    """

    # Route through the single ExecutionGateway money path:
    GATEWAY_SINGLE = "gateway_single"      # gateway.submit() — directional pillars
    GATEWAY_PAIRED = "gateway_paired"      # gateway.submit_paired() — both-or-nothing arb
    GATEWAY_EXTERNAL = "gateway_external"  # gateway.record_external_fill() — concurrently-placed legs
    # Documented, justified gateway bypasses:
    DIRECT_QUOTING = "direct_quoting"      # market maker — resting two-sided quotes (no Signal/risk)
    DIRECT_EQUITY = "direct_equity"        # oddlot tender — IBKR equities, outside the PM gateway


# Modes that MUST run through the ExecutionGateway (no direct exchange.place_order).
GATEWAY_MODES = frozenset({
    ExecutionMode.GATEWAY_SINGLE,
    ExecutionMode.GATEWAY_PAIRED,
    ExecutionMode.GATEWAY_EXTERNAL,
})


@runtime_checkable
class Strategy(Protocol):
    """The uniform pillar contract.

    Thin by design: the five pipeline stages (discover/signal/risk/allocate/
    execute) are INTERNAL phases of ``run_once`` — pillars legitimately fuse them
    differently (MM has no Signal, arb has two legs) — so they are not forced as
    separate methods. What every pillar declares is its identity and HOW it
    executes, which is what the conformance guard checks.
    """

    name: str
    execution_mode: ExecutionMode

    async def run_once(self) -> int:
        """Run one cycle; return the number of entries/quotes placed."""
        ...


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
