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
    # Gateway-submitted — the pillar never calls exchange.place_order itself;
    # the gateway places AND records (this is what the conformance guard asserts):
    GATEWAY_SINGLE = "gateway_single"      # gateway.submit() — directional pillars
    GATEWAY_PAIRED = "gateway_paired"      # gateway.submit_paired() — both-or-nothing arb
    # Places directly for timing/atomicity reasons, but still RECORDS through the
    # gateway (record_external_fill) — so it does call place_order and is NOT a
    # gateway-pure mode:
    GATEWAY_EXTERNAL = "gateway_external"  # concurrently-placed arb legs, then record_external_fill()
    # Fully direct, justified bypasses (own accounting):
    DIRECT_QUOTING = "direct_quoting"      # market maker — resting two-sided quotes (no Signal/risk)
    DIRECT_EQUITY = "direct_equity"        # oddlot tender — IBKR equities, outside the PM gateway


# Gateway-PURE modes: the pillar must not call exchange.place_order at all (the
# gateway submits on its behalf). GATEWAY_EXTERNAL is deliberately excluded — it
# places directly then records — matching the conformance guard's skip set.
GATEWAY_PURE_MODES = frozenset({
    ExecutionMode.GATEWAY_SINGLE,
    ExecutionMode.GATEWAY_PAIRED,
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
