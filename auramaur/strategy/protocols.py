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

    # Every mode below routes ALL placement through the ExecutionGateway — no
    # pillar/flow calls exchange.place_order directly — EXCEPT DIRECT_EQUITY.
    GATEWAY_SINGLE = "gateway_single"      # gateway.submit() — directional pillars
    GATEWAY_PAIRED = "gateway_paired"      # gateway.submit_paired() — both-or-nothing arb
    GATEWAY_EXTERNAL = "gateway_external"  # gateway.place_legs() — concurrent arb legs, then records
    DIRECT_QUOTING = "direct_quoting"      # market maker — gateway.place_quote_pair() (resting two-sided)
    # The one genuine off-gateway exception:
    DIRECT_EQUITY = "direct_equity"        # oddlot tender — IBKR equities, outside the PM gateway


# Modes whose module must NOT contain a raw exchange.place_order — placement goes
# through a gateway method (submit / submit_paired / place_legs / place_quote_pair).
# Only DIRECT_EQUITY (IBKR, genuinely off the prediction-market gateway) is exempt.
NO_DIRECT_PLACE_MODES = frozenset(
    m for m in ExecutionMode if m is not ExecutionMode.DIRECT_EQUITY)


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
