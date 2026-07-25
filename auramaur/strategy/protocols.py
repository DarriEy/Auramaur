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

    # Prediction-market modes route placement through ExecutionGateway. The two
    # DIRECT_* modes use independently gated broker adapters outside that gateway.
    GATEWAY_SINGLE = "gateway_single"  # gateway.submit() — directional pillars
    GATEWAY_PAIRED = "gateway_paired"  # gateway.submit_paired() — both-or-nothing arb
    GATEWAY_EXTERNAL = (
        "gateway_external"  # gateway.place_legs() — concurrent arb legs, then records
    )
    DIRECT_QUOTING = (
        "direct_quoting"  # market maker — gateway.place_quote_pair() (resting two-sided)
    )
    PAPER_SIMULATED = "paper_simulated"  # local fills only; no exchange order method exists
    # Independently gated off-prediction-market adapters:
    DIRECT_EQUITY = "direct_equity"  # oddlot tender — IBKR equities, outside the PM gateway
    DIRECT_SPOT = "direct_spot"  # Kraken adapter with its own triple gates/caps


class DeploymentMode(str, Enum):
    """Capital eligibility for a strategy, independent of the global live gate."""

    PAPER_ONLY = "paper_only"
    GRADUATABLE = "graduatable"
    STRUCTURAL_LIVE = "structural_live"


class StrategyCapability(str, Enum):
    """Auditable stages/services a strategy requires from the runtime."""

    DISCOVERY = "discovery"
    MARKET_DATA = "market_data"
    EVIDENCE = "evidence"
    ANALYSIS = "analysis"
    SIGNAL = "signal"
    RISK = "risk"
    EXECUTION = "execution"
    RECORDING = "recording"
    RECONCILIATION = "reconciliation"
    SETTLEMENT = "settlement"


# Modes whose module must NOT contain a raw exchange.place_order — placement goes
# through a gateway method (submit / submit_paired / place_legs / place_quote_pair).
# DIRECT_EQUITY and DIRECT_SPOT are independently gated broker adapters.
NO_DIRECT_PLACE_MODES = frozenset(
    m for m in ExecutionMode if m not in {ExecutionMode.DIRECT_EQUITY, ExecutionMode.DIRECT_SPOT}
)


@runtime_checkable
class Strategy(Protocol):
    """Common identity/execution contract for every strategy shape.

    Cycle signatures legitimately differ: ordinary pillars use ``run_once``,
    engines and quoting use ``run_cycle``, scanners use ``scan``, and news uses
    ``check_for_news``. ``StrategySpec.runner`` declares and tests that callable.
    """

    name: str
    execution_mode: ExecutionMode


@runtime_checkable
class RunOnceStrategy(Strategy, Protocol):
    """The common cycle shape used by ordinary standalone pillars."""

    async def run_once(self) -> int | None: ...


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
