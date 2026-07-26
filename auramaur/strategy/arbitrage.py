"""Cross-market arbitrage execution."""

from __future__ import annotations

import structlog

from auramaur.db.database import Database
from auramaur.experiments.strategies.arbitrage import (
    ArbitrageMarket,
    PairedArbitrageProposal,
    form_arbitrage_pair,
)
from auramaur.exchange.models import Confidence, Market, OrderSide, Signal
from auramaur.strategy.correlation import CorrelationDetector

log = structlog.get_logger()


class ArbitrageExecutor:
    """Converts detected arbitrage opportunities into tradeable signals.

    Works with two types of opportunities:
    - Conditional violations: P(A) > P(B) when A implies B. Buy B, sell A.
    - Price divergence: same event priced differently. Buy cheap, sell expensive.
    """

    def __init__(self, db: Database, correlator: CorrelationDetector) -> None:
        self._db = db
        self._correlator = correlator

    async def generate_arb_signals(self) -> list[tuple[Signal, Signal, dict]]:
        """Detect and generate paired signals for arbitrage opportunities.

        Returns:
            List of (buy_signal, sell_signal, opportunity_info) tuples.
        """
        opportunities = await self._correlator.detect_arbitrage()
        if not opportunities:
            return []

        pairs: list[tuple[Signal, Signal, dict]] = []
        for opp in opportunities:
            try:
                pair = await self._opportunity_to_signals(opp)
                if pair:
                    pairs.append(pair)
            except Exception as e:
                log.debug("arbitrage.signal_error", error=str(e), opp_type=opp.get("type"))

        if pairs:
            log.info("arbitrage.signals_generated", count=len(pairs))
        return pairs

    async def _opportunity_to_signals(
        self, opp: dict
    ) -> tuple[Signal, Signal, dict] | None:
        """Convert a single opportunity into a buy/sell signal pair."""
        market_a = await self._load_market(opp["market_a"])
        market_b = await self._load_market(opp["market_b"])
        if not market_a or not market_b:
            return None

        proposal = form_arbitrage_pair(
            opp,
            self._experiment_market(market_a),
            self._experiment_market(market_b),
        )
        if proposal is None:
            return None
        return self._proposal_to_signals(proposal, opp)

    @staticmethod
    def _experiment_market(market: Market) -> ArbitrageMarket:
        return ArbitrageMarket(
            market_id=market.id,
            question=market.question,
            yes_price=market.outcome_yes_price,
            exchange=market.exchange,
            yes_token=market.clob_token_yes,
            no_token=market.clob_token_no,
        )

    @staticmethod
    def _proposal_to_signals(
        proposal: PairedArbitrageProposal, opp: dict
    ) -> tuple[Signal, Signal, dict]:
        evidence = f"Arbitrage: {proposal.opportunity_type}"
        buy_signal = Signal(
            market_id=proposal.buy.market_id,
            market_question=proposal.buy.question,
            claude_prob=proposal.buy.fair_probability,
            claude_confidence=Confidence.MEDIUM,
            market_prob=proposal.buy.market_probability,
            edge=proposal.buy.edge_percent,
            evidence_summary=evidence,
            recommended_side=OrderSide.BUY,
            strategy_source="arbitrage",
        )
        sell_signal = Signal(
            market_id=proposal.sell.market_id,
            market_question=proposal.sell.question,
            claude_prob=proposal.sell.fair_probability,
            claude_confidence=Confidence.MEDIUM,
            market_prob=proposal.sell.market_probability,
            edge=proposal.sell.edge_percent,
            evidence_summary=evidence,
            recommended_side=OrderSide.SELL,
            strategy_source="arbitrage",
        )
        return buy_signal, sell_signal, opp

    @staticmethod
    def _pair_is_executable(buy: Market, sell: Market, opp: dict) -> bool:
        """Reject structurally untradeable pairs before they reach the bot loop."""
        missing: list[tuple[str, str]] = []
        if (buy.exchange or "polymarket") == "polymarket" and not buy.clob_token_yes:
            missing.append((buy.id, "YES"))
        if (sell.exchange or "polymarket") == "polymarket" and not sell.clob_token_no:
            missing.append((sell.id, "NO"))
        if missing:
            log.info("arbitrage.candidate_suppressed", type=opp.get("type"),
                     reason="missing_execution_metadata", missing_tokens=missing)
            return False
        return True

    def _make_signal_pair(
        self,
        buy_market: Market,
        sell_market: Market,
        edge: float,
        opp: dict,
    ) -> tuple[Signal, Signal, dict]:
        """Create a matched pair of buy/sell signals."""
        edge_pct = edge * 100

        buy_signal = Signal(
            market_id=buy_market.id,
            market_question=buy_market.question,
            claude_prob=buy_market.outcome_yes_price + edge / 2,  # Our fair value
            claude_confidence=Confidence.MEDIUM,
            market_prob=buy_market.outcome_yes_price,
            edge=edge_pct / 2,  # Split edge across both legs
            evidence_summary=f"Arbitrage: {opp.get('type', 'unknown')}",
            recommended_side=OrderSide.BUY,
            strategy_source="arbitrage",
        )

        sell_signal = Signal(
            market_id=sell_market.id,
            market_question=sell_market.question,
            claude_prob=sell_market.outcome_yes_price - edge / 2,
            claude_confidence=Confidence.MEDIUM,
            market_prob=sell_market.outcome_yes_price,
            edge=edge_pct / 2,
            evidence_summary=f"Arbitrage: {opp.get('type', 'unknown')}",
            recommended_side=OrderSide.SELL,
            strategy_source="arbitrage",
        )

        return (buy_signal, sell_signal, opp)

    async def _load_market(self, market_id: str) -> Market | None:
        """Load a market from the database."""
        row = await self._db.fetchone(
            "SELECT * FROM markets WHERE id = ? AND active = 1", (market_id,)
        )
        if row is None:
            return None

        from datetime import datetime
        end_date = None
        if row["end_date"]:
            try:
                end_date = datetime.fromisoformat(row["end_date"])
            except (ValueError, TypeError):
                pass

        return Market(
            id=row["id"],
            exchange=row["exchange"] or "polymarket",
            condition_id=row["condition_id"],
            ticker=row["ticker"] or "",
            question=row["question"],
            description=row["description"] or "",
            category=row["category"] or "",
            end_date=end_date,
            active=bool(row["active"]),
            outcome_yes_price=row["outcome_yes_price"] or 0.5,
            outcome_no_price=row["outcome_no_price"] or 0.5,
            volume=row["volume"] or 0,
            liquidity=row["liquidity"] or 0,
            clob_token_yes=row["clob_token_yes"] or "",
            clob_token_no=row["clob_token_no"] or "",
        )
