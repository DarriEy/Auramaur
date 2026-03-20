"""Cross-market arbitrage execution."""

from __future__ import annotations

import structlog

from auramaur.db.database import Database
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

        if opp["type"] == "conditional_violation":
            # A implies B, but P(A) > P(B) — buy B (underpriced), sell A (overpriced)
            return self._make_signal_pair(
                buy_market=market_b,
                sell_market=market_a,
                edge=opp["price_a"] - opp["price_b"],
                opp=opp,
            )

        elif opp["type"] == "price_divergence":
            # Same event priced differently — buy cheap, sell expensive
            if opp["price_a"] < opp["price_b"]:
                buy_market, sell_market = market_a, market_b
            else:
                buy_market, sell_market = market_b, market_a
            return self._make_signal_pair(
                buy_market=buy_market,
                sell_market=sell_market,
                edge=opp.get("divergence", abs(opp["price_a"] - opp["price_b"])),
                opp=opp,
            )

        return None

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
            condition_id=row["condition_id"],
            question=row["question"],
            description=row["description"] or "",
            category=row["category"] or "",
            end_date=end_date,
            active=bool(row["active"]),
            outcome_yes_price=row["outcome_yes_price"] or 0.5,
            outcome_no_price=row["outcome_no_price"] or 0.5,
            volume=row["volume"] or 0,
            liquidity=row["liquidity"] or 0,
        )
