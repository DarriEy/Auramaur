"""Cross-platform arbitrage scanner.

Scans across all configured exchanges for price mismatches on equivalent
markets, and checks within-exchange invariants (YES + NO should sum to ~1.0).
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

import structlog

from auramaur.exchange.models import Market
from auramaur.exchange.protocols import MarketDiscovery

log = structlog.get_logger()

# Minimum spread to qualify as a cross-exchange arb opportunity
CROSS_EXCHANGE_MIN_SPREAD = 0.03  # 3%

# If YES + NO < this, buying both is risk-free profit on resolution
INTERNAL_ARB_THRESHOLD = 0.97

# Minimum word overlap ratio to consider two questions equivalent
MATCH_THRESHOLD = 0.60

# Common stop words to ignore in fuzzy matching
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "will", "be", "been",
    "to", "of", "in", "for", "on", "at", "by", "or", "and", "it", "its",
    "this", "that", "with", "from", "as", "do", "does", "did", "not", "no",
    "yes", "has", "have", "had", "can", "could", "would", "should", "may",
    "if", "but", "so", "than", "then", "what", "which", "who", "whom",
    "how", "when", "where", "why", "before", "after", "during", "between",
})


@dataclass
class ArbOpportunity:
    """A detected arbitrage opportunity between two markets."""

    market_a: Market
    market_b: Market
    exchange_a: str
    exchange_b: str
    price_a: float  # YES price on exchange A
    price_b: float  # YES price on exchange B
    spread: float  # absolute price difference
    expected_profit_pct: float  # percentage profit after buying both sides
    question: str  # human-readable description of what's being arbed
    arb_type: str = "cross_exchange"  # "cross_exchange" | "internal"


def _tokenize(text: str) -> set[str]:
    """Extract meaningful words from a question string."""
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    return words - _STOP_WORDS


def _word_overlap_score(text_a: str, text_b: str) -> float:
    """Compute word overlap ratio between two strings.

    Returns a value in [0, 1] where 1 means identical word sets.
    """
    words_a = _tokenize(text_a)
    words_b = _tokenize(text_b)

    if not words_a or not words_b:
        return 0.0

    intersection = words_a & words_b
    # Jaccard-like but weighted toward the smaller set so that
    # "Will X happen?" matches "Will X happen by 2025?"
    smaller = min(len(words_a), len(words_b))
    return len(intersection) / smaller if smaller > 0 else 0.0


class ArbitrageScanner:
    """Scans all connected exchanges for arbitrage opportunities.

    Two types of opportunities are detected:

    1. **Cross-exchange arbs**: The same question is listed on two exchanges
       at different YES prices.  If the spread exceeds 3%, there may be a
       profitable trade.

    2. **Internal arbs**: Within a single exchange, YES + NO should sum to
       ~1.0.  If the sum is < 0.97, buying both YES and NO guarantees a
       profit on resolution (one side pays $1.00, total cost < $0.97).
    """

    def __init__(
        self,
        discoveries: dict[str, MarketDiscovery],
        exchange_fees: dict[str, float] | None = None,
        min_profit_after_fees_pct: float = 1.5,
    ) -> None:
        """Initialize with the discoveries dict from bot components.

        Args:
            discoveries: Mapping of exchange name to MarketDiscovery instance.
                         e.g. {"polymarket": gamma, "kalshi": kalshi_client}
            exchange_fees: Mapping of exchange name to fee rate (0.0-1.0).
                          e.g. {"polymarket": 0.0, "kalshi": 0.07}
            min_profit_after_fees_pct: Minimum profit % after fees to flag
                                       as an opportunity.
        """
        self._discoveries = discoveries
        self._exchange_fees = exchange_fees or {}
        self._min_profit_after_fees_pct = min_profit_after_fees_pct

    async def scan(self) -> list[ArbOpportunity]:
        """Scan all exchanges for price mismatches on equivalent markets.

        Strategy:
        1. Fetch markets from each exchange concurrently.
        2. Match equivalent markets across exchanges by question similarity.
        3. Compare YES prices -- if spread > 3%, it is an arb opportunity.
        4. Also check within-exchange arbs: YES + NO should sum to ~1.0.
           If YES + NO < 0.97, buy both for risk-free profit on resolution.

        Returns:
            List of ArbOpportunity sorted by expected_profit_pct descending.
        """
        opportunities: list[ArbOpportunity] = []

        # Step 1: Fetch markets from all exchanges concurrently
        exchange_markets = await self._fetch_all_markets()

        if not exchange_markets:
            log.debug("arb_scanner.no_markets")
            return []

        # Step 2: Internal arbs (within each exchange)
        for exchange_name, markets in exchange_markets.items():
            discovery = self._discoveries[exchange_name]
            internal = await self.scan_internal_arb(discovery, markets=markets)
            opportunities.extend(internal)

        # Step 3: Cross-exchange arbs (compare every pair of exchanges)
        exchange_names = list(exchange_markets.keys())
        for i in range(len(exchange_names)):
            for j in range(i + 1, len(exchange_names)):
                name_a = exchange_names[i]
                name_b = exchange_names[j]
                markets_a = exchange_markets[name_a]
                markets_b = exchange_markets[name_b]

                matched = self._match_markets(markets_a, markets_b)

                for market_a, market_b in matched:
                    spread = abs(market_a.outcome_yes_price - market_b.outcome_yes_price)

                    if spread < CROSS_EXCHANGE_MIN_SPREAD:
                        continue

                    # Identify cheap/expensive sides for fee calc
                    if market_a.outcome_yes_price <= market_b.outcome_yes_price:
                        cheap_yes, expensive_yes = market_a, market_b
                        cheap_exchange, expensive_exchange = name_a, name_b
                    else:
                        cheap_yes, expensive_yes = market_b, market_a
                        cheap_exchange, expensive_exchange = name_b, name_a

                    # Fee-aware profit: buy YES cheap, buy NO on expensive side
                    # Gross profit = spread (one side resolves $1, other $0)
                    # Fees: each leg's fee applies to the profit from that leg
                    fee_cheap = self._exchange_fees.get(cheap_exchange, 0.0)
                    fee_expensive = self._exchange_fees.get(expensive_exchange, 0.0)
                    # Worst case: profit is taxed at the higher fee on the winning leg
                    max_fee_rate = max(fee_cheap, fee_expensive)
                    net_profit = spread * (1.0 - max_fee_rate)
                    profit_pct = net_profit * 100

                    if profit_pct < self._min_profit_after_fees_pct:
                        continue

                    opp = ArbOpportunity(
                        market_a=market_a,
                        market_b=market_b,
                        exchange_a=name_a,
                        exchange_b=name_b,
                        price_a=market_a.outcome_yes_price,
                        price_b=market_b.outcome_yes_price,
                        spread=spread,
                        expected_profit_pct=profit_pct,
                        question=market_a.question,
                        arb_type="cross_exchange",
                    )
                    opportunities.append(opp)

                    log.info(
                        "arb_scanner.cross_exchange",
                        exchange_a=name_a,
                        exchange_b=name_b,
                        question=market_a.question[:80],
                        price_a=round(market_a.outcome_yes_price, 3),
                        price_b=round(market_b.outcome_yes_price, 3),
                        spread=round(spread, 3),
                        profit_pct=round(profit_pct, 1),
                    )

        # Sort by expected profit descending
        opportunities.sort(key=lambda o: o.expected_profit_pct, reverse=True)

        if opportunities:
            log.info(
                "arb_scanner.scan_complete",
                total_opportunities=len(opportunities),
                cross_exchange=sum(1 for o in opportunities if o.arb_type == "cross_exchange"),
                internal=sum(1 for o in opportunities if o.arb_type == "internal"),
                best_profit_pct=round(opportunities[0].expected_profit_pct, 2),
            )
        else:
            log.debug("arb_scanner.no_opportunities")

        return opportunities

    def _match_markets(
        self, markets_a: list[Market], markets_b: list[Market]
    ) -> list[tuple[Market, Market]]:
        """Fuzzy match markets between exchanges by question similarity.

        Uses simple word overlap scoring with a threshold of >60% overlap.
        Each market is matched at most once (greedy best-match).

        Returns:
            List of (market_a, market_b) tuples that are likely the same question.
        """
        if not markets_a or not markets_b:
            return []

        # Pre-tokenize for performance
        tokens_a = [(m, _tokenize(m.question)) for m in markets_a]
        tokens_b = [(m, _tokenize(m.question)) for m in markets_b]

        # Score all pairs, keep those above threshold
        candidates: list[tuple[float, Market, Market]] = []
        for m_a, words_a in tokens_a:
            if not words_a:
                continue
            for m_b, words_b in tokens_b:
                if not words_b:
                    continue

                intersection = words_a & words_b
                smaller = min(len(words_a), len(words_b))
                score = len(intersection) / smaller if smaller > 0 else 0.0

                if score >= MATCH_THRESHOLD:
                    candidates.append((score, m_a, m_b))

        # Greedy best-match: sort by score descending, assign each market at most once
        candidates.sort(key=lambda x: x[0], reverse=True)
        used_a: set[str] = set()
        used_b: set[str] = set()
        matched: list[tuple[Market, Market]] = []

        for score, m_a, m_b in candidates:
            if m_a.id in used_a or m_b.id in used_b:
                continue
            used_a.add(m_a.id)
            used_b.add(m_b.id)
            matched.append((m_a, m_b))

        if matched:
            log.debug(
                "arb_scanner.matched_markets",
                count=len(matched),
                sample_question=matched[0][0].question[:60] if matched else "",
            )

        return matched

    async def scan_internal_arb(
        self,
        discovery: MarketDiscovery,
        markets: list[Market] | None = None,
    ) -> list[ArbOpportunity]:
        """Check within a single exchange: if YES + NO < 0.97, it is free money.

        When the sum of YES and NO prices is below the threshold, buying both
        outcomes costs less than $1.00 but one side is guaranteed to pay $1.00
        on resolution.

        Args:
            discovery: The MarketDiscovery instance for this exchange.
            markets: Pre-fetched markets (optional, avoids refetching).

        Returns:
            List of internal ArbOpportunity instances.
        """
        if markets is None:
            try:
                markets = await discovery.get_markets(active=True, limit=100)
            except Exception as e:
                log.error("arb_scanner.internal_fetch_error", error=str(e))
                return []

        opportunities: list[ArbOpportunity] = []

        for market in markets:
            yes_price = market.outcome_yes_price
            no_price = market.outcome_no_price
            total = yes_price + no_price

            if total >= INTERNAL_ARB_THRESHOLD:
                continue

            # Cost to buy both: total.  Guaranteed payout: $1.00
            # Profit = 1.0 - total
            profit = 1.0 - total
            profit_pct = profit * 100

            opp = ArbOpportunity(
                market_a=market,
                market_b=market,  # Same market, both outcomes
                exchange_a=market.exchange,
                exchange_b=market.exchange,
                price_a=yes_price,
                price_b=no_price,
                spread=profit,
                expected_profit_pct=profit_pct,
                question=market.question,
                arb_type="internal",
            )
            opportunities.append(opp)

            log.info(
                "arb_scanner.internal_arb",
                exchange=market.exchange,
                market_id=market.id,
                question=market.question[:80],
                yes_price=round(yes_price, 3),
                no_price=round(no_price, 3),
                total=round(total, 3),
                profit_pct=round(profit_pct, 1),
            )

        return opportunities

    async def _fetch_all_markets(self) -> dict[str, list[Market]]:
        """Fetch markets from all exchanges concurrently.

        Returns:
            Dict mapping exchange name to list of markets.
            Exchanges that fail to fetch are omitted (not fatal).
        """

        async def _fetch_one(name: str, discovery: MarketDiscovery) -> tuple[str, list[Market]]:
            try:
                markets = await discovery.get_markets(active=True, limit=100)
                log.debug("arb_scanner.fetched", exchange=name, count=len(markets))
                return (name, markets)
            except Exception as e:
                log.warning("arb_scanner.fetch_failed", exchange=name, error=str(e))
                return (name, [])

        results = await asyncio.gather(
            *[_fetch_one(name, disc) for name, disc in self._discoveries.items()]
        )

        return {name: markets for name, markets in results if markets}
