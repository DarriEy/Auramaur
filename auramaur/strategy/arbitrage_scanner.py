"""Cross-platform arbitrage scanner.

Scans across all configured exchanges for price mismatches on equivalent
markets, and checks within-exchange invariants (YES + NO should sum to ~1.0).
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import structlog

from auramaur.exchange.models import Market
from auramaur.exchange.protocols import MarketDiscovery
from auramaur.nlp.errors import BudgetExhausted
from auramaur.strategy.signals import taker_fee_rate

log = structlog.get_logger()

# Minimum spread to qualify as a cross-exchange arb opportunity
CROSS_EXCHANGE_MIN_SPREAD = 0.03  # 3%

# Risk checks reject arbs inside this window. Filter before logging/execution
# so near-expiry mismatches don't consume LLM matching and risk cycles forever.
CROSS_EXCHANGE_MIN_HOURS_TO_RESOLUTION = 4.0

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


@dataclass
class NegRiskArbOpportunity:
    """A buy-all-NO arbitrage across a NegRisk multi-outcome event.

    A NegRisk event is a set of N mutually-exclusive, collectively-exhaustive
    binary outcomes (exactly one resolves YES). Buying NO on every outcome
    costs ``sum(no_price_i)`` and pays ``$1`` for each of the N-1 losing
    outcomes at resolution. When ``sum(no_price_i) < (N-1)`` the package is
    risk-free profit (equivalently, the YES legs are collectively > $1).
    """

    markets: list[Market]
    exchange: str
    neg_risk_market_id: str
    total_no_cost: float          # sum of NO prices across all legs
    guaranteed_payout: float      # (N - 1), net of fees
    expected_profit_pct: float    # profit / cost, as a percentage
    question: str
    arb_type: str = "negrisk"

    @property
    def n_outcomes(self) -> int:
        return len(self.markets)


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


def _hours_to_resolution(market: Market, now: datetime) -> float | None:
    if market.end_date is None:
        return None
    end = market.end_date
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    return (end - now).total_seconds() / 3600.0


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
        analyzer: Any = None,
        exchange_fees: dict[str, float] | None = None,
        min_profit_after_fees_pct: float = 1.5,
        blocked_categories: list[str] | None = None,
        allowed_categories_live: list[str] | None = None,
    ) -> None:
        """Initialize with the discoveries dict from bot components.

        Args:
            discoveries: Mapping of exchange name to MarketDiscovery instance.
                         e.g. {"polymarket": gamma, "kalshi": kalshi_client}
            analyzer: ClaudeAnalyzer instance for data-driven matching.
            exchange_fees: Mapping of exchange name to fee rate (0.0-1.0).
                          e.g. {"polymarket": 0.0, "kalshi": 0.07}
            min_profit_after_fees_pct: Minimum profit % after fees to flag
                                       as an opportunity.
            blocked_categories: categories never scanned (an arb is hedged
                only when BOTH legs fill — a single-leg fill is directional
                inventory in a banned market; observed live 2026-06-12
                quoting KBO baseball).
            allowed_categories_live: when set (live mode), only these
                categories are scanned — same fail-closed policy as the
                directional books. None disables the allowlist (paper).
        """
        self._discoveries = discoveries
        self._analyzer = analyzer
        self._exchange_fees = exchange_fees or {}
        self._min_profit_after_fees_pct = min_profit_after_fees_pct
        self._blocked_categories = set(blocked_categories or [])
        self._allowed_categories_live = (
            set(allowed_categories_live) if allowed_categories_live is not None
            else None)

    def _category_ok(self, market: Market) -> bool:
        """Category gate for scan candidates (stored label or fresh classify)."""
        from auramaur.strategy.classifier import ensure_category
        category = ensure_category(
            market.question, market.description, market.category)
        if category in self._blocked_categories:
            return False
        if (self._allowed_categories_live is not None
                and category not in self._allowed_categories_live):
            return False
        return True

    async def scan(self) -> list[ArbOpportunity]:
        """Scan all exchanges for price mismatches on equivalent markets.

        Strategy:
        1. Fetch markets from each exchange concurrently.
        2. Match equivalent markets across exchanges using LLM batch pairing.
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
        now = datetime.now(timezone.utc)
        for i in range(len(exchange_names)):
            for j in range(i + 1, len(exchange_names)):
                name_a = exchange_names[i]
                name_b = exchange_names[j]
                markets_a = exchange_markets[name_a]
                markets_b = exchange_markets[name_b]

                matched = await self._match_markets(markets_a, markets_b, name_a, name_b)

                for market_a, market_b in matched:
                    hours_a = _hours_to_resolution(market_a, now)
                    hours_b = _hours_to_resolution(market_b, now)
                    min_hours = min(
                        h for h in (hours_a, hours_b) if h is not None
                    ) if hours_a is not None or hours_b is not None else None
                    if (
                        min_hours is not None
                        and min_hours < CROSS_EXCHANGE_MIN_HOURS_TO_RESOLUTION
                    ):
                        log.debug(
                            "arb_scanner.cross_filtered",
                            reason="near_resolution",
                            exchange_a=name_a,
                            exchange_b=name_b,
                            question=market_a.question[:80],
                            hours_to_resolution=round(min_hours, 2),
                        )
                        continue

                    spread = abs(market_a.outcome_yes_price - market_b.outcome_yes_price)

                    if spread < CROSS_EXCHANGE_MIN_SPREAD:
                        continue

                    # Identify cheap/expensive sides for fee calc
                    if market_a.outcome_yes_price <= market_b.outcome_yes_price:
                        cheap_exchange, expensive_exchange = name_a, name_b
                        cheap_mkt, exp_mkt = market_a, market_b
                    else:
                        cheap_exchange, expensive_exchange = name_b, name_a
                        cheap_mkt, exp_mkt = market_b, market_a

                    # Fee-aware profit: buy YES cheap, buy NO on the expensive side.
                    # Both legs are CROSSING (taker) buys, so each pays its real
                    # taker fee: rate * price * (1-price) per share (p(1-p) is
                    # symmetric, so the NO leg uses the expensive YES price). The
                    # Polymarket rate is per-category (was modeled as 0, which
                    # overstated every cross-exchange arb that crosses a Poly leg).
                    cheap_yes = cheap_mkt.outcome_yes_price
                    exp_yes = exp_mkt.outcome_yes_price
                    fee_cheap = (taker_fee_rate(cheap_exchange, cheap_mkt.category,
                                                self._exchange_fees)
                                 * cheap_yes * (1.0 - cheap_yes))
                    fee_expensive = (taker_fee_rate(expensive_exchange, exp_mkt.category,
                                                    self._exchange_fees)
                                     * exp_yes * (1.0 - exp_yes))
                    net_profit = spread - fee_cheap - fee_expensive
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

    async def _match_markets(
        self,
        markets_a: list[Market],
        markets_b: list[Market],
        name_a: str = "",
        name_b: str = "",
    ) -> list[tuple[Market, Market]]:
        """Match equivalent markets across exchanges.

        Uses LLM batch pairing for high-recall data-driven matching, falling
        back to word-overlap scoring if the analyzer is unavailable.

        Returns:
            List of (market_a, market_b) tuples that are likely the same question.
        """
        if not markets_a or not markets_b:
            return []

        # Strategy A: LLM Batch Pairing (Preferred)
        if self._analyzer:
            try:
                from auramaur.nlp.prompts import BATCH_ARBITRAGE_MATCHING_PROMPT
                from auramaur.nlp.analyzer import _parse_claude_json

                # Format minimal market lists for the prompt to save tokens
                list_a = [{"id": m.id, "question": m.question} for m in markets_a]
                list_b = [{"id": m.id, "question": m.question} for m in markets_b]

                prompt = BATCH_ARBITRAGE_MATCHING_PROMPT.format(
                    exchange_a=name_a or "Exchange A",
                    exchange_b=name_b or "Exchange B",
                    markets_a_json=re.sub(r"\s+", " ", str(list_a)),
                    markets_b_json=re.sub(r"\s+", " ", str(list_b)),
                )

                raw = await self._analyzer._call_claude_cli(prompt)
                matches = _parse_claude_json(raw)

                if not isinstance(matches, list):
                    log.warning("arb_scanner.llm_match_invalid_format", raw=raw[:200])
                    return []

                # Map IDs back to Market objects
                map_a = {m.id: m for m in markets_a}
                map_b = {m.id: m for m in markets_b}
                results: list[tuple[Market, Market]] = []

                for m in matches:
                    id_a, id_b = m.get("id_a"), m.get("id_b")
                    if id_a in map_a and id_b in map_b:
                        results.append((map_a[id_a], map_b[id_b]))

                if results:
                    log.info(
                        "arb_scanner.llm_matched",
                        count=len(results),
                        exchange_a=name_a,
                        exchange_b=name_b,
                    )
                return results

            except BudgetExhausted as e:
                # Expected throttle, not a failure — the word-overlap fallback
                # below still matches pairs without an LLM call.
                log.debug("arb_scanner.llm_budget_skipped", error=str(e))
            except Exception as e:
                log.error("arb_scanner.llm_match_failed", error=str(e))
                # Fall through to word-overlap fallback

        # Strategy B: Word-Overlap Fallback
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
                "arb_scanner.word_overlap_matched",
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
                markets = [m for m in markets if self._category_ok(m)]
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

    async def scan_negrisk(
        self,
        markets_by_exchange: dict[str, list[Market]] | None = None,
    ) -> list[NegRiskArbOpportunity]:
        """Find buy-all-NO arbs across NegRisk multi-outcome events.

        Groups markets by their shared ``neg_risk_market_id`` (only Polymarket
        tags these). For each group of N>=2 mutually-exclusive outcomes, if the
        sum of NO prices is below the guaranteed (N-1) payout — net of the
        exchange fee — the package is risk-free profit.

        Args:
            markets_by_exchange: Pre-fetched markets (avoids refetching). If
                omitted, fetches from all exchanges.
        """
        if markets_by_exchange is None:
            markets_by_exchange = await self._fetch_all_markets()

        opportunities: list[NegRiskArbOpportunity] = []

        for exchange_name, markets in markets_by_exchange.items():
            # Group mutually-exclusive legs by their NegRisk grouping key.
            groups: dict[str, list[Market]] = {}
            for m in markets:
                if not m.neg_risk or not m.neg_risk_market_id:
                    continue
                groups.setdefault(m.neg_risk_market_id, []).append(m)

            for group_id, legs in groups.items():
                n = len(legs)
                if n < 2:
                    continue  # need at least two outcomes for a package

                # Skip groups with unpriced legs — can't trust the sum.
                if any(leg.outcome_no_price <= 0 for leg in legs):
                    continue

                # Per-category taker rate (legs of one group share a category);
                # Polymarket NegRisk legs were modeled fee-free (0.0) before.
                fee = taker_fee_rate(exchange_name, legs[0].category,
                                     self._exchange_fees)

                total_no_cost = sum(leg.outcome_no_price for leg in legs)
                # Exactly one outcome wins: its NO pays $0, the other (n-1) pay
                # $1 each. Fees apply to the winning legs' payout.
                guaranteed_payout = (n - 1) * (1.0 - fee)

                profit = guaranteed_payout - total_no_cost
                if profit <= 0 or total_no_cost <= 0:
                    continue

                profit_pct = (profit / total_no_cost) * 100
                if profit_pct < self._min_profit_after_fees_pct:
                    continue

                question = (legs[0].question or group_id)
                opp = NegRiskArbOpportunity(
                    markets=legs,
                    exchange=exchange_name,
                    neg_risk_market_id=group_id,
                    total_no_cost=total_no_cost,
                    guaranteed_payout=guaranteed_payout,
                    expected_profit_pct=profit_pct,
                    question=question,
                )
                opportunities.append(opp)

                log.info(
                    "arb_scanner.negrisk_arb",
                    exchange=exchange_name,
                    neg_risk_market_id=group_id,
                    n_outcomes=n,
                    total_no_cost=round(total_no_cost, 3),
                    guaranteed_payout=round(guaranteed_payout, 3),
                    profit_pct=round(profit_pct, 2),
                    question=question[:80],
                )

        opportunities.sort(key=lambda o: o.expected_profit_pct, reverse=True)
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
                markets = [m for m in markets if self._category_ok(m)]
                log.debug("arb_scanner.fetched", exchange=name, count=len(markets))
                return (name, markets)
            except Exception as e:
                log.warning("arb_scanner.fetch_failed", exchange=name, error=str(e))
                return (name, [])

        results = await asyncio.gather(
            *[_fetch_one(name, disc) for name, disc in self._discoveries.items()]
        )

        return {name: markets for name, markets in results if markets}
