"""Smart market selection — prioritize markets most likely to be mispriced.

Scoring is two-layer:
  1. Per-market factors (price, liquidity, spread, momentum, timing)
  2. Relational bonus (markets sharing entities with other batch members
     are more valuable — relational reasoning finds edges atomistic
     analysis misses)
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import structlog

from auramaur.exchange.models import Market

log = structlog.get_logger()

# Words too common to be meaningful entities
_STOP_WORDS = frozenset({
    "will", "the", "a", "an", "of", "in", "on", "by", "to", "be", "is",
    "at", "for", "and", "or", "not", "this", "that", "with", "before",
    "after", "between", "more", "than", "its", "has", "have", "from",
    "does", "are", "was", "were", "been", "being", "which", "their",
    "would", "could", "should", "may", "might", "can", "most", "over",
    "under", "above", "below", "any", "each", "every", "other",
})


def _extract_entities(text: str) -> set[str]:
    """Extract likely named entities (capitalized words/phrases)."""
    # Find capitalized words (2+ chars) that aren't sentence-starters
    words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    entities = set()
    for w in words:
        # Split multi-word matches and filter stop words from each part
        parts = [p for p in w.split() if p.lower() not in _STOP_WORDS and len(p) > 2]
        for part in parts:
            entities.add(part.lower())
    # Also grab ALL-CAPS acronyms (3+ letters)
    acronyms = re.findall(r"\b[A-Z]{3,}\b", text)
    for a in acronyms:
        if a.lower() not in _STOP_WORDS:
            entities.add(a.lower())
    return entities


def score_market(
    market: Market,
    price_history: dict[str, list[float]] | None = None,
) -> float:
    """Score a market for likelihood of mispricing. Higher = more promising."""
    score = 0.0
    price = market.outcome_yes_price

    # 1. Price sweet spot — edge is clearer away from 50/50 but not at extremes
    if 0.15 <= price <= 0.40 or 0.60 <= price <= 0.85:
        score += 2.0
    elif 0.40 < price < 0.60:
        score += 0.5
    else:
        score += 1.0

    # 2. Liquidity sweet spot — medium liquidity = less efficient pricing
    # Use max(liquidity, volume) because Kalshi reports thin top-of-book
    # liquidity but high volume on active markets.
    liq = max(market.liquidity or 0, market.volume or 0)
    if 1_000 <= liq < 10_000:
        score += 3.0
    elif 10_000 <= liq < 50_000:
        score += 2.0
    elif 50_000 <= liq < 200_000:
        score += 1.0
    else:
        score += 0.5

    # 3. Spread opportunity — wider spreads = more room for limit orders
    if market.spread > 0.02:
        score += 1.5
    elif market.spread > 0.01:
        score += 0.5

    # 4. Time to resolution — 1-7 day markets with non-extreme prices
    if market.end_date:
        now = datetime.now(timezone.utc)
        end = market.end_date if market.end_date.tzinfo else market.end_date.replace(tzinfo=timezone.utc)
        hours_left = max((end - now).total_seconds() / 3600, 0)

        if 24 < hours_left < 168 and 0.15 < price < 0.85:
            score += 2.0
        elif hours_left < 24:
            score += 0.5

    # 5. Volume / liquidity ratio — active repricing
    if liq > 0:
        vol_ratio = market.volume / liq
        if vol_ratio > 5:
            score += 1.0

    # 6. Recent price movement
    if price_history and market.id in price_history:
        history = price_history[market.id]
        if len(history) >= 2:
            recent_move = abs(history[-1] - history[0])
            if recent_move > 0.05:
                score += 2.5
            elif recent_move > 0.02:
                score += 1.0

        # 7. Momentum — large absolute move = repricing opportunity
        if len(history) >= 3:
            earliest = history[0]
            latest = history[-1]
            if earliest > 0.01:  # Guard against near-zero division
                momentum = abs((latest - earliest) / earliest)
                score += min(momentum * 2.0, 2.0)  # Cap at 2.0

    return score


def _compute_relational_bonus(markets: list[Market]) -> dict[str, float]:
    """Compute per-market bonus based on entity overlap with other markets.

    Markets that share entities with many other markets in the batch
    get a bonus — relational reasoning across a cluster of related
    markets is more likely to surface edges than analyzing isolated ones.
    """
    # Build entity sets per market
    market_entities: dict[str, set[str]] = {}
    for m in markets:
        text = f"{m.question} {m.description[:200]}"
        market_entities[m.id] = _extract_entities(text)

    # Count how many OTHER markets share each entity
    entity_market_count: dict[str, int] = {}
    for entities in market_entities.values():
        for e in entities:
            entity_market_count[e] = entity_market_count.get(e, 0) + 1

    # Shared entities (appear in 2+ markets) are the relational signal
    shared_entities = {e for e, count in entity_market_count.items() if count >= 2}

    if not shared_entities:
        return {}

    # Bonus = number of shared entities this market has, scaled
    bonuses: dict[str, float] = {}
    for m_id, entities in market_entities.items():
        shared_count = len(entities & shared_entities)
        if shared_count > 0:
            # Each shared entity contributes 0.5 points, max 3.0
            bonuses[m_id] = min(shared_count * 0.5, 3.0)

    if bonuses:
        log.debug(
            "market_selector.relational_bonus",
            shared_entities=sorted(shared_entities)[:10],
            markets_boosted=len(bonuses),
        )

    return bonuses


def rank_markets(
    markets: list[Market],
    price_history: dict[str, list[float]] | None = None,
) -> list[tuple[Market, float]]:
    """Rank markets by mispricing potential + relational connectivity."""
    # Phase 1: atomistic scores
    scored = [(m, score_market(m, price_history)) for m in markets]

    # Phase 2: relational bonus — markets sharing entities get boosted
    relational = _compute_relational_bonus(markets)
    scored = [
        (m, s + relational.get(m.id, 0.0))
        for m, s in scored
    ]

    scored.sort(key=lambda x: x[1], reverse=True)

    log.info(
        "market_selector.ranked",
        total=len(scored),
        top_5=[(m.id[:12], round(s, 1)) for m, s in scored[:5]],
        relational_boosts=len(relational),
    )
    return scored
