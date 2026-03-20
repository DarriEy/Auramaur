"""Smart market selection — prioritize markets most likely to be mispriced."""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from auramaur.exchange.models import Market

log = structlog.get_logger()


def score_market(market: Market, price_history: dict[str, list[float]] | None = None) -> float:
    """Score a market for likelihood of mispricing. Higher = more promising.

    Factors:
    - Newness: recently listed markets have less efficient pricing
    - Price extremity: markets near 50% are hardest to trade; 20-40% and 60-80% are sweet spots
    - Liquidity sweet spot: enough to trade ($1k+) but not so much it's efficiently priced (<$100k)
    - Recent price movement: significant recent moves suggest information events
    - Time to resolution: markets near expiry with non-extreme prices may be mispriced
    """
    score = 0.0

    # 1. Price sweet spot — edge is clearer away from 50/50 but not at extremes
    price = market.outcome_yes_price
    if 0.15 <= price <= 0.40 or 0.60 <= price <= 0.85:
        score += 2.0  # Sweet spot for edge detection
    elif 0.40 < price < 0.60:
        score += 0.5  # Hard to have edge on coin-flip markets
    else:
        score += 1.0  # Very extreme prices — could be real or could be mispriced

    # 2. Liquidity sweet spot — medium liquidity = less efficient pricing
    liq = market.liquidity
    if 1_000 <= liq < 10_000:
        score += 3.0  # Low-ish liquidity — least efficient
    elif 10_000 <= liq < 50_000:
        score += 2.0  # Medium — reasonable
    elif 50_000 <= liq < 200_000:
        score += 1.0  # High liquidity — fairly efficient
    else:
        score += 0.5  # Very high — probably very efficient

    # 3. Spread opportunity — wider spreads = more room for limit orders
    if market.spread > 0.02:
        score += 1.5
    elif market.spread > 0.01:
        score += 0.5

    # 4. Time to resolution — markets 1-7 days from expiry with non-extreme prices
    if market.end_date:
        now = datetime.now(timezone.utc)
        end = market.end_date if market.end_date.tzinfo else market.end_date.replace(tzinfo=timezone.utc)
        hours_left = max((end - now).total_seconds() / 3600, 0)

        if 24 < hours_left < 168:  # 1-7 days
            # Near-term markets with non-extreme prices are often mispriced
            if 0.15 < price < 0.85:
                score += 2.0
        elif hours_left < 24:
            score += 0.5  # Too close — resolution risk

    # 5. Volume relative to liquidity — high volume/liquidity ratio suggests active repricing
    if liq > 0:
        vol_ratio = market.volume / liq
        if vol_ratio > 5:
            score += 1.0  # Very active trading relative to depth

    # 6. Recent price movement from history (if available)
    if price_history and market.id in price_history:
        history = price_history[market.id]
        if len(history) >= 2:
            recent_move = abs(history[-1] - history[0])
            if recent_move > 0.05:
                score += 2.5  # Big move — information event
            elif recent_move > 0.02:
                score += 1.0  # Moderate move

    # 7. Momentum scoring — large absolute price movement = repricing opportunity
    if price_history and market.id in price_history:
        history = price_history[market.id]
        if len(history) >= 3:
            earliest = history[0]
            latest = history[-1]
            if earliest > 0:
                momentum = abs((latest - earliest) / earliest)
            else:
                momentum = 0.0
            # Weight momentum at 20% of a strong signal (~2.0 max contribution)
            # Both directions are interesting — abs() already applied
            score += momentum * 10.0 * 0.2  # scale so 100% move = 2.0 points

    return score


def rank_markets(
    markets: list[Market],
    price_history: dict[str, list[float]] | None = None,
) -> list[tuple[Market, float]]:
    """Rank markets by mispricing potential, highest score first."""
    scored = [(m, score_market(m, price_history)) for m in markets]
    scored.sort(key=lambda x: x[1], reverse=True)

    log.info(
        "market_selector.ranked",
        total=len(scored),
        top_3=[(m.id[:12], round(s, 1)) for m, s in scored[:3]],
    )
    return scored
