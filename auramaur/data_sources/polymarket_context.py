"""Polymarket context source — other markets as relational evidence.

Fetches related Polymarket markets and presents their current prices
as evidence.  This gives Claude "what the crowd thinks" about related
questions — a Bayesian prior from the market itself.

Example: When analyzing "Will Trump impose tariffs?", this source
finds related markets like "Will S&P 500 drop 10%?" and "Will China
retaliate?" and shows their prices.  The relational structure of
these prices IS evidence.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import structlog

from auramaur.data_sources.base import NewsItem

logger = structlog.get_logger(__name__)


class PolymarketContextSource:
    """Fetches related Polymarket markets as contextual evidence."""

    source_name: str = "polymarket_context"

    def __init__(self):
        self._gamma = None

    async def _get_gamma(self):
        if self._gamma is None:
            from auramaur.exchange.gamma import GammaClient
            self._gamma = GammaClient()
        return self._gamma

    async def fetch(self, query: str, limit: int = 10) -> list[NewsItem]:
        """Search Polymarket for related markets and return as evidence."""
        try:
            gamma = await self._get_gamma()

            # Extract key entities from query for broader search
            words = query.split()
            # Search with the first 4-5 meaningful words
            search_terms = [w for w in words if len(w) > 2][:5]
            search_query = " ".join(search_terms)

            if not search_query:
                return []

            related = await gamma.search_markets(search_query, limit=limit * 2)

            items: list[NewsItem] = []
            for market in related[:limit]:
                if not market.active:
                    continue

                yes_price = market.outcome_yes_price
                no_price = market.outcome_no_price

                content = (
                    f"POLYMARKET CROWD ESTIMATE: {market.question}\n"
                    f"YES: {yes_price:.0%} | NO: {no_price:.0%}\n"
                    f"Volume: ${market.volume:,.0f} | "
                    f"Liquidity: ${market.liquidity:,.0f}\n"
                    f"Category: {market.category or 'uncategorized'}\n"
                    f"End: {market.end_date.isoformat() if market.end_date else 'unknown'}\n"
                    f"\nThis is what the prediction market crowd currently "
                    f"believes about this related question.  Higher volume "
                    f"and liquidity = more informed crowd."
                )

                item_id = hashlib.sha256(
                    f"poly:{market.id}:{datetime.now().date()}".encode()
                ).hexdigest()[:16]

                # Weight by liquidity — high-liquidity markets are better priors
                relevance = min(3.0, market.liquidity / 5000) if market.liquidity > 0 else 0.5

                items.append(
                    NewsItem(
                        id=item_id,
                        source=self.source_name,
                        title=f"[Crowd: {yes_price:.0%}] {market.question[:80]}",
                        content=content,
                        url=f"https://polymarket.com/event/{market.id}",
                        published_at=datetime.now(timezone.utc),
                        relevance_score=relevance,
                        keywords=[market.category or "", market.id],
                    )
                )

            logger.info(
                "polymarket_context_fetched",
                count=len(items),
                query=search_query,
            )
            return items

        except Exception as e:
            logger.warning("polymarket_context_failed", error=str(e)[:60])
            return []

    async def close(self) -> None:
        if self._gamma:
            await self._gamma.close()
            self._gamma = None
