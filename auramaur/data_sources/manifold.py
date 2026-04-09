"""Manifold Markets community forecast source — calibrated crowd predictions.

Fetches community probabilities from the Manifold Markets API for related
questions and presents them as evidence.  Manifold has deep coverage of niche
prediction markets with well-calibrated probabilities.

No API key required for public market data.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone

import aiohttp
import structlog

from auramaur.data_sources.base import NewsItem

logger = structlog.get_logger(__name__)

# Manifold public API base
_API_BASE = "https://api.manifold.markets/v0"

# Rate limit: be respectful — 1 request per second
_REQUEST_INTERVAL = 1.0


class ManifoldSource:
    """Fetches community probabilities from Manifold Markets as contextual evidence."""

    source_name: str = "manifold"

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None
        self._last_request: float = 0.0

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=15),
            )
        return self._session

    async def _rate_limit(self) -> None:
        """Enforce minimum interval between requests."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request
        if elapsed < _REQUEST_INTERVAL:
            await asyncio.sleep(_REQUEST_INTERVAL - elapsed)
        self._last_request = asyncio.get_event_loop().time()

    async def fetch(self, query: str, limit: int = 10) -> list[NewsItem]:
        """Search Manifold for related binary markets and return community probabilities."""
        try:
            session = await self._get_session()
            await self._rate_limit()

            # Extract key terms for search
            words = query.split()
            search_terms = [w for w in words if len(w) > 2][:5]
            search_query = " ".join(search_terms)

            if not search_query:
                return []

            params = {
                "term": search_query,
                "filter": "open",
                "sort": "liquidity",
                "limit": str(limit * 2),  # Fetch extra, filter to binary
            }

            async with session.get(f"{_API_BASE}/search-markets", params=params) as resp:
                if resp.status != 200:
                    logger.warning(
                        "manifold.api_error",
                        status=resp.status,
                        query=search_query,
                    )
                    return []

                markets = await resp.json()

            items: list[NewsItem] = []

            for market in markets:
                # Only binary markets have a single probability
                if market.get("outcomeType") != "BINARY":
                    continue

                probability = market.get("probability")
                if probability is None:
                    continue

                title_text = market.get("question", "")
                market_id = market.get("id", "")
                slug = market.get("slug", "")
                creator = market.get("creatorUsername", "")
                volume = market.get("volume", 0)
                liquidity = market.get("totalLiquidity", 0)
                unique_bettors = market.get("uniqueBettorCount", 0)

                content = (
                    f"MANIFOLD MARKETS FORECAST: {title_text}\n"
                    f"Community probability: {probability:.0%}\n"
                    f"Unique bettors: {unique_bettors}\n"
                    f"Volume: ${volume:,.0f}\n"
                    f"Liquidity: ${liquidity:,.0f}\n"
                    f"\nManifold Markets uses play money with real-money"
                    f" prizes. Higher liquidity and bettor count = more"
                    f" reliable signal."
                )

                item_id = hashlib.sha256(
                    f"manifold:{market_id}:{datetime.now().date()}".encode()
                ).hexdigest()[:16]

                # Relevance weighted by unique bettors and liquidity
                # 100+ bettors = high relevance, <10 = low
                bettor_score = min(2.0, unique_bettors / 50) if unique_bettors > 0 else 0.2
                liquidity_score = min(1.0, liquidity / 5000) if liquidity > 0 else 0.1
                relevance = min(3.0, bettor_score + liquidity_score)

                url = f"https://manifold.markets/{creator}/{slug}" if creator and slug else ""

                items.append(
                    NewsItem(
                        id=item_id,
                        source=self.source_name,
                        title=f"[Manifold: {probability:.0%}] {title_text[:80]}",
                        content=content,
                        url=url,
                        published_at=datetime.now(timezone.utc),
                        relevance_score=relevance,
                        keywords=[market_id],
                    )
                )

                if len(items) >= limit:
                    break

            logger.info(
                "manifold_fetched",
                count=len(items),
                query=search_query,
            )
            return items

        except asyncio.TimeoutError:
            logger.warning("manifold.timeout", query=query[:40])
            return []
        except Exception as e:
            logger.warning("manifold.fetch_failed", error=str(e)[:60])
            return []

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
