"""Metaculus community forecast source — well-calibrated crowd predictions.

Fetches community predictions from the Metaculus API for related questions
and presents them as evidence.  Metaculus forecasters are among the best-
calibrated public prediction communities, making this a high-value
second-opinion signal.

No API key required for community predictions.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone

import aiohttp
import structlog

from auramaur.data_sources.base import NewsItem

logger = structlog.get_logger(__name__)

# Metaculus public API base
_API_BASE = "https://www.metaculus.com/api2"

# Rate limit: be respectful — 1 request per second
_REQUEST_INTERVAL = 1.0


class MetaculusSource:
    """Fetches community predictions from Metaculus as contextual evidence."""

    source_name: str = "metaculus"

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
        """Search Metaculus for related binary questions and return community predictions."""
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
                "search": search_query,
                "status": "open",
                "type": "binary",
                "limit": limit * 2,  # Fetch extra, filter later
                "order_by": "-activity",
            }

            async with session.get(f"{_API_BASE}/questions/", params=params) as resp:
                if resp.status != 200:
                    logger.warning(
                        "metaculus.api_error",
                        status=resp.status,
                        query=search_query,
                    )
                    return []

                data = await resp.json()

            results = data.get("results", [])
            items: list[NewsItem] = []

            for question in results[:limit]:
                # Extract community prediction
                community_prediction = question.get("community_prediction", {})
                if not community_prediction:
                    continue

                # Full distribution has 'full' key with 'q2' (median)
                full = community_prediction.get("full", {})
                median_prob = full.get("q2")
                if median_prob is None:
                    # Fallback to older API format
                    median_prob = community_prediction.get("full")

                if median_prob is None:
                    continue

                title_text = question.get("title", "")
                question_id = question.get("id", 0)
                forecaster_count = question.get("number_of_forecasters", 0)
                resolution_date = question.get("resolve_time", "")
                created = question.get("created_time", "")

                content = (
                    f"METACULUS COMMUNITY FORECAST: {title_text}\n"
                    f"Community median: {median_prob:.0%}\n"
                    f"Forecasters: {forecaster_count}\n"
                    f"Resolution: {resolution_date[:10] if resolution_date else 'unknown'}\n"
                    f"\nMetaculus community forecasts are well-calibrated. "
                    f"Higher forecaster count = more reliable signal."
                )

                item_id = hashlib.sha256(
                    f"metaculus:{question_id}:{datetime.now().date()}".encode()
                ).hexdigest()[:16]

                # Relevance weighted by forecaster count
                # 50+ forecasters = high relevance, <10 = low
                relevance = min(3.0, forecaster_count / 20) if forecaster_count > 0 else 0.3

                items.append(
                    NewsItem(
                        id=item_id,
                        source=self.source_name,
                        title=f"[Metaculus: {median_prob:.0%}] {title_text[:80]}",
                        content=content,
                        url=f"https://www.metaculus.com/questions/{question_id}/",
                        published_at=datetime.now(timezone.utc),
                        relevance_score=relevance,
                        keywords=[str(question_id)],
                    )
                )

            logger.info(
                "metaculus_fetched",
                count=len(items),
                query=search_query,
            )
            return items

        except asyncio.TimeoutError:
            logger.warning("metaculus.timeout", query=query[:40])
            return []
        except Exception as e:
            logger.warning("metaculus.fetch_failed", error=str(e)[:60])
            return []

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
