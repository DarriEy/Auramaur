"""NewsAPI data source."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone

import aiohttp
import structlog

from auramaur.data_sources.base import DataSource, NewsItem

logger = structlog.get_logger(__name__)

_EVERYTHING_URL = "https://newsapi.org/v2/everything"
_REQUEST_INTERVAL = 1.5
_BACKOFF_DELAYS = [2, 5, 15]


class NewsAPISource:
    """Fetch articles from NewsAPI.org."""

    source_name: str = "newsapi"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._session: aiohttp.ClientSession | None = None
        self._last_request: float = 0.0

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
        return self._session

    async def _rate_limit(self) -> None:
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request
        if elapsed < _REQUEST_INTERVAL:
            await asyncio.sleep(_REQUEST_INTERVAL - elapsed)
        self._last_request = asyncio.get_event_loop().time()

    async def fetch(self, query: str, limit: int = 20) -> list[NewsItem]:
        """Fetch news articles matching *query* from NewsAPI."""
        session = await self._get_session()
        params = {
            "q": query,
            "pageSize": min(limit, 100),
            "sortBy": "publishedAt",
            "apiKey": self._api_key,
        }
        data = None
        for attempt, delay in enumerate(_BACKOFF_DELAYS):
            try:
                await self._rate_limit()
                async with session.get(_EVERYTHING_URL, params=params) as resp:
                    if resp.status == 401:
                        logger.warning("newsapi_unauthorized", hint="Check your NEWSAPI_KEY")
                        return []
                    if resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", delay))
                        logger.warning("newsapi_rate_limited", retry_after=retry_after, attempt=attempt)
                        await asyncio.sleep(retry_after)
                        continue
                    resp.raise_for_status()
                    data = await resp.json()
                    break
            except Exception as e:
                logger.warning("newsapi_fetch_failed", query=query[:50], error=str(e)[:80])
                return []
        if data is None:
            return []

        articles = data.get("articles", [])
        items: list[NewsItem] = []
        for article in articles[:limit]:
            title = article.get("title") or ""
            content = article.get("description") or article.get("content") or ""
            url = article.get("url") or ""
            published_str = article.get("publishedAt") or ""
            try:
                published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                published_at = datetime.now(timezone.utc)

            item_id = hashlib.sha256(f"{url}:{title}".encode()).hexdigest()[:16]
            items.append(
                NewsItem(
                    id=item_id,
                    source=self.source_name,
                    title=title,
                    content=content,
                    url=url,
                    published_at=published_at,
                )
            )
        logger.info("newsapi_fetched", count=len(items), query=query)
        return items

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
