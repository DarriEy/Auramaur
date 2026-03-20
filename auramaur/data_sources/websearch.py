"""Web search data source using DuckDuckGo (no API key required)."""

from __future__ import annotations

import asyncio
import hashlib
import time
from datetime import datetime, timezone

import logging
import structlog
try:
    from ddgs import DDGS  # type: ignore[import-untyped]
except ImportError:
    from duckduckgo_search import DDGS  # type: ignore[import-untyped]

# Suppress noisy Yahoo News parser warnings from ddgs
logging.getLogger("ddgs.engines.yahoo_news").setLevel(logging.ERROR)

from auramaur.data_sources.base import NewsItem

logger = structlog.get_logger(__name__)


class WebSearchSource:
    """Search DuckDuckGo for recent news and web results."""

    source_name: str = "web"

    def __init__(self) -> None:
        self._last_request_time: float = 0.0
        self._min_interval: float = 1.0  # max 1 request per second

    async def _rate_limit(self) -> None:
        """Enforce minimum interval between requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_request_time = time.monotonic()

    async def fetch(self, query: str, limit: int = 20) -> list[NewsItem]:
        """Fetch web search results matching *query*."""
        items: list[NewsItem] = []

        # Fetch news results
        try:
            await self._rate_limit()
            news_items = await asyncio.to_thread(self._search_news, query, limit)
            items.extend(news_items)
        except Exception as e:
            logger.warning("websearch_news_error", error=str(e)[:120], query=query)

        # Fetch general web results if we need more
        if len(items) < limit:
            try:
                await self._rate_limit()
                remaining = limit - len(items)
                web_items = await asyncio.to_thread(self._search_text, query, remaining)
                items.extend(web_items)
            except Exception as e:
                logger.warning("websearch_text_error", error=str(e)[:120], query=query)

        items.sort(key=lambda n: n.published_at, reverse=True)
        logger.info("websearch_fetched", count=len(items[:limit]), query=query)
        return items[:limit]

    def _search_news(self, query: str, limit: int) -> list[NewsItem]:
        """Search DuckDuckGo news (sync, run via to_thread)."""
        items: list[NewsItem] = []
        with DDGS() as ddgs:
            results = ddgs.news(query, max_results=limit)
            for r in results:
                published_at = self._parse_date(r.get("date"))
                title = r.get("title", "")
                url = r.get("url", "")
                item_id = hashlib.sha256(f"web:news:{url}:{title}".encode()).hexdigest()[:16]
                items.append(
                    NewsItem(
                        id=item_id,
                        source=self.source_name,
                        title=title,
                        content=r.get("body", ""),
                        url=url,
                        published_at=published_at,
                    )
                )
        return items

    def _search_text(self, query: str, limit: int) -> list[NewsItem]:
        """Search DuckDuckGo web results (sync, run via to_thread)."""
        items: list[NewsItem] = []
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=limit)
            for r in results:
                title = r.get("title", "")
                url = r.get("href", "")
                item_id = hashlib.sha256(f"web:text:{url}:{title}".encode()).hexdigest()[:16]
                items.append(
                    NewsItem(
                        id=item_id,
                        source=self.source_name,
                        title=title,
                        content=r.get("body", ""),
                        url=url,
                        published_at=datetime.now(timezone.utc),
                    )
                )
        return items

    @staticmethod
    def _parse_date(date_str: str | None) -> datetime:
        """Best-effort parse of date string from DDG results."""
        if not date_str:
            return datetime.now(timezone.utc)
        for fmt in (
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%a, %d %b %Y %H:%M:%S %z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return datetime.now(timezone.utc)

    async def close(self) -> None:
        """No persistent resources to clean up."""
        pass
