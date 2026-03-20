"""RSS feed data source using feedparser + aiohttp."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import aiohttp
import feedparser  # type: ignore[import-untyped]
import structlog

from auramaur.data_sources.base import DataSource, NewsItem

logger = structlog.get_logger(__name__)

_DEFAULT_FEEDS: list[str] = [
    # Major news
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://feeds.npr.org/1001/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "https://www.theguardian.com/world/rss",
    "https://feeds.reuters.com/reuters/topNews",
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "https://www.politico.com/rss/politico-today.xml",
    # Politics & policy
    "https://www.politico.com/rss/congress.xml",
    "https://thehill.com/feed/",
    "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
    # Tech
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://www.theverge.com/rss/index.xml",
    "https://techcrunch.com/feed/",
    # Finance/economics
    "https://feeds.bloomberg.com/markets/news.rss",
    "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    # Crypto
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    # Geopolitics
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    # Science/space
    "https://www.nasa.gov/rss/dyn/breaking_news.rss",
    "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml",
]


class RSSSource:
    """Aggregate and parse multiple RSS feeds."""

    source_name: str = "rss"

    def __init__(self, feed_urls: list[str] | None = None) -> None:
        self._feed_urls = feed_urls or _DEFAULT_FEEDS
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _fetch_feed(self, url: str, query: str, limit: int) -> list[NewsItem]:
        session = await self._get_session()
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                resp.raise_for_status()
                body = await resp.text()
        except Exception as e:
            logger.debug("rss_feed_unreachable", url=url, error=str(e)[:60])
            return []

        feed = feedparser.parse(body)
        items: list[NewsItem] = []

        # Build keyword set from query for fuzzy matching
        _STOP = {"will", "the", "a", "an", "of", "in", "on", "by", "to", "be", "is", "at", "for", "and", "or", "not", "this", "that", "with"}
        query_words = set()
        if query:
            query_words = {w.lower() for w in query.split() if len(w) > 2 and w.lower() not in _STOP}

        for entry in feed.entries[:limit]:
            title = entry.get("title", "")
            summary = entry.get("summary", "")

            # Fuzzy keyword matching: need 2+ query words to appear in title/summary
            if query_words:
                text_lower = f"{title} {summary}".lower()
                matches = sum(1 for w in query_words if w in text_lower)
                if matches < min(2, len(query_words)):
                    continue

            link = entry.get("link", "")
            published_str = entry.get("published", "")
            try:
                published_at = parsedate_to_datetime(published_str)
            except Exception:
                published_at = datetime.now(timezone.utc)

            item_id = hashlib.sha256(f"rss:{link}:{title}".encode()).hexdigest()[:16]
            items.append(
                NewsItem(
                    id=item_id,
                    source=self.source_name,
                    title=title,
                    content=summary,
                    url=link,
                    published_at=published_at,
                )
            )

        return items

    async def fetch(self, query: str, limit: int = 20) -> list[NewsItem]:
        """Fetch and filter RSS entries matching *query*."""
        import asyncio

        tasks = [self._fetch_feed(url, query, limit) for url in self._feed_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        items: list[NewsItem] = []
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("rss_feed_error", error=str(result))
                continue
            items.extend(result)

        items.sort(key=lambda n: n.published_at, reverse=True)
        logger.info("rss_fetched", count=len(items[:limit]), query=query)
        return items[:limit]

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
