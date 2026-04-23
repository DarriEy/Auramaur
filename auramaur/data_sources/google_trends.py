"""Google Trends 'daily trending searches' RSS.

Google publishes a public RSS feed of top trending searches per country.
No auth, no rate limit in practice for light use. We use it as a
relevance signal: does the public have surging interest in a topic
related to the market? Unlike news sources it captures search-intent
rather than journalist-curated coverage.

Note: Google doesn't offer a stable programmatic trends API. Third-party
libraries (pytrends, etc.) can break when the internal endpoint moves.
The RSS feed has been stable for years and is the safest choice.
"""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import aiohttp
import structlog

from auramaur.data_sources.base import NewsItem

logger = structlog.get_logger(__name__)

_FEED_URL = "https://trends.google.com/trending/rss?geo={geo}"


class GoogleTrendsSource:
    """Top trending Google searches for a given region — category-agnostic."""

    source_name: str = "google_trends"
    categories: set[str] | None = None

    def __init__(self, geo: str = "US") -> None:
        self._geo = geo

    async def fetch(self, query: str, limit: int = 20) -> list[NewsItem]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    _FEED_URL.format(geo=self._geo),
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        logger.debug("trends.bad_status", status=resp.status)
                        return []
                    text = await resp.text()
        except Exception as e:
            logger.debug("trends.fetch_error", error=str(e)[:120])
            return []

        try:
            root = ET.fromstring(text)
        except ET.ParseError as e:
            logger.debug("trends.parse_error", error=str(e)[:120])
            return []

        # Trends RSS schema: each <item> contains <title>, <ht:approx_traffic>,
        # <ht:news_item> children with <ht:news_item_title>/<ht:news_item_url>.
        ns = {"ht": "https://trends.google.com/trending/rss"}

        q_tokens = [t for t in (query or "").lower().split() if len(t) > 2]
        items: list[NewsItem] = []

        for el in root.iter("item"):
            title = (el.findtext("title") or "").strip()
            if not title:
                continue

            traffic = el.findtext("ht:approx_traffic", namespaces=ns) or ""
            pub_raw = el.findtext("pubDate") or ""
            try:
                published = datetime.strptime(pub_raw, "%a, %d %b %Y %H:%M:%S %z")
            except Exception:
                published = datetime.now(timezone.utc)

            # Relevance: boost if query tokens appear in the trend title.
            title_lower = title.lower()
            overlap = sum(1 for t in q_tokens if t in title_lower)
            relevance = 1.0 + overlap * 1.5

            # If the caller supplied specific query tokens, skip entries
            # that don't overlap at all — otherwise trends drowns the
            # aggregator with generic celebrity/sports chatter on non-US
            # sports days.
            if q_tokens and overlap == 0:
                continue

            # Pull the first news item URL if the trend has related articles.
            news_title = ""
            news_url = ""
            for ni in el.findall("ht:news_item", namespaces=ns):
                nt = ni.findtext("ht:news_item_title", namespaces=ns) or ""
                nu = ni.findtext("ht:news_item_url", namespaces=ns) or ""
                if nt:
                    news_title = nt
                    news_url = nu
                    break

            body = f"Trending search with {traffic} searches." if traffic else "Trending search."
            if news_title:
                body += f" Lead article: {news_title}"

            items.append(NewsItem(
                id=hashlib.md5(f"gtrends:{title}:{pub_raw}".encode()).hexdigest(),
                source="google_trends",
                title=f"Trending: {title}",
                content=body,
                url=news_url,
                published_at=published,
                relevance_score=relevance,
            ))
            if len(items) >= limit:
                break

        return items

    async def close(self) -> None:
        return None
