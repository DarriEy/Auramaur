"""GDELT Project — global news events via the free DOC 2.0 API.

GDELT monitors global broadcast, print, and web news in 100+ languages and
exposes a free JSON search endpoint. Coverage is far wider than NewsAPI or
DDG, and it tags articles with themes and locations. Category-agnostic —
fires on every query — so it's a general-purpose news upgrade rather than
a domain source.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import aiohttp
import structlog

from auramaur.data_sources.base import NewsItem

logger = structlog.get_logger(__name__)

_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


class GDELTSource:
    """Search GDELT DOC 2.0 for recent articles matching the query."""

    source_name: str = "gdelt"
    # None — fires for every query.
    categories: set[str] | None = None

    def __init__(self, timespan: str = "3d", sourcelang: str = "english") -> None:
        # ``timespan`` uses GDELT's shorthand: 15min/1h/6h/12h/24h/1d/3d/1w/1m/3m
        self._timespan = timespan
        self._sourcelang = sourcelang

    async def fetch(self, query: str, limit: int = 20) -> list[NewsItem]:
        if not query or len(query.strip()) < 3:
            return []

        params = {
            "query": f"{query} sourcelang:{self._sourcelang}",
            "mode": "ArtList",
            "format": "json",
            "maxrecords": str(min(limit, 50)),
            "sort": "DateDesc",
            "timespan": self._timespan,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(_URL, params=params, timeout=aiohttp.ClientTimeout(total=12)) as resp:
                    if resp.status != 200:
                        logger.debug("gdelt.bad_status", status=resp.status)
                        return []
                    try:
                        data = await resp.json(content_type=None)
                    except Exception:
                        return []
        except Exception as e:
            logger.debug("gdelt.fetch_error", error=str(e)[:120])
            return []

        items: list[NewsItem] = []
        for art in (data.get("articles") or [])[:limit]:
            title = art.get("title") or ""
            if not title:
                continue
            url = art.get("url") or ""
            seen_raw = art.get("seendate") or ""
            # seendate format "20260419T204500Z"
            try:
                published = datetime.strptime(seen_raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            except Exception:
                published = datetime.now(timezone.utc)

            items.append(NewsItem(
                id=hashlib.md5(f"gdelt:{url or title}".encode()).hexdigest(),
                source="gdelt",
                title=title,
                content=f"Source: {art.get('domain', 'unknown')}. Language: {art.get('language', 'en')}.",
                url=url,
                published_at=published,
                relevance_score=1.0,
            ))

        return items

    async def close(self) -> None:
        return None
