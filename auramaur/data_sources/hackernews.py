"""Hacker News top-stories source — free, no auth.

Great for tech / crypto / science markets where HN front-page discussion
often leads mainstream news by hours. Uses the official Firebase API.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone

import aiohttp
import structlog

from auramaur.data_sources.base import NewsItem

logger = structlog.get_logger(__name__)

_TOP = "https://hacker-news.firebaseio.com/v0/topstories.json"
_ITEM = "https://hacker-news.firebaseio.com/v0/item/{id}.json"


class HackerNewsSource:
    """Top HN stories, keyword-filtered against the current query."""

    source_name: str = "hackernews"
    categories: set[str] | None = {"tech", "crypto", "science"}

    def __init__(self, max_stories: int = 30) -> None:
        self._max_stories = max_stories

    async def _fetch_ids(self, session: aiohttp.ClientSession) -> list[int]:
        try:
            async with session.get(_TOP, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return []
                return (await resp.json())[: self._max_stories]
        except Exception as e:
            logger.debug("hackernews.ids_error", error=str(e)[:120])
            return []

    async def _fetch_item(self, session: aiohttp.ClientSession, item_id: int) -> dict | None:
        try:
            async with session.get(_ITEM.format(id=item_id), timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    return None
                return await resp.json()
        except Exception:
            return None

    async def fetch(self, query: str, limit: int = 20) -> list[NewsItem]:
        q_lower = (query or "").lower()
        q_tokens = [t for t in q_lower.split() if len(t) > 2]

        async with aiohttp.ClientSession() as session:
            ids = await self._fetch_ids(session)
            if not ids:
                return []

            items_raw = await asyncio.gather(*(self._fetch_item(session, i) for i in ids))

        matches: list[NewsItem] = []
        for raw in items_raw:
            if not raw:
                continue
            title = raw.get("title") or ""
            url = raw.get("url") or ""
            ts = raw.get("time")
            story_id = raw.get("id")
            if not title or not story_id:
                continue

            # If caller supplied a query, require at least one query token to
            # appear in the title. Otherwise include everything up to limit.
            if q_tokens and not any(tok in title.lower() for tok in q_tokens):
                continue

            try:
                published = datetime.fromtimestamp(int(ts), tz=timezone.utc) if ts else datetime.now(timezone.utc)
            except Exception:
                published = datetime.now(timezone.utc)

            matches.append(NewsItem(
                id=hashlib.md5(f"hn:{story_id}".encode()).hexdigest(),
                source="hackernews",
                title=title,
                content=f"HN score: {raw.get('score', 0)}, comments: {raw.get('descendants', 0)}.",
                url=url or f"https://news.ycombinator.com/item?id={story_id}",
                published_at=published,
                relevance_score=min(raw.get("score", 0) / 100.0, 3.0),  # soft cap
            ))
            if len(matches) >= limit:
                break

        return matches

    async def close(self) -> None:
        return None
