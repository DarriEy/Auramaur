"""Bluesky (AT Protocol) post search — free, no auth for reads.

Uses the public appview endpoint ``public.api.bsky.app`` which serves
read-only feed/post queries without requiring a session. Useful for
catching breaking takes and reactions that haven't hit mainstream news
yet (journalists have largely moved there from Twitter).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import aiohttp
import structlog

from auramaur.data_sources.base import NewsItem

logger = structlog.get_logger(__name__)

_SEARCH = "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts"


class BlueskySource:
    """Search Bluesky posts for recent takes on a topic."""

    source_name: str = "bluesky"
    # Category-agnostic — fires for any query, like the rest of the text
    # news sources. Bluesky discussion cuts across every topic we trade.
    categories: set[str] | None = None

    async def fetch(self, query: str, limit: int = 20) -> list[NewsItem]:
        if not query or len(query.strip()) < 3:
            return []

        params = {
            "q": query,
            "limit": str(min(limit, 50)),
            "sort": "latest",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(_SEARCH, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        logger.debug("bluesky.bad_status", status=resp.status)
                        return []
                    data = await resp.json()
        except Exception as e:
            logger.debug("bluesky.fetch_error", error=str(e)[:120])
            return []

        items: list[NewsItem] = []
        for post in (data.get("posts") or [])[:limit]:
            record = post.get("record") or {}
            text = (record.get("text") or "").strip()
            if not text:
                continue
            author = (post.get("author") or {}).get("handle") or "unknown"
            uri = post.get("uri") or ""
            created_at = record.get("createdAt") or ""
            try:
                published = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except Exception:
                published = datetime.now(timezone.utc)

            # Engagement weight — likes + reposts bias upward.
            likes = post.get("likeCount") or 0
            reposts = post.get("repostCount") or 0
            replies = post.get("replyCount") or 0

            # Skip very low-engagement posts to cut noise from random accounts.
            if likes + reposts + replies < 2 and len(text) < 80:
                continue

            # Convert at:// URI to web URL (bsky.app post permalinks).
            # at://<did>/app.bsky.feed.post/<rkey> → https://bsky.app/profile/<did>/post/<rkey>
            web_url = ""
            if uri.startswith("at://"):
                parts = uri[5:].split("/")
                if len(parts) >= 3:
                    web_url = f"https://bsky.app/profile/{parts[0]}/post/{parts[-1]}"

            items.append(NewsItem(
                id=hashlib.md5(f"bsky:{uri or text[:60]}".encode()).hexdigest(),
                source="bluesky",
                title=text.split("\n", 1)[0][:160],
                content=f"@{author}: {text[:400]} (likes:{likes} reposts:{reposts} replies:{replies})",
                url=web_url,
                published_at=published,
                relevance_score=min((likes + reposts * 2) / 50.0, 2.5),
            ))

        return items

    async def close(self) -> None:
        return None
