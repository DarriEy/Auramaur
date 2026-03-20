"""Reddit data source using asyncpraw."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import asyncpraw  # type: ignore[import-untyped]
import structlog

from auramaur.data_sources.base import DataSource, NewsItem

logger = structlog.get_logger(__name__)

_DEFAULT_SUBREDDITS = ["polymarket", "predictions", "news", "worldnews"]


class RedditSource:
    """Search Reddit submissions via async PRAW."""

    source_name: str = "reddit"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str = "auramaur:v0.1 (by /u/auramaur)",
    ) -> None:
        self._reddit = asyncpraw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )

    async def fetch(self, query: str, limit: int = 20) -> list[NewsItem]:
        """Search across default subreddits for submissions matching *query*."""
        items: list[NewsItem] = []
        per_sub = max(1, limit // len(_DEFAULT_SUBREDDITS))

        for sub_name in _DEFAULT_SUBREDDITS:
            try:
                subreddit = await self._reddit.subreddit(sub_name)
                async for submission in subreddit.search(query, limit=per_sub):
                    item_id = hashlib.sha256(
                        f"reddit:{submission.id}".encode()
                    ).hexdigest()[:16]
                    published_at = datetime.fromtimestamp(
                        submission.created_utc, tz=timezone.utc
                    )
                    items.append(
                        NewsItem(
                            id=item_id,
                            source=self.source_name,
                            title=submission.title,
                            content=(submission.selftext or "")[:2000],
                            url=f"https://reddit.com{submission.permalink}",
                            published_at=published_at,
                            relevance_score=submission.score / max(submission.score, 1),
                        )
                    )
            except Exception:
                logger.exception("reddit_sub_fetch_failed", subreddit=sub_name)

        logger.info("reddit_fetched", count=len(items), query=query)
        return items[:limit]

    async def close(self) -> None:
        await self._reddit.close()
