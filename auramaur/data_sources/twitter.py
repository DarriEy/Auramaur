"""Twitter/X data source using tweepy AsyncClient."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import structlog
import tweepy  # type: ignore[import-untyped]

from auramaur.data_sources.base import DataSource, NewsItem

logger = structlog.get_logger(__name__)


class TwitterSource:
    """Search recent tweets via the Twitter v2 API."""

    source_name: str = "twitter"

    def __init__(self, bearer_token: str) -> None:
        self._client = tweepy.AsyncClient(bearer_token=bearer_token)

    async def fetch(self, query: str, limit: int = 20) -> list[NewsItem]:
        """Search recent tweets matching *query*."""
        try:
            response = await self._client.search_recent_tweets(
                query=query,
                max_results=min(max(limit, 10), 100),
                tweet_fields=["created_at", "author_id", "text"],
            )
        except Exception:
            logger.exception("twitter_fetch_failed", query=query)
            return []

        if not response or not response.data:
            logger.info("twitter_no_results", query=query)
            return []

        items: list[NewsItem] = []
        for tweet in response.data[:limit]:
            item_id = hashlib.sha256(f"twitter:{tweet.id}".encode()).hexdigest()[:16]
            published_at = tweet.created_at if tweet.created_at else datetime.now(timezone.utc)
            items.append(
                NewsItem(
                    id=item_id,
                    source=self.source_name,
                    title=tweet.text[:120],
                    content=tweet.text,
                    url=f"https://twitter.com/i/status/{tweet.id}",
                    published_at=published_at,
                )
            )

        logger.info("twitter_fetched", count=len(items), query=query)
        return items

    async def close(self) -> None:
        # tweepy AsyncClient doesn't hold persistent connections
        pass
