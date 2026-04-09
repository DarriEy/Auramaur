"""Aggregator that fans out to multiple DataSource instances."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone

import structlog

from auramaur.data_sources.base import DataSource, NewsItem

logger = structlog.get_logger(__name__)

_PUNCTUATION_RE = re.compile(r"[^\w\s]")


def _normalise_title(title: str) -> str:
    """Lowercase, strip punctuation for dedup comparison."""
    return _PUNCTUATION_RE.sub("", title.lower()).strip()


class Aggregator:
    """Fan-out to multiple data sources, deduplicate, and rank results."""

    source_name: str = "aggregator"

    def __init__(self, sources: list[DataSource]) -> None:
        self._sources = sources

    async def gather(
        self,
        query: str,
        limit_per_source: int = 20,
    ) -> list[NewsItem]:
        """Fetch from all sources concurrently, deduplicate, and rank."""

        async def _safe_fetch(source: DataSource) -> list[NewsItem]:
            try:
                return await source.fetch(query, limit=limit_per_source)
            except Exception:
                logger.exception(
                    "aggregator_source_failed",
                    source=getattr(source, "source_name", str(source)),
                )
                return []

        results = await asyncio.gather(
            *(_safe_fetch(src) for src in self._sources),
            return_exceptions=False,  # exceptions already caught in _safe_fetch
        )

        # Flatten
        all_items: list[NewsItem] = []
        for batch in results:
            all_items.extend(batch)

        # Deduplicate by normalised title
        seen_titles: set[str] = set()
        unique: list[NewsItem] = []
        for item in all_items:
            norm = _normalise_title(item.title)
            if norm and norm not in seen_titles:
                seen_titles.add(norm)
                unique.append(item)

        # Rank: combined score of recency, source reliability, and relevance
        now = datetime.now(tz=timezone.utc)

        # Source reliability tiers (higher = more trustworthy)
        _SOURCE_WEIGHTS = {
            "reuters": 3.0, "ap": 3.0, "bbc": 2.5, "nyt": 2.5,
            "guardian": 2.0, "npr": 2.0, "politico": 2.0, "cnbc": 2.0,
            "bloomberg": 2.5, "ft": 2.5, "wsj": 2.5,
            "web": 1.5, "newsapi": 1.5,
            "rss": 1.0, "reddit": 0.8, "fred": 2.0,
            "market_data": 2.5, "polymarket_context": 2.0,
            "cointelegraph": 1.0, "coindesk": 1.2,
            "manifold": 2.0,
        }

        def _rank(item: NewsItem) -> float:
            pub = item.published_at
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            age_hours = max((now - pub).total_seconds() / 3600, 0.01)
            recency = 1.0 / age_hours
            # Source reliability bonus
            source_lower = item.source.lower()
            source_weight = _SOURCE_WEIGHTS.get(source_lower, 1.0)
            # Check URL for high-quality domains
            url_lower = (item.url or "").lower()
            for domain, weight in _SOURCE_WEIGHTS.items():
                if domain in url_lower:
                    source_weight = max(source_weight, weight)
                    break
            return (recency * source_weight) + item.relevance_score

        unique.sort(key=_rank, reverse=True)

        logger.info(
            "aggregator_gathered",
            total_raw=len(all_items),
            unique=len(unique),
            query=query,
        )
        return unique

    # Implement the DataSource protocol's fetch as well, delegating to gather
    async def fetch(self, query: str, limit: int = 20) -> list[NewsItem]:
        results = await self.gather(query, limit_per_source=limit)
        return results[:limit]

    async def close(self) -> None:
        for source in self._sources:
            try:
                await source.close()
            except Exception:
                logger.exception(
                    "aggregator_source_close_failed",
                    source=getattr(source, "source_name", str(source)),
                )
