"""Aggregator that fans out to multiple DataSource instances."""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from auramaur.data_sources.base import DataSource, NewsItem
if TYPE_CHECKING:
    from auramaur.lineage_observer import LineageObserver

logger = structlog.get_logger(__name__)

_PUNCTUATION_RE = re.compile(r"[^\w\s]")


def _normalise_title(title: str) -> str:
    """Lowercase, strip punctuation for dedup comparison."""
    return _PUNCTUATION_RE.sub("", title.lower()).strip()


class Aggregator:
    """Fan-out to multiple data sources, deduplicate, and rank results."""

    source_name: str = "aggregator"

    def __init__(self, sources: list[DataSource],
                 observer: LineageObserver | None = None) -> None:
        self._sources = sources
        self.observer = observer

    @staticmethod
    def _source_matches_category(source: DataSource, category: str | None) -> bool:
        """Return True if the source should fire for the given market category.

        Sources with ``categories is None`` (or no attr) fire for every query.
        Sources with a concrete set only fire when the current query's
        category is in that set. A query with no category (``None``) falls
        back to firing only the None-gated sources — i.e. domain sources
        don't fire on category-less queries.
        """
        allowed = getattr(source, "categories", None)
        if allowed is None:
            return True
        if category is None:
            return False
        return category in allowed

    async def gather(
        self,
        query: str,
        limit_per_source: int = 20,
        category: str | None = None,
        market_id: str = "",
        market_price: float | None = None,
    ) -> list[NewsItem]:
        """Fetch from matching sources concurrently, deduplicate, and rank.

        ``category`` is the market category of the query. Domain-specific
        sources are only invoked when ``category`` matches their
        ``categories`` set; category-agnostic sources always fire.
        """

        run_id = uuid.uuid4().hex
        started = datetime.now(tz=timezone.utc)
        fetch_rows: list[tuple] = []

        async def _safe_fetch(source: DataSource) -> list[NewsItem]:
            source_name = getattr(source, "source_name", str(source))
            before = time.monotonic()
            try:
                items = await source.fetch(query, limit=limit_per_source)
                mode = getattr(source, "information_mode", "production")
                for item in items:
                    item.information_mode = mode
                fetch_rows.append((run_id, source_name, "ok", len(items),
                                   round((time.monotonic() - before) * 1000), "",
                                   datetime.now(tz=timezone.utc).isoformat(), mode))
                return items
            except Exception as exc:
                fetch_rows.append((run_id, source_name, "error", 0,
                                   round((time.monotonic() - before) * 1000), str(exc)[:500],
                                   datetime.now(tz=timezone.utc).isoformat(),
                                   getattr(source, "information_mode", "production")))
                logger.exception(
                    "aggregator_source_failed",
                    source=source_name,
                )
                return []

        active_sources = [s for s in self._sources if self._source_matches_category(s, category)]
        results = await asyncio.gather(
            *(_safe_fetch(src) for src in active_sources),
            return_exceptions=False,  # exceptions already caught in _safe_fetch
        )

        # Flatten. Shadow sources are persisted below but deliberately withheld
        # from the analyzer until an information-strategy cell graduates.
        all_items: list[NewsItem] = []
        for batch in results:
            all_items.extend(batch)

        # Deduplicate by normalised title
        seen_titles: set[str] = set()
        unique: list[NewsItem] = []
        for item in all_items:
            if item.information_mode == "shadow":
                continue
            norm = _normalise_title(item.title)
            if norm and norm not in seen_titles:
                seen_titles.add(norm)
                unique.append(item)

        # Rank: combined score of recency, source reliability, and relevance.
        # Source authority is centralized in auramaur.nlp.sources so the
        # aggregator and the evidence compressor agree on trust weights.
        from auramaur.nlp.sources import authority

        now = datetime.now(tz=timezone.utc)

        def _rank(item: NewsItem) -> float:
            pub = item.published_at
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            age_hours = max((now - pub).total_seconds() / 3600, 0.01)
            # Preserve the production ranking formula. Timestamp quality is
            # captured for offline trials, not allowed to alter proven books.
            recency = 1.0 / age_hours
            source_weight = authority(item.source, item.url)
            return (recency * source_weight) + item.relevance_score

        unique.sort(key=_rank, reverse=True)

        observed_at = datetime.now(tz=timezone.utc).isoformat()
        for item in unique:
            item.ingestion_run_id = run_id
        if self.observer is not None:
            ranks = {item.id: rank for rank, item in enumerate(unique, start=1)}
            persisted = list({(item.source, item.id): item for item in all_items}.values())
            self.observer.ingestion(
                run_id=run_id, query=query, category=category or "", market_id=market_id,
                started_at=started.isoformat(), observed_at=observed_at,
                fetch_rows=fetch_rows, raw_items=len(all_items),
                items=[{
                    "run_id": run_id, "item_id": item.id, "source": item.source,
                    "title": item.title, "url": item.url,
                    "content_hash": hashlib.sha256(
                        f"{item.title}\n{item.content}".encode("utf-8", "replace")
                    ).hexdigest(),
                    "excerpt": (item.content or "")[:500],
                    "published_at": item.published_at.isoformat(),
                    "observed_at": observed_at, "timestamp_quality": item.timestamp_quality,
                    "relevance_score": item.relevance_score,
                    "rank_position": ranks.get(item.id), "market_id": market_id,
                    "information_mode": item.information_mode,
                } for item in persisted],
                active_sources=[{
                    "name": source.source_name,
                    "mode": getattr(source, "information_mode", "production"),
                    "horizon": getattr(source, "trial_horizon", ""),
                    "event_type": getattr(source, "trial_event_type", "query_evidence"),
                    "had_items": any(i.source == source.source_name for i in all_items),
                    "market_price": market_price,
                } for source in active_sources],
            )

        logger.info(
            "aggregator_gathered",
            total_raw=len(all_items),
            unique=len(unique),
            query=query,
            category=category,
            active_sources=len(active_sources),
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
        if self.observer is not None:
            await self.observer.close()
