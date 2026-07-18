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
from auramaur.db.database import Database

if TYPE_CHECKING:
    from auramaur.information_graduation import InformationGraduation

logger = structlog.get_logger(__name__)

_PUNCTUATION_RE = re.compile(r"[^\w\s]")


def _normalise_title(title: str) -> str:
    """Lowercase, strip punctuation for dedup comparison."""
    return _PUNCTUATION_RE.sub("", title.lower()).strip()


class Aggregator:
    """Fan-out to multiple data sources, deduplicate, and rank results."""

    source_name: str = "aggregator"

    def __init__(self, sources: list[DataSource], db: Database | None = None,
                 information_graduation: InformationGraduation | None = None) -> None:
        self._sources = sources
        self._db = db
        self._information_graduation = information_graduation

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
                                   datetime.now(tz=timezone.utc).isoformat()))
                return items
            except Exception as exc:
                fetch_rows.append((run_id, source_name, "error", 0,
                                   round((time.monotonic() - before) * 1000), str(exc)[:500],
                                   datetime.now(tz=timezone.utc).isoformat()))
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
            recency = min(4.0, 1.0 / age_hours)
            recency *= {"exact": 1.0, "provider_seen": 0.8, "inferred": 0.6,
                        "unknown": 0.25}.get(item.timestamp_quality, 0.5)
            source_weight = authority(item.source, item.url)
            return (recency * source_weight) + item.relevance_score

        unique.sort(key=_rank, reverse=True)

        observed_at = datetime.now(tz=timezone.utc).isoformat()
        for item in unique:
            item.ingestion_run_id = run_id
        if self._db is not None:
            try:
                # Telemetry is deliberately one transaction after the network
                # work. A lock/disk/schema failure must never suppress usable
                # evidence or strand a half-written ingestion run.
                await self._db.execute("SAVEPOINT ingestion_lineage")
                await self._db.execute(
                    "INSERT INTO ingestion_runs "
                    "(id,query,category,market_id,started_at,completed_at,status,"
                    "active_sources,raw_items,unique_items) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (run_id, query, category or "", market_id, started.isoformat(),
                     observed_at,
                     "partial" if any(row[2] == "error" for row in fetch_rows) else "ok",
                     len(active_sources), len(all_items), len(unique)),
                )
                if fetch_rows:
                    await self._db.executemany(
                        "INSERT OR REPLACE INTO source_fetches "
                        "(run_id,source,status,item_count,latency_ms,error,observed_at) "
                        "VALUES (?,?,?,?,?,?,?)", fetch_rows,
                    )
                evidence_rows = []
                persisted_items = list({(item.source, item.id): item for item in all_items}.values())
                ranks = {item.id: rank for rank, item in enumerate(unique, start=1)}
                for item in persisted_items:
                    excerpt = (item.content or "")[:500]
                    content_hash = hashlib.sha256(
                        f"{item.title}\n{item.content}".encode("utf-8", "replace")
                    ).hexdigest()
                    evidence_rows.append((
                        run_id, item.id, item.source, item.title, item.url, content_hash,
                        excerpt, item.published_at.isoformat(), observed_at,
                        item.timestamp_quality, item.relevance_score, ranks.get(item.id),
                        market_id, item.information_mode,
                    ))
                if evidence_rows:
                    await self._db.executemany(
                        "INSERT OR REPLACE INTO evidence_observations "
                        "(run_id,item_id,source,title,url,content_hash,excerpt,published_at,"
                        "observed_at,timestamp_quality,relevance_score,rank_position,market_id,"
                        "information_mode) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", evidence_rows,
                    )
                await self._db.execute("RELEASE SAVEPOINT ingestion_lineage")
                await self._db.commit()
            except Exception as exc:
                try:
                    await self._db.execute("ROLLBACK TO SAVEPOINT ingestion_lineage")
                    await self._db.execute("RELEASE SAVEPOINT ingestion_lineage")
                except Exception:
                    try:
                        await self._db.db.rollback()
                    except Exception:
                        pass
                logger.exception("aggregator_lineage_persist_failed", error=str(exc)[:200])

        # Assignment is immutable and happens after observation but before any
        # future paired analysis consumes the candidate. Shadow evidence itself
        # remains withheld above; this only accrues the eligible trial cohort.
        if (self._information_graduation is not None and market_id
                and market_price is not None):
            for source in active_sources:
                if getattr(source, "information_mode", "production") != "shadow":
                    continue
                if not any(item.source == source.source_name for item in all_items):
                    continue
                try:
                    strategy_id = await self._information_graduation.register(
                        source.source_name, category or "",
                        getattr(source, "trial_horizon", ""),
                        getattr(source, "trial_event_type", "query_evidence"),
                    )
                    await self._information_graduation.assign(
                        strategy_id, market_id, datetime.fromisoformat(observed_at),
                        market_price,
                    )
                except Exception:
                    logger.exception(
                        "information_trial_assignment_failed", source=source.source_name,
                        market_id=market_id,
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
