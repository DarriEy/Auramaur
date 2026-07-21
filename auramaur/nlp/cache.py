"""SQLite-based TTL response cache for NLP analysis results."""

from __future__ import annotations

import asyncio
import hashlib
import json

import aiosqlite
import structlog

from auramaur.db.database import Database

log = structlog.get_logger()


def coarse_evidence_digest(items: list, top_n: int = 8) -> str:
    """Build a churn-tolerant digest from the strongest evidence items.

    Keys on the sorted, normalized titles of the top-N items rather than the
    full evidence blob, so minor reshuffling or low-signal additions don't bust
    the cache. Pair with price-move invalidation (see ``NLPCache.get``) so a
    genuine repricing still forces fresh analysis.

    Args:
        items: Evidence items (NewsItem-like, with a ``.title`` attribute).
        top_n: Number of titles to fold into the digest.

    Returns:
        A short hex digest string.
    """
    titles = []
    for it in items:
        title = getattr(it, "title", None) or ""
        norm = " ".join(title.strip().lower().split())
        if norm:
            titles.append(norm)
    titles = sorted(set(titles))[:top_n]
    raw = "||".join(titles)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def make_cache_key(question: str, evidence_digest: str) -> str:
    """Generate a deterministic cache key from question text and evidence digest.

    Args:
        question: The market question string.
        evidence_digest: A digest (e.g. hash) of the evidence used.

    Returns:
        A hex SHA-256 hash string.
    """
    raw = f"{question.strip().lower()}|{evidence_digest}"
    return hashlib.sha256(raw.encode()).hexdigest()


class NLPCache:
    """TTL-aware cache backed by the nlp_cache SQLite table."""

    def __init__(self, db: Database) -> None:
        self._db = db

    @staticmethod
    def _is_transient_lock(exc: Exception) -> bool:
        detail = str(exc).lower()
        return "database is locked" in detail or "database table is locked" in detail

    async def _read_with_lock_retry(self, sql: str, params: tuple):
        for attempt in range(3):
            try:
                return await self._db.fetchone(sql, params)
            except aiosqlite.OperationalError as exc:
                if not self._is_transient_lock(exc):
                    raise
                if attempt == 2:
                    log.warning("nlp_cache.read_lock_exhausted", error=str(exc))
                    return None
                delay = 0.25 * (2**attempt)
                log.info("nlp_cache.read_lock_retry", attempt=attempt + 1, delay=delay)
                await asyncio.sleep(delay)

    async def _write_with_lock_retry(self, sql: str, params: tuple) -> bool:
        """Retry an idempotent cache upsert; cache failure never loses analysis."""
        for attempt in range(3):
            # Another task's transaction is open on the shared connection.
            # Never commit under it (a legacy commit here lands that task's
            # half-written rows — the exact bleed the contention plan names
            # this file for) and never queue a mere cache write behind
            # transaction()'s multi-second BEGIN wait. Back off; dropping a
            # cache row is acceptable, blocking or bleeding is not.
            if getattr(getattr(self._db, "db", None), "in_transaction", False):
                if attempt == 2:
                    log.warning("nlp_cache.write_busy_dropped")
                    return False
                delay = 0.25 * (2**attempt)
                log.info("nlp_cache.write_busy_retry", attempt=attempt + 1, delay=delay)
                await asyncio.sleep(delay)
                continue
            try:
                async with self._db.transaction():
                    await self._db.execute(sql, params)
                return True
            except aiosqlite.OperationalError as exc:
                if not self._is_transient_lock(exc):
                    raise
                if attempt == 2:
                    log.warning("nlp_cache.write_lock_exhausted", error=str(exc))
                    return False
                delay = 0.25 * (2**attempt)
                log.info("nlp_cache.write_lock_retry", attempt=attempt + 1, delay=delay)
                await asyncio.sleep(delay)
        return False  # pragma: no cover

    async def get(self, cache_key: str, current_price: float | None = None) -> dict | None:
        """Return cached response if it exists, has not expired, and price hasn't moved.

        Args:
            cache_key: The cache key to look up.
            current_price: Optional current market price. If provided and the
                cached analysis was made at a price that differs by >5%, treat
                as a cache miss (stale analysis).

        Returns:
            The cached response dict, or None if missing / expired / stale.
        """
        row = await self._read_with_lock_retry(
            """
            SELECT response, ttl_seconds, created_at, market_price
            FROM nlp_cache
            WHERE cache_key = ?
              AND datetime(created_at, '+' || ttl_seconds || ' seconds') > datetime('now')
            """,
            (cache_key,),
        )
        if row is None:
            return None

        # Price-move invalidation: if market moved >5% since analysis, treat as miss
        if current_price is not None:
            cached_price = row["market_price"] if "market_price" in row.keys() else None
            if cached_price is not None and cached_price > 0:
                price_delta = abs(current_price - cached_price) / cached_price
                if price_delta > 0.05:
                    log.info(
                        "nlp_cache.price_invalidated",
                        cache_key=cache_key[:12],
                        cached_price=round(cached_price, 4),
                        current_price=round(current_price, 4),
                        delta_pct=round(price_delta * 100, 1),
                    )
                    return None

        log.debug("nlp_cache.hit", cache_key=cache_key[:12])
        try:
            return json.loads(row["response"])
        except (json.JSONDecodeError, TypeError):
            log.warning("nlp_cache.corrupt_entry", cache_key=cache_key[:12])
            return None

    async def put(
        self,
        cache_key: str,
        market_id: str,
        response: dict,
        ttl_seconds: int,
        market_price: float = 0.0,
    ) -> None:
        """Store a response in the cache.

        Args:
            cache_key: The cache key.
            market_id: Associated market id.
            response: The response dict to cache.
            ttl_seconds: Time-to-live in seconds.
            market_price: Current market price at analysis time (for price-move invalidation).
        """
        probability = response.get("probability", 0.0)
        confidence = response.get("confidence", "LOW")
        response_json = json.dumps(response)

        written = await self._write_with_lock_retry(
            """
            INSERT OR REPLACE INTO nlp_cache
                (cache_key, market_id, response, probability, confidence, ttl_seconds, market_price, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                cache_key,
                market_id,
                response_json,
                probability,
                confidence,
                ttl_seconds,
                market_price,
            ),
        )
        if written:
            log.debug(
                "nlp_cache.put",
                cache_key=cache_key[:12],
                ttl=ttl_seconds,
                market_price=market_price,
            )

    async def cleanup(self) -> None:
        """Remove all expired cache entries."""
        cursor = await self._db.execute(
            """
            DELETE FROM nlp_cache
            WHERE datetime(created_at, '+' || ttl_seconds || ' seconds') <= datetime('now')
            """,
        )
        deleted = cursor.rowcount
        await self._db.commit()
        if deleted:
            log.info("nlp_cache.cleanup", removed=deleted)
