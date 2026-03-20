"""SQLite-based TTL response cache for NLP analysis results."""

from __future__ import annotations

import hashlib
import json

import structlog

from auramaur.db.database import Database

log = structlog.get_logger()


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

    async def get(self, cache_key: str) -> dict | None:
        """Return cached response if it exists and has not expired.

        Args:
            cache_key: The cache key to look up.

        Returns:
            The cached response dict, or None if missing / expired.
        """
        row = await self._db.fetchone(
            """
            SELECT response, ttl_seconds, created_at
            FROM nlp_cache
            WHERE cache_key = ?
              AND datetime(created_at, '+' || ttl_seconds || ' seconds') > datetime('now')
            """,
            (cache_key,),
        )
        if row is None:
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
    ) -> None:
        """Store a response in the cache.

        Args:
            cache_key: The cache key.
            market_id: Associated market id.
            response: The response dict to cache.
            ttl_seconds: Time-to-live in seconds.
        """
        probability = response.get("probability", 0.0)
        confidence = response.get("confidence", "LOW")
        response_json = json.dumps(response)

        await self._db.execute(
            """
            INSERT OR REPLACE INTO nlp_cache
                (cache_key, market_id, response, probability, confidence, ttl_seconds, created_at)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (cache_key, market_id, response_json, probability, confidence, ttl_seconds),
        )
        await self._db.commit()
        log.debug("nlp_cache.put", cache_key=cache_key[:12], ttl=ttl_seconds)

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
