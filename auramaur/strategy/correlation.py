"""Semantic correlation detection between markets using Claude CLI."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone

import structlog

from auramaur.db.database import Database
from auramaur.exchange.models import Market

log = structlog.get_logger()

# Cache TTL for relationships (24 hours — they're slow-moving)
RELATIONSHIP_CACHE_TTL_HOURS = 24


class CorrelationDetector:
    """Detects semantically related markets using Claude CLI.

    Results are aggressively cached since market relationships
    are slow-moving and Claude CLI calls are expensive.
    """

    def __init__(self, db: Database, model: str = "claude-sonnet-4-20250514") -> None:
        self._db = db
        self._model = model

    async def detect_relationships(
        self, markets: list[Market], batch_size: int = 10
    ) -> list[dict]:
        """Detect relationships between a batch of markets.

        Calls Claude CLI to identify semantic relationships, then stores them.
        Uses aggressive caching — skips markets already analyzed recently.

        Args:
            markets: List of markets to analyze.
            batch_size: Max markets per Claude call.

        Returns:
            List of detected relationships.
        """
        # Filter to markets not recently analyzed
        fresh_markets = []
        for m in markets[:batch_size]:
            row = await self._db.fetchone(
                """SELECT COUNT(*) as n FROM market_relationships
                   WHERE (market_id_a = ? OR market_id_b = ?)
                   AND detected_at > datetime('now', ?)""",
                (m.id, m.id, f"-{RELATIONSHIP_CACHE_TTL_HOURS} hours"),
            )
            if row is None or row["n"] == 0:
                fresh_markets.append(m)

        if not fresh_markets:
            log.debug("correlation.all_cached")
            return []

        # Build prompt
        market_list = "\n".join(
            f"- ID: {m.id} | Q: {m.question} | Category: {m.category}"
            for m in fresh_markets
        )
        prompt = f"""Analyze these prediction markets and identify pairs that are semantically related
(correlated outcomes, same underlying event, conditional relationships, or potential arbitrage).

Markets:
{market_list}

For each related pair, respond with a JSON array of objects:
[{{"market_a": "id", "market_b": "id", "type": "correlated|conditional|same_event|arbitrage", "strength": 0.0-1.0, "description": "why they're related"}}]

If no relationships found, return []. Only return the JSON array, no other text."""

        try:
            proc = await asyncio.create_subprocess_exec(
                "claude", "-p", prompt,
                "--output-format", "text",
                "--model", self._model,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
            raw = stdout.decode().strip()

            # Parse response
            # Try to extract JSON array
            match = re.search(r"\[[\s\S]*\]", raw)
            if not match:
                return []

            relationships = json.loads(match.group(0))
        except Exception as e:
            log.error("correlation.detect_error", error=str(e))
            return []

        # Store relationships
        now = datetime.now(timezone.utc).isoformat()
        stored = []
        for rel in relationships:
            try:
                await self._db.execute(
                    """INSERT OR REPLACE INTO market_relationships
                       (market_id_a, market_id_b, relationship_type, strength, description, detected_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        rel["market_a"], rel["market_b"],
                        rel.get("type", "correlated"),
                        rel.get("strength", 0.5),
                        rel.get("description", ""),
                        now,
                    ),
                )
                stored.append(rel)
            except Exception as e:
                log.debug("correlation.store_error", error=str(e))

        await self._db.commit()
        log.info("correlation.detected", count=len(stored))
        return stored

    async def get_related_markets(self, market_id: str) -> list[dict]:
        """Get all known related markets for a given market ID."""
        rows = await self._db.fetchall(
            """SELECT * FROM market_relationships
               WHERE market_id_a = ? OR market_id_b = ?
               ORDER BY strength DESC""",
            (market_id, market_id),
        )
        return [dict(row) for row in rows]

    async def detect_arbitrage(self) -> list[dict]:
        """Detect potential arbitrage opportunities from stored relationships.

        Looks for conditional probability violations where
        P(X wins primary) = 60% but P(X wins general) = 70%.
        """
        rows = await self._db.fetchall(
            """SELECT mr.*, m1.outcome_yes_price as price_a, m2.outcome_yes_price as price_b
               FROM market_relationships mr
               JOIN markets m1 ON mr.market_id_a = m1.id
               JOIN markets m2 ON mr.market_id_b = m2.id
               WHERE mr.relationship_type IN ('conditional', 'same_event')
               AND m1.active = 1 AND m2.active = 1"""
        )

        opportunities = []
        for row in rows:
            price_a = row["price_a"]
            price_b = row["price_b"]

            # Conditional violation: if A implies B, then P(A) <= P(B)
            if row["relationship_type"] == "conditional":
                if price_a > price_b + 0.05:  # 5% threshold
                    opportunities.append({
                        "type": "conditional_violation",
                        "market_a": row["market_id_a"],
                        "market_b": row["market_id_b"],
                        "price_a": price_a,
                        "price_b": price_b,
                        "description": row["description"],
                    })

            # Same event: prices should be close
            if row["relationship_type"] == "same_event":
                if abs(price_a - price_b) > 0.05:
                    opportunities.append({
                        "type": "price_divergence",
                        "market_a": row["market_id_a"],
                        "market_b": row["market_id_b"],
                        "price_a": price_a,
                        "price_b": price_b,
                        "divergence": abs(price_a - price_b),
                    })

        if opportunities:
            log.info("correlation.arbitrage", count=len(opportunities))
        return opportunities
