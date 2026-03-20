"""Ensemble probability estimation — combines multiple sources with weighted averaging."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

import structlog

from auramaur.db.database import Database

log = structlog.get_logger()


@runtime_checkable
class ProbabilitySource(Protocol):
    """Protocol for probability estimation sources."""

    @property
    def name(self) -> str: ...

    async def estimate(self, question: str, category: str = "") -> float | None:
        """Return a probability estimate, or None if unavailable."""
        ...


class EnsembleEstimator:
    """Combines multiple probability sources using weighted averaging.

    Weights are updated based on per-source Brier scores tracked in the
    source_accuracy table.
    """

    def __init__(self, db: Database, sources: list[ProbabilitySource] | None = None) -> None:
        self._db = db
        self._sources: list[ProbabilitySource] = sources or []
        self._weights: dict[str, float] = {}

    def add_source(self, source: ProbabilitySource) -> None:
        """Register a probability source."""
        self._sources.append(source)

    async def load_weights(self) -> None:
        """Load per-source weights from the database."""
        rows = await self._db.fetchall("SELECT source, weight FROM source_accuracy")
        self._weights = {row["source"]: row["weight"] for row in rows}
        log.debug("ensemble.weights_loaded", weights=self._weights)

    async def estimate(self, question: str, category: str = "") -> dict:
        """Get ensemble probability estimate from all sources.

        Returns:
            Dict with 'probability' (weighted average), 'sources' (per-source estimates),
            and 'weights' (per-source weights used).
        """
        estimates: dict[str, float] = {}

        for source in self._sources:
            try:
                prob = await source.estimate(question, category)
                if prob is not None:
                    estimates[source.name] = prob
            except Exception as e:
                log.debug("ensemble.source_error", source=source.name, error=str(e))

        if not estimates:
            return {"probability": None, "sources": {}, "weights": {}}

        # Weighted average
        total_weight = 0.0
        weighted_sum = 0.0
        used_weights: dict[str, float] = {}

        for name, prob in estimates.items():
            w = self._weights.get(name, 1.0)
            used_weights[name] = w
            weighted_sum += prob * w
            total_weight += w

        ensemble_prob = weighted_sum / total_weight if total_weight > 0 else None

        log.info(
            "ensemble.estimate",
            probability=round(ensemble_prob, 4) if ensemble_prob else None,
            sources=len(estimates),
        )

        return {
            "probability": ensemble_prob,
            "sources": estimates,
            "weights": used_weights,
        }

    async def record_prediction(
        self, source_name: str, market_id: str, predicted_prob: float
    ) -> None:
        """Record a source's prediction for later accuracy tracking."""
        # Store in a lightweight way using the existing calibration table
        await self._db.execute(
            """INSERT INTO calibration (market_id, predicted_prob, category, created_at)
               VALUES (?, ?, ?, datetime('now'))""",
            (f"{source_name}:{market_id}", predicted_prob, f"source:{source_name}"),
        )
        await self._db.commit()

    async def update_source_weights(self) -> None:
        """Update source weights based on Brier scores.

        Better sources (lower Brier) get higher weights.
        Weight = 1 / brier_score, normalized.
        """
        rows = await self._db.fetchall(
            """SELECT category as source,
                      AVG((predicted_prob - actual_outcome) * (predicted_prob - actual_outcome)) as brier,
                      COUNT(*) as n
               FROM calibration
               WHERE actual_outcome IS NOT NULL AND category LIKE 'source:%'
               GROUP BY category
               HAVING n >= 10"""
        )

        if not rows:
            return

        # Compute inverse-Brier weights
        raw_weights: dict[str, float] = {}
        for row in rows:
            source_name = row["source"].replace("source:", "")
            brier = row["brier"]
            if brier > 0:
                raw_weights[source_name] = 1.0 / brier
            else:
                raw_weights[source_name] = 10.0  # Perfect score gets high weight

        # Normalize so average weight = 1.0
        if raw_weights:
            avg_w = sum(raw_weights.values()) / len(raw_weights)
            if avg_w > 0:
                for name in raw_weights:
                    raw_weights[name] /= avg_w

        # Store
        now = datetime.now(timezone.utc).isoformat()
        for name, weight in raw_weights.items():
            await self._db.execute(
                """INSERT OR REPLACE INTO source_accuracy (source, brier_score, weight, updated_at)
                   VALUES (?, (SELECT brier_score FROM source_accuracy WHERE source = ?), ?, ?)""",
                (name, name, weight, now),
            )
        await self._db.commit()

        self._weights = raw_weights
        log.info("ensemble.weights_updated", weights=raw_weights)
