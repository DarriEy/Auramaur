"""Performance feedback loop — learns which prediction categories the bot is good at.

Tracks per-category accuracy (Brier score + directional accuracy), computes
Kelly multipliers for the risk manager, identifies blind-spot categories to
avoid, and generates human-readable calibration summaries for the strategic
prompt.
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from auramaur.db.database import Database

log = structlog.get_logger()

# --- Thresholds ---
_BRIER_GOOD = 0.15      # Below this = well calibrated
_BRIER_BAD = 0.30        # Above this = poorly calibrated
_MULT_GOOD = 1.2         # Kelly multiplier for accurate categories
_MULT_NEUTRAL = 1.0      # Kelly multiplier for average categories
_MULT_BAD = 0.3           # Kelly multiplier for poor categories
_AVOID_MIN_TRADES = 10    # Minimum trades before we consider avoiding
_AVOID_ACCURACY = 0.40    # Directional accuracy below this = avoid


class PerformanceFeedback:
    """Closes the feedback loop: resolution outcomes -> category multipliers -> risk sizing.

    Reads from the ``calibration`` table (populated by CalibrationTracker when
    markets resolve) and the ``category_stats`` table (populated by
    PerformanceAttributor).  Computes per-category Brier scores and
    directional accuracy, then writes updated ``kelly_multiplier`` and
    ``brier_score`` back to ``category_stats`` for the RiskManager to read.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Core update: scan resolutions and recompute per-category stats
    # ------------------------------------------------------------------

    async def update_from_resolutions(self) -> None:
        """Scan calibration table for resolved markets, compute per-category stats.

        For each category with at least one resolved prediction, computes:
        - Brier score: mean((predicted - actual)^2)
        - Directional accuracy: fraction where (p > 0.5 and outcome=1) or (p <= 0.5 and outcome=0)
        - Kelly multiplier: based on Brier score thresholds

        Updates the ``category_stats`` table so the risk manager picks up new
        multipliers on its next evaluation.
        """
        rows = await self._db.fetchall(
            """
            SELECT category,
                   COUNT(*) AS n,
                   AVG((predicted_prob - actual_outcome) * (predicted_prob - actual_outcome)) AS brier,
                   SUM(
                       CASE
                           WHEN (predicted_prob > 0.5 AND actual_outcome = 1)
                             OR (predicted_prob <= 0.5 AND actual_outcome = 0)
                           THEN 1 ELSE 0
                       END
                   ) AS correct
            FROM calibration
            WHERE actual_outcome IS NOT NULL
              AND category != ''
            GROUP BY category
            """
        )

        if not rows:
            log.debug("feedback.no_resolved_predictions")
            return

        now = datetime.now(timezone.utc).isoformat()

        for row in rows:
            category = row["category"]
            n = row["n"]
            brier = row["brier"]
            correct = row["correct"]
            accuracy = correct / n if n > 0 else 0.0

            # Compute Kelly multiplier from Brier score
            kelly_mult = self._compute_kelly_multiplier(brier, accuracy, n)

            # Upsert into category_stats
            existing = await self._db.fetchone(
                "SELECT * FROM category_stats WHERE category = ?",
                (category,),
            )

            if existing is None:
                await self._db.execute(
                    """INSERT INTO category_stats
                       (category, trade_count, brier_score, kelly_multiplier, updated_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (category, n, brier, kelly_mult, now),
                )
            else:
                await self._db.execute(
                    """UPDATE category_stats
                       SET brier_score = ?, kelly_multiplier = ?, updated_at = ?
                       WHERE category = ?""",
                    (brier, kelly_mult, now, category),
                )

            log.info(
                "feedback.category_updated",
                category=category,
                n=n,
                brier=round(brier, 4),
                accuracy=round(accuracy, 2),
                kelly_mult=round(kelly_mult, 2),
            )

        await self._db.commit()

    # ------------------------------------------------------------------
    # Category accuracy report
    # ------------------------------------------------------------------

    async def get_category_accuracy(self) -> dict[str, dict]:
        """Return per-category accuracy stats.

        Returns:
            ``{category: {brier, accuracy, trade_count, kelly_mult}}``
        """
        rows = await self._db.fetchall(
            """
            SELECT category,
                   COUNT(*) AS n,
                   AVG((predicted_prob - actual_outcome) * (predicted_prob - actual_outcome)) AS brier,
                   SUM(
                       CASE
                           WHEN (predicted_prob > 0.5 AND actual_outcome = 1)
                             OR (predicted_prob <= 0.5 AND actual_outcome = 0)
                           THEN 1 ELSE 0
                       END
                   ) AS correct
            FROM calibration
            WHERE actual_outcome IS NOT NULL
              AND category != ''
            GROUP BY category
            """
        )

        result: dict[str, dict] = {}
        for row in rows:
            category = row["category"]
            n = row["n"]
            brier = row["brier"]
            accuracy = row["correct"] / n if n > 0 else 0.0
            kelly_mult = self._compute_kelly_multiplier(brier, accuracy, n)

            result[category] = {
                "brier": round(brier, 4),
                "accuracy": round(accuracy, 4),
                "trade_count": n,
                "kelly_mult": round(kelly_mult, 2),
            }

        return result

    # ------------------------------------------------------------------
    # Calibration summary for the strategic prompt
    # ------------------------------------------------------------------

    async def get_calibration_summary(self) -> str:
        """Return human-readable calibration feedback for the strategic prompt.

        Produces lines like:
            You are WELL CALIBRATED on politics_us (Brier: 0.12, 78% accuracy, 34 trades).
            You are POORLY CALIBRATED on crypto (Brier: 0.35, 41% accuracy, 15 trades) -- be more conservative.
        """
        stats = await self.get_category_accuracy()
        if not stats:
            return "(No per-category calibration data yet.)"

        lines: list[str] = []

        # Sort by trade count descending so most-traded categories appear first
        for category, s in sorted(stats.items(), key=lambda kv: kv[1]["trade_count"], reverse=True):
            brier = s["brier"]
            accuracy = s["accuracy"]
            n = s["trade_count"]

            if brier < _BRIER_GOOD and n >= 5:
                label = "WELL CALIBRATED"
                advice = ""
            elif brier > _BRIER_BAD and n >= 5:
                label = "POORLY CALIBRATED"
                advice = " -- be more conservative"
            else:
                label = "MODERATELY CALIBRATED"
                advice = ""

            lines.append(
                f"You are {label} on {category} "
                f"(Brier: {brier:.2f}, {accuracy:.0%} accuracy, {n} trades){advice}."
            )

        # Add avoid list at the end
        avoid = await self.get_avoid_categories()
        if avoid:
            lines.append(
                f"\nAVOID these categories due to persistent poor performance: "
                f"{', '.join(sorted(avoid))}."
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Avoid list
    # ------------------------------------------------------------------

    async def get_avoid_categories(self) -> set[str]:
        """Categories to skip based on poor historical performance.

        A category is added to the avoid list when:
        - At least ``_AVOID_MIN_TRADES`` resolved predictions exist
        - Directional accuracy is below ``_AVOID_ACCURACY`` (40%)
        """
        rows = await self._db.fetchall(
            """
            SELECT category,
                   COUNT(*) AS n,
                   SUM(
                       CASE
                           WHEN (predicted_prob > 0.5 AND actual_outcome = 1)
                             OR (predicted_prob <= 0.5 AND actual_outcome = 0)
                           THEN 1 ELSE 0
                       END
                   ) AS correct
            FROM calibration
            WHERE actual_outcome IS NOT NULL
              AND category != ''
            GROUP BY category
            HAVING n >= ?
            """,
            (_AVOID_MIN_TRADES,),
        )

        avoid: set[str] = set()
        for row in rows:
            accuracy = row["correct"] / row["n"] if row["n"] > 0 else 0.0
            if accuracy < _AVOID_ACCURACY:
                avoid.add(row["category"])
                log.info(
                    "feedback.avoid_category",
                    category=row["category"],
                    accuracy=round(accuracy, 2),
                    n=row["n"],
                )

        return avoid

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_kelly_multiplier(brier: float, accuracy: float, n: int) -> float:
        """Derive a Kelly multiplier from Brier score and accuracy.

        - Brier < 0.15 and enough data: 1.2x (trade more)
        - Brier > 0.30 and enough data: 0.3x (trade much less)
        - Otherwise: linear interpolation between 0.3 and 1.2
        - Fewer than 5 samples: 1.0 (neutral)
        """
        if n < 5:
            return _MULT_NEUTRAL

        if brier <= _BRIER_GOOD:
            return _MULT_GOOD
        if brier >= _BRIER_BAD:
            return _MULT_BAD

        # Linear interpolation: _BRIER_GOOD -> _MULT_GOOD, _BRIER_BAD -> _MULT_BAD
        t = (brier - _BRIER_GOOD) / (_BRIER_BAD - _BRIER_GOOD)
        return round(_MULT_GOOD + t * (_MULT_BAD - _MULT_GOOD), 2)
