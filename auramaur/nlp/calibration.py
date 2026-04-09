"""Calibration tracking for prediction accuracy with Platt scaling.

Supports both batch Platt scaling (periodic refit every ~6 hours) and
online (continuous) calibration updates via stochastic gradient descent
after each market resolution.
"""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timezone

import structlog

from auramaur.db.database import Database

log = structlog.get_logger()

# Default learning rate for online SGD updates
_ONLINE_LR = 0.01
# Window size for moving Brier score
_BRIER_WINDOW = 20


def _clamp(p: float) -> float:
    """Clamp probability to avoid log(0)."""
    return max(0.001, min(0.999, p))


def _logit(p: float) -> float:
    """Logit transform: log(p / (1-p))."""
    p = _clamp(p)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    """Sigmoid function: 1 / (1 + exp(-x))."""
    # Guard against overflow
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


class CalibrationTracker:
    """Records predictions and resolutions to track calibration quality.

    Supports Platt scaling recalibration: fits a logistic correction
    ``calibrated = sigmoid(a * logit(raw) + b)`` on resolved predictions
    and uses it to adjust future probability estimates.

    Online updates: after each market resolution, ``online_update()``
    performs a single-sample SGD step on the Platt parameters so
    calibration improves immediately rather than waiting for the next
    batch refit.
    """

    def __init__(
        self,
        db: Database,
        min_samples: int = 30,
        online_lr: float = _ONLINE_LR,
    ) -> None:
        self._db = db
        self._min_samples = min_samples
        self._params_cache: dict[str, tuple[float, float]] = {}
        self._online_lr = online_lr
        # Moving Brier score: deque of (predicted_prob, actual_outcome) for
        # the last _BRIER_WINDOW resolutions, used to detect calibration drift.
        self._recent_predictions: deque[tuple[float, float]] = deque(
            maxlen=_BRIER_WINDOW
        )
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Load recent resolved predictions into the Brier window on first use.

        This ensures the moving Brier score is continuous across bot
        restarts instead of resetting to empty each time.
        """
        if self._initialized:
            return
        self._initialized = True
        try:
            rows = await self._db.fetchall(
                """SELECT predicted_prob, actual_outcome
                   FROM calibration
                   WHERE actual_outcome IS NOT NULL
                   ORDER BY resolved_at DESC
                   LIMIT ?""",
                (_BRIER_WINDOW,),
            )
            # Insert oldest-first so the deque is in chronological order
            for row in reversed(rows):
                self._recent_predictions.append(
                    (row["predicted_prob"], float(row["actual_outcome"]))
                )
            if rows:
                brier = self.get_moving_brier_score()
                log.info(
                    "calibration.brier_window_loaded",
                    count=len(rows),
                    moving_brier=round(brier, 4) if brier is not None else None,
                )
        except Exception as e:
            log.warning("calibration.brier_window_load_failed", error=str(e))

    async def record_prediction(
        self, market_id: str, predicted_prob: float, category: str = ""
    ) -> None:
        """Record a new probability prediction for a market.

        Args:
            market_id: The market identifier.
            predicted_prob: Our predicted probability (0-1).
            category: Market category for per-category calibration.
        """
        await self._ensure_initialized()
        await self._db.execute(
            """
            INSERT INTO calibration (market_id, predicted_prob, category, created_at)
            VALUES (?, ?, ?, datetime('now'))
            """,
            (market_id, predicted_prob, category),
        )
        await self._db.commit()
        log.debug(
            "calibration.recorded",
            market_id=market_id,
            prob=predicted_prob,
            category=category,
        )

    async def record_resolution(self, market_id: str, actual_outcome: bool) -> None:
        """Record the actual resolution for a market.

        Updates the most recent unresolved prediction for this market and
        triggers an online calibration update so the Platt scaling
        parameters improve immediately.

        Args:
            market_id: The market identifier.
            actual_outcome: True if the market resolved YES, False otherwise.
        """
        await self._ensure_initialized()
        outcome_int = 1 if actual_outcome else 0

        # Fetch the prediction before marking it resolved so we can run
        # an online parameter update.
        pred_row = await self._db.fetchone(
            """
            SELECT predicted_prob, category
            FROM calibration
            WHERE market_id = ? AND actual_outcome IS NULL
            ORDER BY created_at DESC LIMIT 1
            """,
            (market_id,),
        )

        await self._db.execute(
            """
            UPDATE calibration
            SET actual_outcome = ?, resolved_at = datetime('now')
            WHERE market_id = ? AND actual_outcome IS NULL
            """,
            (outcome_int, market_id),
        )
        await self._db.commit()
        log.info("calibration.resolved", market_id=market_id, outcome=actual_outcome)

        # Trigger online calibration update if we have the prediction data
        if pred_row is not None:
            predicted_prob = pred_row["predicted_prob"]
            category = pred_row["category"] or ""
            await self.online_update(predicted_prob, float(outcome_int), category)

        # After enough resolutions accumulate, trigger a batch refit
        # to complement the online SGD with a proper MLE fit.
        self._resolutions_since_refit = getattr(self, "_resolutions_since_refit", 0) + 1
        if self._resolutions_since_refit >= 10:
            self._resolutions_since_refit = 0
            try:
                await self.refit_all()
                log.info("calibration.auto_refit", trigger="10_resolutions")
            except Exception as e:
                log.warning("calibration.auto_refit_failed", error=str(e))

    async def online_update(
        self, predicted_prob: float, actual_outcome: float, category: str = ""
    ) -> None:
        """Perform a single-sample SGD update to the Platt scaling parameters.

        This is called automatically by ``record_resolution()`` so that
        calibration improves immediately after each market resolves, rather
        than waiting for the next batch refit.

        The update rule (derived from the gradient of binary cross-entropy
        on the Platt-scaled output):

            calibrated = sigmoid(a * logit(pred) + b)
            error = actual_outcome - calibrated
            a += lr * error * logit(pred)
            b += lr * error

        Args:
            predicted_prob: The raw probability we predicted (0-1).
            actual_outcome: 1.0 if YES, 0.0 if NO.
            category: Market category for per-category parameter updates.
        """
        # Update the moving Brier score window
        self._recent_predictions.append((predicted_prob, actual_outcome))

        # Update both the category-specific params and the global params
        categories_to_update = {""}
        if category:
            categories_to_update.add(category)

        x = _logit(predicted_prob)

        for cat in categories_to_update:
            params = await self._load_params(cat)
            if params is None:
                # Seed identity params (a=1, b=0 = no adjustment) so online
                # learning can begin immediately instead of waiting for
                # min_samples batch refit.
                await self._db.execute(
                    """INSERT OR IGNORE INTO calibration_params
                       (category, a, b, fitted_at)
                       VALUES (?, 1.0, 0.0, datetime('now'))""",
                    (cat,),
                )
                await self._db.commit()
                params = (1.0, 0.0)
                self._params_cache[cat] = params
                log.info("calibration.seeded_identity_params", category=cat)

            old_a, old_b = params

            # Compute current calibrated probability
            calibrated = _sigmoid(old_a * x + old_b)

            # Gradient step (ascending on likelihood = descending on loss)
            error = actual_outcome - calibrated
            new_a = old_a + self._online_lr * error * x
            new_b = old_b + self._online_lr * error

            # Update cache
            self._params_cache[cat] = (new_a, new_b)

            # Persist to DB
            now = datetime.now(timezone.utc).isoformat()
            await self._db.execute(
                """
                UPDATE calibration_params
                SET a = ?, b = ?, fitted_at = ?
                WHERE category = ?
                """,
                (new_a, new_b, now, cat),
            )
            await self._db.commit()

            # Compute moving Brier score for logging
            moving_brier = self.get_moving_brier_score()

            log.info(
                "calibration.online_update",
                category=cat,
                predicted_prob=round(predicted_prob, 4),
                actual_outcome=actual_outcome,
                old_a=round(old_a, 4),
                old_b=round(old_b, 4),
                new_a=round(new_a, 4),
                new_b=round(new_b, 4),
                moving_brier=round(moving_brier, 4) if moving_brier is not None else None,
            )

    def get_moving_brier_score(self) -> float | None:
        """Compute Brier score over the last N resolutions (sliding window).

        Tracks recent calibration quality so we can detect drift.  Uses an
        in-memory deque of the most recent predictions/outcomes rather than
        querying the database.

        Returns:
            The moving Brier score, or None if no recent predictions.
        """
        if not self._recent_predictions:
            return None
        total = 0.0
        for pred, actual in self._recent_predictions:
            total += (pred - actual) ** 2
        return total / len(self._recent_predictions)

    async def get_brier_score(self) -> float | None:
        """Compute the Brier score over all resolved predictions.

        Brier score = mean( (predicted_prob - actual_outcome)^2 )
        Lower is better (0 = perfect, 0.25 = coin-flip baseline).

        Returns:
            The Brier score, or None if no resolved predictions exist.
        """
        row = await self._db.fetchone(
            """
            SELECT AVG((predicted_prob - actual_outcome) * (predicted_prob - actual_outcome))
                   AS brier_score,
                   COUNT(*) AS n
            FROM calibration
            WHERE actual_outcome IS NOT NULL
            """,
        )
        if row is None or row["n"] == 0:
            return None

        score = row["brier_score"]
        log.info("calibration.brier_score", score=score, n=row["n"])
        return score

    async def get_calibration_curve(
        self, n_bins: int = 10
    ) -> list[tuple[float, float]]:
        """Compute a calibration curve: predicted vs actual by bin.

        Divides predictions into ``n_bins`` equal-width buckets by predicted
        probability and returns the mean predicted and mean actual outcome
        for each non-empty bucket.

        Args:
            n_bins: Number of bins (default 10).

        Returns:
            A list of (mean_predicted, mean_actual) tuples, one per non-empty bin.
        """
        rows = await self._db.fetchall(
            """
            SELECT predicted_prob, actual_outcome
            FROM calibration
            WHERE actual_outcome IS NOT NULL
            ORDER BY predicted_prob
            """,
        )

        if not rows:
            return []

        # Build bins
        bin_width = 1.0 / n_bins
        bins: dict[int, list[tuple[float, int]]] = {}
        for row in rows:
            prob = row["predicted_prob"]
            outcome = row["actual_outcome"]
            bucket = min(int(prob / bin_width), n_bins - 1)
            bins.setdefault(bucket, []).append((prob, outcome))

        curve: list[tuple[float, float]] = []
        for bucket in sorted(bins):
            entries = bins[bucket]
            mean_pred = sum(p for p, _ in entries) / len(entries)
            mean_actual = sum(o for _, o in entries) / len(entries)
            curve.append((mean_pred, mean_actual))

        log.info("calibration.curve", bins=len(curve), total=len(rows))
        return curve

    # ------------------------------------------------------------------
    # Platt scaling
    # ------------------------------------------------------------------

    async def fit_params(self, category: str | None = None) -> tuple[float, float] | None:
        """Fit Platt scaling parameters using gradient descent.

        Fits ``calibrated = sigmoid(a * logit(raw) + b)`` by minimizing
        cross-entropy loss on resolved predictions.

        Args:
            category: If provided, only use predictions from this category.
                      Use ``""`` or ``None`` for global fit.

        Returns:
            Tuple (a, b) if enough samples, else None.
        """
        cat = category or ""
        if cat:
            rows = await self._db.fetchall(
                """
                SELECT predicted_prob, actual_outcome
                FROM calibration
                WHERE actual_outcome IS NOT NULL AND category = ?
                """,
                (cat,),
            )
        else:
            rows = await self._db.fetchall(
                """
                SELECT predicted_prob, actual_outcome
                FROM calibration
                WHERE actual_outcome IS NOT NULL
                """,
            )

        if len(rows) < self._min_samples:
            log.debug(
                "calibration.fit_skip",
                category=cat,
                n=len(rows),
                min_required=self._min_samples,
            )
            return None

        # Prepare data
        xs = [_logit(row["predicted_prob"]) for row in rows]
        ys = [float(row["actual_outcome"]) for row in rows]
        n = len(xs)

        # Gradient descent to minimize cross-entropy loss
        # Loss = -1/n * sum[ y*log(sig(a*x+b)) + (1-y)*log(1-sig(a*x+b)) ]
        a = 1.0
        b = 0.0
        lr = 0.01

        for _ in range(50):
            grad_a = 0.0
            grad_b = 0.0
            for xi, yi in zip(xs, ys):
                pred = _sigmoid(a * xi + b)
                err = pred - yi
                grad_a += err * xi
                grad_b += err
            grad_a /= n
            grad_b /= n
            a -= lr * grad_a
            b -= lr * grad_b

        # Compute Brier score for this fit
        brier = 0.0
        for xi, yi in zip(xs, ys):
            cal_p = _sigmoid(a * xi + b)
            brier += (cal_p - yi) ** 2
        brier /= n

        # Store params
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """
            INSERT OR REPLACE INTO calibration_params (category, a, b, n, brier_score, fitted_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (cat, a, b, n, brier, now),
        )
        await self._db.commit()

        # Update cache
        self._params_cache[cat] = (a, b)

        log.info(
            "calibration.fitted",
            category=cat,
            a=round(a, 4),
            b=round(b, 4),
            n=n,
            brier=round(brier, 4),
        )
        return (a, b)

    async def _load_params(self, category: str) -> tuple[float, float] | None:
        """Load Platt scaling params, trying category first then global.

        Args:
            category: The category to look up.

        Returns:
            Tuple (a, b) if found, else None.
        """
        # Check in-memory cache first
        if category in self._params_cache:
            return self._params_cache[category]
        if "" in self._params_cache:
            fallback = self._params_cache[""]
        else:
            fallback = None

        # Try category-specific params
        if category:
            row = await self._db.fetchone(
                "SELECT a, b FROM calibration_params WHERE category = ?",
                (category,),
            )
            if row is not None:
                params = (row["a"], row["b"])
                self._params_cache[category] = params
                return params

        # Fall back to global
        row = await self._db.fetchone(
            "SELECT a, b FROM calibration_params WHERE category = ?",
            ("",),
        )
        if row is not None:
            params = (row["a"], row["b"])
            self._params_cache[""] = params
            return params

        return fallback

    async def adjust(self, raw_prob: float, category: str = "") -> float:
        """Apply Platt scaling to a raw probability estimate.

        Args:
            raw_prob: The raw predicted probability (0-1).
            category: Market category for per-category calibration.

        Returns:
            Calibrated probability, or raw_prob if no params are available.
        """
        params = await self._load_params(category)
        if params is None:
            return raw_prob
        a, b = params
        calibrated = _sigmoid(a * _logit(raw_prob) + b)
        log.debug(
            "calibration.adjusted",
            raw=round(raw_prob, 4),
            calibrated=round(calibrated, 4),
            category=category,
        )
        return calibrated

    async def refit_all(self) -> None:
        """Refit global params and per-category params where enough samples exist."""
        # Global fit
        await self.fit_params(category=None)

        # Per-category fits
        rows = await self._db.fetchall(
            """
            SELECT category, COUNT(*) AS n
            FROM calibration
            WHERE actual_outcome IS NOT NULL AND category != ''
            GROUP BY category
            HAVING n >= ?
            """,
            (self._min_samples,),
        )
        for row in rows:
            await self.fit_params(category=row["category"])

        log.info("calibration.refit_all_done", categories=len(rows))
