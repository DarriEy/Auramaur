"""Multi-LLM ensemble analyzer — runs multiple models in parallel and weights by Brier scores.

Uses the user's 2 Max+ Claude accounts to run opus and sonnet concurrently,
then blends their probability estimates weighted by per-model calibration accuracy.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import date, datetime, timezone

import structlog
from pydantic import BaseModel, Field

from auramaur.data_sources.base import NewsItem
from auramaur.db.database import Database
from auramaur.exchange.models import Market
from auramaur.nlp.analyzer import AnalysisResult, _parse_claude_json
from auramaur.nlp.prompts import PROBABILITY_ESTIMATION_PROMPT, format_evidence
from config.settings import Settings

log = structlog.get_logger()


class ModelResult(BaseModel):
    """Result from a single model in the ensemble."""

    model: str
    probability: float = Field(ge=0, le=1)
    confidence: str = "MEDIUM"
    reasoning: str = ""
    key_factors: list[str] = Field(default_factory=list)
    time_sensitivity: str = "MEDIUM"
    weight: float = 0.5
    error: str | None = None


class EnsembleResult(BaseModel):
    """Blended result from all models in the ensemble."""

    probability: float = Field(ge=0, le=1)
    confidence: str = "MEDIUM"
    reasoning: str = ""
    key_factors: list[str] = Field(default_factory=list)
    time_sensitivity: str = "MEDIUM"
    model_results: list[ModelResult] = Field(default_factory=list)
    blend_method: str = "weighted_brier"


class EnsembleAnalyzer:
    """Runs multiple LLMs in parallel and weights their probability estimates.

    Primary model: opus (highest intelligence, deeper reasoning)
    Secondary model: sonnet (faster, different perspective)

    Weights start at 50/50 and adjust based on per-model Brier scores
    tracked in the ensemble_predictions table.
    """

    def __init__(self, settings: Settings, db: Database) -> None:
        self._settings = settings
        self._db = db
        self._models = settings.llm_ensemble.models
        self._model_weights: dict[str, float] = {}
        self._daily_calls = 0
        self._daily_calls_date = ""

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    async def load_model_weights(self) -> dict[str, float]:
        """Load per-model weights from recent resolved predictions.

        Uses the last 30 resolutions per model with exponential decay —
        recent predictions weighted more than old ones. This prevents a
        model that was sharp months ago but has drifted from carrying
        stale weight.
        """
        min_samples = self._settings.llm_ensemble.min_samples_for_weights
        default_weight = self._settings.llm_ensemble.default_weight

        # Fetch last 30 resolved predictions per model, ordered by recency
        rows = await self._db.fetchall(
            """SELECT model, probability, actual_outcome,
                      ROW_NUMBER() OVER (PARTITION BY model ORDER BY timestamp DESC) AS recency_rank
               FROM ensemble_predictions
               WHERE actual_outcome IS NOT NULL
               ORDER BY model, timestamp DESC""",
        )

        if not rows:
            self._model_weights = {m: default_weight for m in self._models}
            log.debug("ensemble.using_default_weights", weights=self._model_weights)
            return self._model_weights

        # Compute exponentially-decayed Brier scores per model
        # Decay factor: most recent prediction has weight 1.0, 30th-oldest has weight ~0.22
        import math
        decay_rate = 0.05  # exp(-0.05 * rank)
        max_window = 30

        model_scores: dict[str, dict] = {}  # model -> {weighted_brier, total_weight, n}
        for row in rows:
            model = row["model"]
            rank = row["recency_rank"]
            if rank > max_window:
                continue

            if model not in model_scores:
                model_scores[model] = {"weighted_brier": 0.0, "total_weight": 0.0, "n": 0}

            prob = float(row["probability"])
            actual = int(row["actual_outcome"])
            brier_i = (prob - actual) ** 2
            decay_weight = math.exp(-decay_rate * (rank - 1))

            model_scores[model]["weighted_brier"] += brier_i * decay_weight
            model_scores[model]["total_weight"] += decay_weight
            model_scores[model]["n"] += 1

        # Convert to inverse-Brier weights
        raw_weights: dict[str, float] = {}
        for model, scores in model_scores.items():
            if scores["n"] < min_samples:
                continue
            brier = scores["weighted_brier"] / scores["total_weight"] if scores["total_weight"] > 0 else 0.5
            n = scores["n"]
            if brier > 0:
                raw_weights[model] = 1.0 / brier
            else:
                raw_weights[model] = 10.0

            log.info(
                "ensemble.model_brier",
                model=model,
                brier=round(brier, 4),
                n=n,
            )

        # Normalize so weights sum to 1.0
        total = sum(raw_weights.values())
        if total > 0:
            for model in raw_weights:
                raw_weights[model] /= total

        # Fill in defaults for models without enough data
        for model in self._models:
            if model not in raw_weights:
                raw_weights[model] = default_weight

        # Re-normalize after adding defaults
        total = sum(raw_weights.values())
        if total > 0:
            for model in raw_weights:
                raw_weights[model] /= total

        self._model_weights = raw_weights
        log.info("ensemble.weights_loaded", weights={m: round(w, 4) for m, w in raw_weights.items()})
        return self._model_weights

    async def load_model_weights_by_category(self, category: str) -> dict[str, float]:
        """Load per-model weights filtered by market category.

        Falls back to global weights if not enough category-specific data.
        """
        min_samples = self._settings.llm_ensemble.min_samples_for_weights

        rows = await self._db.fetchall(
            """SELECT model,
                      COUNT(*) AS n,
                      AVG((probability - actual_outcome) * (probability - actual_outcome)) AS brier
               FROM ensemble_predictions
               WHERE actual_outcome IS NOT NULL AND category = ?
               GROUP BY model
               HAVING n >= ?""",
            (category, min_samples),
        )

        if not rows:
            # Fall back to global weights
            return await self.load_model_weights()

        raw_weights: dict[str, float] = {}
        for row in rows:
            model = row["model"]
            brier = row["brier"]
            if brier > 0:
                raw_weights[model] = 1.0 / brier
            else:
                raw_weights[model] = 10.0

        # Normalize
        total = sum(raw_weights.values())
        if total > 0:
            for model in raw_weights:
                raw_weights[model] /= total

        # Fill missing models with small default
        default_weight = self._settings.llm_ensemble.default_weight
        for model in self._models:
            if model not in raw_weights:
                raw_weights[model] = default_weight

        total = sum(raw_weights.values())
        if total > 0:
            for model in raw_weights:
                raw_weights[model] /= total

        log.info(
            "ensemble.category_weights",
            category=category,
            weights={m: round(w, 4) for m, w in raw_weights.items()},
        )
        return raw_weights

    # ------------------------------------------------------------------
    # Core estimation
    # ------------------------------------------------------------------

    async def estimate_probability(
        self,
        market: Market,
        evidence: list[NewsItem],
    ) -> AnalysisResult:
        """Run all models in parallel, weight by per-model Brier scores, blend.

        If a model fails, use the surviving model's result at 100%.
        Returns an AnalysisResult compatible with the existing pipeline.
        """
        evidence_text = format_evidence(evidence)
        prompt = PROBABILITY_ESTIMATION_PROMPT.format(
            question=market.question,
            description=market.description,
            market_price=market.outcome_yes_price,
            evidence=evidence_text,
        )

        # Load category-specific weights if we have a category
        category = getattr(market, "category", "") or ""
        if category:
            weights = await self.load_model_weights_by_category(category)
        else:
            weights = await self.load_model_weights()

        # Launch all models in parallel
        tasks = [self._call_model(model, prompt) for model in self._models]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful results
        model_results: list[ModelResult] = []
        for model, result in zip(self._models, raw_results):
            if isinstance(result, Exception):
                log.warning(
                    "ensemble.model_failed",
                    model=model,
                    error=str(result),
                )
                model_results.append(ModelResult(
                    model=model,
                    probability=0.5,
                    weight=0.0,
                    error=str(result),
                ))
            else:
                weight = weights.get(model, self._settings.llm_ensemble.default_weight)
                model_results.append(ModelResult(
                    model=model,
                    probability=result["probability"],
                    confidence=result.get("confidence", "MEDIUM"),
                    reasoning=result.get("reasoning", ""),
                    key_factors=result.get("key_factors", []),
                    time_sensitivity=result.get("time_sensitivity", "MEDIUM"),
                    weight=weight,
                ))

        # Blend results
        successful = [mr for mr in model_results if mr.error is None]

        if not successful:
            log.error("ensemble.all_models_failed", market_id=market.id)
            return AnalysisResult(
                probability=0.5,
                confidence="LOW",
                skipped_reason="All ensemble models failed",
            )

        blended = self._weighted_blend(successful)

        # Record per-model predictions for future Brier scoring
        for mr in successful:
            await self.record_model_prediction(market.id, mr.model, mr.probability, category)

        # Build combined reasoning
        combined_reasoning = self._merge_reasoning(successful)

        # Combine key factors (deduplicate)
        all_factors: list[str] = []
        seen: set[str] = set()
        for mr in successful:
            for factor in mr.key_factors:
                if factor.lower() not in seen:
                    all_factors.append(factor)
                    seen.add(factor.lower())

        # Use highest confidence from successful models
        confidence_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        best_confidence = max(
            (mr.confidence for mr in successful),
            key=lambda c: confidence_order.get(c, 0),
        )

        # Use the most conservative time sensitivity
        sensitivity_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        max_sensitivity = max(
            (mr.time_sensitivity for mr in successful),
            key=lambda s: sensitivity_order.get(s, 0),
        )

        log.info(
            "ensemble.blended",
            market_id=market.id,
            blended_prob=round(blended, 4),
            model_probs={mr.model: round(mr.probability, 4) for mr in successful},
            model_weights={mr.model: round(mr.weight, 4) for mr in successful},
            spread=round(max(mr.probability for mr in successful) - min(mr.probability for mr in successful), 4),
        )

        return AnalysisResult(
            probability=blended,
            confidence=best_confidence,
            reasoning=combined_reasoning,
            key_factors=all_factors[:10],
            time_sensitivity=max_sensitivity,
        )

    # ------------------------------------------------------------------
    # Model calling
    # ------------------------------------------------------------------

    async def _call_model(self, model: str, prompt: str) -> dict:
        """Call a specific model via Claude CLI.

        Same as ClaudeAnalyzer._call_claude_cli but with configurable --model flag.
        """
        today = date.today().isoformat()
        if self._daily_calls_date != today:
            self._daily_calls = 0
            self._daily_calls_date = today

        budget = self._settings.nlp.daily_claude_call_budget
        if budget > 0 and self._daily_calls >= budget:
            raise RuntimeError(f"Daily Claude call budget ({budget}) exhausted")

        max_attempts = 3
        backoff_seconds = [5, 10, 20]
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "claude", "-p", prompt,
                    "--output-format", "text",
                    "--model", model,
                    "--effort", "max",
                    "--max-turns", "1",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=300,  # 5 min per model
                )

                if proc.returncode != 0:
                    err_msg = stderr.decode().strip()
                    raise RuntimeError(f"Claude CLI ({model}) failed (rc={proc.returncode}): {err_msg}")

                self._daily_calls += 1
                log.info(
                    "ensemble.model_call",
                    model=model,
                    daily_calls=self._daily_calls,
                )

                raw_text = stdout.decode().strip()
                return _parse_claude_json(raw_text)

            except (TimeoutError, asyncio.TimeoutError, RuntimeError) as e:
                last_error = e
                if attempt < max_attempts:
                    delay = backoff_seconds[attempt - 1]
                    log.warning(
                        "ensemble.model_retry",
                        model=model,
                        attempt=attempt,
                        delay=delay,
                        error=str(e),
                    )
                    await asyncio.sleep(delay)

        raise last_error  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Blending
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_blend(results: list[ModelResult]) -> float:
        """Blend probabilities weighted by model weights (derived from inverse Brier).

        If only one model succeeded, returns its probability directly.
        """
        if len(results) == 1:
            return results[0].probability

        total_weight = sum(r.weight for r in results)
        if total_weight <= 0:
            # Equal weighting fallback
            return sum(r.probability for r in results) / len(results)

        weighted_sum = sum(r.probability * r.weight for r in results)
        return weighted_sum / total_weight

    @staticmethod
    def _merge_reasoning(results: list[ModelResult]) -> str:
        """Merge reasoning from multiple models into a coherent summary."""
        if len(results) == 1:
            return results[0].reasoning

        parts: list[str] = []
        for mr in results:
            if mr.reasoning:
                parts.append(f"[{mr.model} ({mr.probability:.0%})]: {mr.reasoning}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Prediction tracking
    # ------------------------------------------------------------------

    async def record_model_prediction(
        self, market_id: str, model: str, prob: float, category: str = "",
    ) -> None:
        """Store per-model prediction for later Brier scoring."""
        await self._db.execute(
            """INSERT INTO ensemble_predictions (market_id, model, category, probability, timestamp)
               VALUES (?, ?, ?, ?, datetime('now'))""",
            (market_id, model, category, prob),
        )
        await self._db.commit()

    async def update_model_outcomes(self, market_id: str, actual_outcome: int) -> None:
        """After market resolution, set the actual outcome on all model predictions.

        Called by the calibration/resolution pipeline when a market resolves.
        """
        await self._db.execute(
            """UPDATE ensemble_predictions
               SET actual_outcome = ?
               WHERE market_id = ? AND actual_outcome IS NULL""",
            (actual_outcome, market_id),
        )
        await self._db.commit()
        log.info(
            "ensemble.outcome_recorded",
            market_id=market_id,
            actual_outcome=actual_outcome,
        )

    async def get_model_stats(self) -> dict[str, dict]:
        """Get per-model performance statistics.

        Returns:
            {model: {brier, n, weight}} for each model with resolved predictions.
        """
        rows = await self._db.fetchall(
            """SELECT model,
                      COUNT(*) AS n,
                      AVG((probability - actual_outcome) * (probability - actual_outcome)) AS brier
               FROM ensemble_predictions
               WHERE actual_outcome IS NOT NULL
               GROUP BY model"""
        )

        stats: dict[str, dict] = {}
        for row in rows:
            model = row["model"]
            brier = row["brier"]
            n = row["n"]
            stats[model] = {
                "brier": round(brier, 4),
                "n": n,
                "weight": self._model_weights.get(model, self._settings.llm_ensemble.default_weight),
            }

        return stats

    async def get_model_stats_by_category(self) -> dict[str, dict[str, dict]]:
        """Get per-model, per-category performance statistics.

        Returns:
            {category: {model: {brier, n}}}
        """
        rows = await self._db.fetchall(
            """SELECT category, model,
                      COUNT(*) AS n,
                      AVG((probability - actual_outcome) * (probability - actual_outcome)) AS brier
               FROM ensemble_predictions
               WHERE actual_outcome IS NOT NULL AND category != ''
               GROUP BY category, model"""
        )

        stats: dict[str, dict[str, dict]] = {}
        for row in rows:
            category = row["category"]
            model = row["model"]
            if category not in stats:
                stats[category] = {}
            stats[category][model] = {
                "brier": round(row["brier"], 4),
                "n": row["n"],
            }

        return stats
