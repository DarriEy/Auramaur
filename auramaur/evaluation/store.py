"""SQLite persistence for paired prospective evaluation records."""

from __future__ import annotations

import hashlib
import json

from auramaur.evaluation.domain import (
    EpisodeSnapshot, EvaluationForecast, EvaluationOutcome, EvaluationRun,
)
from auramaur.evaluation.scoring import ForecastScore, score_forecast
from auramaur.evaluation.evidence import MarketOutcome, OutcomeRepository


def _ts(value) -> str:
    return value.isoformat()


class EvaluationStore:
    def __init__(self, db) -> None:
        self._db = db

    async def put_episode(self, episode: EpisodeSnapshot) -> str:
        raw = episode.canonical_json()
        async with self._db.transaction():
            await self._db.execute(
                """INSERT OR IGNORE INTO evaluation_episodes
                   (episode_hash, venue, market_id, event_family, observed_at,
                    market_prob_yes, snapshot_json) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (episode.episode_hash, episode.venue, episode.market_id,
                 episode.event_family, _ts(episode.observed_at),
                 episode.market_prob_yes, raw),
            )
            row = await self._db.fetchone(
                "SELECT snapshot_json FROM evaluation_episodes WHERE episode_hash = ?",
                (episode.episode_hash,),
            )
            if row is None or row["snapshot_json"] != raw:
                raise ValueError("episode hash collision or immutable snapshot conflict")
        return episode.episode_hash

    async def get_episode(self, episode_hash: str) -> EpisodeSnapshot | None:
        row = await self._db.fetchone(
            "SELECT snapshot_json FROM evaluation_episodes WHERE episode_hash = ?",
            (episode_hash,),
        )
        return None if row is None else EpisodeSnapshot.model_validate_json(row["snapshot_json"])

    async def put_run(self, run: EvaluationRun) -> None:
        async with self._db.transaction():
            await self._db.execute(
                """INSERT INTO evaluation_runs
                   (run_id, arm_name, model, quantization, exploration_policy,
                    seed, prompt_version, output_schema_version, status,
                    input_tokens, output_tokens, tool_calls, duration_ms,
                    compute_seconds, error, started_at, completed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(run_id) DO UPDATE SET status=excluded.status,
                    input_tokens=excluded.input_tokens, output_tokens=excluded.output_tokens,
                    tool_calls=excluded.tool_calls, duration_ms=excluded.duration_ms,
                    compute_seconds=excluded.compute_seconds, error=excluded.error,
                    completed_at=excluded.completed_at""",
                (run.run_id, run.arm_name, run.model, run.quantization,
                 run.exploration_policy, run.seed, run.prompt_version,
                 run.output_schema_version, run.status.value, run.input_tokens,
                 run.output_tokens, run.tool_calls, run.duration_ms,
                 run.compute_seconds, run.error, _ts(run.started_at),
                 None if run.completed_at is None else _ts(run.completed_at)),
            )

    async def put_forecast(self, forecast: EvaluationForecast) -> None:
        if await self.get_episode(forecast.episode_hash) is None:
            raise ValueError("unknown episode")
        async with self._db.transaction():
            await self._db.execute(
                """INSERT INTO evaluation_forecasts
                   (forecast_id, run_id, episode_hash, prob_yes, action,
                    min_acceptable_price, max_acceptable_price, thesis,
                    uncertainty, evidence_ids_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (forecast.forecast_id, forecast.run_id, forecast.episode_hash,
                 forecast.prob_yes, forecast.action, forecast.min_acceptable_price,
                 forecast.max_acceptable_price, forecast.thesis,
                 forecast.uncertainty, json.dumps(forecast.evidence_ids)),
            )

    async def put_attempt(self, run_id, episode_hash, attempt) -> None:
        """Retain raw samples, critics, telemetry, and failures."""
        attempt_id = hashlib.sha256(
            f"{run_id}:{attempt.stage}:{attempt.sample_index}".encode()).hexdigest()
        forecast = attempt.forecast
        async with self._db.transaction():
            await self._db.execute(
                """INSERT OR REPLACE INTO evaluation_attempts
                   (attempt_id,run_id,episode_hash,stage,sample_index,seed,prob_yes,
                    action,confidence,thesis,telemetry_json,error)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (attempt_id, run_id, episode_hash, attempt.stage, attempt.sample_index,
                 attempt.seed, None if forecast is None else forecast.prob_yes,
                 None if forecast is None else forecast.action,
                 None if forecast is None else forecast.confidence,
                 "" if forecast is None else (forecast.thesis or ""),
                 json.dumps(dict(attempt.telemetry), sort_keys=True), attempt.error or ""))

    async def latest_market_observations(self) -> dict[tuple[str, str], dict]:
        rows = await self._db.fetchall(
            """SELECT e.venue,e.market_id,e.observed_at,e.market_prob_yes,e.event_family
                 FROM evaluation_episodes e JOIN
                 (SELECT venue,market_id,MAX(observed_at) observed_at
                    FROM evaluation_episodes GROUP BY venue,market_id) latest
                 ON latest.venue=e.venue AND latest.market_id=e.market_id
                 AND latest.observed_at=e.observed_at""")
        return {(r["venue"].lower(), r["market_id"]): dict(r) for r in rows}

    async def put_cycle(self, values: dict) -> None:
        keys = ("cycle_id","started_at","completed_at","eligible_markets",
                "selected_markets","unique_families","forecasts","attempts",
                "failed_attempts","duration_ms","compute_seconds")
        async with self._db.transaction():
            await self._db.execute(
                "INSERT INTO evaluation_cycles VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                tuple(values[key] for key in keys))

    async def settle(self, outcome: EvaluationOutcome) -> None:
        if await self.get_episode(outcome.episode_hash) is None:
            raise ValueError("unknown episode")
        episode = await self.get_episode(outcome.episode_hash)
        await OutcomeRepository(self._db).record(MarketOutcome(
            venue=episode.venue, market_id=episode.market_id,
            outcome=outcome.outcome, resolved_at=outcome.resolved_at,
            source=outcome.source or "evaluation_compat",
            event_family=episode.event_family or episode.market_id,
        ))
        async with self._db.transaction():
            existing = await self._db.fetchone(
                "SELECT outcome, resolved_at, source FROM evaluation_outcomes WHERE episode_hash = ?",
                (outcome.episode_hash,),
            )
            values = (outcome.outcome, _ts(outcome.resolved_at), outcome.source)
            if existing is not None:
                if (existing["outcome"], existing["resolved_at"], existing["source"]) != values:
                    raise ValueError("conflicting outcome settlement")
                return
            await self._db.execute(
                "INSERT INTO evaluation_outcomes (episode_hash, outcome, resolved_at, source) VALUES (?, ?, ?, ?)",
                (outcome.episode_hash, *values),
            )

    async def score(self, forecast_id: str) -> ForecastScore | None:
        row = await self._db.fetchone(
            """SELECT f.prob_yes, e.market_prob_yes, o.outcome
               FROM evaluation_forecasts f
               JOIN evaluation_episodes e ON e.episode_hash=f.episode_hash
               LEFT JOIN market_outcomes o
                 ON o.event_key=lower(e.venue) || ':' || e.market_id
               WHERE f.forecast_id=?""", (forecast_id,))
        if row is None or row["outcome"] is None:
            return None
        return score_forecast(row["prob_yes"], row["market_prob_yes"], row["outcome"])

    async def summary(self) -> list[dict]:
        """Read-only per-arm resolved scorecard for CLI/reporting consumers."""
        rows = await self._db.fetchall(
            """WITH ranked AS (
                 SELECT u.*, ROW_NUMBER() OVER (
                   PARTITION BY u.arm,u.event_family
                   ORDER BY u.observed_at ASC,u.forecast_key ASC) AS rn
                 FROM unified_forecast_evidence u
                 WHERE u.stream='intelligence_eval' AND u.outcome IS NOT NULL
               )
               SELECT arm AS arm_name,model,COUNT(*) AS forecasts,
                      AVG((probability-outcome)*(probability-outcome)) AS brier,
                      AVG((market_probability-outcome)*
                          (market_probability-outcome)) AS market_brier,
                      SUM(abstained) AS abstains
                 FROM ranked WHERE rn=1
                GROUP BY arm,model ORDER BY brier ASC""")
        return [dict(row) for row in rows]
