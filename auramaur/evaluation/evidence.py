"""Canonical outcomes and versioned cross-stream forecast scoring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from auramaur.evaluation.scoring import score_forecast

SCORE_VERSION = "binary-proper-v1"


def event_key(venue: str, market_id: str) -> str:
    venue_norm = (venue or "polymarket").strip().lower()
    market_norm = market_id.strip()
    if not venue_norm or not market_norm:
        raise ValueError("venue and market_id are required")
    return f"{venue_norm}:{market_norm}"


@dataclass(frozen=True)
class MarketOutcome:
    venue: str
    market_id: str
    outcome: int
    resolved_at: datetime
    event_family: str = ""
    source: str = "venue"
    resolution_version: str = "venue-v1"

    def __post_init__(self) -> None:
        if self.outcome not in (0, 1):
            raise ValueError("outcome must be 0 or 1")
        if self.resolved_at.tzinfo is None or self.resolved_at.utcoffset() is None:
            raise ValueError("resolved_at must be timezone-aware")

    @property
    def event_key(self) -> str:
        return event_key(self.venue, self.market_id)


class OutcomeRepository:
    def __init__(self, db) -> None:
        self._db = db

    async def record(self, outcome: MarketOutcome) -> bool:
        """Record immutable venue truth; identical repeats are no-ops."""
        async with self._db.transaction():
            cursor = await self._db.execute(
                """INSERT OR IGNORE INTO market_outcomes
                   (event_key,venue,market_id,event_family,outcome,resolved_at,
                    source,resolution_version)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (outcome.event_key, outcome.venue.strip().lower(), outcome.market_id,
                 outcome.event_family or outcome.market_id, outcome.outcome,
                 outcome.resolved_at.astimezone(timezone.utc).isoformat(),
                 outcome.source, outcome.resolution_version),
            )
            rowcount = cursor.rowcount
            # aiosqlite reports 1/0. Lightweight Database protocol fakes used
            # by strategy tests may not expose a numeric rowcount; an INSERT
            # that did not raise is their success contract.
            if not isinstance(rowcount, int):
                return True
            inserted = rowcount == 1
            if inserted:
                return True
            existing = await self._db.fetchone(
                "SELECT outcome FROM market_outcomes WHERE event_key=?",
                (outcome.event_key,),
            )
            if existing is None or int(existing["outcome"]) != outcome.outcome:
                raise ValueError(f"conflicting canonical outcome for {outcome.event_key}")
        return False


def _horizon_bucket(observed_at: str, resolved_at: str) -> str:
    observed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    resolved = datetime.fromisoformat(resolved_at.replace("Z", "+00:00"))
    days = max(0.0, (resolved - observed).total_seconds() / 86400)
    if days <= 1:
        return "0-1d"
    if days <= 7:
        return "1-7d"
    if days <= 30:
        return "7-30d"
    return "30d+"


class ForecastScoreMaterializer:
    """Idempotently rebuild comparable score facts from the semantic view."""

    def __init__(self, db, score_version: str = SCORE_VERSION) -> None:
        self._db = db
        self._version = score_version

    async def refresh(self) -> int:
        rows = await self._db.fetchall(
            """SELECT * FROM unified_forecast_evidence
               WHERE outcome IS NOT NULL AND status='succeeded'""")
        facts = []
        for row in rows:
            score = score_forecast(
                float(row["probability"]), float(row["market_probability"]),
                int(row["outcome"]),
            )
            facts.append((
                row["forecast_key"], row["event_key"], row["event_family"] or row["event_key"],
                row["stream"], row["arm"] or "", row["probability_kind"],
                row["observed_at"], _horizon_bucket(row["observed_at"], row["resolved_at"]),
                row["outcome"], score.brier, score.log_loss, score.market_brier,
                score.brier_delta, score.brier_skill, self._version,
            ))
        async with self._db.transaction():
            await self._db.execute(
                "DELETE FROM forecast_score_facts WHERE score_version=?", (self._version,))
            if facts:
                await self._db.executemany(
                    """INSERT INTO forecast_score_facts
                       (forecast_key,event_key,event_family,stream,arm,probability_kind,
                        observed_at,horizon_bucket,outcome,brier,log_loss,market_brier,
                        brier_delta,brier_skill,score_version)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", facts)
        return len(facts)

    async def event_weighted_summary(self) -> list[dict]:
        """One predeclared earliest observation per event-family/horizon/arm."""
        rows = await self._db.fetchall(
            """WITH ranked AS (
                 SELECT *, ROW_NUMBER() OVER (
                   PARTITION BY stream,arm,probability_kind,event_family,horizon_bucket
                   ORDER BY observed_at ASC,forecast_key ASC) AS rn
                 FROM forecast_score_facts WHERE score_version=?
               )
               SELECT stream,arm,probability_kind,horizon_bucket,COUNT(*) AS events,
                      AVG(brier) AS brier,AVG(log_loss) AS log_loss,
                      AVG(market_brier) AS market_brier,AVG(brier_delta) AS brier_delta
                 FROM ranked WHERE rn=1
                GROUP BY stream,arm,probability_kind,horizon_bucket
                ORDER BY stream,arm,probability_kind,horizon_bucket""",
            (self._version,),
        )
        return [dict(row) for row in rows]
