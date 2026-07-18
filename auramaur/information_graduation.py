"""Graduation for source × category × horizon × event-type information cells."""

from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from auramaur.db.database import Database


@dataclass(frozen=True)
class InformationDecision:
    status: str
    influence_multiplier: float
    reason: str


class InformationGraduation:
    """Registers immutable paired trials and graduates incremental information.

    Unlike capital graduation, this ladder changes evidence influence only.
    It can never loosen risk checks or directly increase position size.
    """

    def __init__(self, db: Database, *, min_resolved: int = 30,
                 min_paired: int = 50, min_success_rate: float = 0.98,
                 probation_multiplier: float = 0.25) -> None:
        self._db = db
        self._min_resolved = min_resolved
        self._min_paired = min_paired
        self._min_success_rate = min_success_rate
        self._probation_multiplier = probation_multiplier

    async def register(self, source: str, category: str, horizon: str,
                       event_type: str = "") -> str:
        key = f"{source}|{category}|{horizon}|{event_type}"
        strategy_id = hashlib.sha256(key.encode()).hexdigest()[:24]
        await self._db.execute(
            """INSERT OR IGNORE INTO information_strategies
               (id,source,category,horizon,event_type,mode)
               VALUES (?,?,?,?,?,'shadow')""",
            (strategy_id, source, category, horizon, event_type),
        )
        await self._db.execute(
            "INSERT OR IGNORE INTO information_graduation_state (strategy_id) VALUES (?)",
            (strategy_id,),
        )
        await self._db.commit()
        return strategy_id

    async def assign(self, strategy_id: str, market_id: str, observed_at: datetime,
                     market_price: float) -> tuple[str, str]:
        """Create a deterministic, immutable 50/50 control/treatment assignment."""
        stamp = observed_at.astimezone(timezone.utc).isoformat()
        digest = hashlib.sha256(f"{strategy_id}|{market_id}|{stamp}".encode()).hexdigest()
        assignment = "treatment" if int(digest[:8], 16) % 2 else "control"
        trial_id = uuid.uuid5(uuid.NAMESPACE_URL, f"auramaur:{digest}").hex
        await self._db.execute(
            """INSERT OR IGNORE INTO information_trials
               (id,strategy_id,market_id,observed_at,assignment,assignment_hash,market_price)
               VALUES (?,?,?,?,?,?,?)""",
            (trial_id, strategy_id, market_id, stamp, assignment, digest, market_price),
        )
        await self._db.commit()
        return trial_id, assignment

    async def record_forecast(self, trial_id: str, arm: str, probability: float,
                              forecast_id: int | None = None,
                              net_paper_pnl: float | None = None) -> None:
        if arm not in {"control", "treatment"}:
            raise ValueError("arm must be control or treatment")
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0,1]")
        await self._db.execute(
            """INSERT OR REPLACE INTO paired_forecasts
               (trial_id,arm,probability,forecast_id,net_paper_pnl)
               VALUES (?,?,?,?,?)""",
            (trial_id, arm, probability, forecast_id, net_paper_pnl),
        )
        await self._db.commit()

    async def resolve(self, trial_id: str, outcome: bool) -> None:
        actual = int(outcome)
        await self._db.execute(
            "UPDATE information_trials SET resolved_outcome=?,resolved_at=datetime('now') WHERE id=?",
            (actual, trial_id),
        )
        rows = await self._db.fetchall(
            "SELECT arm,probability,net_paper_pnl FROM paired_forecasts WHERE trial_id=?",
            (trial_id,),
        )
        arms = {r["arm"]: r for r in rows}
        if {"control", "treatment"} <= arms.keys():
            eps = 1e-9
            def scores(row):
                p = min(1 - eps, max(eps, float(row["probability"])))
                return (p - actual) ** 2, -(actual * math.log(p) + (1-actual) * math.log(1-p))
            cb, cl = scores(arms["control"])
            tb, tl = scores(arms["treatment"])
            source = await self._db.fetchone(
                """SELECT s.source FROM information_trials t
                   JOIN information_strategies s ON s.id=t.strategy_id WHERE t.id=?""",
                (trial_id,),
            )
            cp = arms["control"]["net_paper_pnl"] or 0
            tp = arms["treatment"]["net_paper_pnl"] or 0
            await self._db.execute(
                """INSERT OR REPLACE INTO source_contributions
                   (trial_id,source,control_brier,treatment_brier,control_log_loss,
                    treatment_log_loss,incremental_brier,incremental_log_loss,incremental_pnl)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (trial_id, source["source"], cb, tb, cl, tl, cb-tb, cl-tl, tp-cp),
            )
        await self._db.commit()

    async def evaluate(self, strategy_id: str) -> InformationDecision:
        row = await self._db.fetchone(
            """SELECT s.source,
                 COUNT(DISTINCT CASE WHEN t.resolved_outcome IS NOT NULL THEN t.market_id END) resolved_n,
                 COUNT(DISTINCT c.trial_id) paired_n,
                 AVG(c.incremental_brier) inc_brier,
                 AVG(c.incremental_log_loss) inc_log,
                 SUM(c.incremental_pnl) inc_pnl
               FROM information_strategies s
               LEFT JOIN information_trials t ON t.strategy_id=s.id
               LEFT JOIN source_contributions c ON c.trial_id=t.id
               WHERE s.id=? GROUP BY s.id""",
            (strategy_id,),
        )
        if row is None:
            raise KeyError(strategy_id)
        health = await self._db.fetchone(
            """SELECT AVG(CASE WHEN status='ok' THEN 1.0 ELSE 0.0 END) rate
               FROM source_fetches WHERE source=? AND observed_at>=datetime('now','-30 days')""",
            (row["source"],),
        )
        success = float(health["rate"] or 0) if health else 0.0
        resolved, paired = int(row["resolved_n"] or 0), int(row["paired_n"] or 0)
        ib, il, pnl = row["inc_brier"], row["inc_log"], float(row["inc_pnl"] or 0)
        if success < self._min_success_rate:
            decision = InformationDecision("shadow", 0, f"source health {success:.1%} below {self._min_success_rate:.0%}")
        elif resolved < self._min_resolved or paired < self._min_paired:
            decision = InformationDecision("paired_trial", 0, f"samples resolved={resolved}/{self._min_resolved}, paired={paired}/{self._min_paired}")
        elif float(ib or 0) > 0 and float(il or 0) > 0 and pnl > 0:
            decision = InformationDecision(
                "probation", self._probation_multiplier,
                "positive incremental Brier, log loss, and paper P&L")
        else:
            decision = InformationDecision("demoted", 0, "candidate failed incremental-value gates")
        await self._db.execute(
            """UPDATE information_graduation_state SET status=?,influence_multiplier=?,
               resolved_trials=?,paired_forecasts=?,incremental_brier=?,incremental_log_loss=?,
               incremental_pnl=?,source_success_rate=?,reason=?,updated_at=datetime('now')
               WHERE strategy_id=?""",
            (decision.status, decision.influence_multiplier, resolved, paired, ib, il,
             pnl, success, decision.reason, strategy_id),
        )
        await self._db.commit()
        return decision

    async def report(self) -> list[dict]:
        rows = await self._db.fetchall(
            """SELECT s.source,s.category,s.horizon,s.event_type,g.*
               FROM information_strategies s
               JOIN information_graduation_state g ON g.strategy_id=s.id
               ORDER BY s.source,s.category,s.horizon"""
        )
        return [dict(r) for r in rows]
