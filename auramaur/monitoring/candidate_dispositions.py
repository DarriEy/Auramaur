"""Persistent, reason-coded terminal accounting for discovered candidates."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from uuid import uuid4
import structlog

log = structlog.get_logger()
TERMINAL_DISPOSITIONS = frozenset({"executed", "risk-blocked", "filtered", "throttled", "malformed", "unavailable", "failed"})

@dataclass
class CandidateDispositionCycle:
    db: object
    exchange: str
    strategy: str = "engine"
    retention_days: int = 30
    summary_retention_days: int = 90
    cycle_id: str = field(default_factory=lambda: uuid4().hex)
    _rows: dict[str, tuple[str, str, str]] = field(default_factory=dict)

    def offer(self, market_id: str) -> None:
        self.mark(market_id, "unavailable", "discovery", "no terminal decision")

    def mark(self, market_id: str, disposition: str, stage: str, reason: str = "") -> None:
        if disposition not in TERMINAL_DISPOSITIONS:
            raise ValueError(f"invalid candidate disposition: {disposition}")
        if market_id:
            self._rows[str(market_id)] = (disposition, stage[:80], reason[:500])

    async def flush(self) -> dict[str, int]:
        counts = Counter(row[0] for row in self._rows.values())
        try:
            async with self.db.transaction(owner="candidate_dispositions"):
                for market_id, (disposition, stage, reason) in self._rows.items():
                    await self.db.execute("""INSERT INTO candidate_dispositions
                        (cycle_id,market_id,exchange,strategy,disposition,stage,reason) VALUES (?,?,?,?,?,?,?)
                        ON CONFLICT(cycle_id,market_id,strategy) DO UPDATE SET
                        disposition=excluded.disposition,stage=excluded.stage,reason=excluded.reason""",
                        (self.cycle_id,market_id,self.exchange,self.strategy,disposition,stage,reason))
                await self.db.execute("""INSERT OR REPLACE INTO candidate_cycle_summaries
                    (cycle_id,exchange,strategy,discovered,executed,risk_blocked,filtered,throttled,malformed,unavailable,failed)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (self.cycle_id,self.exchange,self.strategy,len(self._rows),counts["executed"],counts["risk-blocked"],counts["filtered"],counts["throttled"],counts["malformed"],counts["unavailable"],counts["failed"]))
                # Both time columns are indexed. Running these bounded deletes
                # per cycle is cheap when nothing has expired and prevents an
                # interrupted deployment from missing a separate cleanup job.
                await self.db.execute(
                    "DELETE FROM candidate_dispositions "
                    "WHERE observed_at < datetime('now', ?)",
                    (f"-{max(1, int(self.retention_days))} days",))
                await self.db.execute(
                    "DELETE FROM candidate_cycle_summaries "
                    "WHERE completed_at < datetime('now', ?)",
                    (f"-{max(1, int(self.summary_retention_days))} days",))
        except Exception as exc:
            log.error("candidate_dispositions.write_failed", cycle_id=self.cycle_id, error=str(exc))
            return dict(counts)
        log.info("candidate_dispositions.cycle", cycle_id=self.cycle_id, exchange=self.exchange, discovered=len(self._rows), counts=dict(counts))
        return dict(counts)
