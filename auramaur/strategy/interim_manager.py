"""Interim manager — operator-proposed entries, ladder-evaluated.

The disciplined channel for manual balance deployment while the strategy
book earns its graduation records (docs/INTERIM_MANAGER.md is the charter).
Proposals arrive via ``auramaur manager propose``; this pillar validates
each one against the charter's mechanical rules, the risk gateway, and the
graduation ladder, then executes and books it under
``strategy_source = "interim_manager"`` so the ladder judges the manager
like any other strategy.

Two rules make it self-retiring:

  * DELEGATION — a category where any non-exempt strategy holds a live or
    probation cell belongs to that strategy; manager proposals there are
    skipped permanently.
  * SUNSET — once ``sunset_after_live_cells`` non-exempt strategy cells have
    graduated, the manager stops executing and expires its queue.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import structlog

from auramaur.broker.execution_gateway import ExecutionGateway, TradeIntent
from auramaur.exchange.models import Confidence, OrderSide, Signal
from auramaur.risk.graduation import GraduationLadder
from auramaur.strategy.classifier import ensure_category
from auramaur.strategy.protocols import ExecutionMode

log = structlog.get_logger()

_GRADUATED = {"live", "probation"}
_MIN_THESIS_CHARS = 40


class InterimManagerPillar:
    """Executes queued operator proposals through the standard rails."""

    name = "interim_manager"
    execution_mode = ExecutionMode.GATEWAY_SINGLE

    def __init__(self, db, settings, discoveries, exchanges,
                 risk_manager, pnl_tracker, calibration) -> None:
        self._db = db
        self._settings = settings
        self._discoveries = discoveries or {}
        self._exchanges = exchanges or {}
        self._risk = risk_manager
        self._pnl = pnl_tracker
        self._calibration = calibration
        self._ladder = GraduationLadder(db, settings)
        self._gateways: dict[str, ExecutionGateway] = {}

    def _gateway(self, venue: str) -> ExecutionGateway | None:
        if venue not in self._gateways:
            exchange = self._exchanges.get(venue)
            if exchange is None:
                return None
            self._gateways[venue] = ExecutionGateway(
                router=None, exchange=exchange, exchange_name=venue,
                settings=self._settings, db=self._db, pnl_tracker=self._pnl,
            )
        return self._gateways.get(venue)

    # ------------------------------------------------------------------

    async def run_once(self) -> int:
        cfg = self._settings.interim_manager
        if not cfg.enabled:
            return 0

        graduated = await self._graduated_cells()
        if len(graduated) >= cfg.sunset_after_live_cells:
            await self._expire_pending(
                f"sunset: {len(graduated)} graduated strategy cells")
            log.info("interim_manager.sunset", graduated=len(graduated))
            return 0

        owned_categories = {category: strategy
                           for strategy, category in graduated}
        pending = await self._db.fetchall(
            """SELECT * FROM manager_proposals WHERE status = 'pending'
               ORDER BY created_at LIMIT ?""",
            (max(1, cfg.max_entries_per_cycle),))
        executed = 0
        for row in pending or []:
            try:
                if await self._decide_one(dict(row), owned_categories):
                    executed += 1
            except Exception as e:  # noqa: BLE001 — one proposal must not kill the cycle
                log.error("interim_manager.proposal_error",
                          proposal_id=row["id"], error=str(e))
                await self._resolve(row["id"], "skipped", f"error: {e}"[:200])
        return executed

    async def _graduated_cells(self) -> list[tuple[str, str]]:
        """(strategy, category) pairs holding live/probation cells, excluding
        exempt strategies and the manager itself."""
        exempt = set(self._settings.graduation.exempt_strategies) | {self.name}
        cells = []
        for cell in await self._ladder.report():
            status = str(cell.get("status", ""))
            # observe-mode statuses look like "observe:<status>".
            status = status.split(":", 1)[-1] if ":" in status else status
            if status in _GRADUATED and cell.get("strategy") not in exempt:
                cells.append((cell["strategy"], cell.get("category") or ""))
        return cells

    # ------------------------------------------------------------------

    async def _decide_one(self, p: dict, owned: dict[str, str]) -> bool:
        cfg = self._settings.interim_manager
        pid = p["id"]

        created = self._parse_ts(p.get("created_at"))
        if created is not None and datetime.now(timezone.utc) - created > timedelta(
                hours=cfg.proposal_ttl_hours):
            await self._resolve(pid, "expired", "proposal ttl elapsed")
            return False

        thesis = (p.get("thesis") or "").strip()
        if len(thesis) < _MIN_THESIS_CHARS:
            await self._resolve(pid, "skipped",
                                "charter: no nameable mispricing mechanism")
            return False

        venue = p["venue"]
        discovery = self._discoveries.get(venue)
        gateway = self._gateway(venue)
        if discovery is None or gateway is None:
            await self._resolve(pid, "skipped", f"venue {venue} not composed")
            return False

        market = await discovery.get_market(p["market_id"])
        if market is None or not (0.0 < market.outcome_yes_price < 1.0):
            await self._resolve(pid, "skipped", "market missing or no live mid")
            return False

        category = ensure_category(market.question, market.description,
                                   market.category)
        if category in owned:
            await self._resolve(pid, "skipped", f"delegated to {owned[category]}")
            return False

        open_now = await self._open_positions()
        if open_now >= cfg.max_open_positions:
            log.info("interim_manager.position_cap", open=open_now)
            return False  # stays pending; retried next cycle

        side = OrderSide.BUY if str(p["side"]).upper() == "BUY" else OrderSide.SELL
        signal = Signal(
            market_id=market.id,
            market_question=market.question,
            claude_prob=max(0.01, min(0.99, float(p["fair_prob"]))),
            claude_confidence=Confidence.MEDIUM,
            market_prob=market.outcome_yes_price,
            edge=abs(float(p["fair_prob"]) - market.outcome_yes_price) * 100.0,
            evidence_summary=f"[interim_manager proposal {pid}] {thesis}"[:500],
            recommended_side=side,
            strategy_source=self.name,
        )
        decision = await self._risk.evaluate(signal, market)
        if not decision.approved or decision.position_size <= 0:
            await self._resolve(pid, "skipped", f"risk: {decision.reason}"[:200])
            return False

        size = min(decision.position_size, cfg.stake_usd, float(p["stake_usd"]))
        force_paper = cfg.paper or getattr(decision, "force_paper", False)
        res = await gateway.submit(TradeIntent(
            signal=signal, market=market, size_dollars=size,
            force_paper=force_paper))
        if res.status not in ("filled", "paper", "partial", "pending"):
            await self._resolve(pid, "skipped", f"order: {res.reason}"[:200])
            return False

        await self._resolve(pid, "executed",
                            f"{'paper' if res.result.is_paper else 'LIVE'} "
                            f"${size:.2f} @ {res.order.price}")
        log.info("interim_manager.entered", proposal_id=pid,
                 market_id=market.id, side=side.value, size=size,
                 paper=res.result.is_paper)
        try:
            await self._calibration.record_prediction(
                market.id, signal.claude_prob, category)
        except Exception as e:  # noqa: BLE001 — bookkeeping only
            log.debug("interim_manager.calibration_error", error=str(e))
        return True

    # ------------------------------------------------------------------

    async def _open_positions(self) -> int:
        row = await self._db.fetchone(
            """SELECT COUNT(DISTINCT s.market_id) AS n FROM signals s
               JOIN portfolio p ON p.market_id = s.market_id AND p.size > 0
               WHERE s.strategy_source = ?""", (self.name,))
        return int(row["n"] or 0) if row else 0

    async def _expire_pending(self, reason: str) -> None:
        await self._db.execute(
            """UPDATE manager_proposals SET status = 'expired', reason = ?,
               decided_at = datetime('now') WHERE status = 'pending'""",
            (reason,))
        await self._db.commit()

    async def _resolve(self, proposal_id: int, status: str, reason: str) -> None:
        await self._db.execute(
            """UPDATE manager_proposals SET status = ?, reason = ?,
               decided_at = datetime('now') WHERE id = ?""",
            (status, reason, proposal_id))
        await self._db.commit()

    @staticmethod
    def _parse_ts(raw) -> datetime | None:
        if not raw:
            return None
        try:
            ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
