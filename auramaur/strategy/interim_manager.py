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

from datetime import datetime, timezone

import structlog

from auramaur.broker.execution_gateway import ExecutionGateway, TradeIntent
from auramaur.experiments.strategies.interim_manager import (
    InterimManagerInputs,
    InterimManagerRejection,
    InterimManagerRules,
    assess_interim_manager,
)
from auramaur.exchange.models import Confidence, OrderSide, Signal
from auramaur.risk.graduation import GraduationLadder
from auramaur.strategy.classifier import ensure_category
from auramaur.strategy.protocols import ExecutionMode

log = structlog.get_logger()

_GRADUATED = {"live", "probation"}


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
        await self._auto_propose(owned_categories)
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
        from auramaur.strategy.signals import taker_fee_rate

        cfg = self._settings.interim_manager
        pid = p["id"]

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
        open_now = await self._open_positions()
        assessment = assess_interim_manager(
            InterimManagerInputs(
                proposal_id=pid, market_id=market.id,
                thesis=p.get("thesis") or "", side=str(p["side"]),
                fair_probability=float(p["fair_prob"]),
                requested_stake_usd=float(p["stake_usd"]),
                market_yes_price=float(market.outcome_yes_price),
                market_liquidity=float(market.liquidity or 0), category=category,
                created_at=self._parse_ts(p.get("created_at")),
                sunset_at=self._parse_ts(p.get("sunset_at")),
                as_of=datetime.now(timezone.utc), delegated_to=owned.get(category),
                open_positions=open_now,
                open_positions_in_category=await self._open_positions_in_category(category),
                fee_rate=taker_fee_rate(market.exchange or "kalshi", category),
                confidence_lo=float(p["confidence_lo"]) if p.get("confidence_lo") is not None else None,
                confidence_hi=float(p["confidence_hi"]) if p.get("confidence_hi") is not None else None,
                max_entry_price=float(p["max_entry_price"]) if p.get("max_entry_price") is not None else None,
            ),
            InterimManagerRules(
                proposal_ttl_hours=cfg.proposal_ttl_hours,
                max_open_positions=cfg.max_open_positions,
                min_robust_edge=cfg.min_robust_edge,
                default_uncertainty_buffer=cfg.default_uncertainty_buffer,
                slippage_buffer=cfg.slippage_buffer,
                liquidity_penalty=cfg.liquidity_penalty,
                thin_liquidity_usd=cfg.thin_liquidity_usd,
                correlation_penalty_per_position=cfg.correlation_penalty_per_position,
                stake_usd=cfg.stake_usd, paper=cfg.paper,
            ),
        )
        if assessment.proposal is None:
            if assessment.robust_edge is not None:
                # 2026-08-05 (#353 phase 5 survey): classification B — a
                # single UPDATE, already atomic and durable under the
                # autocommit connection; the trailing commit() is dead
                # (no-op since eed51b8) but stays untouched: this live-money
                # file ships zero behavior change (classification-only).
                await self._db.execute(
                    "UPDATE manager_proposals SET robust_edge = ?, decision_price = ? "
                    "WHERE id = ?",
                    (round(assessment.robust_edge, 4), assessment.decision_price, pid),
                )
                await self._db.commit()
            if assessment.rejection == InterimManagerRejection.POSITION_CAP:
                log.info("interim_manager.position_cap", open=open_now)
                return False
            expired = assessment.rejection in {
                InterimManagerRejection.EXPIRED,
                InterimManagerRejection.SUNSET_REACHED,
            }
            await self._resolve(pid, "expired" if expired else "skipped", assessment.reason)
            return False
        proposal = assessment.proposal
        # 2026-08-05 (#353 phase 5 survey): classification B — single
        # UPDATE + dead commit (no-op since eed51b8); no span needed.
        await self._db.execute(
            "UPDATE manager_proposals SET robust_edge = ?, decision_price = ? "
            "WHERE id = ?", (round(proposal.robust_edge, 4), proposal.entry_price, pid))
        await self._db.commit()
        side = OrderSide.BUY if proposal.buy_yes else OrderSide.SELL
        signal = Signal(
            market_id=market.id,
            market_question=market.question,
            claude_prob=proposal.fair_probability,
            claude_confidence=Confidence.MEDIUM,
            market_prob=market.outcome_yes_price,
            edge=abs(float(p["fair_prob"]) - market.outcome_yes_price) * 100.0,
            evidence_summary=f"[interim_manager proposal {pid}] {proposal.thesis}"[:500],
            recommended_side=side,
            strategy_source=self.name,
        )
        decision = await self._risk.evaluate(signal, market)
        if not decision.approved or decision.position_size <= 0:
            await self._resolve(pid, "skipped", f"risk: {decision.reason}"[:200])
            return False

        size = min(decision.position_size, proposal.requested_stake_usd)
        force_paper = proposal.force_paper or getattr(decision, "force_paper", False)
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

    # Counted through `signals`, but this pillar NEVER writes a signals row —
    # every other pillar does, so the join silently returned 0 forever. That
    # made max_open_positions non-binding and zeroed the correlation penalty
    # in _robust_edge (penalty x 0 positions). Count from trades, which the
    # gateway writes for every placement this pillar makes.
    _COUNT_SQL = """SELECT COUNT(DISTINCT p.market_id) AS n FROM portfolio p
                    WHERE p.size > 0
                      AND EXISTS (SELECT 1 FROM cost_basis cb
                                  WHERE cb.market_id = p.market_id
                                    AND UPPER(cb.token) = UPPER(p.token)
                                    AND cb.is_paper = p.is_paper AND cb.size > 0)
                      AND EXISTS (SELECT 1 FROM trades t
                                  WHERE t.market_id = p.market_id
                                    AND t.strategy_source = ?
                                    AND t.is_paper = p.is_paper)"""

    async def _open_positions_in_category(self, category: str) -> int:
        row = await self._db.fetchone(
            self._COUNT_SQL + " AND p.category = ?", (self.name, category))
        return int(row["n"] or 0) if row else 0

    async def _open_positions(self) -> int:
        row = await self._db.fetchone(self._COUNT_SQL, (self.name,))
        return int(row["n"] or 0) if row else 0

    async def _expire_pending(self, reason: str) -> None:
        # 2026-08-05 (#353 phase 5 survey): classification B — one UPDATE
        # covering all pending rows in a single statement, atomic under
        # autocommit; the trailing commit() is dead (no-op since eed51b8).
        await self._db.execute(
            """UPDATE manager_proposals SET status = 'expired', reason = ?,
               decided_at = datetime('now') WHERE status = 'pending'""",
            (reason,))
        await self._db.commit()

    async def _resolve(self, proposal_id: int, status: str, reason: str) -> None:
        # 2026-08-05 (#353 phase 5 survey): classification B — single
        # UPDATE + dead commit (no-op since eed51b8); no span needed.
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

    async def _auto_propose(self, owned: dict[str, str]) -> None:
        """Author queue entries from strong calibrated signals (charter
        amendment 2026-07-20). Confidence is operationalized: HIGH-confidence
        source signal, calibrated edge above the floor, category not owned by
        a graduate, market not already held or proposed. The proposal then
        faces the identical gauntlet as an operator proposal — the CI width
        is a configured, deliberately honest haircut, and the robust-edge
        gate remains the arbiter. Capped per day; scored separately via the
        proposer tag.
        """
        cfg = self._settings.interim_manager
        if not cfg.auto_propose or self._calibration is None:
            return
        # 2026-08-05 (#353 phase 5 survey): classification B — a loop of
        # INDEPENDENT proposal INSERTs; partial completion just means fewer
        # auto-proposals this cycle, so there is no batch to protect. The
        # trailing commit() is dead (no-op since eed51b8).
        try:
            row = await self._db.fetchone(
                """SELECT COUNT(*) AS n FROM manager_proposals
                   WHERE proposer = 'auto' AND created_at >= date('now')""")
            if int(row["n"] or 0) >= cfg.auto_daily_cap:
                return
            budget = cfg.auto_daily_cap - int(row["n"] or 0)
            signals = await self._db.fetchall(
                """SELECT s.market_id, s.exchange, s.claude_prob, s.market_prob,
                          s.evidence_summary, s.strategy_source,
                          COALESCE(m.category, '') AS category
                   FROM signals s LEFT JOIN markets m ON m.id = s.market_id
                   WHERE s.claude_confidence = ?
                     AND s.timestamp >= datetime('now', '-6 hours')
                     AND s.market_id NOT IN
                         (SELECT market_id FROM manager_proposals
                           WHERE created_at >= datetime('now', '-48 hours'))
                     AND s.market_id NOT IN
                         (SELECT market_id FROM portfolio WHERE size > 0)
                   ORDER BY s.timestamp DESC LIMIT 25""",
                (cfg.auto_min_confidence,))
            for sig in signals or []:
                if budget <= 0:
                    break
                category = sig["category"] or ""
                if category in owned:
                    continue
                exempt = set(self._settings.graduation.exempt_strategies)
                if (sig["strategy_source"] or "") in exempt:
                    continue  # structural strategies place their own orders
                calibrated = await self._calibration.adjust(
                    float(sig["claude_prob"]), category)
                market_p = float(sig["market_prob"] or 0)
                if not (0.0 < market_p < 1.0):
                    continue
                edge = calibrated - market_p
                if abs(edge) < cfg.auto_min_calibrated_edge:
                    continue
                half = cfg.auto_ci_width / 2.0
                lo = max(0.01, calibrated - half)
                hi = min(0.99, calibrated + half)
                side = "BUY" if edge > 0 else "SELL"
                thesis = (
                    f"[auto] Source {sig['strategy_source'] or 'llm'} HIGH-confidence "
                    f"signal: raw {float(sig['claude_prob']):.2f} -> calibrated "
                    f"{calibrated:.2f} vs market {market_p:.2f} "
                    f"(edge {edge:+.2f} after Platt scaling on category "
                    f"'{category or 'global'}'). Mechanism per source evidence: "
                    f"{(sig['evidence_summary'] or '')[:200]}")
                await self._db.execute(
                    """INSERT INTO manager_proposals
                       (venue, market_id, side, fair_prob, stake_usd, thesis,
                        thesis_class, confidence_lo, confidence_hi,
                        sunset_at, proposer)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
                               datetime('now', '+24 hours'), 'auto')""",
                    (sig["exchange"] or "polymarket", sig["market_id"], side,
                     round(calibrated, 4), cfg.auto_stake_usd, thesis,
                     "forecast_divergence" if category in ("weather", "economics")
                     else "unclassified", round(lo, 4), round(hi, 4)))
                budget -= 1
                log.info("interim_manager.auto_proposed",
                         market_id=sig["market_id"], side=side,
                         calibrated=round(calibrated, 3),
                         market=round(market_p, 3), category=category)
            await self._db.commit()
        except Exception as e:  # noqa: BLE001 — authorship must not break the cycle
            log.warning("interim_manager.auto_propose_error", error=str(e)[:150])
