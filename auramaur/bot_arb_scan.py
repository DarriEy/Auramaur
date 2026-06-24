"""Arb scan/drive loop + depth-research task (Phase-6 split).

Pure structural move out of ArbExecutionMixin: the scan loop (_task_arb_scanner,
which dispatches to the per-arb-type executors) and the depth-research task live
here as ArbScanLoopMixin, mixed back into AuramaurBot via ArbExecutionMixin.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from auramaur.nlp.errors import BudgetExhausted
from auramaur.strategy.arbitrage_scanner import ArbitrageScanner

if TYPE_CHECKING:
    from auramaur.db.database import Database
    from auramaur.monitoring.alerts import AlertManager
    from auramaur.risk.manager import RiskManager
    from auramaur.strategy.engine import TradingEngine

log = structlog.get_logger()


class ArbScanLoopMixin:
    """Scan/drive loop + depth research for ArbExecutionMixin (see module docstring)."""

    async def _task_depth_research(self) -> None:
        """Run deep research on the most promising markets.

        Complements the strategic loop (breadth) with deep-dive analysis
        on markets where the strategic batch found potential edge but
        confidence was low or the market is high-value.
        """
        from datetime import datetime

        from auramaur.strategy.agent_analyzer import AgentAnalyzer

        depth_agent: AgentAnalyzer = self._components.depth_agent
        db: Database = self._components.db
        engines: dict[str, TradingEngine] = self._components.engines
        engine = engines.get("polymarket")
        if engine is None:
            return

        # Dedup: deep_research is the heaviest LLM call (multi-turn web research).
        # A persistent high-edge/low-confidence market would otherwise be
        # re-researched every cycle for hours; remember what we've researched and
        # skip repeats within the window.
        researched: dict[str, datetime] = {}
        research_ttl = 86400.0  # 24h

        while self._running:
            if await self._check_kill_switch():
                return
            try:
                # Find markets with high edge but low confidence from recent signals.
                # CRITICAL: filter to the same exchange as the engine (polymarket).
                # Without this filter, Kalshi signals get routed through the
                # Polymarket CLOB client and fail with "No CLOB token_id".
                rows = await db.fetchall(
                    """SELECT s.market_id, m.question, m.description, m.category,
                              m.outcome_yes_price, m.outcome_no_price, m.end_date,
                              m.volume, m.liquidity,
                              s.edge, s.claude_confidence
                       FROM signals s
                       JOIN markets m ON s.market_id = m.id
                       WHERE s.timestamp > datetime('now', '-6 hours')
                         AND ABS(s.edge) >= 12
                         AND s.claude_confidence IN ('LOW', 'MEDIUM_LOW', 'MEDIUM')
                         AND m.active = 1
                         AND m.exchange = 'polymarket'
                         AND COALESCE(m.liquidity, 0) >= 1000
                       ORDER BY ABS(s.edge) DESC
                       LIMIT 3"""
                )

                now_ts = datetime.now(timezone.utc)
                researched = {k: v for k, v in researched.items()
                              if (now_ts - v).total_seconds() < research_ttl}
                for row in rows:
                    if await self._check_kill_switch():
                        return

                    # Skip markets researched within the dedup window.
                    last = researched.get(row["market_id"])
                    if last is not None and (now_ts - last).total_seconds() < research_ttl:
                        continue
                    researched[row["market_id"]] = now_ts

                    # Build market from DB + enrich from Gamma
                    from auramaur.exchange.models import Market
                    market = Market(
                        id=row["market_id"],
                        question=row["question"] or "",
                        description=row["description"] or "",
                        category=row["category"] or "",
                        outcome_yes_price=row["outcome_yes_price"] or 0.5,
                        outcome_no_price=row["outcome_no_price"] or 0.5,
                        volume=row["volume"] or 0,
                        liquidity=row["liquidity"] or 0,
                    )
                    try:
                        end_str = row["end_date"]
                        if end_str:
                            from datetime import datetime
                            market.end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                    except Exception:
                        pass
                    # Enrich with Gamma data for CLOB tokens
                    try:
                        discovery = self._components.discovery
                        full_market = await discovery.get_market(market.id)
                        if full_market:
                            market.clob_token_yes = full_market.clob_token_yes
                            market.clob_token_no = full_market.clob_token_no
                            market.condition_id = full_market.condition_id
                            if full_market.description and len(full_market.description) > len(market.description):
                                market.description = full_market.description
                    except Exception:
                        pass

                    log.info(
                        "depth.researching",
                        market_id=market.id,
                        question=market.question[:60],
                        initial_edge=row["edge"],
                    )

                    candidate = await depth_agent.deep_research(market)
                    if candidate:
                        # Run through risk checks and execution
                        results = await engine._execute_candidates([candidate])
                        trades = [r for r in results if r.get("order")]
                        if trades:
                            log.info(
                                "depth.trade_placed",
                                market_id=market.id,
                                edge=round(candidate.signal.edge, 1),
                            )
                            alerts: AlertManager = self._components.alerts
                            await alerts.send(
                                f"Depth research trade: {market.question[:40]} "
                                f"edge={candidate.signal.edge:.1f}%",
                                level="info",
                            )

            except BudgetExhausted as e:
                # Expected daily-budget throttle, not a failure — skip quietly.
                log.debug("depth.budget_skipped", error=str(e))
            except Exception as e:
                log.error("depth.error", error=str(e))
            await asyncio.sleep(1800)  # Every 30 minutes

    async def _task_arb_scanner(self) -> None:
        """Periodically scan all exchanges for arbitrage opportunities."""
        scanner: ArbitrageScanner = self._components.arb_scanner
        alerts: AlertManager = self._components.alerts
        risk_manager: RiskManager = self._components.risk_manager
        engines: dict[str, TradingEngine] = self._components.engines

        while self._running:
            if await self._check_kill_switch():
                return
            try:
                opportunities = await scanner.scan()

                for opp in opportunities:
                    # Log every opportunity
                    log.info(
                        "arb_scanner.opportunity",
                        arb_type=opp.arb_type,
                        question=opp.question[:80],
                        exchange_a=opp.exchange_a,
                        exchange_b=opp.exchange_b,
                        price_a=round(opp.price_a, 3),
                        price_b=round(opp.price_b, 3),
                        spread=round(opp.spread, 3),
                        profit_pct=round(opp.expected_profit_pct, 2),
                    )

                    # Alert on significant opportunities (> 5% expected profit)
                    if opp.expected_profit_pct > 5.0:
                        await alerts.send(
                            f"Arb opportunity ({opp.arb_type}): "
                            f"{opp.question[:60]} | "
                            f"{opp.exchange_a} {opp.price_a:.2f} vs "
                            f"{opp.exchange_b} {opp.price_b:.2f} | "
                            f"profit: {opp.expected_profit_pct:.1f}%",
                            level="warning",
                        )

                    # Auto-execute internal arbs (YES+NO < 0.97) if risk checks pass
                    if opp.arb_type == "internal":
                        await self._execute_internal_arb(opp, risk_manager, engines)
                    elif opp.arb_type == "cross_exchange" and self.settings.arbitrage.cross_exchange_auto_execute:
                        await self._execute_cross_exchange_arb(opp, risk_manager, engines)

                # NegRisk multi-outcome arbs: buy NO on every leg of a
                # mutually-exclusive event when the legs sum below the (N-1)
                # guaranteed payout. (Separate fetch; different opportunity shape.)
                negrisk_opps = await scanner.scan_negrisk()
                for nopp in negrisk_opps:
                    log.info(
                        "arb_scanner.opportunity",
                        arb_type=nopp.arb_type,
                        question=nopp.question[:80],
                        exchange=nopp.exchange,
                        n_outcomes=nopp.n_outcomes,
                        total_no_cost=round(nopp.total_no_cost, 3),
                        profit_pct=round(nopp.expected_profit_pct, 2),
                    )
                    if nopp.expected_profit_pct > 5.0:
                        await alerts.send(
                            f"NegRisk arb: {nopp.question[:50]} | "
                            f"{nopp.n_outcomes} legs, buy-all-NO cost "
                            f"{nopp.total_no_cost:.2f} < payout "
                            f"{nopp.guaranteed_payout:.2f} | "
                            f"profit: {nopp.expected_profit_pct:.1f}%",
                            level="warning",
                        )
                    if self.settings.arbitrage.negrisk_auto_execute:
                        await self._execute_negrisk_arb(nopp, risk_manager, engines)
                    else:
                        log.info(
                            "arb_scanner.negrisk_auto_disabled",
                            group=nopp.neg_risk_market_id,
                            n_outcomes=nopp.n_outcomes,
                            profit_pct=round(nopp.expected_profit_pct, 2),
                        )

            except Exception as e:
                log.error("arb_scanner.task_error", error=str(e))
            interval = self.settings.hybrid.arb_scan_seconds if self._hybrid else 300
            await asyncio.sleep(interval)

