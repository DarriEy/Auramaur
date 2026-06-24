"""Arbitrage execution — extracted from AuramaurBot as a mixin (Phase 5 split).

Pure structural move: the cross-exchange / internal / neg-risk / conditional arb
execution paths, the arb scanner + depth-research loops, and concentrated-position
rebalancing live here as ArbExecutionMixin, mixed into AuramaurBot. Behavior is
unchanged — the methods still operate on the bot's self (components, settings,
arb-attempt tracking, kill switch). Callers that stay in bot.py
(_task_trading_cycle, _task_correlation_scan) reach these via self.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from auramaur.exchange.models import (
    Confidence, Market, Order, OrderSide, Signal, TokenType,
)
from auramaur.monitoring.display import console
from auramaur.nlp.errors import BudgetExhausted
from auramaur.strategy.arbitrage_scanner import (
    ArbOpportunity, ArbitrageScanner, NegRiskArbOpportunity,
)

if TYPE_CHECKING:
    from auramaur.db.database import Database
    from auramaur.monitoring.alerts import AlertManager
    from auramaur.risk.manager import RiskManager
    from auramaur.strategy.engine import TradingEngine

log = structlog.get_logger()


class ArbExecutionMixin:
    """Arbitrage-execution methods for AuramaurBot (see module docstring)."""

    async def _execute_conditional_arb(
        self,
        buy_signal: "Signal",
        sell_signal: "Signal",
        buy_market: "Market",
        sell_market: "Market",
        opp: dict,
        risk_manager: RiskManager,
        engines: dict[str, TradingEngine],
    ) -> None:
        """Execute a conditional / divergence arb: buy the underpriced leg, sell
        the overpriced leg.

        Both legs pass risk checks independently and are sized to the smaller
        approved size (capped at arbitrage.max_arb_size). Orders are built via
        each exchange's ``prepare_order`` (tick rounding, token resolution,
        SELL→buy-NO conversion) and carry ``dry_run = not is_live`` so paper
        mode is honored. These legs are a relative-value pair, not a guaranteed
        $1 hedge, so a half-fill raises a critical alert for manual review
        rather than attempting an automatic rollback.
        """
        buy_exchange = buy_market.exchange or "polymarket"
        sell_exchange = sell_market.exchange or "polymarket"
        buy_engine = engines.get(buy_exchange)
        sell_engine = engines.get(sell_exchange)
        if buy_engine is None or sell_engine is None:
            log.debug("arbitrage.no_engine", buy=buy_exchange, sell=sell_exchange)
            return

        alerts: AlertManager = self._components.alerts

        # Risk checks on both legs using exchange-local cash for sizing.
        buy_cash = await buy_engine._get_available_cash()
        sell_cash = await sell_engine._get_available_cash()
        buy_decision = await risk_manager.evaluate(
            buy_signal, buy_market, available_cash=buy_cash
        )
        sell_decision = await risk_manager.evaluate(
            sell_signal, sell_market, available_cash=sell_cash
        )
        if not buy_decision.approved or not sell_decision.approved:
            log.debug(
                "arbitrage.risk_rejected",
                type=opp.get("type"),
                buy_approved=buy_decision.approved,
                sell_approved=sell_decision.approved,
                buy_reason=buy_decision.reason,
                sell_reason=sell_decision.reason,
            )
            return

        max_size = self.settings.arbitrage.max_arb_size
        position_size = min(
            buy_decision.position_size, sell_decision.position_size, max_size
        )
        if position_size <= 0:
            return

        # Dedup: 1-hour cooldown per market pair, set before placement so
        # concurrent/rapid scans don't double-fire on the same arb.
        import time as _time
        arb_key = f"cond|{buy_market.id}|{sell_market.id}"
        now_ts = _time.monotonic()
        expiry = self._arb_attempts.get(arb_key)
        if expiry and expiry > now_ts:
            log.debug("arbitrage.dedup_skip", buy=buy_market.id, sell=sell_market.id)
            return
        self._arb_attempts[arb_key] = now_ts + 3600
        if len(self._arb_attempts) > 200:
            self._arb_attempts = {
                k: v for k, v in self._arb_attempts.items() if v > now_ts
            }

        is_live = self.settings.is_live
        buy_client = buy_engine.exchange
        sell_client = sell_engine.exchange
        buy_order = buy_client.prepare_order(buy_signal, buy_market, position_size, is_live)
        sell_order = sell_client.prepare_order(sell_signal, sell_market, position_size, is_live)
        if buy_order is None or sell_order is None:
            log.debug(
                "arbitrage.prepare_failed",
                type=opp.get("type"),
                buy_built=buy_order is not None,
                sell_built=sell_order is not None,
            )
            return

        log.info(
            "arbitrage.executing",
            type=opp.get("type"),
            buy_market=buy_market.id,
            sell_market=sell_market.id,
            size=round(position_size, 2),
            is_paper=not is_live,
        )

        try:
            (buy_result, _), (sell_result, _) = await self._exit_gateway.place_legs(
                [(buy_order, buy_client, buy_order.exchange or "polymarket"),
                 (sell_order, sell_client, sell_order.exchange or "polymarket")],
                strategy_source="conditional_arb", concurrent=True)
            buy_ok = buy_result.status not in ("rejected", "error")
            sell_ok = sell_result.status not in ("rejected", "error")
            mode = "PAPER" if not is_live else "LIVE"
            if buy_ok and sell_ok:
                await alerts.send(
                    f"[{mode}] Arb executed ({opp.get('type')}): "
                    f"buy {buy_market.id[:12]} / sell {sell_market.id[:12]} "
                    f"size {position_size:.2f}",
                    level="warning",
                )
            elif buy_ok != sell_ok:
                log.warning(
                    "arbitrage.half_fill",
                    type=opp.get("type"),
                    buy_status=buy_result.status,
                    sell_status=sell_result.status,
                )
                await alerts.send(
                    f"[{mode}] ARB HALF-FILL ({opp.get('type')}): "
                    f"buy_ok={buy_ok} sell_ok={sell_ok} — "
                    f"buy {buy_market.id[:12]} / sell {sell_market.id[:12]}. "
                    f"Manual review required (no auto-rollback for relative-value legs).",
                    level="critical",
                )
        except Exception as e:
            log.error("arbitrage.execution_error", error=str(e))

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

    async def _execute_negrisk_arb(
        self,
        opp: NegRiskArbOpportunity,
        risk_manager: RiskManager,
        engines: dict[str, TradingEngine],
    ) -> None:
        """Execute a NegRisk buy-all-NO arb: buy NO on every outcome of a
        mutually-exclusive event so the (N-1) guaranteed payout exceeds the cost.

        Every leg passes its own risk check, all legs are sized to the same
        share quantity (so each losing leg pays an identical $1), and each NO
        order carries ``dry_run = not is_live``. The profit guarantee only holds
        with the COMPLETE set, so if any leg fails risk checks the whole package
        is skipped, and a partial fill raises a critical alert for manual review.
        """
        from auramaur.exchange.models import Signal

        engine = engines.get(opp.exchange)
        if engine is None:
            log.debug("arb_scanner.negrisk_no_engine", exchange=opp.exchange)
            return
        legs = opp.markets
        if len(legs) < 2:
            return
        if any(not leg.clob_token_no or leg.outcome_no_price <= 0 for leg in legs):
            log.debug("arb_scanner.negrisk_unpriced_or_no_token", group=opp.neg_risk_market_id)
            return

        alerts: AlertManager = self._components.alerts

        # Dedup: 1-hour cooldown per NegRisk event, set before placement.
        import time as _time
        arb_key = f"negrisk|{opp.exchange}|{opp.neg_risk_market_id}"
        now_ts = _time.monotonic()
        expiry = self._arb_attempts.get(arb_key)
        if expiry and expiry > now_ts:
            log.debug("arb_scanner.negrisk_dedup_skip", group=opp.neg_risk_market_id)
            return

        # Risk-check every leg as a BUY-NO. The package edge is carried on each
        # leg so the generic min-edge gate doesn't reject valid arb legs.
        available_cash = await engine._get_available_cash()
        decisions = []
        for leg in legs:
            no_price = leg.outcome_no_price
            fair = min(0.98, no_price * (1.0 + opp.expected_profit_pct / 100.0))
            signal = Signal(
                market_id=leg.id,
                market_question=leg.question,
                claude_prob=fair,
                claude_confidence=Confidence.HIGH,
                market_prob=no_price,
                edge=opp.expected_profit_pct,
                evidence_summary=(
                    f"NegRisk arb: {opp.n_outcomes} legs, buy-all-NO cost "
                    f"{opp.total_no_cost:.3f} < payout {opp.guaranteed_payout:.3f}"
                ),
                recommended_side=OrderSide.BUY,
            )
            decision = await risk_manager.evaluate(signal, leg, available_cash=available_cash)
            if not decision.approved:
                log.info(
                    "arb_scanner.negrisk_risk_rejected",
                    group=opp.neg_risk_market_id,
                    market_id=leg.id,
                    reason=decision.reason,
                )
                return  # incomplete package is not an arb — skip entirely
            decisions.append((leg, decision))

        # Size all legs to a common SHARE quantity so each losing leg pays an
        # identical $1. Per leg, shares are bounded by both the approved dollar
        # size and the max per-leg arb size.
        max_size = self.settings.arbitrage.max_arb_size
        qty = min(
            min(decision.position_size, max_size) / leg.outcome_no_price
            for leg, decision in decisions
        )
        if qty < 1:
            log.debug("arb_scanner.negrisk_size_too_small", group=opp.neg_risk_market_id, qty=round(qty, 3))
            return

        self._arb_attempts[arb_key] = now_ts + 3600
        if len(self._arb_attempts) > 200:
            self._arb_attempts = {k: v for k, v in self._arb_attempts.items() if v > now_ts}

        is_live = self.settings.is_live
        exchange_client = engine.exchange
        orders = [
            Order(
                market_id=leg.id,
                exchange=opp.exchange,
                token_id=leg.clob_token_no,
                side=OrderSide.BUY,
                token=TokenType.NO,
                size=qty,
                # Round to a valid Polymarket tick (1-cent, 0.01-0.99), matching
                # PolymarketClient.prepare_order so live orders aren't rejected.
                price=max(0.01, min(0.99, round(leg.outcome_no_price, 2))),
                dry_run=not is_live,
            )
            for leg, _ in decisions
        ]

        log.info(
            "arb_scanner.negrisk_executing",
            group=opp.neg_risk_market_id,
            n_outcomes=opp.n_outcomes,
            shares_per_leg=round(qty, 2),
            profit_pct=round(opp.expected_profit_pct, 2),
            is_paper=not is_live,
        )

        try:
            placed = await self._exit_gateway.place_legs(
                [(o, exchange_client, o.exchange or "polymarket") for o in orders],
                strategy_source="negrisk_arb", concurrent=True)
            results = [r for r, _ in placed]
            n_ok = sum(1 for r in results if r.status not in ("rejected", "error"))
            mode = "PAPER" if not is_live else "LIVE"
            if n_ok == len(orders):
                await alerts.send(
                    f"[{mode}] NegRisk arb executed: {opp.question[:45]} | "
                    f"{opp.n_outcomes} legs @ {round(qty, 1)} shares | "
                    f"profit: {opp.expected_profit_pct:.1f}%",
                    level="warning",
                )
            elif n_ok > 0:
                log.warning(
                    "arb_scanner.negrisk_partial_fill",
                    group=opp.neg_risk_market_id,
                    filled_legs=n_ok,
                    total_legs=len(orders),
                )
                await alerts.send(
                    f"[{mode}] NEGRISK PARTIAL FILL: {n_ok}/{len(orders)} legs filled "
                    f"for {opp.question[:40]}. Package no longer guaranteed — "
                    f"manual review required.",
                    level="critical",
                )
        except Exception as e:
            log.error("arb_scanner.negrisk_execution_error", group=opp.neg_risk_market_id, error=str(e))

    async def _execute_internal_arb(
        self,
        opp: ArbOpportunity,
        risk_manager: RiskManager,
        engines: dict[str, TradingEngine],
    ) -> None:
        """Execute an internal arb: buy both YES and NO when their sum < 0.97.

        Both legs must pass risk checks independently before execution.
        """
        from auramaur.exchange.models import Signal

        market = opp.market_a  # Same market for both sides
        exchange_name = opp.exchange_a
        engine = engines.get(exchange_name)
        if engine is None:
            log.debug("arb_scanner.no_engine", exchange=exchange_name)
            return

        # Cross-attempt pairing guard (2026-06-12): re-quoting a market whose
        # previous attempt PARTIALLY filled completes the pair at the new
        # cycle's worse prices — observed live locking 0.82 + 0.21 = $1.03
        # against a $1.00 payout. If we already hold either token of this
        # market, skip: the one-sided leg is the exit machinery's problem,
        # not a fresh "arb".
        db: Database = self._components.db
        is_paper_flag = 0 if self.settings.is_live else 1
        held = await db.fetchone(
            "SELECT 1 FROM portfolio WHERE market_id = ? AND is_paper = ? "
            "AND size > 0",
            (market.id, is_paper_flag),
        )
        if held is not None:
            log.info("arb_scanner.skip_partial_inventory", market_id=market.id)
            return

        alerts: AlertManager = self._components.alerts

        # Build synthetic signals for risk checks.  expected_profit_pct is the
        # net package return; halving it per leg makes valid package arbs fail
        # the generic minimum-edge gate.
        edge_pct = opp.expected_profit_pct
        yes_signal = Signal(
            market_id=market.id,
            market_question=market.question,
            claude_prob=market.outcome_yes_price + opp.spread / 2,
            claude_confidence=Confidence.HIGH,
            market_prob=market.outcome_yes_price,
            edge=edge_pct,
            evidence_summary=f"Internal arb: YES+NO={opp.price_a + opp.price_b:.3f}",
            recommended_side=OrderSide.BUY,
        )

        # Build a synthetic signal for the NO leg (buy NO)
        no_signal = Signal(
            market_id=market.id,
            market_question=market.question,
            claude_prob=market.outcome_no_price + opp.spread / 2,
            claude_confidence=Confidence.HIGH,
            market_prob=market.outcome_no_price,
            edge=edge_pct,
            evidence_summary=f"Internal arb: YES+NO={opp.price_a + opp.price_b:.3f}",
            recommended_side=OrderSide.BUY,
        )

        # Run risk checks on both legs
        available_cash = await engine._get_available_cash()
        yes_decision = await risk_manager.evaluate(
            yes_signal,
            market,
            available_cash=available_cash,
        )
        no_decision = await risk_manager.evaluate(
            no_signal,
            market,
            available_cash=available_cash,
        )

        if not yes_decision.approved or not no_decision.approved:
            log.info(
                "arb_scanner.internal_risk_rejected",
                market_id=market.id,
                expected_profit_pct=round(opp.expected_profit_pct, 2),
                yes_approved=yes_decision.approved,
                no_approved=no_decision.approved,
                yes_reason=yes_decision.reason,
                no_reason=no_decision.reason,
                available_cash=round(available_cash, 2),
            )
            return

        # Execute both legs -- use the smaller position size for balance
        position_size = min(yes_decision.position_size, no_decision.position_size)
        if position_size <= 0:
            return

        exchange_client = engine.exchange

        # YES leg
        yes_order = Order(
            market_id=market.id,
            exchange=exchange_name,
            token_id=market.clob_token_yes or market.id,
            side=OrderSide.BUY,
            size=position_size / market.outcome_yes_price if market.outcome_yes_price > 0 else 0,
            price=market.outcome_yes_price,
            dry_run=not self.settings.is_live,
        )

        # NO leg
        no_order = Order(
            market_id=market.id,
            exchange=exchange_name,
            token_id=market.clob_token_no or market.id,
            side=OrderSide.BUY,
            size=position_size / market.outcome_no_price if market.outcome_no_price > 0 else 0,
            price=market.outcome_no_price,
            dry_run=not self.settings.is_live,
        )

        if yes_order.size < 1 or no_order.size < 1:
            return

        try:
            (yes_result, _), (no_result, _) = await self._exit_gateway.place_legs(
                [(yes_order, exchange_client, yes_order.exchange or "polymarket"),
                 (no_order, exchange_client, no_order.exchange or "polymarket")],
                strategy_source="internal_arb", concurrent=False)

            log.info(
                "arb_scanner.internal_executed",
                market_id=market.id,
                question=market.question[:60],
                yes_status=yes_result.status,
                no_status=no_result.status,
                yes_size=round(yes_order.size, 2),
                no_size=round(no_order.size, 2),
                profit_pct=round(opp.expected_profit_pct, 2),
                is_paper=yes_order.dry_run,
            )

            mode = "PAPER" if yes_order.dry_run else "LIVE"
            await alerts.send(
                f"[{mode}] Internal arb executed: {market.question[:50]} | "
                f"YES@{opp.price_a:.2f} + NO@{opp.price_b:.2f} = "
                f"{opp.price_a + opp.price_b:.3f} | "
                f"profit: {opp.expected_profit_pct:.1f}%",
                level="warning",
            )
        except Exception as e:
            log.error(
                "arb_scanner.internal_execution_error",
                market_id=market.id,
                error=str(e),
            )

    async def _execute_cross_exchange_arb(
        self,
        opp: ArbOpportunity,
        risk_manager: RiskManager,
        engines: dict[str, TradingEngine],
    ) -> None:
        """Execute a cross-exchange arb: buy cheap YES on one exchange, buy
        equivalent NO on the other.

        Both legs go through each exchange's ``prepare_order`` so tick rounding,
        token-id resolution, and minimum-size checks match what the exchange
        accepts. Both legs must pass risk checks independently. Execution is
        concurrent; half-fills trigger a cancel with a critical alert if
        cancel can't confirm.
        """
        from auramaur.exchange.models import Signal

        # Identify which side is cheap (lower YES price)
        if opp.price_a <= opp.price_b:
            cheap_market, expensive_market = opp.market_a, opp.market_b
            cheap_exchange, expensive_exchange = opp.exchange_a, opp.exchange_b
        else:
            cheap_market, expensive_market = opp.market_b, opp.market_a
            cheap_exchange, expensive_exchange = opp.exchange_b, opp.exchange_a

        engine_cheap = engines.get(cheap_exchange)
        engine_expensive = engines.get(expensive_exchange)
        if engine_cheap is None or engine_expensive is None:
            log.debug(
                "arb_scanner.cross_no_engine",
                cheap=cheap_exchange,
                expensive=expensive_exchange,
            )
            return

        alerts: AlertManager = self._components.alerts
        max_size = self.settings.arbitrage.max_arb_size

        # Synthetic signals for risk checks and prepare_order.  The scanner's
        # expected_profit_pct is already the net package return after fees; do
        # not halve it per leg or guaranteed arbs fall below the normal edge
        # floor before sizing.
        edge_pct = opp.expected_profit_pct
        evidence = (
            f"Cross-exchange arb: {cheap_exchange} YES@{cheap_market.outcome_yes_price:.3f} "
            f"vs {expensive_exchange} YES@{expensive_market.outcome_yes_price:.3f}"
        )
        yes_signal = Signal(
            market_id=cheap_market.id,
            market_question=cheap_market.question,
            claude_prob=cheap_market.outcome_yes_price + opp.spread / 2,
            claude_confidence=Confidence.HIGH,
            market_prob=cheap_market.outcome_yes_price,
            edge=edge_pct,
            evidence_summary=evidence,
            recommended_side=OrderSide.BUY,
        )
        # SELL signal → each exchange's prepare_order produces a BUY NO
        # (Polymarket token-swap, Kalshi new-bearish-position branch).
        no_signal = Signal(
            market_id=expensive_market.id,
            market_question=expensive_market.question,
            claude_prob=expensive_market.outcome_no_price + opp.spread / 2,
            claude_confidence=Confidence.HIGH,
            market_prob=expensive_market.outcome_no_price,
            edge=edge_pct,
            evidence_summary=evidence,
            recommended_side=OrderSide.SELL,
        )

        # Risk checks on both legs using exchange-local cash for sizing.
        cheap_cash = await engine_cheap._get_available_cash()
        expensive_cash = await engine_expensive._get_available_cash()
        yes_decision = await risk_manager.evaluate(
            yes_signal,
            cheap_market,
            available_cash=cheap_cash,
        )
        no_decision = await risk_manager.evaluate(
            no_signal,
            expensive_market,
            available_cash=expensive_cash,
        )
        if not yes_decision.approved or not no_decision.approved:
            log.info(
                "arb_scanner.cross_risk_rejected",
                question=opp.question[:60],
                expected_profit_pct=round(opp.expected_profit_pct, 2),
                yes_approved=yes_decision.approved,
                no_approved=no_decision.approved,
                yes_reason=yes_decision.reason,
                no_reason=no_decision.reason,
                cheap_cash=round(cheap_cash, 2),
                expensive_cash=round(expensive_cash, 2),
            )
            return

        position_size = min(
            yes_decision.position_size,
            no_decision.position_size,
            max_size,
        )
        if position_size <= 0:
            return

        cheap_client = engine_cheap.exchange
        expensive_client = engine_expensive.exchange
        is_live = self.settings.is_live

        yes_order = cheap_client.prepare_order(yes_signal, cheap_market, position_size, is_live)
        no_order = expensive_client.prepare_order(no_signal, expensive_market, position_size, is_live)
        if yes_order is None or no_order is None:
            log.debug(
                "arb_scanner.cross_prepare_failed",
                question=opp.question[:60],
                yes_built=yes_order is not None,
                no_built=no_order is not None,
            )
            return

        # Dedup: 1-hour cooldown per (question, exchange-pair). Set before
        # placement so concurrent scans don't double-fire on the same arb.
        import time as _time
        arb_key = f"{opp.question[:80]}|{cheap_exchange}|{expensive_exchange}"
        now_ts = _time.monotonic()
        expiry = self._arb_attempts.get(arb_key)
        if expiry and expiry > now_ts:
            log.debug(
                "arb_scanner.cross_dedup_skip",
                question=opp.question[:60],
                cheap_exchange=cheap_exchange,
                expensive_exchange=expensive_exchange,
            )
            return
        self._arb_attempts[arb_key] = now_ts + 3600
        if len(self._arb_attempts) > 200:
            self._arb_attempts = {
                k: v for k, v in self._arb_attempts.items() if v > now_ts
            }

        try:
            # Place + record both legs through the gateway (the single placement
            # choke point). place_legs gathers them concurrently to minimize the
            # leg-risk window and records each with the cross_exchange_arb
            # invariant (fills + cost_basis + pnl_ledger + the pending
            # trades-mirror the monitor finalizes). Raw results drive the
            # half-fill rollback below.
            (yes_result, _), (no_result, _) = await self._exit_gateway.place_legs(
                [(yes_order, cheap_client, cheap_exchange),
                 (no_order, expensive_client, expensive_exchange)],
                strategy_source="cross_exchange_arb", concurrent=True)

            yes_ok = yes_result.status not in ("rejected", "error")
            no_ok = no_result.status not in ("rejected", "error")
            if yes_ok != no_ok:
                rollback_client = cheap_client if yes_ok else expensive_client
                rollback_result = yes_result if yes_ok else no_result
                rollback_exchange = cheap_exchange if yes_ok else expensive_exchange
                log.warning(
                    "arb_scanner.cross_half_fill_rollback",
                    question=opp.question[:60],
                    yes_status=yes_result.status,
                    no_status=no_result.status,
                    rollback_exchange=rollback_exchange,
                    rollback_order_id=rollback_result.order_id,
                )
                cancelled = False
                # "unknown"/"ERROR" are sentinel IDs the exchange clients
                # return when they couldn't parse a real order_id — cancel
                # would no-op. Skip straight to the critical alert path.
                real_id = rollback_result.order_id not in ("", "unknown", "ERROR", "BLOCKED", "SKIP_DUP")
                if real_id:
                    try:
                        cancelled = bool(await rollback_client.cancel_order(rollback_result.order_id))
                    except Exception as e:
                        log.error(
                            "arb_scanner.cross_rollback_failed",
                            question=opp.question[:60],
                            error=str(e),
                        )
                if not cancelled:
                    # Unhedged directional exposure — operator must intervene.
                    await alerts.send(
                        f"[CRITICAL] Unhedged arb leg on {rollback_exchange}: "
                        f"{opp.question[:60]} order_id={rollback_result.order_id} "
                        f"size={rollback_result.filled_size or 'unknown'} — "
                        "cancel failed or id unknown. Manual flatten required.",
                        level="critical",
                    )

            log.info(
                "arb_scanner.cross_executed",
                question=opp.question[:60],
                cheap_exchange=cheap_exchange,
                expensive_exchange=expensive_exchange,
                yes_status=yes_result.status,
                no_status=no_result.status,
                yes_size=round(yes_order.size, 2),
                no_size=round(no_order.size, 2),
                spread=round(opp.spread, 3),
                profit_pct=round(opp.expected_profit_pct, 2),
                is_paper=yes_order.dry_run,
            )

            if yes_ok and no_ok:
                mode = "PAPER" if yes_order.dry_run else "LIVE"
                await alerts.send(
                    f"[{mode}] Cross-exchange arb executed: {opp.question[:50]} | "
                    f"BUY YES@{yes_order.price:.2f} on {cheap_exchange}, "
                    f"BUY NO@{no_order.price:.2f} on {expensive_exchange} | "
                    f"profit: {opp.expected_profit_pct:.1f}%",
                    level="warning",
                )
        except Exception as e:
            log.error(
                "arb_scanner.cross_execution_error",
                question=opp.question[:60],
                error=str(e),
            )

    async def _rebalance_concentrated_positions(self, engine: TradingEngine) -> None:
        """Trim oversized positions to free up capital for diversification.

        If any single event exceeds MAX_EVENT_PCT of total portfolio exposure,
        sell contracts to bring it down to TARGET_EVENT_PCT.  This prevents
        concentration risk from locking up all capital in one bet.
        """
        from auramaur.exchange.models import Signal, TokenType as TT

        MAX_EVENT_PCT = 0.30   # trigger rebalance above 30%
        TARGET_EVENT_PCT = 0.20  # trim down to 20%

        db = engine.db
        rows = await db.fetchall(
            """SELECT p.market_id, p.token, p.size, p.avg_price, p.current_price,
                      m.outcome_yes_price, m.outcome_no_price, m.spread, m.question
               FROM portfolio p
               JOIN markets m ON p.market_id = m.id
               WHERE p.size > 0 AND m.exchange = 'kalshi'"""
        )

        if not rows:
            return

        # Group by event and compute exposure
        events: dict[str, list] = {}
        total_exposure = 0.0
        for r in rows:
            mid = r["market_id"]
            token = r["token"] or "YES"
            size = r["size"]
            price = r["current_price"] or r["avg_price"] or 0
            exposure = size * price

            # Extract event base from ticker
            if mid.count("-") >= 2:
                event_key = mid.rsplit("-", 1)[0]
            else:
                event_key = mid

            events.setdefault(event_key, []).append({
                "row": r, "exposure": exposure, "mid": mid,
                "token": token, "size": size, "price": price,
            })
            total_exposure += exposure

        if total_exposure <= 0:
            return

        # Check each event for overconcentration
        for event_key, positions in events.items():
            event_exposure = sum(p["exposure"] for p in positions)
            event_pct = event_exposure / total_exposure

            if event_pct <= MAX_EVENT_PCT:
                continue

            target_exposure = total_exposure * TARGET_EVENT_PCT
            excess = event_exposure - target_exposure

            log.info(
                "rebalance.triggered",
                event=event_key,
                event_pct=f"{event_pct:.0%}",
                exposure=round(event_exposure, 2),
                target=round(target_exposure, 2),
                excess=round(excess, 2),
            )
            from datetime import datetime
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            console.print(
                f"[dim]{ts}[/] [bold yellow]REBALANCE[/] {event_key} "
                f"at [red]{event_pct:.0%}[/] of portfolio — trimming to {TARGET_EVENT_PCT:.0%}"
            )

            # Sell from largest position in this event first
            sorted_positions = sorted(positions, key=lambda p: -p["exposure"])
            remaining_excess = excess

            for pos in sorted_positions:
                if remaining_excess <= 0:
                    break

                r = pos["row"]
                token = pos["token"]
                price = pos["price"]
                if price <= 0:
                    continue

                # How many contracts to sell
                contracts_to_sell = min(
                    int(remaining_excess / price),
                    int(pos["size"]) - 1,  # keep at least 1
                )
                if contracts_to_sell < 1:
                    continue

                exit_token = TT.NO if token == "NO" else TT.YES
                exit_signal = Signal(
                    market_id=pos["mid"],
                    market_question=r.get("question", ""),
                    claude_prob=0.5,
                    claude_confidence=Confidence.MEDIUM,
                    market_prob=0.5,
                    edge=5.0,
                    evidence_summary=f"Rebalance: {event_key} at {event_pct:.0%}",
                    recommended_side=OrderSide.SELL,
                    exit_token=exit_token,
                )

                from auramaur.exchange.models import Market
                market = Market(
                    id=pos["mid"],
                    exchange="kalshi",
                    ticker=pos["mid"],
                    question=r.get("question", ""),
                    outcome_yes_price=r["outcome_yes_price"] or 0.5,
                    outcome_no_price=r["outcome_no_price"] or 0.5,
                    spread=r["spread"] or 0,
                )

                order = engine.exchange.prepare_order(
                    exit_signal, market, contracts_to_sell * price, self.settings.is_live,
                )
                if order is None:
                    continue

                order.size = min(order.size, contracts_to_sell)
                if order.size < 1:
                    continue

                (result, _), = await self._exit_gateway.place_legs(
                    [(order, engine.exchange, "kalshi")],
                    strategy_source="rebalance", concurrent=False, show=True)

                if result.status not in ("rejected",):
                    sell_value = order.size * order.price
                    remaining_excess -= sell_value

                    # Block re-entry into this event for 24 hours (DB-persisted)
                    await engine.db.execute(
                        """INSERT OR REPLACE INTO rebalance_blocks
                           (event_key, blocked_until, reason)
                           VALUES (?, datetime('now', '+24 hours'), ?)""",
                        (event_key, f"rebalanced from {event_pct:.0%}"),
                    )
                    await engine.db.commit()

                    console.print(
                        f"         [yellow]Trimmed[/] {pos['mid']} "
                        f"—{int(order.size)} contracts (${sell_value:.2f}) "
                        f"[dim](blocked 24h)[/]"
                    )

                    alerts = self._components.alerts
                    if alerts:
                        await alerts.send(
                            f"Rebalance: trimmed {pos['mid']} "
                            f"by {int(order.size)} contracts — "
                            f"event was {event_pct:.0%} of portfolio",
                            level="warning",
                        )

