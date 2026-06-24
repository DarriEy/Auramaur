"""Arb trade-execution helpers — the per-arb-type executors (Phase-6 split).

Pure structural move out of ArbExecutionMixin: the negrisk / internal /
cross-exchange execution methods live here as ArbTradeExecutionMixin, mixed back
into AuramaurBot via ArbExecutionMixin. They operate on the bot's shared ``self``
(the exit gateway, components, alerts) exactly as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from auramaur.exchange.models import (
    Confidence, Order, OrderSide, Signal, TokenType,
)
from auramaur.strategy.arbitrage_scanner import ArbOpportunity, NegRiskArbOpportunity

if TYPE_CHECKING:
    from auramaur.db.database import Database
    from auramaur.monitoring.alerts import AlertManager
    from auramaur.risk.manager import RiskManager
    from auramaur.strategy.engine import TradingEngine

log = structlog.get_logger()


class ArbTradeExecutionMixin:
    """Per-arb-type execution helpers for ArbExecutionMixin (see module docstring)."""

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

