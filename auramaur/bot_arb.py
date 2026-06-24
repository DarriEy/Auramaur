"""Arb execution — composition point + conditional/rebalance helpers.

The bulk of the arb logic was extracted (Phase-6) into ArbTradeExecutionMixin
(bot_arb_execute.py) and ArbScanLoopMixin (bot_arb_scan.py). ArbExecutionMixin
inherits both and adds the conditional-arb and rebalance helpers, preserving the
exact public method set AuramaurBot mixes in.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from auramaur.bot_arb_execute import ArbTradeExecutionMixin
from auramaur.bot_arb_scan import ArbScanLoopMixin
from auramaur.exchange.models import (
    Confidence, Market, OrderSide, Signal,
)
from auramaur.monitoring.display import console

if TYPE_CHECKING:
    from auramaur.monitoring.alerts import AlertManager
    from auramaur.risk.manager import RiskManager
    from auramaur.strategy.engine import TradingEngine

log = structlog.get_logger()


class ArbExecutionMixin(ArbTradeExecutionMixin, ArbScanLoopMixin):
    """Composition point + conditional-arb and rebalance helpers."""

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

