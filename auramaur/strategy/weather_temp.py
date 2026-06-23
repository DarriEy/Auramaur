"""Weather-temperature pillar (#weather spike) — price Polymarket daily
city-temperature bins from an Open-Meteo ensemble, trade the mispricings.

Measurement-first: every priced bin is logged (model vs market) so we can score
against realized highs before trusting it. PAPER-FORCED by default; a divergence
cap blocks implausibly large gaps (likely a bin-rounding or station-match
artifact, not edge) until the paper record validates the model.
"""

from __future__ import annotations

from datetime import timezone

import structlog

from auramaur.broker.execution_gateway import ExecutionGateway, TradeIntent
from auramaur.exchange.models import Confidence, Market, OrderSide, Signal
from auramaur.strategy.classifier import ensure_category
from auramaur.strategy.weather_pricing import (
    bin_probability,
    parse_temp_market,
)

log = structlog.get_logger()


class WeatherTempPillar:
    def __init__(self, db, settings, discovery, exchange, risk_manager,
                 pnl_tracker, calibration, weather) -> None:
        self._db = db
        self._settings = settings
        self._discovery = discovery
        self._exchange = exchange
        self._risk = risk_manager
        self._pnl = pnl_tracker
        self._calibration = calibration
        # Shared execution tail; no router (passive entry at the observed price).
        self._gateway = ExecutionGateway(
            router=None, exchange=exchange, exchange_name="polymarket",
            settings=settings, db=db, pnl_tracker=pnl_tracker,
        )
        self._weather = weather

    async def run_once(self) -> int:
        cfg = self._settings.weather_temp
        if not cfg.enabled or self._weather is None:
            return 0
        markets = await self._discovery.get_markets(limit=cfg.scan_limit)
        entered = 0
        for m in markets:
            if entered >= cfg.max_entries_per_cycle:
                break
            if (m.exchange or "polymarket") != "polymarket":
                continue
            if m.end_date is None:
                continue
            end = m.end_date if m.end_date.tzinfo else m.end_date.replace(tzinfo=timezone.utc)
            spec = parse_temp_market(m.question, end.date())
            if spec is None:
                continue
            try:
                if await self._price_and_enter(m, spec):
                    entered += 1
            except Exception as e:
                log.error("weather_temp.error", market_id=m.id, error=str(e))
        if entered:
            log.info("weather_temp.cycle_done", entered=entered)
        return entered

    def _required_edge(self, market: Market) -> float:
        from auramaur.strategy.signals import taker_fee_rate
        cfg = self._settings.weather_temp
        p = market.outcome_yes_price
        fee = taker_fee_rate("polymarket", market.category or "",
                             self._settings.arbitrage.exchange_fees) * p * (1.0 - p)
        return cfg.min_edge + fee

    async def _price_and_enter(self, market: Market, spec) -> bool:
        cfg = self._settings.weather_temp
        members = await self._weather.daily_ensemble(
            spec.lat, spec.lon, spec.target, spec.kind, spec.unit)
        model_p = bin_probability(members, spec.lo, spec.hi)
        if model_p is None:
            return False
        market_p = market.outcome_yes_price
        edge = model_p - market_p
        # Always log model vs market — this is the measurement record.
        log.info("weather_temp.priced", market_id=market.id, city=spec.city,
                 kind=spec.kind, model_p=round(model_p, 3), market_p=round(market_p, 3),
                 members=len(members))
        if abs(edge) < self._required_edge(market):
            return False
        if abs(edge) > cfg.max_divergence:
            log.info("weather_temp.implausible_divergence", market_id=market.id,
                     model_p=round(model_p, 3), market_p=round(market_p, 3))
            return False
        if await self._already_entered_or_held(market.id):
            return False
        side = OrderSide.BUY if edge > 0 else OrderSide.SELL
        signal = Signal(
            market_id=market.id, market_question=market.question,
            claude_prob=max(0.01, min(0.99, model_p)),
            claude_confidence=Confidence.MEDIUM, market_prob=market_p,
            edge=abs(edge) * 100.0,
            evidence_summary=(f"GEFS ensemble P(bin)={model_p:.2f} ({len(members)} members) "
                              f"vs market {market_p:.2f} for {spec.city} {spec.kind} temp."),
            recommended_side=side, strategy_source="weather_temp",
        )
        await self._persist_signal(signal, market)
        decision = await self._risk.evaluate(signal, market)
        if not decision.approved or decision.position_size <= 0:
            log.info("weather_temp.risk_rejected", market_id=market.id, reason=decision.reason)
            return False
        size = min(decision.position_size, cfg.stake_usd)
        force_paper = cfg.paper or getattr(decision, "force_paper", False)
        res = await self._gateway.submit(TradeIntent(
            signal=signal, market=market, size_dollars=size, force_paper=force_paper))
        if res.status not in ("filled", "paper", "partial", "pending"):
            log.warning("weather_temp.order_rejected", market_id=market.id, status=res.status)
            return False
        await self._record_position(signal, market, res.order, res.result)
        log.info("weather_temp.entered", market_id=market.id, side=side.value,
                 model_p=round(model_p, 3), market_p=round(market_p, 3), paper=res.result.is_paper)
        return True

    async def _already_entered_or_held(self, market_id: str) -> bool:
        row = await self._db.fetchone(
            "SELECT 1 FROM signals WHERE market_id = ? AND strategy_source = 'weather_temp' LIMIT 1",
            (market_id,))
        if row is not None:
            return True
        row = await self._db.fetchone(
            "SELECT 1 FROM portfolio WHERE market_id = ? LIMIT 1", (market_id,))
        return row is not None

    async def _persist_signal(self, signal: Signal, market: Market) -> None:
        await self._db.execute(
            """INSERT OR IGNORE INTO markets (id, exchange, condition_id, question,
               description, category, active, outcome_yes_price, outcome_no_price,
               volume, liquidity, last_updated)
               VALUES (?, 'polymarket', ?, ?, ?, ?, 1, ?, ?, ?, ?, datetime('now'))""",
            (market.id, market.condition_id, market.question,
             (market.description or "")[:500],
             ensure_category(market.question, market.description, market.category),
             market.outcome_yes_price, market.outcome_no_price,
             market.volume, market.liquidity))
        await self._db.execute(
            """INSERT INTO signals (market_id, claude_prob, claude_confidence,
               market_prob, edge, evidence_summary, action, strategy_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'weather_temp')""",
            (signal.market_id, signal.claude_prob, signal.claude_confidence.value,
             signal.market_prob, signal.edge, signal.evidence_summary,
             signal.recommended_side.value))
        await self._db.commit()

    async def _record_position(self, signal: Signal, market: Market, order, result) -> None:
        # Fill (-> cost_basis -> pnl_ledger) and the trades-mirror are owned by
        # the ExecutionGateway; this keeps the portfolio row (resolution tracker
        # settles against it; sync is mode-scoped) + the calibration prediction.
        fill_size = result.filled_size if result.filled_size > 0 else order.size
        fill_price = result.filled_price if result.filled_price > 0 else order.price
        is_paper = bool(result.is_paper)
        await self._db.execute(
            """INSERT INTO portfolio (market_id, exchange, side, size, avg_price,
               current_price, unrealized_pnl, category, token, token_id, is_paper, updated_at)
               VALUES (?, 'polymarket', ?, ?, ?, ?, 0, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(market_id, is_paper, token) DO UPDATE SET
                   size = excluded.size, avg_price = excluded.avg_price,
                   current_price = excluded.current_price, updated_at = excluded.updated_at""",
            (order.market_id, order.side.value, fill_size, fill_price, fill_price,
             market.category or "", order.token.value, order.token_id,
             1 if is_paper else 0))
        await self._db.commit()
        try:
            await self._calibration.record_prediction(
                order.market_id, signal.claude_prob, market.category or "")
        except Exception as e:
            log.debug("weather_temp.calibration_error", error=str(e))
