"""Data-driven economic-indicator pricing pillar (#2 Phase B).

Prices Kalshi "Above X" econ ladders from the official FRED history and trades
bins the crowd misprices. Pure math lives in econ_pricing.py; this pillar wires
it to live data + the standard rails:

  FRED history -> indicator distribution -> P(Above X) per bin
                -> edge vs the market mid -> fee-cleared signal -> risk gate
                -> Kalshi order (paper-forced) -> fills/trades/portfolio/calibration

Guardrails for model risk: only the NEAREST release with live quotes is priced
(far bins have no book), the latest FRED print is pulled every cycle (a surprise
relocates the mode), the distribution has a sigma floor, the edge must clear the
Kalshi taker fee, and the pillar is PAPER-FORCED by default — the paper ledger
and calibration prove or kill the edge before any live capital.
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from auramaur.broker.execution_gateway import ExecutionGateway, TradeIntent
from auramaur.exchange.models import Confidence, Market, OrderSide, Signal
from auramaur.strategy.classifier import ensure_category
from auramaur.strategy.econ_pricing import (
    estimate_distribution,
    indicator_series,
    prob_above,
    spec_for_series,
)
from auramaur.strategy.entailment_arb import parse_kalshi_ladder

log = structlog.get_logger()


class EconIndicatorPillar:
    """Periodic data-driven scanner for Kalshi economic-indicator bins."""

    def __init__(self, db, settings, kalshi_discovery, fred_source, exchange,
                 risk_manager, pnl_tracker, calibration) -> None:
        self._db = db
        self._settings = settings
        self._kalshi = kalshi_discovery
        self._fred = fred_source
        self._exchange = exchange
        self._risk = risk_manager
        self._pnl = pnl_tracker
        self._calibration = calibration
        # Shared execution tail; no router (Kalshi taker entry via prepare_order).
        self._gateway = ExecutionGateway(
            router=None, exchange=exchange, exchange_name="kalshi",
            settings=settings, db=db, pnl_tracker=pnl_tracker,
        )

    async def run_once(self) -> int:
        cfg = self._settings.econ_indicator
        if not cfg.enabled or self._kalshi is None or self._fred is None:
            return 0
        series_list = cfg.series or list(_registered_series())
        entered = 0
        for prefix in series_list:
            if entered >= cfg.max_entries_per_cycle:
                break
            try:
                entered += await self._scan_series(prefix, cfg.max_entries_per_cycle - entered)
            except Exception as e:
                log.error("econ_indicator.series_error", series=prefix, error=str(e))
        if entered:
            log.info("econ_indicator.cycle_done", entered=entered)
        return entered

    async def _scan_series(self, prefix: str, budget: int) -> int:
        cfg = self._settings.econ_indicator
        spec = spec_for_series(prefix)
        if spec is None:
            return 0
        obs = await self._fred.get_observations(spec.fred_series, n=cfg.history_n)
        if len(obs) < 4:
            log.debug("econ_indicator.thin_history", series=prefix, n=len(obs))
            return 0
        last_obs_date = obs[-1][0]
        indicator = indicator_series([v for _, v in obs], spec)
        if len(indicator) < 3:
            return 0

        bins = await self._kalshi.get_markets_by_series(prefix)
        period = self._nearest_period(bins)
        if period is None:
            return 0
        release, members = period
        horizon = self._horizon_periods(last_obs_date, release, spec.periods_per_year)
        dist = estimate_distribution(indicator, horizon_periods=horizon)
        if dist is None:
            return 0
        mean, sigma = dist
        log.info("econ_indicator.priced", series=prefix, bins=len(members),
                 mean=round(mean, 3), sigma=round(sigma, 3), horizon=horizon)

        entered = 0
        for threshold, market in sorted(members, reverse=True):
            if entered >= budget:
                break
            model_p = prob_above(threshold, mean, sigma)
            if await self._enter_if_edge(market, model_p):
                entered += 1
        return entered

    # -- selection helpers ---------------------------------------------------

    def _nearest_period(self, bins: list[Market]):
        """Group bins by release period (ticker family) and return the soonest
        future release that has >= 2 live-quoted bins, as (release_dt, members)
        where members = [(threshold, market)]."""
        groups: dict[str, list[tuple[float, Market]]] = {}
        ends: dict[str, datetime] = {}
        now = datetime.now(timezone.utc)
        for m in bins:
            parsed = parse_kalshi_ladder(m.ticker)
            if parsed is None or not (0.0 < m.outcome_yes_price < 1.0):
                continue  # non-ladder or no live mid
            if m.end_date is None:
                continue
            end = m.end_date if m.end_date.tzinfo else m.end_date.replace(tzinfo=timezone.utc)
            if end <= now:
                continue
            (_, family), value = parsed
            groups.setdefault(family, []).append((value, m))
            ends[family] = min(ends.get(family, end), end)
        eligible = [(ends[f], mem) for f, mem in groups.items() if len(mem) >= 2]
        if not eligible:
            return None
        eligible.sort(key=lambda x: x[0])
        return eligible[0]

    @staticmethod
    def _horizon_periods(last_obs: datetime, release: datetime, periods_per_year: int) -> int:
        if last_obs.tzinfo is None:
            last_obs = last_obs.replace(tzinfo=timezone.utc)
        days = max(0.0, (release - last_obs).total_seconds() / 86400.0)
        days_per_period = 365.0 / max(1, periods_per_year)
        return max(1, round(days / days_per_period))

    def _required_edge(self, market: Market) -> float:
        cfg = self._settings.econ_indicator
        from auramaur.strategy.signals import taker_fee_rate
        p = market.outcome_yes_price
        fee = taker_fee_rate("kalshi", market.category or "",
                             self._settings.arbitrage.exchange_fees) * p * (1.0 - p)
        return cfg.min_edge + fee

    # -- entry (paper-forced; same rails as bias_harvest) --------------------

    async def _enter_if_edge(self, market: Market, model_p: float) -> bool:
        cfg = self._settings.econ_indicator
        market_p = market.outcome_yes_price
        edge = model_p - market_p
        if abs(edge) < self._required_edge(market):
            return False
        # Implausible-disagreement guard: a random-walk nowcast that disagrees
        # with the market by a huge margin is naive (the crowd prices forward
        # info the model lacks), not an edge — trust the market and skip.
        if abs(edge) > cfg.max_divergence:
            log.info("econ_indicator.implausible_divergence", market_id=market.id,
                     model_p=round(model_p, 3), market_p=round(market_p, 3))
            return False
        if await self._already_entered_or_held(market.id):
            return False
        side = OrderSide.BUY if edge > 0 else OrderSide.SELL  # SELL -> Kalshi buys NO
        signal = Signal(
            market_id=market.id,
            market_question=market.question,
            claude_prob=max(0.01, min(0.99, model_p)),
            claude_confidence=Confidence.MEDIUM,
            market_prob=market_p,
            edge=abs(edge) * 100.0,
            evidence_summary=(
                f"Econ-indicator model P(above)={model_p:.2f} vs market "
                f"{market_p:.2f} from {spec_for_series(market.ticker.split('-')[0]).fred_series} history."
            ),
            recommended_side=side,
            strategy_source="econ_indicator",
        )
        await self._persist_signal(signal, market)
        decision = await self._risk.evaluate(signal, market)
        if not decision.approved or decision.position_size <= 0:
            log.info("econ_indicator.risk_rejected", market_id=market.id,
                     reason=decision.reason)
            return False
        size = min(decision.position_size, cfg.stake_usd)
        force_paper = cfg.paper or getattr(decision, "force_paper", False)
        res = await self._gateway.submit(TradeIntent(
            signal=signal, market=market, size_dollars=size, force_paper=force_paper))
        if res.status not in ("filled", "paper", "partial", "pending"):
            log.warning("econ_indicator.order_rejected", market_id=market.id,
                        status=res.status, error=res.reason)
            return False
        await self._record_position(signal, market, res.order, res.result)
        log.info("econ_indicator.entered", market_id=market.id, side=side.value,
                 model_p=round(model_p, 3), market_p=round(market_p, 3),
                 paper=res.result.is_paper)
        return True

    async def _already_entered_or_held(self, market_id: str) -> bool:
        row = await self._db.fetchone(
            "SELECT 1 FROM signals WHERE market_id = ? AND strategy_source = 'econ_indicator' LIMIT 1",
            (market_id,),
        )
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
               VALUES (?, 'kalshi', ?, ?, ?, ?, 1, ?, ?, ?, ?, datetime('now'))""",
            (market.id, market.condition_id, market.question,
             (market.description or "")[:500],
             ensure_category(market.question, market.description, market.category),
             market.outcome_yes_price, market.outcome_no_price,
             market.volume, market.liquidity),
        )
        await self._db.execute(
            """INSERT INTO signals (market_id, claude_prob, claude_confidence,
               market_prob, edge, evidence_summary, action, strategy_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'econ_indicator')""",
            (signal.market_id, signal.claude_prob, signal.claude_confidence.value,
             signal.market_prob, signal.edge, signal.evidence_summary,
             signal.recommended_side.value),
        )
        await self._db.commit()

    async def _record_position(self, signal: Signal, market: Market, order, result) -> None:
        # Fill + trades-mirror owned by the ExecutionGateway; this keeps the
        # portfolio row (resolution tracker settles it) + calibration.
        fill_size = result.filled_size if result.filled_size > 0 else order.size
        fill_price = result.filled_price if result.filled_price > 0 else order.price
        is_paper = bool(result.is_paper)
        await self._db.execute(
            """INSERT INTO portfolio (market_id, exchange, side, size, avg_price,
               current_price, unrealized_pnl, category, token, token_id,
               is_paper, updated_at)
               VALUES (?, 'kalshi', ?, ?, ?, ?, 0, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(market_id, is_paper, token) DO UPDATE SET
                   size = excluded.size, avg_price = excluded.avg_price,
                   current_price = excluded.current_price, updated_at = excluded.updated_at""",
            (order.market_id, order.side.value, fill_size, fill_price, fill_price,
             market.category or "", order.token.value, order.token_id,
             1 if is_paper else 0),
        )
        await self._db.commit()
        try:
            await self._calibration.record_prediction(
                order.market_id, signal.claude_prob, market.category or "")
        except Exception as e:
            log.debug("econ_indicator.calibration_error", error=str(e))


def _registered_series():
    from auramaur.strategy.econ_pricing import ECON_SERIES
    return ECON_SERIES.keys()
