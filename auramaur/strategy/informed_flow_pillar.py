"""Informed-flow follower over Kalshi — mimic the informed side of abnormal flow.

Consumes the data layer in :mod:`auramaur.strategy.informed_flow` (the Kalshi
trade tape + pure abnormal-trade-size detector). When a market shows abnormally
large, one-sided order flow — the ATS proxy for informed (non-liquidity) trading
that predicts resolution (Delvecchio CMC thesis #4166; Bartlett & O'Hara,
"Adverse Selection in Prediction Markets") — we FOLLOW that side with a small
uplift. Forecast-free: we claim no probability of our own, only that informed
flow favors the detected side.

Why it's hedged: the evidence is a single in-sample thesis (medium confidence),
and adverse selection cuts both ways — the edge exists only if we are reliably on
the INFORMED side, not the picked-off one. So this is PAPER-FORCED with its own
graduation cell, the uplift stays under the divergence-filter floor (it's not a
high-confidence call), and the scan is bounded (a trade tape is pulled per
eligible market, so cheap filters run FIRST). No-ops cleanly without a Kalshi
venue. Rides the full RiskManager gate and the same rails as the other pillars.
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from auramaur.broker.execution_gateway import ExecutionGateway, TradeIntent
from auramaur.exchange.models import Confidence, Market, OrderSide, Signal
from auramaur.strategy.classifier import blocked_category_hit, ensure_category
from auramaur.strategy.informed_flow import KalshiTradeTape
from auramaur.strategy.protocols import ExecutionMode

log = structlog.get_logger()


class InformedFlowPillar:
    """Periodic Kalshi scanner that follows abnormal informed order flow."""

    # Uniform Strategy contract (see strategy/protocols.py).
    name = "informed_flow"
    execution_mode = ExecutionMode.GATEWAY_SINGLE

    def __init__(self, db, settings, kalshi_discovery, exchange, risk_manager,
                 pnl_tracker, calibration) -> None:
        self._db = db
        self._settings = settings
        self._kalshi = kalshi_discovery
        self._exchange = exchange
        self._risk = risk_manager
        self._pnl = pnl_tracker
        self._calibration = calibration
        self._tape = KalshiTradeTape(exchange)
        self._gateway = ExecutionGateway(
            router=None, exchange=exchange, exchange_name="kalshi",
            settings=settings, db=db, pnl_tracker=pnl_tracker,
        )

    async def run_once(self) -> int:
        cfg = self._settings.informed_flow
        if not cfg.enabled or self._kalshi is None or self._exchange is None:
            return 0
        open_count = await self._open_position_count()
        if open_count >= cfg.max_open:
            log.info("informed_flow.cycle", scanned=0, entered=0,
                     book_full=True, open=open_count, cap=cfg.max_open)
            return 0
        # Scan the NEAR-DATED slice by close-time window, not get_markets() — the
        # /events scan get_markets() uses surfaces only ~18-yr novelty markets
        # (median horizon ~18 years), so every market failed the horizon gate and
        # the tape was never pulled. /markets honors the close window and returns
        # the actually-tradeable near-dated markets (econ ladders, MVE, events).
        now_ts = int(datetime.now(timezone.utc).timestamp())
        min_close_ts = now_ts + int(cfg.min_hours_to_resolution * 3600)
        max_close_ts = now_ts + int(cfg.max_days_to_resolution * 86400)
        if hasattr(self._kalshi, "get_markets_by_close_window"):
            markets = await self._kalshi.get_markets_by_close_window(
                min_close_ts, max_close_ts, limit=cfg.scan_limit)
        else:  # fallback (e.g. a discovery without the windowed fetch / tests)
            markets = await self._kalshi.get_markets(limit=cfg.scan_limit)
        entered = 0
        for market in markets:
            if entered >= cfg.max_entries_per_cycle:
                break
            if open_count + entered >= cfg.max_open:
                break
            try:
                if await self._try_enter(market):
                    entered += 1
            except Exception as e:
                log.error("informed_flow.entry_error", market_id=market.id, error=str(e))
        log.info("informed_flow.cycle", scanned=len(markets), entered=entered)
        return entered

    # ------------------------------------------------------------------
    # Eligibility (CHEAP filters first — a trade tape is fetched only after)
    # ------------------------------------------------------------------

    def _eligible(self, market: Market) -> bool:
        cfg = self._settings.informed_flow
        if not market.active:
            return False
        if (market.exchange or "kalshi") != "kalshi":
            return False
        if not market.ticker:
            return False  # need the ticker to pull the tape
        p_yes = market.outcome_yes_price
        if not (cfg.band_lo <= p_yes <= cfg.band_hi):
            return False  # extremes: ATS is noise / already near-resolved
        if market.liquidity < cfg.min_liquidity:
            return False
        hit = blocked_category_hit(set(self._settings.risk.blocked_categories),
                                   market.question, market.description, market.category)
        if hit:
            log.debug("informed_flow.skip_category", market_id=market.id, category=hit)
            return False
        if market.end_date is None:
            return False
        end = market.end_date
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        hours_left = (end - datetime.now(timezone.utc)).total_seconds() / 3600.0
        if hours_left < cfg.min_hours_to_resolution:
            return False
        if hours_left > cfg.max_days_to_resolution * 24.0:
            return False
        return True

    async def _try_enter(self, market: Market) -> bool:
        cfg = self._settings.informed_flow
        if not self._eligible(market):
            return False
        if await self._already_entered_or_held(market.id):
            return False

        # Only NOW pull the tape (bounded API cost) and detect informed flow.
        flow = await self._tape.informed_flow(
            market.ticker, limit=cfg.trades_limit,
            min_sample=cfg.min_abnormal_sample, size_mult=cfg.size_mult,
            min_dominance=cfg.min_dominance,
        )
        if not flow.has_signal or flow.informed_side is None:
            return False

        fav_is_yes = flow.informed_side == "yes"
        p_yes = market.outcome_yes_price
        # Forecast-free: follow the informed side with a small uplift (stays under
        # the 0.05 divergence-filter floor so a MEDIUM-confidence follow isn't
        # blocked, as in bias_harvest).
        claude_prob = (min(0.99, p_yes + cfg.uplift) if fav_is_yes
                       else max(0.01, p_yes - cfg.uplift))
        signal = Signal(
            market_id=market.id,
            market_question=market.question,
            claude_prob=claude_prob,
            claude_confidence=Confidence.MEDIUM,
            market_prob=p_yes,
            edge=cfg.uplift * 100.0,
            evidence_summary=(
                f"Informed-flow follow: {flow.abnormal_count} abnormal trades "
                f"({flow.signal_volume:.0f} contracts) on the {flow.informed_side.upper()} "
                f"side vs a {flow.baseline_size:.0f}-size baseline (n={flow.sample})."
            ),
            recommended_side=OrderSide.BUY if fav_is_yes else OrderSide.SELL,
            strategy_source="informed_flow",
            mispricing_reason=(
                "structural: abnormal-trade-size (informed) order flow favors the "
                "followed side (ATS proxy)"
            ),
        )
        await self._persist_signal(signal, market)

        decision = await self._risk.evaluate(signal, market)
        if not decision.approved or decision.position_size <= 0:
            log.info("informed_flow.risk_rejected", market_id=market.id,
                     reason=decision.reason)
            return False
        size = min(decision.position_size, cfg.stake_usd)

        force_paper = cfg.paper or getattr(decision, "force_paper", False)
        res = await self._gateway.submit(TradeIntent(
            signal=signal, market=market, size_dollars=size, force_paper=force_paper))
        if res.status not in ("filled", "paper", "partial", "pending"):
            log.warning("informed_flow.order_rejected", market_id=market.id,
                        status=res.status, error=res.reason)
            return False

        await self._record_position(signal, market, res.order, res.result)
        log.info("informed_flow.entered", market_id=market.id,
                 side=signal.recommended_side.value, informed=flow.informed_side,
                 abnormal=flow.abnormal_count, paper=res.result.is_paper)
        return True

    # ------------------------------------------------------------------
    # Bookkeeping — same rails as the other pillars
    # ------------------------------------------------------------------

    async def _already_entered_or_held(self, market_id: str) -> bool:
        row = await self._db.fetchone(
            "SELECT 1 FROM signals WHERE market_id = ? AND strategy_source = 'informed_flow' LIMIT 1",
            (market_id,),
        )
        if row is not None:
            return True
        row = await self._db.fetchone(
            "SELECT 1 FROM portfolio WHERE market_id = ? LIMIT 1", (market_id,),
        )
        return row is not None

    async def _open_position_count(self) -> int:
        row = await self._db.fetchone(
            """SELECT COUNT(*) AS n FROM portfolio p
               WHERE EXISTS (SELECT 1 FROM signals s
                             WHERE s.market_id = p.market_id
                               AND s.strategy_source = 'informed_flow')""",
        )
        return int(row["n"]) if row else 0

    async def _persist_signal(self, signal: Signal, market: Market) -> None:
        await self._db.execute(
            """INSERT OR IGNORE INTO markets (id, exchange, condition_id, question,
               description, category, active, outcome_yes_price, outcome_no_price,
               volume, liquidity, last_updated)
               VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, datetime('now'))""",
            (market.id, market.exchange or "kalshi", market.condition_id,
             market.question, (market.description or "")[:500],
             ensure_category(market.question, market.description, market.category),
             market.outcome_yes_price, market.outcome_no_price,
             market.volume, market.liquidity),
        )
        await self._db.execute(
            """INSERT INTO signals (market_id, claude_prob, claude_confidence,
               market_prob, edge, evidence_summary, action, strategy_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'informed_flow')""",
            (signal.market_id, signal.claude_prob, signal.claude_confidence.value,
             signal.market_prob, signal.edge, signal.evidence_summary,
             signal.recommended_side.value),
        )
        await self._db.commit()

    async def _record_position(self, signal: Signal, market: Market,
                               order, result) -> None:
        fill_size = result.filled_size if result.filled_size > 0 else order.size
        fill_price = result.filled_price if result.filled_price > 0 else order.price
        is_paper = bool(result.is_paper)
        await self._db.execute(
            """INSERT INTO portfolio (market_id, exchange, side, size, avg_price,
               current_price, unrealized_pnl, category, token, token_id,
               is_paper, updated_at)
               VALUES (?, 'kalshi', 'BUY', ?, ?, ?, 0, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(market_id, is_paper, token) DO UPDATE SET
                   size = excluded.size,
                   avg_price = excluded.avg_price,
                   current_price = excluded.current_price,
                   updated_at = excluded.updated_at""",
            (order.market_id, fill_size, fill_price, fill_price,
             market.category or "", order.token.value, order.token_id,
             1 if is_paper else 0),
        )
        await self._db.commit()
        try:
            await self._calibration.record_prediction(
                order.market_id, signal.claude_prob, market.category or "")
        except Exception as e:
            log.debug("informed_flow.calibration_error", error=str(e))
