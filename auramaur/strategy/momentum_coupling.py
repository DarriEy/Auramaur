"""Momentum-coupling pillar — the FAST PATH (spot -> prediction lead-lag).

Distinct from the slow LLM loop: a fast-cadence momentum signal. When crypto
spot moves over a short window, the coupled near-the-money prediction market
("Will BTC be above $X") is expected to reprice minutes later (coupling_discovery
found spot leads). We'd take the matching side early.

OFF by default and DETECTION-ONLY (logs the signal, places nothing) until
coupling_tradeability.py confirms after-cost edge. Its signals carry
strategy_source="momentum_coupling" so they never run through the LLM-divergence
filter (that's scoped to LLM disagreement, which this isn't).
"""

from __future__ import annotations

import re
import time
from collections import defaultdict

import structlog

from auramaur.broker.execution_gateway import ExecutionGateway, TradeIntent
from auramaur.strategy.protocols import ExecutionMode

log = structlog.get_logger()

_PAIR = {"BTC": "XBTUSD", "ETH": "ETHUSD"}
_NAME = {"BTC": "Bitcoin", "ETH": "Ethereum"}


class MomentumCouplingPillar:
    # Uniform Strategy contract (see strategy/protocols.py). It is a directional,
    # risk-evaluated pillar like the others, so it routes through the single
    # ExecutionGateway — not a justified direct-placement exception.
    name = "momentum_coupling"
    execution_mode = ExecutionMode.GATEWAY_SINGLE

    def __init__(self, settings, console=None, polymarket_client=None,
                 risk_manager=None, bot=None, db=None, pnl_tracker=None):
        self._s = settings
        self._console = console
        self._poly = polymarket_client   # PolymarketClient (client.py) for execution
        self._risk = risk_manager        # RiskManager gateway — no trade bypasses it
        self._bot = bot                  # for available cash
        self._db = db
        self._pnl = pnl_tracker
        self._gateway = None             # lazily built once db+pnl+client present
        self._spot_hist: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self._kraken = None
        self._gamma = None

    async def run_once(self) -> None:
        cfg = self._s.momentum_coupling
        if not cfg.enabled:
            return
        if self._kraken is None:
            from auramaur.exchange.kraken import KrakenSpotClient
            from auramaur.exchange.gamma import GammaClient
            self._kraken = KrakenSpotClient(self._s)
            self._gamma = GammaClient()

        now = time.time()
        for asset in cfg.assets:
            pair = _PAIR.get(asset)
            if not pair:
                continue
            try:
                price = await self._kraken.get_price(pair)
            except Exception as e:  # noqa: BLE001
                log.debug("coupling.spot_error", asset=asset, error=str(e)[:80])
                continue
            if not price:
                continue
            hist = self._spot_hist[asset]
            hist.append((now, price))
            hist[:] = [(t, p) for t, p in hist if now - t <= cfg.lookback_seconds]
            if len(hist) < 2:
                continue
            move = (price - hist[0][1]) / hist[0][1] * 100
            if abs(move) >= cfg.move_threshold_pct:
                await self._emit(asset, price, move)

    async def _emit(self, asset: str, spot: float, move: float) -> None:
        cfg = self._s.momentum_coupling
        try:
            markets = await self._gamma.search_markets(f"{_NAME[asset]} above", limit=20)
        except Exception:
            return
        for m in markets:
            mt = re.search(r"\$?([\d,]{4,})", m.question)
            if not mt:
                continue
            strike = float(mt.group(1).replace(",", ""))
            if strike <= 0 or abs(strike - spot) / spot > cfg.near_money_pct:
                continue  # only near-the-money markets are spot-sensitive
            if not (0.10 < m.outcome_yes_price < 0.90):
                continue  # skip pinned markets — prob must be live to move with spot
            direction = "BUY_YES" if move > 0 else "BUY_NO"
            log.info("coupling.signal", asset=asset, move_pct=round(move, 2), spot=round(spot),
                     market=m.question[:50], strike=strike,
                     market_prob=round(m.outcome_yes_price, 3), direction=direction,
                     size_usd=cfg.max_position_usd, execute=cfg.execute)
            if self._console:
                self._console.print(
                    f"[cyan]coupling[/] {asset} spot {move:+.1f}% -> {direction} "
                    f"'{m.question[:38]}' (prob {m.outcome_yes_price:.2f})"
                    + ("" if cfg.execute else " [dim](detect-only)[/]"))
            if cfg.execute:
                await self._execute(m, direction, move)
            break  # one signal per asset per cycle

    async def _execute(self, market, direction: str, move: float) -> None:
        """Place the coupling trade through the ExecutionGateway (execute=True).

        The momentum signal is framed as a spot-implied prob nudge vs the market,
        so it flows through risk_manager.evaluate() — every check, the divergence
        filter, Kelly sizing, AND the risk_tolerance lever apply — and then the
        single ExecutionGateway, which co-writes fill+trades and honors the
        graduation ladder's force_paper (the old direct place_order skipped both,
        and used raw is_live so an ungraduated cell could trade live). Detect-only
        if risk/client/db/pnl aren't wired.
        """
        if (self._poly is None or self._risk is None
                or self._db is None or self._pnl is None):
            log.warning("coupling.no_route", reason="no risk/client/db/pnl; detect-only")
            return
        from auramaur.exchange.models import Signal, OrderSide, Confidence
        mp = market.outcome_yes_price
        shift = (1 if direction == "BUY_YES" else -1) * min(abs(move) / 100.0 * 5, 0.15)
        claude_prob = max(0.02, min(0.98, mp + shift))
        signal = Signal(
            market_id=market.id, market_question=market.question,
            claude_prob=claude_prob, market_prob=mp,
            claude_confidence=Confidence.HIGH if abs(move) >= 1.0 else Confidence.MEDIUM,
            edge=abs(claude_prob - mp) * 100,
            recommended_side=OrderSide.BUY if direction == "BUY_YES" else OrderSide.SELL,
            strategy_source="momentum_coupling")
        cash = getattr(self._bot, "_last_known_cash", None) if self._bot else None
        try:
            decision = await self._risk.evaluate(signal, market, available_cash=cash)
            if not decision.approved or decision.position_size <= 0:
                log.info("coupling.risk_rejected", market=market.id,
                         reason=decision.reason[:80])
                return
            if self._gateway is None:
                self._gateway = ExecutionGateway(
                    router=None, exchange=self._poly, exchange_name="polymarket",
                    settings=self._s, db=self._db, pnl_tracker=self._pnl)
            res = await self._gateway.submit(TradeIntent(
                signal=signal, market=market,
                size_dollars=decision.position_size,
                force_paper=getattr(decision, "force_paper", False)))
            log.warning("coupling.executed", market=market.id, direction=direction,
                        size=decision.position_size, status=res.status,
                        is_paper=res.result.is_paper if res.result else None)
        except Exception as e:  # noqa: BLE001
            log.error("coupling.execute_error", market=market.id, error=str(e)[:120])
