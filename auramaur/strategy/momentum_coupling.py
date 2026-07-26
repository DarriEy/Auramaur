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

import time
from collections import defaultdict

import structlog

from auramaur.broker.execution_gateway import ExecutionGateway, TradeIntent
from auramaur.experiments.strategies.momentum_coupling import (
    CouplingCandidate, CouplingInputs, CouplingProposal, CouplingRules,
    assess_momentum_coupling,
)
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
        assessment = assess_momentum_coupling(
            CouplingInputs(
                asset=asset, spot=spot, move_pct=move,
                candidates=tuple(CouplingCandidate(
                    market_id=m.id, question=m.question,
                    yes_price=m.outcome_yes_price,
                ) for m in markets),
            ),
            CouplingRules(cfg.move_threshold_pct, cfg.near_money_pct, cfg.max_position_usd),
        )
        proposal = assessment.proposal
        if proposal is not None:
            m = next(m for m in markets if m.id == proposal.market_id)
            direction = proposal.direction
            log.info("coupling.signal", asset=asset, move_pct=round(move, 2), spot=round(spot),
                     market=m.question[:50], strike=proposal.strike,
                     market_prob=round(m.outcome_yes_price, 3), direction=direction,
                     size_usd=cfg.max_position_usd, execute=cfg.execute)
            if self._console:
                self._console.print(
                    f"[cyan]coupling[/] {asset} spot {move:+.1f}% -> {direction} "
                    f"'{m.question[:38]}' (prob {m.outcome_yes_price:.2f})"
                    + ("" if cfg.execute else " [dim](detect-only)[/]"))
            if cfg.execute:
                await self._execute(m, proposal)

    async def _execute(self, market, proposal: CouplingProposal) -> None:
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
        mp = proposal.market_probability
        signal = Signal(
            market_id=market.id, market_question=market.question,
            claude_prob=proposal.fair_probability, market_prob=mp,
            claude_confidence=Confidence(proposal.confidence),
            edge=proposal.edge_percent,
            recommended_side=OrderSide.BUY if proposal.direction == "BUY_YES" else OrderSide.SELL,
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
            log.warning("coupling.executed", market=market.id, direction=proposal.direction,
                        size=decision.position_size, status=res.status,
                        is_paper=res.result.is_paper if res.result else None)
        except Exception as e:  # noqa: BLE001
            log.error("coupling.execute_error", market=market.id, error=str(e)[:120])
