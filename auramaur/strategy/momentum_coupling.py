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

log = structlog.get_logger()

_PAIR = {"BTC": "XBTUSD", "ETH": "ETHUSD"}
_NAME = {"BTC": "Bitcoin", "ETH": "Ethereum"}


class MomentumCouplingPillar:
    def __init__(self, settings, console=None):
        self._s = settings
        self._console = console
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
            # Execution intentionally not wired until the coupling validates.
            # When cfg.execute: route a strategy_source='momentum_coupling' order
            # through the Polymarket engine here, bounded by max_position_usd.
            break  # one signal per asset per cycle
