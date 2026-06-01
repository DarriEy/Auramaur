"""Dual-purpose Kraken pillar for the hybrid loop.

TREASURY (always on when kraken.enabled):
  - sync + log Kraken balances (first-class visibility)
  - auto-convert idle fiat -> USDC toward a target reserve (capped, gated)
  - when Polymarket cash is low, ALERT to refill from Kraken USDC (no auto
    withdrawal — transfers keep their own gate + per-move approval)

DIRECTIONAL (gated behind kraken.directional_enabled, default OFF):
  - simple momentum signal per configured pair -> capped long entry/exit
  - shipped OFF: Path-B research found no durable directional edge, so this is
    opt-in and experimental. Every order still passes the spot adapter's gates
    (per-order cap, directional gate, three-gate live model, kill switch).

All real orders flow through KrakenSpotClient, so its safety model applies here.
"""

from __future__ import annotations

import structlog

from auramaur.exchange.models import OrderSide

log = structlog.get_logger()


class KrakenPillar:
    def __init__(self, settings, kraken_client, bot=None, console=None):
        self._s = settings
        self._k = kraken_client
        self._bot = bot          # for _last_known_cash
        self._console = console
        # In-memory open directional longs: pair -> entry price. Experimental;
        # resets on restart (directional is off by default).
        self._dir_long: dict[str, float] = {}

    async def run_once(self) -> None:
        try:
            await self._treasury()
        except Exception as e:  # noqa: BLE001 — a pillar must not kill the loop
            log.error("kraken.treasury.error", error=str(e))
        if self._s.kraken.directional_enabled and self._s.kraken.directional_pairs:
            try:
                await self._directional()
            except Exception as e:  # noqa: BLE001
                log.error("kraken.directional.error", error=str(e))

    # ------------------------------------------------------------------
    # Treasury
    # ------------------------------------------------------------------

    async def _treasury(self) -> None:
        kcfg = self._s.kraken
        bal = await self._k.get_balance()
        if not bal:
            return
        log.info("kraken.treasury.balances", **{a: round(v, 4) for a, v in bal.items() if v > 0})
        usdc = bal.get("USDC", 0.0)

        # Idle fiat -> USDC toward the target reserve (one conversion per cycle).
        if kcfg.auto_convert and usdc < kcfg.target_usdc:
            for fiat in kcfg.fiat_assets:
                have = bal.get(fiat, 0.0)
                if have <= 0:
                    continue
                pair = f"USDC{fiat.lstrip('XZ')}"   # ZCAD -> USDCCAD
                price = await self._k.get_price(pair)   # fiat per USDC
                if not price or price <= 0:
                    continue
                need = kcfg.target_usdc - usdc
                # Bound volume so volume*price (fiat spend) stays under the USD
                # cap and within the fiat we actually hold.
                vol = round(min(need, kcfg.max_order_usd / price, have / price), 2)
                if vol < 5:   # below typical Kraken min order size
                    continue
                res = await self._k.place_spot_order(
                    pair, OrderSide.BUY, volume=vol, ordertype="market", purpose="treasury",
                )
                log.info("kraken.treasury.convert", pair=pair, usdc=vol, status=res.status,
                         paper=res.is_paper, err=res.error_message[:80])
                break

        # Cash-starved Polymarket refill — ALERT only (human approves the move).
        cash = getattr(self._bot, "_last_known_cash", None) if self._bot else None
        if (cash is not None and cash < kcfg.refill_cash_floor
                and usdc >= self._s.transfers.min_transfer_usd):
            self._refill_alert(cash, usdc)

    def _refill_alert(self, cash: float, usdc: float) -> None:
        amt = min(usdc, self._s.transfers.per_transfer_cap_usd)
        log.warning("kraken.refill_suggested", poly_cash=round(cash, 2), kraken_usdc=round(usdc, 2))
        if self._console:
            self._console.print(
                f"\n[yellow]╔══ TREASURY: low Polymarket cash ══╗[/]\n"
                f"[yellow]║[/] Poly cash ${cash:.2f} < floor; ${usdc:.2f} USDC sitting on Kraken\n"
                f"[yellow]║[/] → python scripts/kraken_transfer.py --to polymarket-usdc "
                f"--amount {amt:.0f} --execute\n"
                f"[yellow]╚════════════════════════════════════╝[/]"
            )

    # ------------------------------------------------------------------
    # Directional (gated, experimental, no validated edge)
    # ------------------------------------------------------------------

    async def _directional(self) -> None:
        kcfg = self._s.kraken
        for pair in kcfg.directional_pairs:
            mom = await self._momentum(pair)
            if mom is None:
                continue
            price = await self._k.get_price(pair)
            if not price or price <= 0:
                continue
            holding = pair in self._dir_long

            if not holding and mom >= kcfg.directional_momentum_pct:
                vol = round(kcfg.max_order_usd / price, 6)
                res = await self._k.place_spot_order(
                    pair, OrderSide.BUY, volume=vol, ordertype="market", purpose="directional")
                if res.order_id not in ("ERROR", "BLOCKED"):
                    self._dir_long[pair] = price
                log.info("kraken.directional.entry", pair=pair, momentum=round(mom, 2),
                         status=res.status, err=res.error_message[:80])

            elif holding and mom <= -kcfg.directional_momentum_pct:
                entry = self._dir_long.get(pair) or price
                vol = round(kcfg.max_order_usd / entry, 6)
                res = await self._k.place_spot_order(
                    pair, OrderSide.SELL, volume=vol, ordertype="market", purpose="directional")
                if res.order_id not in ("ERROR", "BLOCKED"):
                    self._dir_long.pop(pair, None)
                log.info("kraken.directional.exit", pair=pair, momentum=round(mom, 2),
                         status=res.status, err=res.error_message[:80])

    async def _momentum(self, pair: str) -> float | None:
        """Percent change of close over the lookback window (hourly candles)."""
        data = await self._k._public("OHLC", {"pair": pair, "interval": 60})
        candles = next((v for k, v in data.items() if k != "last" and isinstance(v, list)), None)
        lb = self._s.kraken.directional_lookback
        if not candles or len(candles) < lb + 1:
            return None
        closes = [float(c[4]) for c in candles]
        old, cur = closes[-1 - lb], closes[-1]
        return None if old <= 0 else (cur - old) / old * 100.0
