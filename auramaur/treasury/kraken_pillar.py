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
        # Open directional longs: pair -> entry-price proxy. Reconciled from
        # actual Kraken balances each cycle (see _reconcile_positions), so it
        # survives restarts instead of silently over-allocating.
        self._dir_long: dict[str, float] = {}
        self._pair_base: dict[str, str] | None = None  # pair -> base asset code
        self._pair_min: dict[str, float] = {}          # pair -> ordermin (base units)

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

        # Surface Kraken in the live console (not just the JSON log). Show actual
        # crypto holdings (= open speculative exposure), not the in-memory count.
        if self._console:
            cad = bal.get("ZCAD", 0.0)
            crypto = [a for a, v in bal.items()
                      if v > 0 and a != "USDC" and not a.startswith("Z")]
            if kcfg.directional_enabled:
                spec = f"[red]spec: {','.join(crypto)}[/]" if crypto else "spec: flat"
            else:
                spec = "spec off"
            self._console.print(
                f"[magenta]kraken[/] [green]${usdc:.2f}[/] USDC"
                f"{f' + {cad:.0f} CAD' if cad else ''} | {spec}"
            )

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

    async def _reconcile_positions(self, bal: dict) -> None:
        """Sync _dir_long to actual Kraken holdings so restarts don't lose track.

        A directional pair is "held" when we own a non-dust amount of its base
        asset. Adds discovered holdings, drops ones that were closed externally.
        """
        kcfg = self._s.kraken
        if self._pair_base is None:
            self._pair_base = {}
            try:
                info = await self._k._public(
                    "AssetPairs", {"pair": ",".join(kcfg.directional_pairs)})
                for _, meta in info.items():
                    alt = meta.get("altname")
                    if alt:
                        self._pair_base[alt] = meta.get("base")
                        try:
                            self._pair_min[alt] = float(meta.get("ordermin", 0) or 0)
                        except (TypeError, ValueError):
                            self._pair_min[alt] = 0.0
            except Exception as e:  # noqa: BLE001
                log.warning("kraken.reconcile_pairs_error", error=str(e))

        held: dict[str, float] = {}
        held_prices: dict[str, tuple[float, float, float]] = {}  # pair -> (entry, current, qty)
        for pair in kcfg.directional_pairs:
            base = self._pair_base.get(pair)
            amt = bal.get(base, 0.0) if base else 0.0
            if amt <= 0:
                continue
            price = await self._k.get_price(pair)
            if price and amt * price >= 2.0:   # non-dust threshold
                entry = self._dir_long.get(pair, price)  # keep known entry
                held[pair] = entry
                held_prices[pair] = (entry, price, amt)

        closed = [pair for pair in self._dir_long if pair not in held]
        for pair in closed:
            self._dir_long.pop(pair, None)   # closed externally
        self._dir_long.update(held)

        # Mirror the spec book into the portfolio table so it's visible in the
        # dashboard with unrealized P&L. Kraken isn't a discovery, so check_exits
        # never touches these rows — the momentum logic owns the exits.
        await self._mirror_to_portfolio(held_prices, closed)

    async def _mirror_to_portfolio(
        self, held_prices: dict[str, tuple[float, float, float]], closed: list[str],
    ) -> None:
        db = self._bot._components.get("db") if self._bot else None
        if db is None:
            return
        is_paper = 0 if self._s.is_live else 1
        try:
            for pair, (entry, current, qty) in held_prices.items():
                await db.execute(
                    """INSERT INTO portfolio
                       (market_id, exchange, side, size, avg_price, current_price,
                        unrealized_pnl, category, token, token_id, is_paper, updated_at)
                       VALUES (?, 'kraken', 'BUY', ?, ?, ?, ?, 'crypto', 'YES', ?, ?, datetime('now'))
                       ON CONFLICT(market_id, is_paper) DO UPDATE SET
                           size = excluded.size,
                           avg_price = excluded.avg_price,
                           current_price = excluded.current_price,
                           unrealized_pnl = excluded.unrealized_pnl,
                           updated_at = excluded.updated_at""",
                    (pair, qty, entry, current, (current - entry) * qty, pair, is_paper),
                )
            for pair in closed:
                await db.execute(
                    "DELETE FROM portfolio WHERE market_id = ? AND exchange = 'kraken' AND is_paper = ?",
                    (pair, is_paper),
                )
            await db.commit()
        except Exception as e:  # noqa: BLE001 — visibility only; never break the pillar
            log.debug("kraken.portfolio_mirror_error", error=str(e)[:100])

    async def _directional(self) -> None:
        kcfg = self._s.kraken
        bal = await self._k.get_balance()
        await self._reconcile_positions(bal)
        for pair in kcfg.directional_pairs:
            mom = await self._momentum(pair)
            if mom is None:
                continue
            price = await self._k.get_price(pair)
            if not price or price <= 0:
                continue
            holding = pair in self._dir_long

            # Asymmetric long bias: enter on a smaller up-move, hold through a
            # larger down-move (ride winners). Fall back to the legacy symmetric
            # threshold if the split ones aren't configured.
            entry_thr = getattr(kcfg, "directional_entry_momentum_pct", None) or kcfg.directional_momentum_pct
            exit_thr = getattr(kcfg, "directional_exit_momentum_pct", None) or kcfg.directional_momentum_pct

            if not holding and mom >= entry_thr:
                # Budget ceiling: each open position ~= max_order_usd. Scaled by
                # the global risk-tolerance lever so it moves with all exposure.
                from auramaur.risk.tolerance import scale_budget, current_tolerance
                budget = scale_budget(kcfg.directional_budget_usd, current_tolerance(self._s))
                allocated = len(self._dir_long) * kcfg.max_order_usd
                if allocated + kcfg.max_order_usd > budget:
                    log.info("kraken.directional.budget_full", pair=pair,
                             allocated=allocated, budget=round(budget, 2))
                    continue
                # Size in USD terms via the pair's quote currency (works for any
                # quote: USDC/USDT/EUR/…). 2% buffer so rounding never trips the
                # per-order cap.
                vol = await self._k.size_for_usd(pair, kcfg.max_order_usd * 0.98, price)
                if not vol or vol <= 0:
                    continue
                # Respect the pair's minimum lot. Bump up to it only if the min
                # still fits ~1.5x the per-order cap in USD; else the pair is too
                # expensive per lot to trade within budget.
                ordermin = self._pair_min.get(pair, 0.0)
                if ordermin and vol < ordermin:
                    min_usd = await self._k.usd_notional(pair, ordermin, price)
                    if min_usd and min_usd > kcfg.max_order_usd * 1.5:
                        log.info("kraken.directional.min_lot_too_large", pair=pair,
                                 ordermin=ordermin, min_usd=round(min_usd, 2))
                        continue
                    vol = ordermin
                vol = round(vol, 8)
                res = await self._k.place_spot_order(
                    pair, OrderSide.BUY, volume=vol, ordertype="market", purpose="directional")
                if res.order_id not in ("ERROR", "BLOCKED"):
                    self._dir_long[pair] = price
                log.info("kraken.directional.entry", pair=pair, momentum=round(mom, 2),
                         status=res.status, err=res.error_message[:80])

            elif holding and mom <= -exit_thr:
                entry = self._dir_long.get(pair) or price
                vol = await self._k.size_for_usd(pair, kcfg.max_order_usd * 0.98, entry)
                if not vol or vol <= 0:
                    continue
                vol = round(vol, 8)
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
