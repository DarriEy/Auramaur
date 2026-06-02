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

import time

import structlog

from auramaur.exchange.models import Fill, OrderSide, TokenType

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
        self._valid_pairs: list[str] = []              # configured pairs Kraken recognizes
        self._pnl = None                               # lazy PnLTracker (needs db)
        self._cooldown_until: dict[str, float] = {}    # pair -> monotonic re-entry gate

    def _db(self):
        return self._bot._components.get("db") if self._bot else None

    def _in_cooldown(self, pair: str) -> bool:
        return self._cooldown_until.get(pair, 0.0) > time.monotonic()

    def _set_cooldown(self, pair: str) -> None:
        mins = getattr(self._s.kraken, "directional_reentry_cooldown_min", 0.0) or 0.0
        if mins > 0:
            self._cooldown_until[pair] = time.monotonic() + mins * 60.0

    async def _update_and_get_peak(self, pair: str, gain_pct: float) -> float:
        """Record the running high-water-mark gain% for a held pair (reusing
        position_peaks) and return it, so a trailing stop can fire on give-back.
        Survives restarts because the peak is persisted."""
        db = self._db()
        if db is None:
            return gain_pct
        try:
            await db.execute(
                """INSERT INTO position_peaks (market_id, peak_pnl_pct, updated_at)
                   VALUES (?, ?, datetime('now'))
                   ON CONFLICT(market_id) DO UPDATE SET
                       peak_pnl_pct = MAX(excluded.peak_pnl_pct, position_peaks.peak_pnl_pct),
                       updated_at = excluded.updated_at""",
                (pair, gain_pct),
            )
            await db.commit()
            row = await db.fetchone(
                "SELECT peak_pnl_pct FROM position_peaks WHERE market_id = ?", (pair,))
            return float(row["peak_pnl_pct"]) if row else gain_pct
        except Exception:  # noqa: BLE001
            return gain_pct

    async def _clear_peak(self, pair: str) -> None:
        db = self._db()
        if db is None:
            return
        try:
            await db.execute("DELETE FROM position_peaks WHERE market_id = ?", (pair,))
            await db.commit()
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _exit_reason(mom, gain_pct, peak_pct, exit_thr, kcfg) -> str | None:
        """First applicable exit trigger for a held pair, or None to hold.

        Priority: cut losers (stop) → bank a target (TP, net of fees) → protect
        a winner's gains (trailing) → follow momentum down. TP/trailing are
        measured net of round-trip fees so they only fire on a real edge.
        """
        stop_pct = getattr(kcfg, "directional_stop_loss_pct", 0.0) or 0.0
        tp_pct = getattr(kcfg, "directional_take_profit_pct", 0.0) or 0.0
        trail_pct = getattr(kcfg, "directional_trailing_stop_pct", 0.0) or 0.0
        rt_fee = 2.0 * (getattr(kcfg, "directional_fee_pct", 0.0) or 0.0)
        if stop_pct > 0 and gain_pct <= -stop_pct:
            return "stop_loss"
        if tp_pct > 0 and (gain_pct - rt_fee) >= tp_pct:
            return "take_profit"
        if trail_pct > 0 and peak_pct > rt_fee and (peak_pct - gain_pct) >= trail_pct:
            return "trailing_stop"
        if mom <= -exit_thr:
            return "momentum"
        return None

    def _pnl_tracker(self):
        """Lazy PnLTracker so directional fills land in fills/cost_basis/daily_stats
        via the same machinery the binary venues use. None if no db is wired."""
        if self._pnl is None:
            db = self._bot._components.get("db") if self._bot else None
            if db is None:
                return None
            from auramaur.broker.pnl import PnLTracker
            self._pnl = PnLTracker(db, self._s)
        return self._pnl

    async def _record_directional_fill(self, pair, side, res, est_price, qty) -> float | None:
        """Record a directional entry/exit as a Fill so fills + realized P&L land
        in the books (cost_basis / daily_stats), the same path the binary venues
        use. Uses the real Kraken fill (price/vol/fee) when available; otherwise
        the ticker estimate + a modeled taker fee. Returns the fill price.

        Scoped to live mode: paper orders are validate-only (never execute), so
        the real balance never reflects them and there is nothing to record.
        """
        if not self._s.is_live:
            return None
        pnl = self._pnl_tracker()
        if pnl is None:
            return None
        fee_rate = (getattr(self._s.kraken, "directional_fee_pct", 0.0) or 0.0) / 100.0
        price, vol, fee = est_price, qty, est_price * qty * fee_rate
        actual = await self._k.query_fill(getattr(res, "order_id", ""))
        if actual:
            price = actual["price"] or est_price
            vol = actual["vol"] or qty
            fee = actual["fee"]
        fill = Fill(
            order_id=str(getattr(res, "order_id", "")), market_id=pair, token_id=pair,
            side=side, token=TokenType.YES, size=vol, price=price, fee=fee,
            is_paper=not self._s.is_live,
        )
        try:
            await pnl.record_fill(fill)
        except Exception as e:  # noqa: BLE001 — bookkeeping must not break the pillar
            log.warning("kraken.directional.fill_record_error", pair=pair, error=str(e)[:120])
        return price

    async def _recover_entry(self, pair: str) -> float | None:
        """True entry basis for a held pair, recovered from cost_basis.avg_cost.

        After a restart ``_dir_long`` is empty; without this the entry would
        default to the *current* price, silently resetting cost basis (and
        defeating the stop-loss). cost_basis is the persisted, fee-weighted
        entry written on the BUY fill.
        """
        pnl = self._pnl_tracker()
        if pnl is None:
            return None
        try:
            avg_cost, _ = await pnl.get_cost_basis(pair)
            return avg_cost if avg_cost and avg_cost > 0 else None
        except Exception:  # noqa: BLE001
            return None

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

    async def _resolve_pairs(self, pairs: list[str]) -> None:
        """Validate configured pairs against Kraken's catalog (once).

        Queries the full AssetPairs catalog (no pair filter) and matches by
        altname, so a single bad pair can't fail the whole batch — and, more
        importantly, invalid pairs are pruned from ``_valid_pairs`` instead of
        spamming "Unknown asset pair" on every OHLC/Ticker call each cycle.
        """
        self._pair_base = {}
        self._valid_pairs = []
        try:
            info = await self._k._public("AssetPairs")
            by_altname = {}
            for meta in info.values():
                alt = meta.get("altname")
                if alt:
                    by_altname[alt] = meta
            for pair in pairs:
                meta = by_altname.get(pair)
                if meta is None:
                    continue
                self._valid_pairs.append(pair)
                self._pair_base[pair] = meta.get("base")
                try:
                    self._pair_min[pair] = float(meta.get("ordermin", 0) or 0)
                except (TypeError, ValueError):
                    self._pair_min[pair] = 0.0
            dropped = sorted(set(pairs) - set(self._valid_pairs))
            if dropped:
                log.warning("kraken.directional.pruned_unknown_pairs",
                            count=len(dropped), pairs=dropped)
        except Exception as e:  # noqa: BLE001
            log.warning("kraken.reconcile_pairs_error", error=str(e))
            self._valid_pairs = list(pairs)  # fall back to all; per-call errors absorb

    async def _reconcile_positions(self, bal: dict) -> dict[str, tuple[float, float, float]]:
        """Sync _dir_long to actual Kraken holdings so restarts don't lose track.

        A directional pair is "held" when we own a non-dust amount of its base
        asset. Adds discovered holdings, drops ones that were closed externally.
        Returns the held book as ``pair -> (entry, current, qty)`` so the exit
        path can sell the actual quantity held.
        """
        kcfg = self._s.kraken
        if self._pair_base is None:
            await self._resolve_pairs(kcfg.directional_pairs)

        held: dict[str, float] = {}
        held_prices: dict[str, tuple[float, float, float]] = {}  # pair -> (entry, current, qty)
        for pair in (self._valid_pairs or kcfg.directional_pairs):
            base = self._pair_base.get(pair)
            amt = bal.get(base, 0.0) if base else 0.0
            if amt <= 0:
                continue
            price = await self._k.get_price(pair)
            if price and amt * price >= 2.0:   # non-dust threshold
                # Keep the in-memory entry; on a cold start recover the true
                # fee-weighted basis from cost_basis before falling back to the
                # current price (which would silently reset cost basis).
                entry = self._dir_long.get(pair)
                if entry is None:
                    entry = await self._recover_entry(pair) or price
                held[pair] = entry
                held_prices[pair] = (entry, price, amt)

        closed = [pair for pair in self._dir_long if pair not in held]
        for pair in closed:
            self._dir_long.pop(pair, None)   # closed externally
            await self._clear_peak(pair)     # drop stale trailing high-water mark
        self._dir_long.update(held)

        # Mirror the spec book into the portfolio table so it's visible in the
        # dashboard with unrealized P&L. Kraken isn't a discovery, so check_exits
        # never touches these rows — the momentum logic owns the exits.
        await self._mirror_to_portfolio(held_prices, closed)
        return held_prices

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
        held_prices = await self._reconcile_positions(bal)
        for pair in (self._valid_pairs or kcfg.directional_pairs):
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

            if holding:
                # Exit side: stop-loss / take-profit / trailing-stop / momentum,
                # all measured against the real entry basis and the persisted
                # peak. _exit_reason owns the priority + fee-netting.
                entry_px = self._dir_long.get(pair) or price
                gain_pct = (price - entry_px) / entry_px * 100.0 if entry_px > 0 else 0.0
                peak_pct = await self._update_and_get_peak(pair, gain_pct)
                reason = self._exit_reason(mom, gain_pct, peak_pct, exit_thr, kcfg)
                if not reason:
                    continue
                # Sell the ACTUAL held quantity (from balance reconciliation), not
                # a recomputed USD notional — otherwise ordermin-bumped or
                # fee-trimmed positions under-sell and leave a residual, or a
                # too-large recompute trips insufficient-funds. Pass max_usd with
                # headroom so an appreciated winner isn't blocked by the per-order
                # entry cap (exits must always be allowed to fully close).
                qty = held_prices.get(pair, (None, None, 0.0))[2]
                if not qty or qty <= 0:
                    continue
                vol = round(qty, 8)
                notional = await self._k.usd_notional(pair, vol, price) or (vol * price)
                res = await self._k.place_spot_order(
                    pair, OrderSide.SELL, volume=vol, ordertype="market",
                    purpose="directional", max_usd=max(notional, kcfg.max_order_usd) * 1.1)
                if res.order_id not in ("ERROR", "BLOCKED"):
                    # Record the SELL fill — this realizes P&L into cost_basis +
                    # daily_stats so the spec book is finally measurable.
                    await self._record_directional_fill(pair, OrderSide.SELL, res, price, vol)
                    self._dir_long.pop(pair, None)
                    await self._clear_peak(pair)
                    self._set_cooldown(pair)   # damp whipsaw re-entry
                log.info("kraken.directional.exit", pair=pair, reason=reason,
                         momentum=round(mom, 2), gain_pct=round(gain_pct, 2),
                         peak_pct=round(peak_pct, 2),
                         status=res.status, err=res.error_message[:80])

            elif mom >= entry_thr and not self._in_cooldown(pair):
                # Budget ceiling: cap TOTAL open directional exposure at the
                # tolerance-scaled budget, measured by ACTUAL notional (count×cap
                # understates appreciated winners).
                from auramaur.risk.tolerance import scale_budget, current_tolerance
                budget = scale_budget(kcfg.directional_budget_usd, current_tolerance(self._s))
                allocated = sum(cur * q for (_e, cur, q) in held_prices.values())
                if allocated + kcfg.max_order_usd > budget:
                    log.info("kraken.directional.budget_full", pair=pair,
                             allocated=round(allocated, 2), budget=round(budget, 2))
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
                    # Record the fill (live) and anchor the entry to the actual
                    # fill price so cost basis + the stops use real numbers.
                    fill_price = await self._record_directional_fill(
                        pair, OrderSide.BUY, res, price, vol)
                    self._dir_long[pair] = fill_price or price
                log.info("kraken.directional.entry", pair=pair, momentum=round(mom, 2),
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
