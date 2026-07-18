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

import asyncio
import math
import time

import structlog

from auramaur.exchange.models import Fill, OrderSide, TokenType

log = structlog.get_logger()


class KrakenPillar:
    def __init__(self, settings, kraken_client, bot=None, console=None,
                 paper_comparator=None):
        self._s = settings
        self._k = kraken_client
        self._bot = bot          # for _last_known_cash
        self._console = console
        self._paper_comparator = paper_comparator
        # Open directional longs: pair -> entry-price proxy. Reconciled from
        # actual Kraken balances each cycle (see _reconcile_positions), so it
        # survives restarts instead of silently over-allocating.
        self._dir_long: dict[str, float] = {}
        self._pair_base: dict[str, str] | None = None  # pair -> base asset code
        self._pair_min: dict[str, float] = {}          # pair -> ordermin (base units)
        self._pair_lot_dec: dict[str, int] = {}        # pair -> lot_decimals (volume precision)
        self._valid_pairs: list[str] = []              # configured pairs Kraken recognizes
        # Reverse index over the FULL Kraken catalog (base asset -> a preferred
        # sellable pair), so a held position whose pair was pruned or de-configured
        # can still be mapped to a pair and liquidated. Built in _resolve_pairs.
        self._base_to_pair: dict[str, str] = {}        # base asset -> liquidation pair (altname)
        self._base_pair_meta: dict[str, tuple] = {}    # pair -> (base, ordermin, lot_decimals)
        self._pnl = None                               # lazy PnLTracker (needs db)
        self._cooldown_until: dict[str, float] = {}    # pair -> monotonic re-entry gate
        # LLM directional view cache: pair -> (monotonic_ts, prob, confidence).
        # Throttles LLM calls to directional_llm_refresh_hours per pair (cost).
        self._llm_views: dict[str, tuple[float, float, str]] = {}

    # Kraken pair -> human asset name for the LLM directional question. Only
    # pairs listed here are eligible for the LLM signal (keeps it to assets the
    # news/LLM pipeline can actually reason about).
    _PAIR_ASSET = {
        "XBTUSDC": "Bitcoin", "ETHUSDC": "Ethereum", "SOLUSDC": "Solana",
        "XRPUSDC": "XRP", "ADAUSDC": "Cardano", "AVAXUSDC": "Avalanche",
        "DOTUSDC": "Polkadot", "LINKUSDC": "Chainlink", "LTCUSDC": "Litecoin",
        "BCHUSDC": "Bitcoin Cash", "ATOMUSDC": "Cosmos", "TONUSDC": "Toncoin",
        "XTZUSDC": "Tezos", "MANAUSDC": "Decentraland", "APEUSDC": "ApeCoin",
        "VETUSDC": "VeChain", "XDGUSDC": "Dogecoin", "SHIBUSDC": "Shiba Inu",
    }

    # Confidence floor ordering (mirrors exchange.models.Confidence).
    _CONF_RANK = {
        "LOW": 0, "MEDIUM_LOW": 1, "MEDIUM": 2, "MEDIUM_HIGH": 3, "HIGH": 4,
    }

    # Cash-like assets that are NOT directional crypto exposure: stablecoins,
    # Kraken fiat (Z-prefixed), and staked/earn variants ("DOT.S", "ETH2.S").
    _STABLES = frozenset({
        "USDC", "USDT", "DAI", "USDG", "PYUSD", "RLUSD", "TUSD", "USDP", "USD", "ZUSD",
    })

    def _is_cash_asset(self, asset: str) -> bool:
        a = (asset or "").upper()
        return a in self._STABLES or a.startswith("Z") or "." in a

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

    def _floor_lot(self, pair: str, qty: float) -> float:
        """Floor a volume to the pair's lot precision (lot_decimals).

        Kraken rejects orders whose volume has more precision than the pair
        allows; flooring (not rounding) also guarantees a sell never exceeds the
        free balance by a sliver. Used by both entry and exit sizing.
        """
        scale = 10 ** self._pair_lot_dec.get(pair, 8)
        return math.floor(qty * scale) / scale

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

    async def _llm_exit_reason(self, pair: str, gain_pct: float, kcfg,
                               peak_pct: float | None = None) -> str | None:
        """Exit a held LLM-driven long.

        The GAIN-based exits (stop-loss, fee-netted take-profit, trailing
        stop) apply regardless of signal source — they protect the position,
        not the thesis. BUG fixED 2026-06-10: the LLM pivot (#76) routed
        exits here WITHOUT them, so a winner could never bank a target (TON
        sat at +40% over basis with no exit able to fire — the same failure
        mode #68 fixed once before on the momentum path) and a missing LLM
        view froze exits entirely. Priority matches _exit_reason: cut losers
        -> bank the target -> protect the peak -> then ask the LLM.
        A missing view still holds (don't churn on a transient gap) — but
        only after the gain-based exits had their chance.
        """
        stop_pct = getattr(kcfg, "directional_stop_loss_pct", 0.0) or 0.0
        tp_pct = getattr(kcfg, "directional_take_profit_pct", 0.0) or 0.0
        trail_pct = getattr(kcfg, "directional_trailing_stop_pct", 0.0) or 0.0
        rt_fee = 2.0 * (getattr(kcfg, "directional_fee_pct", 0.0) or 0.0)
        if stop_pct > 0 and gain_pct <= -stop_pct:
            return "stop_loss"
        if tp_pct > 0 and (gain_pct - rt_fee) >= tp_pct:
            return "take_profit"
        if (trail_pct > 0 and peak_pct is not None and peak_pct > rt_fee
                and (peak_pct - gain_pct) >= trail_pct):
            return "trailing_stop"
        view = await self._llm_view(pair)
        if view is None:
            return None
        prob, _conf = view
        exit_prob = getattr(kcfg, "directional_llm_exit_prob", 0.45) or 0.45
        return "llm_bearish" if prob < exit_prob else None

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

    async def _await_fill(self, order_id: str, retries: int = 5, delay: float = 1.0) -> dict | None:
        """Poll the exchange for a CONFIRMED fill (``vol_exec > 0``).

        ``place_spot_order`` returns ``pending`` the instant Kraken accepts the
        order — *before* it knows the order actually executed. A market order
        fills near-instantly, but QueryOrders can lag a beat, so retry briefly.
        Returns the real fill dict, or None if it never confirms (the caller then
        records nothing — see ``_record_directional_fill``).
        """
        if not order_id:
            return None
        for attempt in range(retries):
            actual = await self._k.query_fill(order_id)
            if actual and actual.get("vol", 0) > 0:
                return actual
            if attempt < retries - 1:
                await asyncio.sleep(delay)
        return None

    async def _record_directional_fill(self, pair, side, res, est_price, qty, paper=False) -> float | None:
        """Record a directional entry/exit as a Fill so fills + realized P&L land
        in the books (cost_basis / daily_stats), the same path the binary venues
        use. Returns the fill price, or None if nothing was recorded.

        ``paper=True`` (LLM paper-validation, or a paper bot run) books a
        SIMULATED fill at the estimate price with is_paper=1 — no real order
        executed, but `kraken pnl` (paper) can then measure the strategy. Live
        mode (paper=False) only records a CONFIRMED fill.

        Scoped to live mode: paper orders are validate-only (never execute), so
        the real balance never reflects them and there is nothing to record.

        Only a CONFIRMED fill is recorded. Previously this fell back to the
        pre-trade ticker estimate whenever ``query_fill`` hadn't confirmed yet,
        which booked phantom closes (zeroing a sold-but-not-actually-filled
        position's cost basis) and fabricated entry bases — the root cause of the
        ``basis_unknown`` cascade that silently disabled every gain-based exit. If
        the fill can't be confirmed we record nothing and return None; the caller
        leaves the position untouched so balance reconciliation re-detects it.
        """
        if paper:
            # Simulated fill — no real execution to confirm. Book at the estimate
            # price with is_paper=1 so the strategy is measurable via `kraken pnl`.
            pnl = self._pnl_tracker()
            if pnl is not None:
                fee_pct = (getattr(self._s.kraken, "directional_fee_pct", 0.26) or 0.0) / 100.0
                fill = Fill(
                    order_id=str(getattr(res, "order_id", "")) or f"paper-{pair}",
                    market_id=pair, token_id=pair, side=side, token=TokenType.YES,
                    size=qty, price=est_price, fee=est_price * qty * fee_pct, is_paper=True,
                )
                try:
                    await pnl.record_fill(fill)
                except Exception as e:  # noqa: BLE001 — bookkeeping must not break the pillar
                    log.warning("kraken.directional.paper_fill_record_error", pair=pair, error=str(e)[:120])
            return est_price
        if not self._s.is_live:
            return None
        # Confirmation comes from the EXCHANGE, not from bookkeeping. An order
        # that filled must clear the position even if recording P&L fails — else
        # the position would be re-sold next cycle (double close). So await the
        # fill first; only then (best-effort) record it.
        actual = await self._await_fill(getattr(res, "order_id", ""))
        if not actual:
            log.warning("kraken.directional.fill_unconfirmed", pair=pair,
                        side=getattr(side, "value", str(side)),
                        order_id=str(getattr(res, "order_id", "")),
                        note="fill not confirmed — recording nothing to avoid a "
                             "phantom close / fabricated basis")
            return None
        price = actual["price"] or est_price
        vol = actual["vol"] or qty
        fee = actual["fee"]
        pnl = self._pnl_tracker()
        if pnl is not None:
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
        self._base_to_pair = {}
        self._base_pair_meta = {}
        try:
            info = await self._k._public("AssetPairs")
            by_altname = {}
            # Quote preference for the orphan-liquidation reverse index (lower = better).
            _quote_rank = {"ZUSD": 0, "USD": 0, "USDC": 1, "USDT": 2, "ZEUR": 3, "EUR": 3}
            best_rank: dict[str, int] = {}
            for meta in info.values():
                alt = meta.get("altname")
                base = meta.get("base")
                quote = meta.get("quote")
                if alt:
                    by_altname[alt] = meta
                # Build base -> preferred sellable pair across the WHOLE catalog.
                if alt and base:
                    rank = _quote_rank.get(quote, 9)
                    if base not in best_rank or rank < best_rank[base]:
                        try:
                            omin = float(meta.get("ordermin", 0) or 0)
                        except (TypeError, ValueError):
                            omin = 0.0
                        try:
                            ldec = int(meta.get("lot_decimals", 8))
                        except (TypeError, ValueError):
                            ldec = 8
                        best_rank[base] = rank
                        self._base_to_pair[base] = alt
                        self._base_pair_meta[alt] = (base, omin, ldec)
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
                try:
                    self._pair_lot_dec[pair] = int(meta.get("lot_decimals", 8))
                except (TypeError, ValueError):
                    self._pair_lot_dec[pair] = 8
            dropped = sorted(set(pairs) - set(self._valid_pairs))
            if dropped:
                log.warning("kraken.directional.pruned_unknown_pairs",
                            count=len(dropped), pairs=dropped)
        except Exception as e:  # noqa: BLE001
            log.warning("kraken.reconcile_pairs_error", error=str(e))
            self._valid_pairs = list(pairs)  # fall back to all; per-call errors absorb

    def _register_orphan_pair(self, pair: str) -> None:
        """Populate pair metadata for an orphan liquidation pair from the reverse
        index, so exit sizing (base lookup, ordermin, lot precision) works."""
        meta = self._base_pair_meta.get(pair)
        if not meta:
            return
        base, omin, ldec = meta
        if self._pair_base is None:
            self._pair_base = {}
        self._pair_base.setdefault(pair, base)
        self._pair_min.setdefault(pair, omin)
        self._pair_lot_dec.setdefault(pair, ldec)

    def _managed_pairs(self, bal: dict) -> list[str]:
        """Pairs to evaluate this cycle: the configured/valid set UNION every
        pair we currently hold the base asset for.

        The exit loop previously iterated only configured pairs, so a held
        position whose pair was pruned (or removed from config) was never checked
        and sat unsold. Here we add any non-cash crypto balance as an "orphan"
        pair mapped through the catalog reverse index, so it gets liquidated.
        """
        kcfg = self._s.kraken
        pairs = list(self._valid_pairs or kcfg.directional_pairs)
        seen = set(pairs)
        covered_bases = {
            (self._pair_base or {}).get(p) for p in pairs
        }
        for asset, amt in bal.items():
            if amt <= 0 or self._is_cash_asset(asset):
                continue
            if asset in covered_bases:
                continue  # already handled by a managed pair
            lp = self._base_to_pair.get(asset)
            if not lp:
                log.warning("kraken.directional.orphan_unmappable", asset=asset,
                            amt=round(amt, 8),
                            note="held crypto with no sellable pair in catalog")
                continue
            if lp not in seen:
                self._register_orphan_pair(lp)
                pairs.append(lp)
                seen.add(lp)
        return pairs

    async def _reconcile_positions(
        self, bal: dict, managed: list[str] | None = None,
    ) -> dict[str, tuple[float, float, float]]:
        """Sync _dir_long to actual Kraken holdings so restarts don't lose track.

        A directional pair is "held" when we own a non-dust amount of its base
        asset. Adds discovered holdings, drops ones that were closed externally.
        Returns the held book as ``pair -> (entry, current, qty)`` so the exit
        path can sell the actual quantity held.

        ``managed`` is the union of configured pairs and currently-held (incl.
        orphan) pairs — iterating it means a held position whose pair was pruned
        is reconciled and exited instead of being silently dropped as "closed".
        Defaults to the computed union when not supplied.
        """
        if self._pair_base is None:
            await self._resolve_pairs(self._s.kraken.directional_pairs)
        if managed is None:
            managed = self._managed_pairs(bal)
        held: dict[str, float] = {}
        held_prices: dict[str, tuple[float, float, float]] = {}  # pair -> (entry, current, qty)
        for pair in managed:
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
                    entry = await self._recover_entry(pair)
                    if entry is None:
                        # No persisted basis to recover. Anchoring to the current
                        # price zeroes gain% every cycle, which silently DISABLES
                        # the stop-loss and trailing-stop (they measure against
                        # entry). Surface it loudly instead of masking a held,
                        # unprotected position as a healthy 0-P&L row.
                        entry = price
                        log.warning("kraken.directional.basis_unknown", pair=pair,
                                    anchored_to_price=round(price, 6),
                                    note="stop-loss/trailing disabled until cost_basis is backfilled")
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
                       ON CONFLICT(market_id, is_paper, token) DO UPDATE SET
                           size = excluded.size,
                           avg_price = excluded.avg_price,
                           current_price = excluded.current_price,
                           unrealized_pnl = excluded.unrealized_pnl,
                           updated_at = excluded.updated_at""",
                    (pair, qty, entry, current, (current - entry) * qty, pair, is_paper),
                )
            # Reconcile the mirror against WALLET truth, not in-memory state:
            # `closed` only sees pairs tracked by THIS process, so positions
            # closed in a previous session (manual flatten, prior bot run, the
            # 2026-06-10 duplicate-process incident) left immortal rows — 12
            # phantom positions / ~$280 of fake exposure. Any kraken row whose
            # pair isn't currently held gets deleted.
            rows = await db.fetchall(
                "SELECT market_id FROM portfolio WHERE exchange = 'kraken' AND is_paper = ?",
                (is_paper,),
            )
            stale = [r["market_id"] for r in (rows or [])
                     if r["market_id"] not in held_prices]
            for pair in stale:
                await db.execute(
                    "DELETE FROM portfolio WHERE market_id = ? AND exchange = 'kraken' AND is_paper = ?",
                    (pair, is_paper),
                )
                if pair not in closed:
                    log.info("kraken.portfolio_mirror.stale_removed", pair=pair)
            await db.commit()
        except Exception as e:  # noqa: BLE001 — visibility only; never break the pillar
            log.debug("kraken.portfolio_mirror_error", error=str(e)[:100])

    async def _directional(self) -> None:
        kcfg = self._s.kraken
        # Free (tradable) balances: total minus anything reserved by open orders.
        # Sizing exits/entries off the free amount avoids requesting more than is
        # actually sellable/spendable (which Kraken rejects as insufficient funds).
        bal = await self._k.get_free_balance()
        if self._pair_base is None:
            await self._resolve_pairs(kcfg.directional_pairs)
        # Evaluate the UNION of configured pairs and everything we actually hold,
        # so a position whose pair was pruned/de-configured still gets exited.
        managed = self._managed_pairs(bal)
        held_prices = await self._reconcile_positions(bal, managed)
        valid_set = set(self._valid_pairs or kcfg.directional_pairs)
        liquidate_orphans = getattr(kcfg, "directional_liquidate_orphans", True)

        # Asymmetric long bias: enter on a smaller up-move, hold through a larger
        # down-move (ride winners). Fall back to the legacy symmetric threshold.
        entry_thr = getattr(kcfg, "directional_entry_momentum_pct", None) or kcfg.directional_momentum_pct
        exit_thr = getattr(kcfg, "directional_exit_momentum_pct", None) or kcfg.directional_momentum_pct

        # LLM/news gate. When enabled it REPLACES the price-momentum entry/exit
        # signal with the bot's news->LLM crypto read. While unproven it is
        # paper-forced: orders go in validate-only and positions are simulated +
        # booked as paper fills, so `kraken pnl` measures it without risking
        # capital even when the bot runs live for other venues.
        llm_on = bool(getattr(kcfg, "directional_llm_enabled", False))
        llm_paper = llm_on and bool(getattr(kcfg, "directional_llm_paper", True))
        effective_paper = (not self._s.is_live) or llm_paper

        # LLM-eligible pairs (those the news/LLM pipeline can reason about).
        # Drives both the calibration feedback loop and the conviction-weighted
        # budget. conv_mult is 1.0 unless the conviction budget is enabled.
        eligible = [p for p in (self._valid_pairs or kcfg.directional_pairs)
                    if p in self._PAIR_ASSET]
        if llm_on:
            await self._track_dir_signals(eligible)
        conv_mult = self._conviction_mult(eligible)

        for pair in managed:
            orphaned = pair not in valid_set
            price = await self._k.get_price(pair)
            if not price or price <= 0:
                if (pair in self._dir_long) and orphaned:
                    log.warning("kraken.directional.orphan_no_price", pair=pair,
                                note="held position Kraken won't price; manual close may be needed")
                continue
            holding = pair in self._dir_long
            mom: float | None = None

            if holding:
                # Exit side. For orphans (pair pruned/de-configured while still
                # held) force-close regardless of momentum — we no longer manage
                # it. Otherwise stop-loss / take-profit / trailing-stop / momentum
                # via _exit_reason (priority + fee-netting), against real basis.
                entry_px = self._dir_long.get(pair) or price
                gain_pct = (price - entry_px) / entry_px * 100.0 if entry_px > 0 else 0.0
                if orphaned:
                    if not liquidate_orphans:
                        log.warning("kraken.directional.orphan_detected", pair=pair,
                                    gain_pct=round(gain_pct, 2),
                                    note="liquidation disabled; position left open")
                        continue
                    reason = "orphaned"
                    peak_pct = gain_pct
                else:
                    peak_pct = await self._update_and_get_peak(pair, gain_pct)
                    if llm_on:
                        reason = await self._llm_exit_reason(pair, gain_pct, kcfg, peak_pct=peak_pct)
                    else:
                        mom = await self._momentum(pair)
                        if mom is None:
                            continue
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
                # Floor to lot precision so the request never exceeds the free
                # balance by a rounding sliver. If the held amount is below the
                # pair's ordermin, a single market sell can't close it — log once
                # and skip rather than spamming rejected orders forever.
                vol = self._floor_lot(pair, qty)
                ordermin = self._pair_min.get(pair, 0.0)
                if vol <= 0 or (ordermin and vol < ordermin):
                    log.warning("kraken.directional.exit_below_ordermin", pair=pair,
                                reason=reason, held=round(qty, 8), ordermin=ordermin,
                                note="position too small to close with one market sell")
                    continue
                notional = await self._k.usd_notional(pair, vol, price) or (vol * price)
                res = await self._k.place_spot_order(
                    pair, OrderSide.SELL, volume=vol, ordertype="market",
                    purpose="directional", max_usd=max(notional, kcfg.max_order_usd) * 1.1,
                    dry_run=True if effective_paper else None)
                placed = res.order_id not in ("ERROR", "BLOCKED")
                fill_price = None
                if placed:
                    # Record the SELL fill — realizes P&L into cost_basis +
                    # daily_stats so the spec book is measurable. Live confirms
                    # the fill; paper books a simulated one at the est price.
                    fill_price = await self._record_directional_fill(
                        pair, OrderSide.SELL, res, price, vol, paper=effective_paper)
                    if effective_paper and self._paper_comparator is not None:
                        await self._paper_comparator.record_shadow_fill(
                            pair, OrderSide.SELL, vol)
                # Only forget the position once the sell is real. In paper mode
                # nothing executes, so simulate the close unconditionally; in live
                # mode require a confirmed fill — otherwise a sell that never
                # actually filled would drop the position from the book (phantom
                # close) and leave the held crypto stranded + basis-blind.
                closed = placed and (effective_paper or fill_price is not None)
                if closed:
                    self._dir_long.pop(pair, None)
                    await self._clear_peak(pair)
                    self._set_cooldown(pair)   # damp whipsaw re-entry
                elif placed:   # live, but the fill never confirmed
                    log.warning("kraken.directional.exit_unconfirmed", pair=pair,
                                reason=reason, order_id=str(res.order_id),
                                note="sell not confirmed filled — position retained, "
                                     "retried next cycle")
                log.info("kraken.directional.exit", pair=pair, reason=reason,
                         orphaned=orphaned, confirmed=closed,
                         momentum=(round(mom, 2) if mom is not None else None),
                         gain_pct=round(gain_pct, 2),
                         peak_pct=round(peak_pct, 2),
                         status=res.status, err=res.error_message[:80])

            elif not orphaned:
                # Entry side — configured pairs only; never re-enter an orphan.
                if self._in_cooldown(pair):
                    continue
                if llm_on:
                    view = await self._llm_view(pair)
                    if view is None:
                        continue
                    prob, conf = view
                    min_prob = getattr(kcfg, "directional_llm_min_prob", 0.60) or 0.60
                    if prob < min_prob or not self._conf_ok(conf):
                        continue
                else:
                    mom = await self._momentum(pair)
                    if mom is None or mom < entry_thr:
                        continue
                # Budget ceiling: cap TOTAL open directional exposure at the
                # tolerance-scaled budget, measured by ACTUAL notional (count×cap
                # understates appreciated winners).
                from auramaur.risk.tolerance import scale_budget, current_tolerance
                # Conviction-weighted: conv_mult in [min_mult, 1.0] shrinks the
                # ceiling when the LLM book is broadly neutral, holding more USDC.
                # It is <=1.0, so it can only reduce exposure vs the static budget.
                budget = scale_budget(kcfg.directional_budget_usd, current_tolerance(self._s)) * conv_mult
                allocated = sum(cur * q for (_e, cur, q) in held_prices.values())
                if allocated + kcfg.max_order_usd > budget:
                    log.info("kraken.directional.budget_full", pair=pair,
                             allocated=round(allocated, 2), budget=round(budget, 2),
                             conv_mult=round(conv_mult, 3))
                    continue
                # Size in USD terms via the pair's quote currency (works for any
                # quote: USDC/USDT/EUR/…). 2% buffer so rounding never trips the
                # per-order cap.
                vol = await self._k.size_for_usd(pair, kcfg.max_order_usd * 0.98, price)
                if not vol or vol <= 0:
                    continue
                # Floor to the pair's lot precision first (Kraken rejects an
                # over-precise volume), then respect the minimum lot — bump up to
                # ordermin only if it still fits ~1.5x the per-order cap in USD;
                # else the pair is too expensive per lot to trade within budget.
                vol = self._floor_lot(pair, vol)
                ordermin = self._pair_min.get(pair, 0.0)
                if ordermin and vol < ordermin:
                    min_usd = await self._k.usd_notional(pair, ordermin, price)
                    if min_usd and min_usd > kcfg.max_order_usd * 1.5:
                        log.info("kraken.directional.min_lot_too_large", pair=pair,
                                 ordermin=ordermin, min_usd=round(min_usd, 2))
                        continue
                    vol = ordermin  # ordermin is already a valid lot size
                if vol <= 0:
                    continue
                # Funding gate: only fire if we actually hold enough FREE quote
                # currency to pay for it. The notional budget ceiling above is far
                # larger than available cash, so without this the loop spams orders
                # Kraken rejects as insufficient funds every cycle.
                # Paper validation simulates entries, so it needs no real quote
                # balance — skip the funding gate (else a fully-deployed wallet
                # would block the paper book from ever opening a position).
                if not effective_paper:
                    quote = await self._k.get_pair_quote(pair) or "ZUSD"
                    free_quote = bal.get(quote, 0.0)
                    need_quote = vol * price * 1.005   # +0.5% for taker fee headroom
                    if free_quote < need_quote:
                        log.info("kraken.directional.skip_unfunded", pair=pair, quote=quote,
                                 free=round(free_quote, 2), need=round(need_quote, 2))
                        continue
                res = await self._k.place_spot_order(
                    pair, OrderSide.BUY, volume=vol, ordertype="market", purpose="directional",
                    dry_run=True if effective_paper else None)
                if res.order_id not in ("ERROR", "BLOCKED"):
                    # Record the fill and anchor the entry to the actual fill price
                    # so cost basis + the stops use real numbers. Paper books a
                    # simulated fill at the current price.
                    fill_price = await self._record_directional_fill(
                        pair, OrderSide.BUY, res, price, vol, paper=effective_paper)
                    if effective_paper and self._paper_comparator is not None:
                        await self._paper_comparator.record_shadow_fill(
                            pair, OrderSide.BUY, vol)
                    if effective_paper:
                        self._dir_long[pair] = price          # paper: simulate the entry
                    elif fill_price is not None:
                        self._dir_long[pair] = fill_price     # live: track only a CONFIRMED buy
                    else:
                        # Live buy we couldn't confirm: don't fabricate a basis.
                        # If it did fill, balance reconciliation re-detects the
                        # holding next cycle (basis anchored to price, surfaced via
                        # the basis_unknown warning) — and it still counts against
                        # the budget since that's measured off actual balances.
                        log.warning("kraken.directional.entry_unconfirmed", pair=pair,
                                    order_id=str(res.order_id),
                                    note="buy not confirmed — not tracking; reconcile re-detects")
                log.info("kraken.directional.entry", pair=pair,
                         momentum=(round(mom, 2) if mom is not None else None),
                         signal=("llm" if llm_on else "momentum"), paper=effective_paper,
                         confirmed=(pair in self._dir_long),
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

    def _conf_ok(self, conf: str) -> bool:
        kcfg = self._s.kraken
        floor = getattr(kcfg, "directional_llm_min_confidence", "MEDIUM")
        return self._CONF_RANK.get(str(conf).upper(), 0) >= self._CONF_RANK.get(str(floor).upper(), 2)

    async def _llm_view(self, pair: str, force: bool = False) -> tuple[float, str] | None:
        """LLM/news-driven P(asset higher over the horizon) for a pair.

        Reuses the proven news->LLM pipeline (aggregator + analyzer + calibration)
        the prediction-market side runs — the same path that scores ~72% on
        resolved crypto markets. Throttled to directional_llm_refresh_hours per
        pair to cap LLM spend; cached views are reused between refreshes. Returns
        (prob_up, confidence) or None if unavailable. Records each fresh read into
        calibration (market_id "kraken-dir:<pair>") so accuracy is measurable.
        """
        asset = self._PAIR_ASSET.get(pair)
        if asset is None or self._bot is None:
            return None
        kcfg = self._s.kraken
        refresh_s = (getattr(kcfg, "directional_llm_refresh_hours", 8.0) or 8.0) * 3600.0
        cached = self._llm_views.get(pair)
        if cached and not force and (time.monotonic() - cached[0]) < refresh_s:
            return (cached[1], cached[2])

        comp = self._bot._components
        aggregator = comp.get("aggregator")
        analyzer = comp.get("analyzer")
        cache = comp.get("cache")
        if aggregator is None or analyzer is None:
            return None

        horizon = getattr(kcfg, "directional_llm_horizon_days", 3)
        from auramaur.exchange.models import Market
        market = Market(
            id=f"kraken-dir:{pair}",
            exchange="kraken",
            question=(f"Will {asset} trade higher {horizon} days from now than it "
                      f"does today?"),
            description=(f"Short-term directional read on {asset} ({pair}) spot price "
                         f"over the next {horizon} days, for a long-only spot position."),
            category="crypto",
            outcome_yes_price=0.5,
            outcome_no_price=0.5,
        )
        try:
            queries = [f"{asset} crypto price outlook", f"{asset} news"]
            evidence: list = []
            seen: set[str] = set()
            for q in queries:
                items = await aggregator.gather(q, limit_per_source=3, category="crypto")
                for it in items:
                    if it.id not in seen:
                        seen.add(it.id)
                        evidence.append(it)
            analysis = await analyzer.analyze(market, evidence, cache)
        except Exception as e:
            log.warning("kraken.llm_view_error", pair=pair, error=str(e))
            return None
        if analysis is None or analysis.skipped_reason:
            return None

        prob = float(analysis.probability)
        conf = str(analysis.confidence)
        self._llm_views[pair] = (time.monotonic(), prob, conf)
        # Calibration recording is owned by _track_dir_signals (one tracked,
        # horizon-resolved bet per pair) so the feedback loop actually closes —
        # see kraken_dir_signals.
        log.info("kraken.llm_view", pair=pair, asset=asset, prob=round(prob, 3), confidence=conf)
        return (prob, conf)

    async def _track_dir_signals(self, eligible: list[str]) -> None:
        """Close the calibration feedback loop for the LLM directional book.

        Keeps at most one outstanding tracked prediction per pair in
        kraken_dir_signals: when a bet's horizon elapses, resolve it (spot now
        vs the reference price snapshotted at open) and feed the up/down outcome
        to calibration, then open a fresh bet from the current cached LLM view.
        Pure measurement — no orders, no fund movement. Recorded under the
        ``kraken_spot`` category so it never pollutes the Polymarket ``crypto``
        calibration that drives prediction-market sizing.
        """
        db = self._db()
        if db is None or not eligible:
            return
        comp = self._bot._components if self._bot else {}
        calibration = comp.get("calibration")
        horizon = int(getattr(self._s.kraken, "directional_llm_horizon_days", 3) or 3)

        # 1) Resolve bets whose horizon has elapsed.
        try:
            due = await db.fetchall(
                "SELECT pair, ref_price FROM kraken_dir_signals WHERE due_at <= datetime('now')"
            )
        except Exception:
            due = []
        for row in due:
            pair = row["pair"]
            price = await self._safe_price(pair)
            if price is None:
                continue   # can't price now; retry next cycle (row stays due)
            went_up = price > row["ref_price"]
            if calibration is not None:
                try:
                    await calibration.record_resolution(f"kraken-dir:{pair}", went_up)
                except Exception as e:
                    log.warning("kraken.dir_signal.resolve_error", pair=pair, error=str(e))
            await db.execute("DELETE FROM kraken_dir_signals WHERE pair = ?", (pair,))
            await db.commit()
            log.info("kraken.dir_signal.resolved", pair=pair,
                     ref=round(row["ref_price"], 6), now=round(price, 6), went_up=went_up)

        # 2) Open a fresh bet for any eligible pair that has a warm LLM view but
        #    no outstanding row.
        try:
            open_rows = await db.fetchall("SELECT pair FROM kraken_dir_signals")
            open_pairs = {r["pair"] for r in open_rows}
        except Exception:
            open_pairs = set()
        for pair in eligible:
            if pair in open_pairs:
                continue
            view = self._llm_views.get(pair)
            if view is None:
                continue   # no warm view yet — nothing to track
            prob = view[1]
            price = await self._safe_price(pair)
            if price is None:
                continue
            await db.execute(
                "INSERT OR REPLACE INTO kraken_dir_signals "
                "(pair, prob, ref_price, opened_at, due_at) "
                "VALUES (?, ?, ?, datetime('now'), datetime('now', ?))",
                (pair, prob, price, f"+{horizon} days"),
            )
            if calibration is not None:
                try:
                    await calibration.record_prediction(f"kraken-dir:{pair}", prob, "kraken_spot")
                except Exception:
                    pass
            await db.commit()
            log.info("kraken.dir_signal.opened", pair=pair, prob=round(prob, 3),
                     ref=round(price, 6), horizon_days=horizon)

    async def _safe_price(self, pair: str) -> float | None:
        try:
            price = await self._k.get_price(pair)
        except Exception:
            return None
        return price if price and price > 0 else None

    def _conviction_mult(self, eligible: list[str]) -> float:
        """Aggregate-conviction multiplier in [min_mult, 1.0] for the crypto budget.

        Reads only CACHED LLM views (no extra LLM cost). Each eligible pair
        contributes clamp((P(up) - 0.5) / 0.5, 0, 1) when its confidence clears
        the floor, else 0; pairs with no warm view count as 0. The mean over all
        eligible pairs maps onto [min_mult, 1.0]. <=1.0 by construction, so it
        can only shrink the budget (hold more USDC), never grow it.
        """
        kcfg = self._s.kraken
        if not getattr(kcfg, "directional_conviction_budget_enabled", False):
            return 1.0
        min_mult = max(0.0, min(1.0, float(getattr(kcfg, "directional_conviction_min_mult", 0.34) or 0.34)))
        if not eligible:
            return min_mult
        total = 0.0
        for pair in eligible:
            view = self._llm_views.get(pair)
            if view is None:
                continue
            _, prob, conf = view
            if not self._conf_ok(conf):
                continue
            total += max(0.0, min(1.0, (prob - 0.5) / 0.5))
        conviction = total / len(eligible)
        return min_mult + (1.0 - min_mult) * conviction
