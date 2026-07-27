"""Paper-only broad-market ETF book using IBKR quotes and LLM/news signals."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, time as wall_time, timezone
import json
import time
from uuid import uuid4
from zoneinfo import ZoneInfo

import structlog

from auramaur.exchange.models import Market, OrderSide
from auramaur.experiments.strategies.ibkr_etf_paper import (
    entry_proposal, exit_proposal,
)
from auramaur.strategy.protocols import ExecutionMode
from auramaur.risk.ibkr_math import (
    annualized_volatility, normalized_momentum,
)

log = structlog.get_logger()
_ET = ZoneInfo("America/New_York")


class IBKRETFPaperPillar:
    """A structurally paper-only ETF strategy; this class cannot place orders."""

    _INSTRUMENTS = {
        "SPY": ("S&P 500", "us_broad"), "QQQ": ("Nasdaq-100", "us_broad"),
        "IWM": ("Russell 2000", "us_broad"), "DIA": ("Dow Jones", "us_broad"),
        "VTI": ("US total stock market", "us_broad"),
        "XLK": ("US technology sector", "us_sector"),
        "XLF": ("US financial sector", "us_sector"),
        "XLV": ("US health care sector", "us_sector"),
        "XLE": ("US energy sector", "us_sector"),
        "XLI": ("US industrial sector", "us_sector"),
        "XLY": ("US consumer discretionary sector", "us_sector"),
        "XLP": ("US consumer staples sector", "defensive"),
        "XLU": ("US utilities sector", "defensive"),
        "XLB": ("US materials sector", "us_sector"),
        "XLRE": ("US real estate sector", "real_assets"),
        "XLC": ("US communication services sector", "us_sector"),
        "VEA": ("developed markets ex-US", "international"),
        "VWO": ("emerging markets", "international"),
        "EWC": ("Canadian equities", "international"),
        "EWJ": ("Japanese equities", "international"),
        "TLT": ("long-term US Treasuries", "rates"),
        "IEF": ("intermediate US Treasuries", "rates"),
        "SHY": ("short-term US Treasuries", "rates"),
        "TIP": ("US inflation-protected Treasuries", "rates"),
        "GLD": ("gold", "real_assets"), "SLV": ("silver", "real_assets"),
        "DBC": ("broad commodities", "real_assets"),
        "VNQ": ("US real estate investment trusts", "real_assets"),
    }
    _CONF = {"LOW": 0, "MEDIUM_LOW": 1, "MEDIUM": 2,
             "MEDIUM_HIGH": 3, "HIGH": 4}
    name = "ibkr_etf_paper"
    execution_mode = ExecutionMode.PAPER_SIMULATED

    def __init__(self, settings, client, db, aggregator, analyzer, cache=None,
                 model_alias: str = "baseline", evidence_cache=None) -> None:
        self._s = settings
        self._client = client
        self._db = db
        self._aggregator = aggregator
        self._analyzer = analyzer
        self._cache = cache
        self._alias = model_alias
        self._evidence_cache = evidence_cache if evidence_cache is not None else {}
        self._views: dict[str, tuple[float, float, str]] = {}
        self._cooldown: dict[str, float] = {}
        self._refresh_cursor = 0
        self._state_loaded = False

    @staticmethod
    def market_open(now: datetime | None = None) -> bool:
        local = (now or datetime.now(timezone.utc)).astimezone(_ET)
        return local.weekday() < 5 and wall_time(9, 30) <= local.time() < wall_time(16, 0)

    def _market_id(self, symbol: str) -> str:
        return f"ibkr-etf:{self._alias}:{symbol}"

    @property
    def strategy_source(self) -> str:
        return f"ibkr_etf_{self._alias}"

    @property
    def model_alias(self) -> str:
        return self._alias

    @property
    def _min_prob(self) -> float:
        return float(self._s.ibkr.etf_arm_min_prob.get(
            self._alias, self._s.ibkr.etf_min_prob))

    @property
    def _min_conf_rank(self) -> int:
        return self._CONF.get(str(self._s.ibkr.etf_arm_min_confidence.get(
            self._alias, self._s.ibkr.etf_min_confidence)).upper(), 2)

    async def _view(self, symbol: str, allow_refresh: bool = True,
                    reference_price: float | None = None) -> tuple[float, str] | None:
        cfg = self._s.ibkr
        refresh = cfg.etf_signal_refresh_hours * 3600
        cached = self._views.get(symbol)
        if cached and time.time() - cached[0] < refresh:
            return cached[1], cached[2]
        if not allow_refresh:
            return (cached[1], cached[2]) if cached else None
        deterministic = getattr(self._analyzer, "analyze_symbol", None)
        if deterministic is not None:
            analysis = await deterministic(self._client, symbol)
            if analysis is None:
                return None
            view = (float(analysis.probability), str(analysis.confidence))
            self._views[symbol] = (time.time(), *view)
            # The control arm records its forecast on the SAME terms as the LLM
            # arms. It used to return here, so momentum_control wrote zero rows
            # to ibkr_etf_forecasts (282 rows, none of them the control, as of
            # 2026-07-27) — which left the intelligence-cap A/B with no control
            # to compare against. A control that is never scored is not a
            # control.
            await self._record_forecast(symbol, view, analysis, reference_price)
            return view
        call_count = await self._db.fetchone(
            """SELECT COUNT(*) AS n FROM ibkr_etf_openai_attempts
               WHERE model_alias = ? AND date(started_at) = date('now')""",
            (self._alias,))
        arm_limit = max(1, self._s.ibkr.etf_openai_daily_call_limit // max(
            1, len(self._s.ibkr.etf_models)))
        if call_count and call_count["n"] >= arm_limit:
            log.warning("ibkr_etf.openai_daily_limit",
                        model_alias=self._alias, limit=arm_limit)
            return (cached[1], cached[2]) if cached else None
        if self._aggregator is None or self._analyzer is None:
            return None
        name = self._INSTRUMENTS.get(symbol, (symbol, "other"))[0]
        horizon = cfg.etf_signal_horizon_days
        market = Market(
            id=self._market_id(symbol), exchange="ibkr",
            question=(f"Will {name}'s adjusted close {horizon} trading sessions "
                      "after today be higher than today's adjusted close?"),
            description=(f"Long-only directional paper forecast for {name}; scored "
                         "close-to-close on IBKR adjusted daily bars."),
            category="traditional_markets", outcome_yes_price=0.5,
            outcome_no_price=0.5,
        )
        evidence = []
        seen: set[str] = set()
        for query in (f"{name} market outlook", "US stock market macroeconomic news"):
            try:
                if query not in self._evidence_cache:
                    self._evidence_cache[query] = await self._aggregator.gather(
                        query, limit_per_source=3, category="finance",
                        market_id=symbol, market_price=reference_price,
                        consumer=self.name)
                items = self._evidence_cache[query]
            except Exception as exc:  # noqa: BLE001
                log.warning("ibkr_etf.paper.evidence_error", symbol=symbol,
                            error=str(exc)[:100])
                continue
            for item in items:
                if item.id not in seen:
                    seen.add(item.id)
                    evidence.append(item)
        try:
            analysis = await self._analyzer.analyze(market, evidence, self._cache)
        except Exception as exc:  # noqa: BLE001
            log.warning("ibkr_etf.paper.analysis_error", symbol=symbol,
                        error=str(exc)[:120])
            return None
        if analysis is None or analysis.skipped_reason:
            return None
        view = (float(analysis.probability), str(analysis.confidence))
        self._views[symbol] = (time.time(), *view)
        await self._record_forecast(symbol, view, analysis, reference_price)
        return view

    async def _record_forecast(self, symbol: str, view: tuple[float, str],
                               analysis, reference_price: float | None) -> None:
        """Persist one scoreable forecast. Shared by the LLM and control arms."""
        if not reference_price or reference_price <= 0:
            return
        days = max(1, round(self._s.ibkr.etf_signal_horizon_days * 7 / 5))
        await self._db.execute(
            """INSERT INTO ibkr_etf_forecasts
               (model_alias, model, symbol, probability, confidence, thesis,
                risks_json, reference_price, intelligence_cost_usd,
                opened_session_date, horizon_sessions, due_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now', ?))""",
            (self._alias, getattr(self._analyzer, "model", "unknown"), symbol,
             view[0], view[1], str(getattr(analysis, "thesis", "")),
             json.dumps(list(getattr(analysis, "key_risks", []) or [])),
             reference_price, float(getattr(analysis, "intelligence_cost_usd", 0)),
             datetime.now(timezone.utc).astimezone(_ET).date().isoformat(),
             self._s.ibkr.etf_signal_horizon_days, f"+{days} days"))

    async def _resolve_forecasts(self, symbol: str) -> None:
        rows = await self._db.fetchall(
            """SELECT id, reference_price, opened_session_date, horizon_sessions
               FROM ibkr_etf_forecasts
               WHERE model_alias = ? AND symbol = ? AND actual_outcome IS NULL
                 AND opened_session_date < ?""",
            (self._alias, symbol,
             datetime.now(timezone.utc).astimezone(_ET).date().isoformat()))
        if not rows:
            return
        closes = await self._client.get_adjusted_daily_closes(symbol)
        for row in rows:
            base = next((close for day, close in closes
                         if day == row["opened_session_date"]), None)
            if base is None:
                continue
            later = [(day, close) for day, close in closes
                     if day > row["opened_session_date"]]
            horizon = int(row["horizon_sessions"])
            if len(later) < horizon:
                continue
            target_day, final_price = later[horizon - 1]
            await self._db.execute(
                """UPDATE ibkr_etf_forecasts SET sessions_elapsed=?,
                     last_session_date=?, reference_price=?, final_price=?,
                     actual_outcome=CASE WHEN ? > ? THEN 1 ELSE 0 END,
                     resolved_at=datetime('now') WHERE id=?""",
                (horizon, target_day, base, final_price,
                 final_price, base, row["id"]))

    def _asset_class(self, symbol: str) -> str:
        return self._INSTRUMENTS.get(symbol, (symbol, "other"))[1]

    async def _ensure_state(self) -> None:
        if self._state_loaded:
            return
        state = await self._db.fetchone(
            "SELECT refresh_cursor FROM ibkr_etf_state WHERE model_alias = ?",
            (self._alias,))
        if state:
            self._refresh_cursor = int(state["refresh_cursor"] or 0)
        rows = await self._db.fetchall(
            """SELECT symbol, probability, confidence,
                      CAST(strftime('%s', opened_at) AS REAL) AS opened_epoch
               FROM ibkr_etf_forecasts WHERE model_alias = ?
               ORDER BY id DESC""", (self._alias,))
        refresh = self._s.ibkr.etf_signal_refresh_hours * 3600
        now = time.time()
        for row in rows:
            symbol = row["symbol"]
            opened = float(row["opened_epoch"] or 0)
            if symbol not in self._views and now - opened < refresh:
                self._views[symbol] = (
                    opened, float(row["probability"]), str(row["confidence"]))
        cooldowns = await self._db.fetchall(
            "SELECT symbol, until_epoch FROM ibkr_etf_cooldowns WHERE model_alias = ?",
            (self._alias,))
        self._cooldown = {r["symbol"]: float(r["until_epoch"])
                          for r in cooldowns if float(r["until_epoch"]) > now}
        await self._db.execute(
            "DELETE FROM ibkr_etf_cooldowns WHERE model_alias = ? AND until_epoch <= ?",
            (self._alias, now))
        self._state_loaded = True

    async def _save_cursor(self) -> None:
        await self._db.execute(
            """INSERT INTO ibkr_etf_state (model_alias, refresh_cursor, updated_at)
               VALUES (?, ?, datetime('now')) ON CONFLICT(model_alias) DO UPDATE SET
                 refresh_cursor=excluded.refresh_cursor, updated_at=excluded.updated_at""",
            (self._alias, self._refresh_cursor))

    async def _daily_realized(self) -> float:
        row = await self._db.fetchone(
            """SELECT COALESCE(SUM(pnl), 0) AS pnl FROM ibkr_etf_ledger
               WHERE model_alias = ?
                 AND date(realized_at) = date('now')""", (self._alias,))
        if row is None:
            return 0.0
        return float(row["pnl"] or 0.0) if row else 0.0

    async def _daily_equity_change(self) -> float:
        """Realized and current open losses both consume the daily envelope."""
        realized = await self._daily_realized()
        row = await self._db.fetchone(
            """SELECT COALESCE(SUM(unrealized_pnl), 0) AS pnl
                 FROM ibkr_etf_positions WHERE model_alias = ?""", (self._alias,))
        return realized + float(row["pnl"] or 0) if row else realized

    async def _positions(self) -> dict[str, tuple[float, float]]:
        rows = await self._db.fetchall(
            """SELECT symbol, quantity, avg_cost FROM ibkr_etf_positions
               WHERE model_alias = ? AND quantity > 0""", (self._alias,))
        return {r["symbol"]: (float(r["quantity"]), float(r["avg_cost"]))
                for r in rows}

    async def _peak(self, symbol: str, gain: float) -> float:
        await self._db.execute(
            """UPDATE ibkr_etf_positions SET
               peak_pnl_pct = MAX(peak_pnl_pct, ?),
               updated_at = datetime('now')
               WHERE model_alias = ? AND symbol = ?""",
            (gain, self._alias, symbol))
        row = await self._db.fetchone(
            """SELECT peak_pnl_pct FROM ibkr_etf_positions
               WHERE model_alias = ? AND symbol = ?""", (self._alias, symbol))
        return float(row["peak_pnl_pct"])

    async def _fill(self, symbol: str, side: OrderSide, qty: float, price: float,
                    *, stop_price: float = 0, initial_risk_usd: float = 0) -> None:
        fee = self._s.ibkr.etf_fee_per_order_usd
        fill_ref = f"ibkr-etf-paper-{self._alias}-{symbol}-{uuid4().hex}"
        await self._db.execute(
            """INSERT INTO ibkr_etf_fills
               (model_alias, symbol, side, quantity, price, commission_usd, fill_ref)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (self._alias, symbol, side.value, qty, price, fee, fill_ref))
        await self._db.execute(
            """INSERT INTO ibkr_etf_ledger (model_alias, kind, pnl, source_ref)
               VALUES (?, 'commission', ?, ?)""",
            (self._alias, -fee, f"{fill_ref}:commission"))
        if side == OrderSide.BUY:
            await self._db.execute(
                """INSERT INTO ibkr_etf_positions
                    (model_alias, symbol, quantity, avg_cost, current_price,
                     stop_price, initial_risk_usd)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (self._alias, symbol, qty, price, price, stop_price, initial_risk_usd))
        else:
            position = await self._db.fetchone(
                """SELECT quantity, avg_cost FROM ibkr_etf_positions
                   WHERE model_alias = ? AND symbol = ?""", (self._alias, symbol))
            if position is None:
                raise ValueError(f"cannot sell absent ETF paper position: {symbol}")
            realized = (price - float(position["avg_cost"])) * qty
            await self._db.execute(
                """INSERT INTO ibkr_etf_ledger (model_alias, kind, pnl, source_ref)
                   VALUES (?, 'trade', ?, ?)""",
                (self._alias, realized, f"{fill_ref}:trade"))
            await self._db.execute(
                "DELETE FROM ibkr_etf_positions WHERE model_alias = ? AND symbol = ?",
                (self._alias, symbol))

    async def _mirror(self, symbol: str, qty: float, entry: float,
                      current: float) -> None:
        await self._db.execute(
            """UPDATE ibkr_etf_positions SET current_price=?, unrealized_pnl=?,
                 updated_at=datetime('now')
               WHERE model_alias=? AND symbol=?""",
            (current, (current - entry) * qty, self._alias, symbol))

    async def run_once(self) -> int:
        if not self._s.ibkr.etf_paper_enabled or not self.market_open():
            return 0
        cfg = self._s.ibkr
        await self._ensure_state()
        entries = 0
        # Why a cycle produced no entry. Without this the arm reported only
        # "0 entries" and the cause was invisible: ibkr_etf_openai ran 382
        # evaluations to zero fills over four days (2026-07-27) and nothing in
        # the logs or the DB said the probability gate was rejecting 100% of
        # forecasts. A quiet arm must say which gate is binding.
        blocked: Counter[str] = Counter()
        positions = await self._positions()
        allocated = sum(qty * entry for qty, entry in positions.values())
        class_allocated: dict[str, float] = {}
        for symbol, (qty, entry) in positions.items():
            asset_class = self._asset_class(symbol)
            class_allocated[asset_class] = class_allocated.get(asset_class, 0.0) + qty * entry
        position_count = len(positions)
        daily_loss_hit = await self._daily_equity_change() <= -cfg.etf_daily_loss_limit_usd
        unmarked_positions = set(positions)

        # Refresh only a bounded slice per cycle so a 28-instrument universe
        # cannot trigger 28 LLM calls at once. Held positions get first claim;
        # the remaining slots rotate through the opportunity set.
        symbols = list(dict.fromkeys(cfg.etf_symbols))
        held_order = [s for s in symbols if s in positions]
        candidates = [s for s in symbols if s not in positions]
        if candidates:
            cursor = self._refresh_cursor % len(candidates)
            candidates = candidates[cursor:] + candidates[:cursor]
            self._refresh_cursor = (cursor + cfg.etf_max_signal_refreshes_per_cycle) % len(candidates)
            await self._save_cursor()
        refresh_order = held_order + candidates
        refresh_set = set(refresh_order[:cfg.etf_max_signal_refreshes_per_cycle])

        # Held positions must receive fresh marks before any new risk is opened.
        for symbol in held_order + candidates:
            try:
                quote = await self._client.get_quote(symbol)
            except Exception as exc:  # noqa: BLE001
                log.warning("ibkr_etf.paper.quote_error", symbol=symbol,
                            error=str(exc)[:100])
                continue
            quote_age = time.time() - quote.timestamp if quote is not None else None
            if quote is None or quote_age is None or quote_age < -60 or quote_age > 20 * 60:
                continue
            if getattr(quote, "source", "ibkr_live") not in {"ibkr_live", "alpaca_iex"}:
                continue
            from auramaur.data_edge import DataDelivery, record_delivery
            quote_at = datetime.fromtimestamp(float(quote.timestamp), timezone.utc)
            await record_delivery(self._db, DataDelivery(
                strategy=self.name, component="equity_quote", status="ok",
                provider=getattr(quote, "source", "ibkr_live"),
                market_id=symbol, source_at=quote_at, item_count=1,
                required_fields=("bid", "ask"),
            ))
            mid = (quote.bid + quote.ask) / 2
            await self._resolve_forecasts(symbol)
            spread_bps = (quote.ask - quote.bid) / mid * 10_000
            held = positions.get(symbol)
            # The spread gate bounds ENTRY quality only. It used to sit above
            # the held-position branch, so any widening past 20 bps — an
            # auction, a thin tape, a stress print — silently disabled
            # stop_loss, take_profit, trailing_stop and llm_bearish for a
            # position already at risk, exactly when an exit matters most.
            # The multiasset book already marks and exits held positions
            # before its own spread check; match it.
            if spread_bps > cfg.etf_max_spread_bps and not held:
                continue
            view = await self._view(
                symbol, allow_refresh=symbol in refresh_set, reference_price=mid)
            prob = view[0] if view else None
            confidence = view[1] if view else ""
            conf_ok = bool(view) and self._CONF.get(
                confidence.upper(), 0) >= self._min_conf_rank
            if held:
                qty, entry = held
                gain = (quote.bid - entry) / entry * 100
                peak = await self._peak(symbol, gain)
                risk_row = await self._db.fetchone(
                    """SELECT stop_price FROM ibkr_etf_positions
                         WHERE model_alias=? AND symbol=?""", (self._alias, symbol))
                stored_stop = float(risk_row["stop_price"] or 0) if risk_row else 0
                proposal = exit_proposal(
                    symbol=symbol, quantity=qty, entry_price=entry,
                    bid=quote.bid, ask=quote.ask, stored_stop=stored_stop,
                    peak_gain_pct=peak, probability=prob,
                    stop_loss_pct=cfg.etf_stop_loss_pct,
                    take_profit_pct=cfg.etf_take_profit_pct,
                    trailing_stop_pct=cfg.etf_trailing_stop_pct,
                    exit_probability=cfg.etf_exit_prob,
                    slippage_bps=cfg.etf_slippage_bps)
                if proposal:
                    await self._fill(symbol, OrderSide.SELL, proposal.quantity,
                                     proposal.price)
                    self._cooldown[symbol] = time.time() + (
                        cfg.etf_reentry_cooldown_hours * 3600)
                    await self._db.execute(
                        """INSERT INTO ibkr_etf_cooldowns
                           (model_alias, symbol, until_epoch) VALUES (?, ?, ?)
                           ON CONFLICT(model_alias, symbol) DO UPDATE SET
                             until_epoch=excluded.until_epoch""",
                        (self._alias, symbol, self._cooldown[symbol]))
                    cost = qty * entry
                    allocated -= cost
                    asset_class = self._asset_class(symbol)
                    class_allocated[asset_class] = max(
                        0.0, class_allocated.get(asset_class, 0.0) - cost)
                    position_count -= 1
                    unmarked_positions.discard(symbol)
                    log.info("ibkr_etf.paper.exit", symbol=symbol,
                             reason=proposal.reason,
                             model_alias=self._alias,
                             gain_pct=round(gain, 2),
                             probability=(round(prob, 3) if prob is not None else None))
                else:
                    await self._mirror(symbol, qty, entry, quote.bid)
                    unmarked_positions.discard(symbol)
                daily_loss_hit = await self._daily_equity_change() <= -cfg.etf_daily_loss_limit_usd
            elif view is None:
                blocked["no_view"] += 1
            elif not conf_ok:
                blocked["confidence"] += 1
            elif prob < self._min_prob:
                blocked["probability"] += 1
            elif daily_loss_hit:
                blocked["daily_loss"] += 1
            elif unmarked_positions:
                blocked["unmarked_position"] += 1
            elif position_count >= cfg.etf_max_positions:
                blocked["max_positions"] += 1
            elif time.time() < self._cooldown.get(symbol, 0):
                blocked["cooldown"] += 1
            else:
                deployment_cap = cfg.etf_paper_budget_usd * cfg.etf_max_deployment_pct / 100.0
                asset_class = self._asset_class(symbol)
                class_cap = cfg.etf_paper_budget_usd * cfg.etf_max_asset_class_pct / 100.0
                closes_raw = await self._client.get_adjusted_daily_closes(symbol)
                closes = [float(close) for _, close in closes_raw if close and close > 0]
                annual_vol = annualized_volatility(closes)
                momentum = normalized_momentum(closes)
                if annual_vol is None or momentum is None or momentum <= 0:
                    blocked["momentum"] += 1
                    continue
                risk_budget = cfg.etf_paper_budget_usd * cfg.etf_risk_per_position_pct / 100
                open_risk = await self._db.fetchone(
                    """SELECT COALESCE(SUM(initial_risk_usd), 0) AS risk
                         FROM ibkr_etf_positions WHERE model_alias=?""", (self._alias,))
                max_risk = cfg.etf_paper_budget_usd * cfg.etf_max_portfolio_risk_pct / 100
                proposal = entry_proposal(
                    symbol=symbol, bid=quote.bid, ask=quote.ask,
                    probability=prob, confidence=confidence,
                    confidence_rank=self._CONF.get(confidence.upper(), 0),
                    # Per-arm, not global: entry_proposal re-checks both
                    # thresholds, so passing cfg.* here would silently veto any
                    # per-arm override the outer gate just allowed through.
                    min_confidence_rank=self._min_conf_rank,
                    min_probability=self._min_prob,
                    annual_volatility=annual_vol, momentum=momentum,
                    remaining_deployment_usd=deployment_cap - allocated,
                    remaining_class_usd=class_cap - class_allocated.get(asset_class, 0.0),
                    max_entry_usd=cfg.etf_max_entry_usd,
                    fee_usd=cfg.etf_fee_per_order_usd,
                    slippage_bps=cfg.etf_slippage_bps,
                    stop_vol_multiple=cfg.etf_stop_vol_multiple,
                    min_stop_pct=cfg.etf_min_stop_pct,
                    risk_budget_usd=risk_budget,
                    open_risk_usd=float(open_risk["risk"] or 0),
                    max_portfolio_risk_usd=max_risk,
                    controls_allow_entry=True)
                if proposal is None:
                    blocked["sizing"] += 1
                    continue
                await self._fill(symbol, OrderSide.BUY, proposal.quantity,
                                 proposal.price, stop_price=proposal.stop_price,
                                 initial_risk_usd=proposal.initial_risk_usd)
                await self._mirror(symbol, proposal.quantity, proposal.price, quote.bid)
                actual_cost = proposal.quantity * proposal.price + cfg.etf_fee_per_order_usd
                allocated += actual_cost
                class_allocated[asset_class] = (
                    class_allocated.get(asset_class, 0.0) + actual_cost)
                position_count += 1
                entries += 1
                log.info("ibkr_etf.paper.entry", symbol=symbol, usd=round(actual_cost, 2),
                         model_alias=self._alias,
                         probability=round(prob, 3), confidence=confidence,
                         spread_bps=round(spread_bps, 2))
        await self._db.commit()
        # Unconditional cycle log (repo convention, #285): a quiet log must be
        # distinguishable from a dead task.
        log.info("ibkr_etf.cycle", model_alias=self._alias,
                 positions=position_count, entries=entries,
                 min_prob=self._min_prob, min_conf_rank=self._min_conf_rank,
                 blocked=dict(blocked),
                 blocked_by=(blocked.most_common(1)[0][0] if blocked else None))
        return entries
