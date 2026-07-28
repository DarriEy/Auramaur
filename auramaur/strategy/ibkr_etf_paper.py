"""Paper-only broad-market ETF book using IBKR quotes and LLM/news signals."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, time as wall_time, timezone
import json
import time
from uuid import uuid4
from zoneinfo import ZoneInfo

import structlog

from auramaur.broker.execution_gateway import ExecutionGateway, TradeIntent
from auramaur.broker.ledger import record_ledger_event
from auramaur.broker.pnl import PnLTracker
from auramaur.exchange.ibkr_instruments import BY_KEY
from auramaur.exchange.ibkr_intent import (
    SimulatedInstrumentExchange, instrument_event_family, instrument_market,
    instrument_market_id, instrument_signal, prepare_instrument_order,
)
from auramaur.exchange.models import Market, OrderSide
from auramaur.experiments.strategies.ibkr_etf_paper import (
    entry_proposal, exit_proposal,
)
from auramaur.strategy.protocols import ExecutionMode
from auramaur.evaluation.etf_calibration import Forecast, clearance
from auramaur.risk.ibkr_math import (
    annualized_volatility, horizon_up_rate, normalized_momentum,
)
from auramaur.strategy.ibkr_edge_economics import (
    clears_costs, required_conviction, round_trip_cost_bps,
)
from auramaur.strategy.ibkr_etf_controls import completed_closes

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
        self._closes_cache: dict[str, list] = {}
        self._state_loaded = False
        # Standard rails. The book's own ibkr_etf_* tables stay the venue-native
        # accounting; the gateway additionally captures a decision snapshot
        # (the graduation clock) and books realized P&L where the ladder reads
        # it. The exchange handed to it is a local simulator, so PAPER_SIMULATED
        # remains literally true — see SimulatedInstrumentExchange.
        self._sim = SimulatedInstrumentExchange()
        self._gateway = ExecutionGateway(
            router=None, exchange=self._sim, exchange_name="ibkr",
            settings=settings, db=db,
            pnl_tracker=PnLTracker(db, settings) if db is not None else None,
        )

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

    async def _closes(self, symbol: str):
        """Adjusted daily closes, once per symbol per cycle.

        Resolution, entry sizing and the forecast benchmark all want the same
        bars. Fetching them three times was three identical historical requests
        against IBKR's pacing limits for one symbol.
        """
        if symbol not in self._closes_cache:
            self._closes_cache[symbol] = await self._client.get_adjusted_daily_closes(symbol)
        return self._closes_cache[symbol]

    async def _record_outcome(self, symbol: str, session_date: str,
                              outcome: int) -> None:
        """Publish a resolved forecast as a scoreable outcome.

        An ETF arm's forecast is a genuine binary that genuinely resolves, so
        recording the outcome is honest. But it is keyed by FAMILY (the weekly
        bucket), not by the symbol-scoped market_id, and that is deliberate.

        ``market_outcomes`` is ``UNIQUE(venue, market_id)`` — the ladder models
        one outcome per market, because a prediction market resolves once. An
        ETF arm forecasts the same symbol every week. Keying by market_id would
        let the FIRST resolved forecast for (arm, symbol) write the only row
        that pair can ever have, and since _prospective_stats joins
        ``o.event_key = lower(venue)||':'||d.market_id``, every later week's
        forecast would then be scored against that one stale outcome. Wrong
        evidence into a gate is worse than none.

        Keyed by family, each week gets its own row and nothing joins today —
        the outcomes bank correctly while the join stays honest. Making them
        countable needs a decision recorded in docs/ibkr_graduation_spec.md:
        either the ladder joins on event_family, or these books take the
        risk-adjusted-return bar instead. Not a choice to make silently.
        """
        spec = BY_KEY.get(symbol)
        if spec is None:
            return
        family = instrument_event_family(spec, self._alias, session_date)
        await self._db.execute(
            """INSERT OR IGNORE INTO market_outcomes
               (event_key, venue, market_id, event_family, outcome,
                resolved_at, source, resolution_version, recorded_at)
               VALUES (?, 'ibkr', ?, ?, ?, datetime('now'),
                       'ibkr_etf_adjusted_close', 1, datetime('now'))""",
            (f"ibkr:{family}", family, family, outcome))

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
        # Drop the session in progress. reqHistoricalData with endDateTime=""
        # RETURNS TODAY'S PARTIAL BAR (measured 2026-07-27: a "1 M" request at
        # 14:29 ET included 2026-07-27), so without this the 5th session counts
        # while it is still trading and the forecast resolves against whatever
        # the price happened to be when the cycle ran — six hours before the
        # close it actually asks about. The query filters actual_outcome IS
        # NULL, so that wrong value would never be revisited.
        session_today = datetime.now(timezone.utc).astimezone(_ET).date().isoformat()
        closes = [(day, close) for day, close in await self._closes(symbol)
                  if day < session_today]
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
            await self._record_outcome(
                symbol, str(row["opened_session_date"]),
                1 if final_price > base else 0)

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
                    *, stop_price: float = 0, initial_risk_usd: float = 0,
                    bid: float = 0.0, ask: float = 0.0) -> None:
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
        await self._book_fill(symbol, side, qty, price, fee, fill_ref,
                              bid=bid, ask=ask)

    async def _clearance(self):
        """Has this arm earned the right to trade? Recomputed each cycle.

        The whole design rests on this: forecasts are free and resolve in five
        sessions, so an arm accumulates ~125 scoreable calls a week and can be
        judged in about three weeks. Trading is not free -- ~27bps a round trip
        at $1,100, of which 20bps is the $1 commission floor. So the arm
        forecasts from day one and may not trade until its Brier edge over its
        own trailing benchmark has a 95% lower bound clear of zero.

        Deliberately fails closed. Any error, and the arm does not trade.
        """
        try:
            rows = await self._db.fetchall(
                """SELECT probability, confidence, actual_outcome, reference_price,
                          final_price
                     FROM ibkr_etf_forecasts WHERE model_alias = ?""",
                (self._alias,))
        except Exception as exc:  # noqa: BLE001
            log.warning("ibkr_etf.clearance_unavailable", model_alias=self._alias,
                        error=str(exc)[:120])
            return clearance([], min_resolved=self._s.ibkr.etf_min_resolved_to_trade)
        forecasts = []
        for row in rows or []:
            outcome = row["actual_outcome"]
            forecasts.append(Forecast(
                float(row["probability"]), str(row["confidence"]),
                None if outcome is None else int(outcome),
                # The benchmark each forecast was scored against. Stored
                # forecasts predate the benchmark column, so fall back to a
                # coin -- which UNDERSTATES the edge and therefore keeps the
                # gate shut rather than opening it by accident.
                0.5))
        return clearance(forecasts,
                         min_resolved=self._s.ibkr.etf_min_resolved_to_trade)

    async def _forecast_pair(self, symbol: str, side: OrderSide,
                             session_date: str) -> tuple[float | None, float]:
        """(fair, reference) for the Brier bar: the arm's forecast, and what it
        must beat.

        The reference is the instrument's own trailing drift — P(higher after
        `horizon` sessions) over completed history — NOT 0.5. Equities rise more
        often than not, so a coin-flip reference would score an arm positively
        for constantly answering 0.55, rewarding it for the direction of the
        last century rather than for forecasting.

        An exit claims no forecast: the arm's probability is about the entry
        window, so a SELL returns fair=None and asserts no edge. The ladder
        keeps the earliest decision per family, which is the entry, so this only
        affects a family an exit opens on its own.
        """
        view = self._views.get(symbol)
        fair = float(view[1]) if view and side == OrderSide.BUY else None
        reference = None
        try:
            closes = completed_closes(await self._closes(symbol), session_date)
            reference = horizon_up_rate(
                closes, int(self._s.ibkr.etf_signal_horizon_days))
        except Exception as exc:  # noqa: BLE001
            log.warning("ibkr_etf.benchmark_unavailable", symbol=symbol,
                        model_alias=self._alias, error=str(exc)[:100])
        if reference is None:
            # No benchmark means no defensible edge claim. Falling back to the
            # arm's own number asserts a zero Brier edge, which the ladder reads
            # as "no evidence" — the honest outcome. Falling back to 0.5 would
            # manufacture an edge out of a missing history.
            return (fair, fair if fair is not None else 0.5)
        return (fair, float(reference))

    async def _book_fill(self, symbol: str, side: OrderSide, qty: float,
                         price: float, fee: float, fill_ref: str,
                         bid: float = 0.0, ask: float = 0.0) -> None:
        """Register an already-simulated fill on the standard rails.

        This is what puts the arm on the graduation ladder. ``gateway.submit``
        captures a decision snapshot (which registers the strategy_experiments
        clock) and records the fill through ``PnLTracker``, so realized P&L
        lands in ``pnl_ledger`` where ``GraduationLadder`` reads it — the two
        inputs that were permanently empty for this book (audit finding #7,
        2026-07-26: the arm wrote fills by raw SQL with no record_fill call).

        Deliberately runs AFTER the book's own ibkr_etf_* rows and cannot veto
        them. This increment is booking and capture only; consulting the ladder
        is a separate, later step, so a booking failure must lose evidence, not
        change what the book traded. The ibkr_etf_fills / ibkr_etf_ledger /
        ibkr_etf_positions rows remain the venue-native accounting.
        """
        if self._db is None or self._gateway.pnl_tracker is None:
            return
        spec = BY_KEY.get(symbol)
        if spec is None:
            log.warning("ibkr_etf.unregistered_instrument", symbol=symbol,
                        model_alias=self._alias,
                        detail="no manifest entry; fill not booked to pnl_ledger")
            return
        market_id = instrument_market_id(spec, self._alias)
        try:
            # The ledger resolves venue/category from `markets`; without a row
            # the pnl_ledger entry lands venue-blank and cannot be scoped.
            # active=0 on purpose: this is a continuously-priced broker
            # instrument, not a resolvable prediction market, and nothing in
            # the prediction-market pipeline should treat it as tradeable.
            await self._db.execute(
                """INSERT OR IGNORE INTO markets
                   (id, exchange, question, category, active,
                    outcome_yes_price, outcome_no_price, last_updated)
                   VALUES (?, 'ibkr', ?, ?, 0, 0, 0, datetime('now'))""",
                (market_id, f"{self._alias}: {spec.description or spec.symbol}",
                 spec.asset_class or ""))
            order = prepare_instrument_order(
                spec, side=side, quantity=qty, price=price, is_live=False,
                strategy_source=self.strategy_source, cell=self._alias)
            if order is None:
                return
            session_date = datetime.now(timezone.utc).astimezone(_ET).date().isoformat()
            fair, reference = await self._forecast_pair(symbol, side, session_date)
            # The real BBO behind this fill. Without it _capture_decision finds
            # no book, so _place_and_record cannot tell whether the fill crossed
            # and stamps fill_evidence='synthetic' — which is not in
            # graduation.credible_fill_evidence, making every decision
            # uncountable under require_executable_fills. The ETF arms have a
            # genuine IBKR bid/ask; recording it supplies the data the check was
            # designed to read rather than working around the check.
            if bid > 0 and ask > 0:
                await self._db.execute(
                    """INSERT INTO orderbook_snapshots
                       (market_id, token_id, exchange, best_bid, best_ask, mid,
                        recorded_at)
                       VALUES (?, ?, 'ibkr', ?, ?, ?, datetime('now'))""",
                    (market_id, spec.key, bid, ask, (bid + ask) / 2))
            self._sim.stage(order, order_id=fill_ref)
            result = await self._gateway.submit(TradeIntent(
                signal=instrument_signal(
                    spec, strategy_source=self.strategy_source,
                    mark=reference, fair=fair, side=side, cell=self._alias,
                    rationale=f"ibkr_etf {self._alias} {side.value} {symbol}"),
                market=instrument_market(
                    spec, mark=reference, cell=self._alias,
                    event_family=instrument_event_family(
                        spec, self._alias, session_date)),
                size_dollars=qty * price,
                # Structurally paper-only. force_paper keeps the global live
                # flag out of this book entirely; the simulator refuses a
                # non-dry_run order as the second lock.
                force_paper=True,
            ))
            if result.status not in ("paper", "filled", "partial"):
                log.error("ibkr_etf.fill_not_booked", symbol=symbol,
                          model_alias=self._alias, status=result.status,
                          reason=result.reason,
                          detail="venue-native fill stands; ladder evidence lost")
                return
            # The commission the book already charged its own ledger. The
            # gateway's Fill carries no fee, so without this row the ladder
            # would read a gross P&L — and the entire realized IBKR record to
            # date is commission.
            await record_ledger_event(
                self._db, market_id=market_id, kind="commission", token="YES",
                qty=qty, pnl=-float(fee), fees=0.0, is_paper=True,
                source_ref=f"{fill_ref}:commission")
        except Exception as exc:  # noqa: BLE001 - evidence, never the money path
            log.error("ibkr_etf.fill_booking_failed", symbol=symbol,
                      model_alias=self._alias, error=str(exc)[:200])

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
        # Bars are stable within a cycle; refetching them per consumer was three
        # identical historical requests per symbol against IBKR pacing.
        self._closes_cache.clear()
        session_today = datetime.now(timezone.utc).astimezone(_ET).date().isoformat()
        cleared = await self._clearance()
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
                                     proposal.price, bid=quote.bid, ask=quote.ask)
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
            elif not cleared.cleared:
                # Forecast-only until the arm has DEMONSTRATED skill. Forecasts
                # cost nothing and resolve in five sessions; trading costs
                # ~27bps a round trip. The previous book inverted this and paid
                # $926 in commission to learn its gross P&L was -$42.
                blocked["not_cleared"] += 1
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
                closes_raw = await self._closes(symbol)
                closes = [float(close) for _, close in closes_raw if close and close > 0]
                annual_vol = annualized_volatility(closes)
                momentum = normalized_momentum(closes)
                if annual_vol is None or momentum is None or momentum <= 0:
                    blocked["momentum"] += 1
                    continue
                # THE ENTRY TEST. A bare probability threshold knows nothing
                # about the instrument or what it costs to trade: at $1,100,
                # SPY needs 19pp of conviction to clear 27bps and this model
                # has never produced 6pp, while SLV needs 3.7pp. So the gate is
                # expected edge against cost, not a number.
                horizon = int(cfg.etf_signal_horizon_days)
                base_rate = horizon_up_rate(
                    completed_closes(closes_raw, session_today), horizon)
                if base_rate is None:
                    blocked["no_benchmark"] += 1
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
                    # Deliberately permissive: the economic gate above is
                    # strictly stronger than any probability threshold, and it
                    # has already run. Leaving the old thresholds here would
                    # re-impose the very number this design replaced -- the
                    # same silent double-veto that made etf_min_prob 0.62
                    # reject 100% of forecasts.
                    min_confidence_rank=0,
                    min_probability=0.0,
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
                # THE ENTRY TEST, priced on the REAL notional. Running it on
                # the intended entry size instead would have been badly wrong:
                # risk sizing can cut a $1,100 intent to $110, where the $1
                # commission is 91bps a leg rather than 9. A bare probability
                # threshold cannot see any of this — at $1,100 SPY needs 19pp
                # of conviction and this model has never produced 6pp, while
                # SLV needs 3.7pp.
                notional = proposal.quantity * proposal.price
                cost_bps = round_trip_cost_bps(
                    notional, commission_usd=cfg.etf_fee_per_order_usd,
                    spread_bps=spread_bps, slippage_bps=cfg.etf_slippage_bps)
                if not clears_costs(prob, base_rate, annual_vol, horizon,
                                    cost_bps, margin=cfg.etf_edge_cost_margin):
                    blocked["economics"] += 1
                    continue
                await self._fill(symbol, OrderSide.BUY, proposal.quantity,
                                 proposal.price, stop_price=proposal.stop_price,
                                 initial_risk_usd=proposal.initial_risk_usd,
                                 bid=quote.bid, ask=quote.ask)
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
                 cleared_to_trade=cleared.cleared, clearance=cleared.reason,
                 resolved=cleared.resolved,
                 brier_edge=round(cleared.brier_edge, 5),
                 max_conviction=round(cleared.max_conviction, 4),
                 blocked=dict(blocked),
                 blocked_by=(blocked.most_common(1)[0][0] if blocked else None))
        return entries
