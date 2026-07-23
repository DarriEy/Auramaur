"""Six isolated, read-only IBKR market-data / local paper-execution books."""

from __future__ import annotations

from datetime import datetime, time as wall_time, timezone
import json
import math
import time
from uuid import uuid4
from zoneinfo import ZoneInfo

import structlog

from auramaur.exchange.ibkr_instruments import (
    BY_BOOK, BY_KEY, ContractKind, IBKRBook, InstrumentSpec,
)
from auramaur.strategy.protocols import ExecutionMode
from auramaur.risk.ibkr_math import (
    adverse_fill, annualized_volatility, closes_from_bars,
    normalized_momentum, risk_quantity, stop_distance,
)

log = structlog.get_logger()


async def warn_stranded_positions(db, enabled_books: set[str]) -> list[str]:
    """Warn for paper positions stranded in books that are NOT running.

    A disabled book never cycles, so its open positions are never re-marked
    and never exit-managed (2026-07-22: a SHEL.L row sat frozen in the held
    international_equity book from the day of the hold). Visibility only —
    flattening stays an operator decision, because a temporary hold must not
    destroy positions. Returns the stranded book names (for tests)."""
    stranded: list[str] = []
    try:
        rows = await db.fetchall(
            "SELECT book, COUNT(*) AS n FROM ibkr_paper_positions GROUP BY book")
    except Exception as e:
        log.debug("ibkr_multiasset.stranded_check_error", error=str(e))
        return stranded
    for row in rows or []:
        book = str(row["book"] or "")
        if book and book not in enabled_books and int(row["n"] or 0) > 0:
            stranded.append(book)
            log.warning(
                "ibkr_multiasset.stranded_positions",
                book=book, count=int(row["n"]),
                detail="book disabled — positions stay unmarked and "
                       "unmanaged until the book is re-enabled or the "
                       "operator closes them")
    return stranded


class IBKRMultiAssetPaperBook:
    """One asset-specific ledger; this class has no broker order capability."""

    execution_mode = ExecutionMode.PAPER_SIMULATED

    def __init__(self, settings, client, db, book: IBKRBook,
                 rates_provider=None, executor=None):
        self._settings = settings
        self._client = client
        self._db = db
        self.book = book
        self.name = f"ibkr_{book.value}_paper"
        self._rates_provider = rates_provider
        self._executor = executor

    @property
    def config(self):
        return self._settings.ibkr.multiasset_books[self.book.value]

    @staticmethod
    def _weekday_session(zone: str, start: wall_time, end: wall_time,
                         now: datetime) -> bool:
        local = now.astimezone(ZoneInfo(zone))
        return local.weekday() < 5 and start <= local.time() < end

    def market_open(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        if self.book is IBKRBook.FX:
            et = now.astimezone(ZoneInfo("America/New_York"))
            return not (et.weekday() == 4 and et.time() >= wall_time(17)) and not (
                et.weekday() == 5 or (et.weekday() == 6 and et.time() < wall_time(17)))
        if self.book is IBKRBook.FUTURES:
            et = now.astimezone(ZoneInfo("America/New_York"))
            if et.weekday() == 5 or (et.weekday() == 4 and et.time() >= wall_time(17)):
                return False
            if et.weekday() == 6 and et.time() < wall_time(18):
                return False
            return not (wall_time(17) <= et.time() < wall_time(18))
        if self.book is IBKRBook.INTERNATIONAL_EQUITY:
            # Individual closed venues simply return no fresh BBO; the task is
            # active whenever at least one major configured region is open.
            sessions = (
                ("America/Toronto", wall_time(9, 30), wall_time(16)),
                ("Europe/London", wall_time(8), wall_time(16, 30)),
                ("Europe/Berlin", wall_time(9), wall_time(17, 30)),
                ("Asia/Tokyo", wall_time(9), wall_time(15, 30)),
                ("Asia/Hong_Kong", wall_time(9, 30), wall_time(16)),
                ("Australia/Sydney", wall_time(10), wall_time(16)),
            )
            return any(self._weekday_session(zone, start, end, now)
                       for zone, start, end in sessions)
        return self._weekday_session("America/New_York", wall_time(9, 30),
                                     wall_time(16), now)

    async def _positions(self):
        rows = await self._db.fetchall(
            "SELECT * FROM ibkr_paper_positions WHERE book = ?", (self.book.value,))
        return {row["instrument_key"]: row for row in rows}

    async def _loss_exposure(self) -> float:
        """Today's realized P&L plus the latest marks on every open position."""
        row = await self._db.fetchone(
            """SELECT
                 (SELECT COALESCE(SUM(pnl_usd), 0) FROM ibkr_paper_ledger
                   WHERE book = ? AND date(realized_at) = date('now'))
                 +
                 (SELECT COALESCE(SUM(unrealized_pnl_usd), 0)
                    FROM ibkr_paper_positions WHERE book = ?) AS pnl""",
            (self.book.value, self.book.value))
        return float(row["pnl"] or 0)

    def _quote_fresh(self, quote) -> bool:
        age = time.time() - float(quote.timestamp)
        return (getattr(quote, "source", "") == "ibkr_live"
                and 0 <= age <= self._settings.ibkr.multiasset_max_quote_age_seconds)

    @staticmethod
    def _serialize_spec(spec: InstrumentSpec) -> str:
        values = {name: getattr(spec, name) for name in spec.__dataclass_fields__}
        values["book"] = spec.book.value
        values["kind"] = spec.kind.value
        return json.dumps(values, sort_keys=True)

    @staticmethod
    def _position_spec(row) -> InstrumentSpec | None:
        known = BY_KEY.get(row["instrument_key"])
        if known is not None:
            return known
        raw = row["instrument_spec_json"]
        if not raw:
            return None
        values = json.loads(raw)
        values["book"] = IBKRBook(values["book"])
        values["kind"] = ContractKind(values["kind"])
        return InstrumentSpec(**values)

    @staticmethod
    def _momentum(bars) -> float | None:
        return normalized_momentum(closes_from_bars(bars))

    @staticmethod
    def _commission(spec, quantity: float, notional_usd: float) -> float:
        if spec.kind is ContractKind.FOREX:
            return max(0.20, notional_usd * 0.00002)
        if spec.kind is ContractKind.FUTURE:
            return max(1.0, quantity * 2.50)
        if spec.kind is ContractKind.OPTION:
            return max(1.0, quantity * 0.65)
        if spec.kind is ContractKind.BOND:
            return max(1.0, quantity)
        return max(1.0, min(notional_usd * 0.001, 10.0))

    @staticmethod
    def _capital_per_unit(spec, price: float, fx: float) -> float:
        if spec.paper_capital_per_unit_usd > 0:
            return spec.paper_capital_per_unit_usd
        return price * spec.multiplier * fx

    @staticmethod
    def _quantity(spec, target_usd: float, unit_capital: float) -> float:
        raw = target_usd / unit_capital
        if spec.kind is ContractKind.STOCK:
            return math.floor(raw * 10000) / 10000
        return float(math.floor(raw))

    async def _fill(self, spec, quote, side: str, quantity: float,
                    fx: float, entry_price: float | None = None,
                    stop_price: float = 0, initial_risk_usd: float = 0) -> None:
        execution_ref = ""
        broker_price = 0.0
        requested_quantity = quantity
        if (self._executor is not None
                and not self._executor.gate_reason(self.book)
                and await self._executor.graduated(self.book)):
            result = await self._executor.place(spec, side, quantity)
            if not result.accepted:
                raise RuntimeError(f"IBKR live execution rejected: {result.reason}")
            quantity = result.filled_quantity
            if requested_quantity > 0 and quantity < requested_quantity:
                initial_risk_usd *= quantity / requested_quantity
            broker_price = result.filled_price
            execution_ref = result.execution_ref
        price = broker_price or adverse_fill(
            quote.bid, quote.ask, side, self.config.slippage_bps)
        position = None
        held_quantity = quantity
        if side == "SELL":
            position = await self._db.fetchone(
                "SELECT quantity, entry_commission_usd, entry_fill_ref, opened_at "
                "FROM ibkr_paper_positions WHERE book=? AND instrument_key=?",
                (self.book.value, spec.key))
            held_quantity = float(position["quantity"] or 0) if position else quantity
            if quantity > held_quantity + 1e-9:
                log.critical("ibkr_multiasset.live_overfill",
                             book=self.book.value, key=spec.key,
                             filled=quantity, tracked=held_quantity)
            quantity = min(quantity, held_quantity)
        capital = quantity * self._capital_per_unit(spec, price, fx)
        fee = self._commission(spec, quantity, capital)
        fill_ref = execution_ref or f"ibkr-{self.book.value}-paper-{uuid4().hex}"
        await self._db.execute(
            """INSERT INTO ibkr_paper_fills
               (book, instrument_key, con_id, side, quantity, multiplier, price,
                currency, fx_to_usd, commission_usd, price_source, fill_ref)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (self.book.value, spec.key, quote.con_id, side, quantity,
             quote.multiplier, price, spec.currency, fx, fee, quote.source, fill_ref))
        await self._db.execute(
            "INSERT INTO ibkr_paper_ledger (book, kind, pnl_usd, source_ref) VALUES (?, 'commission', ?, ?)",
            (self.book.value, -fee, f"{fill_ref}:commission"))
        if side == "BUY":
            await self._db.execute(
                """INSERT INTO ibkr_paper_positions
                   (book, instrument_key, con_id, currency, quantity, multiplier,
                    fx_to_usd, avg_cost, current_price, price_source,
                    instrument_spec_json, stop_price, initial_risk_usd,
                    entry_commission_usd, entry_fill_ref)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (self.book.value, spec.key, quote.con_id, spec.currency, quantity,
                 quote.multiplier, fx, price, price, quote.source,
                 self._serialize_spec(spec), stop_price, initial_risk_usd, fee,
                 fill_ref))
        else:
            pnl = (price - float(entry_price)) * quantity * quote.multiplier * fx
            await self._db.execute(
                "INSERT INTO ibkr_paper_ledger (book, kind, pnl_usd, source_ref) VALUES (?, 'trade', ?, ?)",
                (self.book.value, pnl, f"{fill_ref}:trade"))
            entry_fee = float(position["entry_commission_usd"] or 0) if position else 0.0
            allocated_entry_fee = (entry_fee * quantity / held_quantity
                                   if held_quantity > 0 else entry_fee)
            entry_fill_ref = str(position["entry_fill_ref"] or "") if position else ""
            if position is None:
                # Understates elapsed_days for the evidence contract; loud so a
                # recurring miss is investigated rather than absorbed.
                log.warning("ibkr_multiasset.round_trip_missing_entry",
                            book=self.book.value, instrument=spec.key)
            opened_at = position["opened_at"] if position else datetime.now(timezone.utc).isoformat()
            # OR IGNORE: exit_fill_ref is UNIQUE, so a crash-replayed exit books
            # the round trip once instead of failing the cycle.
            await self._db.execute(
                """INSERT OR IGNORE INTO ibkr_paper_round_trips
                   (book, instrument_key, entry_fill_ref, exit_fill_ref, gross_pnl_usd,
                    entry_commission_usd, exit_commission_usd, net_pnl_usd,
                    opened_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (self.book.value, spec.key, entry_fill_ref, fill_ref, pnl,
                 allocated_entry_fee, fee, pnl - allocated_entry_fee - fee, opened_at))
            remaining = max(0.0, held_quantity - quantity)
            if remaining > 1e-9:
                await self._db.execute(
                    """UPDATE ibkr_paper_positions
                          SET quantity=?, entry_commission_usd=?, current_price=?,
                              updated_at=datetime('now')
                        WHERE book=? AND instrument_key=?""",
                    (remaining, max(0.0, entry_fee - allocated_entry_fee), price,
                     self.book.value, spec.key))
            else:
                await self._db.execute(
                    "DELETE FROM ibkr_paper_positions WHERE book=? AND instrument_key=?",
                    (self.book.value, spec.key))
        await self._db.commit()
        if execution_ref:
            await self._executor.acknowledge(execution_ref)

    async def run_once(self) -> int:
        cfg = self.config
        if not self._settings.ibkr.multiasset_paper_enabled or not cfg.enabled:
            return 0
        if not hasattr(self._client, "is_market_open") and not self.market_open():
            return 0
        positions = await self._positions()
        state = await self._db.fetchone(
            "SELECT refresh_cursor FROM ibkr_paper_state WHERE book = ?",
            (self.book.value,))
        cursor = int(state["refresh_cursor"] or 0) if state else 0
        disabled = set(self._settings.ibkr.multiasset_disabled_instruments)
        eligible = None
        if self._settings.ibkr.multiasset_registry_required:
            from auramaur.exchange.ibkr_registry import eligible_keys
            eligible = await eligible_keys(self._db)
        universe = tuple(spec for spec in BY_BOOK[self.book]
                         if spec.key not in disabled
                         and (eligible is None or spec.key in eligible))
        held_specs = {}
        for key, row in positions.items():
            spec = self._position_spec(row)
            if spec is None:
                log.error("ibkr_multiasset.unmanageable_orphan", book=self.book.value,
                          key=key)
                continue
            held_specs[key] = spec
        if not universe and not held_specs:
            return 0
        count = min(len(universe), self._settings.ibkr.multiasset_refreshes_per_cycle)
        selected = [universe[(cursor + i) % len(universe)] for i in range(count)]
        # Open positions are always managed, even when outside the refresh slice.
        keys = {spec.key for spec in selected}
        selected.extend(spec for key, spec in held_specs.items() if key not in keys)
        # Mark/manage every held position before considering new risk.
        selected.sort(key=lambda spec: spec.key not in positions)
        unmarked_positions = set(positions)
        entries = 0
        error = ""
        candidates: list = []
        for spec in selected:
            try:
                held = positions.get(spec.key)
                held_con_id = int(held["con_id"]) if held else 0
                if hasattr(self._client, "is_market_open") and not await self._client.is_market_open(
                        spec, con_id=held_con_id):
                    continue
                quote = (await self._client.get_quote_by_con_id(
                    spec, held_con_id) if held
                    else await self._client.get_quote(spec))
                if quote is None or not self._quote_fresh(quote):
                    continue
                if held and int(quote.con_id) != held_con_id:
                    raise RuntimeError(
                        f"held contract mismatch: expected {held_con_id}, got {quote.con_id}")
                fx = await self._client.get_fx_to_usd(spec.currency)
                if held:
                    # Protective exits depend only on an executable live quote.
                    # Use the entry FX snapshot if the cross is temporarily absent.
                    mark_fx = float(fx if fx is not None else held["fx_to_usd"])
                    entry = float(held["avg_cost"])
                    gain_pct = (quote.bid / entry - 1) * 100
                    pnl = ((quote.bid - entry) * float(held["quantity"])
                           * quote.multiplier * mark_fx)
                    await self._db.execute(
                        """UPDATE ibkr_paper_positions SET current_price=?,
                           unrealized_pnl_usd=?, price_source=?, updated_at=datetime('now')
                           WHERE book=? AND instrument_key=?""",
                        (quote.bid, pnl, quote.source, self.book.value, spec.key))
                    unmarked_positions.discard(spec.key)
                    stored_stop = float(held["stop_price"] or 0)
                    hard_exit = ((stored_stop > 0 and quote.bid <= stored_stop)
                                 or gain_pct <= -cfg.stop_loss_pct
                                 or gain_pct >= cfg.take_profit_pct)
                    momentum_exit = False
                    if not hard_exit:
                        bars = await self._client.get_daily_bars_by_con_id(spec, held_con_id)
                        momentum = self._momentum(bars)
                        momentum_exit = (momentum is not None and momentum <=
                                         self._settings.ibkr.multiasset_exit_normalized_momentum)
                    if hard_exit or momentum_exit:
                        await self._fill(spec, quote, "SELL", float(held["quantity"]),
                                         mark_fx, entry_price=entry)
                        positions.pop(spec.key, None)
                    continue
                spread_bps = (quote.ask - quote.bid) / ((quote.ask + quote.bid) / 2) * 10_000
                if fx is None or spread_bps > cfg.max_spread_bps:
                    continue
                bars = await self._client.get_daily_bars(spec)
                closes = closes_from_bars(bars)
                momentum = normalized_momentum(closes)
                annual_vol = annualized_volatility(closes)
                if momentum is None or annual_vol is None:
                    continue
                if momentum < self._settings.ibkr.multiasset_min_normalized_momentum:
                    continue
                # Qualifiers are COLLECTED, not entered: with more qualifying
                # signals than slots, entry order must be strongest-first, not
                # rotation-cursor order (audit 2026-07-20).
                candidates.append((momentum, spec, quote, fx, annual_vol))
            except Exception as exc:  # noqa: BLE001
                error = str(exc)[:300]
                log.warning("ibkr_multiasset.instrument_error", book=self.book.value,
                            key=spec.key, error=error)

        candidates.sort(key=lambda item: -item[0])
        for momentum, spec, quote, fx, annual_vol in candidates:
            try:
                loss_exposure = await self._loss_exposure()
                if (unmarked_positions or loss_exposure <= -cfg.daily_loss_limit_usd
                        or len(positions) >= cfg.max_positions):
                    break
                deployed = sum(
                    float(row["quantity"]) * self._capital_per_unit(
                        held_specs.get(key) or BY_KEY[key],
                        float(row["avg_cost"]), float(row["fx_to_usd"]))
                    for key, row in positions.items())
                deployment_cap = cfg.budget_usd * cfg.max_deployment_pct / 100
                target = min(cfg.budget_usd * cfg.max_position_pct / 100,
                             deployment_cap - deployed)
                unit_capital = self._capital_per_unit(spec, quote.ask, fx)
                entry_price = adverse_fill(
                    quote.bid, quote.ask, "BUY", cfg.slippage_bps)
                distance = stop_distance(entry_price, annual_vol,
                                         cfg.stop_vol_multiple, cfg.min_stop_pct)
                risk_budget = cfg.budget_usd * cfg.risk_per_position_pct / 100
                class_risk = sum(
                    float(row["initial_risk_usd"] or 0)
                    for key, row in positions.items()
                    if (held_specs.get(key) or BY_KEY[key]).asset_class == spec.asset_class
                )
                class_risk_cap = cfg.budget_usd * cfg.max_asset_class_risk_pct / 100
                risk_budget = min(risk_budget, class_risk_cap - class_risk)
                quantity = min(
                    self._quantity(spec, target, unit_capital),
                    risk_quantity(risk_budget, distance, quote.multiplier, fx,
                                  fractional=spec.kind is ContractKind.STOCK),
                )
                if quantity <= 0:
                    continue
                initial_risk = quantity * distance * quote.multiplier * fx
                await self._fill(spec, quote, "BUY", quantity, fx,
                                 stop_price=entry_price - distance,
                                 initial_risk_usd=initial_risk)
                positions = await self._positions()
                entries += 1
            except Exception as exc:  # noqa: BLE001
                error = str(exc)[:300]
                log.warning("ibkr_multiasset.instrument_error", book=self.book.value,
                            key=spec.key, error=error)
        next_cursor = (cursor + count) % len(universe) if universe else 0
        await self._db.execute(
            """INSERT INTO ibkr_paper_state
               (book, refresh_cursor, last_cycle_at, last_success_at, last_error)
               VALUES (?, ?, datetime('now'), datetime('now'), ?)
               ON CONFLICT(book) DO UPDATE SET refresh_cursor=excluded.refresh_cursor,
                 last_cycle_at=excluded.last_cycle_at,
                 last_success_at=CASE WHEN excluded.last_error='' THEN datetime('now')
                                      ELSE last_success_at END,
                 last_error=excluded.last_error, updated_at=datetime('now')""",
            (self.book.value, next_cursor, error))
        await self._db.commit()
        await self._record_daily_mark()
        await self._record_research_signals()
        loss_exposure = await self._loss_exposure()
        log.info("ibkr_multiasset.cycle", book=self.book.value, entries=entries,
                 positions=len(positions), loss_exposure=round(loss_exposure, 2))
        return entries

    async def _record_daily_mark(self) -> None:
        """Upsert the current UTC date's marked-to-market equity.

        Idempotent per (book, date); the last cycle of the day wins, so the
        mark reflects end-of-day state without any scheduler. This is the
        observation stream for evaluate_ibkr_daily_evidence.
        """
        if self._db is None:
            return
        try:
            realized = await self._db.fetchone(
                "SELECT COALESCE(SUM(pnl_usd), 0) AS pnl FROM ibkr_paper_ledger "
                "WHERE book = ?", (self.book.value,))
            unrealized = await self._db.fetchone(
                "SELECT COALESCE(SUM(unrealized_pnl_usd), 0) AS pnl "
                "FROM ibkr_paper_positions WHERE book = ?", (self.book.value,))
            r = float(realized["pnl"] or 0)
            u = float(unrealized["pnl"] or 0)
            await self._db.execute(
                """INSERT INTO ibkr_paper_daily_marks
                   (book, mark_date, equity_usd, realized_cum_usd, unrealized_usd)
                   VALUES (?, date('now'), ?, ?, ?)
                   ON CONFLICT(book, mark_date) DO UPDATE SET
                     equity_usd=excluded.equity_usd,
                     realized_cum_usd=excluded.realized_cum_usd,
                     unrealized_usd=excluded.unrealized_usd,
                     marked_at=datetime('now')""",
                (self.book.value, r + u, r, u))
            await self._db.commit()
        except Exception as e:  # noqa: BLE001 - bookkeeping must not break the cycle
            log.warning("ibkr_multiasset.mark_error", book=self.book.value,
                      error=str(e)[:120])

    async def _record_research_signals(self) -> None:
        """Once daily for the FX book: record execution-free comparator signals.

        Records carry+trend (when a rates provider is wired and both legs
        resolve) beside the live book's own trend signal, so the comparative
        record exists BEFORE anything is wired into entries — the
        pre-registered upgrade path (docs/ibkr_multiasset_paper.md).
        """
        if self._db is None or self.book is not IBKRBook.FX:
            return
        try:
            row = await self._db.fetchone(
                "SELECT 1 AS x FROM ibkr_research_signals "
                "WHERE signal_date = date('now') LIMIT 1")
            if row is not None:
                return  # today already recorded
            from datetime import datetime, timedelta, timezone

            from auramaur.research.ibkr_signals import PricePoint, fx_carry_trend
            now = datetime.now(timezone.utc)
            # Gather all bars/rates over the network FIRST; the write+commit
            # span at the end stays pure-DB so the shared connection never
            # holds an open transaction across gateway/rates awaits.
            signal_rows: list[tuple] = []
            for spec in BY_BOOK[self.book]:
                try:
                    bars = await self._client.get_daily_bars(spec)
                    closes = closes_from_bars(bars)
                    momentum = normalized_momentum(closes)
                    if momentum is not None:
                        signal_rows.append(
                            (spec.key, "trend_normalized",
                             1 if momentum > 0 else -1 if momentum < 0 else 0,
                             abs(momentum), f"normalized_momentum={momentum:.4f}"))
                    if self._rates_provider is None or len(spec.key) < 6:
                        continue
                    base_rate = await self._rates_provider.rate(spec.key[:3])
                    quote_rate = await self._rates_provider.rate(spec.key[3:6])
                    if base_rate is None or quote_rate is None:
                        continue
                    window = closes[-121:]
                    points = [
                        PricePoint(
                            observed_at=now - timedelta(days=len(window) - 1 - i),
                            value=c)
                        for i, c in enumerate(window)]
                    signal = fx_carry_trend(points, as_of=now, base_rate=base_rate,
                                            quote_rate=quote_rate)
                    signal_rows.append(
                        (spec.key, "fx_carry_trend", signal.direction,
                         signal.score, signal.rationale[:200]))
                except Exception as e:  # noqa: BLE001 - one pair must not stop the sweep
                    log.warning("ibkr_multiasset.research_signal_error",
                              key=spec.key, error=str(e)[:100])
            for key, name, direction, strength, detail in signal_rows:
                await self._db.execute(
                    """INSERT OR IGNORE INTO ibkr_research_signals
                       (instrument_key, signal_date, signal_name,
                        direction, strength, detail)
                       VALUES (?, date('now'), ?, ?, ?, ?)""",
                    (key, name, direction, strength, detail))
            await self._db.commit()
        except Exception as e:  # noqa: BLE001 - research-only
            log.warning("ibkr_multiasset.research_error", error=str(e)[:120])
