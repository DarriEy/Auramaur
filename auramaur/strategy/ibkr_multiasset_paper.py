"""Six isolated, read-only IBKR market-data / local paper-execution books."""

from __future__ import annotations

from datetime import datetime, time as wall_time, timezone
import math
from uuid import uuid4
from zoneinfo import ZoneInfo

import structlog

from auramaur.exchange.ibkr_instruments import BY_BOOK, ContractKind, IBKRBook
from auramaur.strategy.protocols import ExecutionMode

log = structlog.get_logger()


class IBKRMultiAssetPaperBook:
    """One asset-specific ledger; this class has no broker order capability."""

    execution_mode = ExecutionMode.PAPER_SIMULATED

    def __init__(self, settings, client, db, book: IBKRBook):
        self._settings = settings
        self._client = client
        self._db = db
        self.book = book
        self.name = f"ibkr_{book.value}_paper"

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
            return et.weekday() < 5 and not (wall_time(17) <= et.time() < wall_time(18))
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

    async def _daily_pnl(self) -> float:
        row = await self._db.fetchone(
            """SELECT COALESCE(SUM(pnl_usd), 0) AS pnl FROM ibkr_paper_ledger
               WHERE book = ? AND date(realized_at) = date('now')""",
            (self.book.value,))
        return float(row["pnl"] or 0)

    @staticmethod
    def _momentum(bars) -> float | None:
        closes = [float(close) for _, close in bars if close and close > 0]
        if len(closes) < 21:
            return None
        return (closes[-1] / closes[-21] - 1) * 100

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
                    fx: float, entry_price: float | None = None) -> None:
        price = quote.ask if side == "BUY" else quote.bid
        capital = quantity * self._capital_per_unit(spec, price, fx)
        fee = self._commission(spec, quantity, capital)
        fill_ref = f"ibkr-{self.book.value}-paper-{uuid4().hex}"
        await self._db.execute(
            """INSERT INTO ibkr_paper_fills
               (book, instrument_key, con_id, side, quantity, multiplier, price,
                currency, fx_to_usd, commission_usd, fill_ref)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (self.book.value, spec.key, quote.con_id, side, quantity,
             quote.multiplier, price, spec.currency, fx, fee, fill_ref))
        await self._db.execute(
            "INSERT INTO ibkr_paper_ledger (book, kind, pnl_usd, source_ref) VALUES (?, 'commission', ?, ?)",
            (self.book.value, -fee, f"{fill_ref}:commission"))
        if side == "BUY":
            await self._db.execute(
                """INSERT INTO ibkr_paper_positions
                   (book, instrument_key, con_id, currency, quantity, multiplier,
                    fx_to_usd, avg_cost, current_price)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (self.book.value, spec.key, quote.con_id, spec.currency, quantity,
                 quote.multiplier, fx, price, price))
        else:
            pnl = (price - float(entry_price)) * quantity * quote.multiplier * fx
            await self._db.execute(
                "INSERT INTO ibkr_paper_ledger (book, kind, pnl_usd, source_ref) VALUES (?, 'trade', ?, ?)",
                (self.book.value, pnl, f"{fill_ref}:trade"))
            await self._db.execute(
                "DELETE FROM ibkr_paper_positions WHERE book = ? AND instrument_key = ?",
                (self.book.value, spec.key))
        await self._db.commit()

    async def run_once(self) -> int:
        cfg = self.config
        if not self._settings.ibkr.multiasset_paper_enabled or not cfg.enabled:
            return 0
        if not self.market_open():
            return 0
        positions = await self._positions()
        daily_pnl = await self._daily_pnl()
        state = await self._db.fetchone(
            "SELECT refresh_cursor FROM ibkr_paper_state WHERE book = ?",
            (self.book.value,))
        cursor = int(state["refresh_cursor"] or 0) if state else 0
        universe = BY_BOOK[self.book]
        count = min(len(universe), self._settings.ibkr.multiasset_refreshes_per_cycle)
        selected = [universe[(cursor + i) % len(universe)] for i in range(count)]
        # Open positions are always managed, even when outside the refresh slice.
        keys = {spec.key for spec in selected}
        selected.extend(spec for spec in universe if spec.key in positions and spec.key not in keys)
        entries = 0
        error = ""
        for spec in selected:
            try:
                quote = await self._client.get_quote(spec)
                if quote is None:
                    continue
                spread_bps = (quote.ask - quote.bid) / ((quote.ask + quote.bid) / 2) * 10_000
                fx = await self._client.get_fx_to_usd(spec.currency)
                if fx is None or spread_bps > cfg.max_spread_bps:
                    continue
                bars = await self._client.get_daily_bars(spec)
                momentum = self._momentum(bars)
                if momentum is None:
                    continue
                held = positions.get(spec.key)
                if held:
                    entry = float(held["avg_cost"])
                    gain_pct = (quote.bid / entry - 1) * 100
                    pnl = (quote.bid - entry) * float(held["quantity"]) * quote.multiplier * fx
                    await self._db.execute(
                        """UPDATE ibkr_paper_positions SET current_price=?,
                           unrealized_pnl_usd=?, updated_at=datetime('now')
                           WHERE book=? AND instrument_key=?""",
                        (quote.bid, pnl, self.book.value, spec.key))
                    if (gain_pct <= -cfg.stop_loss_pct or gain_pct >= cfg.take_profit_pct
                            or momentum <= self._settings.ibkr.multiasset_exit_momentum_pct):
                        await self._fill(spec, quote, "SELL", float(held["quantity"]),
                                         fx, entry_price=entry)
                        positions.pop(spec.key, None)
                    continue
                if daily_pnl <= -cfg.daily_loss_limit_usd or len(positions) >= cfg.max_positions:
                    continue
                if momentum < self._settings.ibkr.multiasset_min_momentum_pct:
                    continue
                deployed = sum(
                    float(row["quantity"]) * self._capital_per_unit(
                        next(item for item in universe if item.key == key),
                        float(row["avg_cost"]), float(row["fx_to_usd"]))
                    for key, row in positions.items())
                deployment_cap = cfg.budget_usd * cfg.max_deployment_pct / 100
                target = min(cfg.budget_usd * cfg.max_position_pct / 100,
                             deployment_cap - deployed)
                unit_capital = self._capital_per_unit(spec, quote.ask, fx)
                quantity = self._quantity(spec, target, unit_capital)
                if quantity <= 0:
                    continue
                await self._fill(spec, quote, "BUY", quantity, fx)
                positions = await self._positions()
                entries += 1
            except Exception as exc:  # noqa: BLE001
                error = str(exc)[:300]
                log.warning("ibkr_multiasset.instrument_error", book=self.book.value,
                            key=spec.key, error=error)
        next_cursor = (cursor + count) % len(universe)
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
        log.info("ibkr_multiasset.cycle", book=self.book.value, entries=entries,
                 positions=len(positions), daily_pnl=round(daily_pnl, 2))
        return entries
