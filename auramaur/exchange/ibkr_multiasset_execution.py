"""Independently gated IBKR execution for graduated multi-asset books."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import uuid4

import structlog

from auramaur.exchange.ibkr_instruments import IBKRBook, InstrumentSpec
from auramaur.killswitch import kill_switch_present
from auramaur.risk.ibkr_evidence import (
    evaluate_ibkr_daily_evidence, evaluate_ibkr_evidence,
)

log = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class IBKRExecutionResult:
    accepted: bool
    order_id: str = ""
    reason: str = ""
    execution_ref: str = ""
    filled_quantity: float = 0.0
    filled_price: float = 0.0


class IBKRMultiAssetExecution:
    """Order-capable socket kept separate from the read-only quote client."""

    def __init__(self, settings, resolver, db):
        self._settings = settings
        self._resolver = resolver
        self._db = db
        self._ib = None
        self._lock = asyncio.Lock()

    def gate_reason(self, book: IBKRBook) -> str:
        cfg = self._settings.ibkr
        if book.value not in {"global_etf", "futures", "international_equity"}:
            return "book has no supported live execution model"
        if not cfg.multiasset_execution_enabled:
            return "ibkr.multiasset_execution_enabled=false"
        if not cfg.multiasset_execution_confirm_live:
            return "ibkr.multiasset_execution_confirm_live=false"
        if book.value not in cfg.multiasset_execution_books:
            return "book absent from ibkr.multiasset_execution_books"
        if not self._settings.is_live:
            return "global live gates are closed"
        if cfg.environment != "live":
            return "ibkr.environment is not live"
        if kill_switch_present():
            return "kill switch"
        return ""

    async def graduated(self, book: IBKRBook) -> bool:
        """Require the same daily + round-trip contract shown in preflight/UI."""
        cfg = self._settings.ibkr.multiasset_books[book.value]
        marks = await self._db.fetchall(
            "SELECT mark_date,equity_usd FROM ibkr_paper_daily_marks "
            "WHERE book=? ORDER BY mark_date", (book.value,))
        if not marks:
            return False
        from datetime import date
        equities = [float(row["equity_usd"]) for row in marks]
        daily = [b - a for a, b in zip(equities, equities[1:])]
        elapsed = max(0, (date.today() - date.fromisoformat(
            str(marks[0]["mark_date"])[:10])).days)
        daily_result = evaluate_ibkr_daily_evidence(
            daily, elapsed_days=elapsed, budget_usd=cfg.budget_usd)
        rows = await self._db.fetchall(
            "SELECT net_pnl_usd FROM ibkr_paper_round_trips "
            "WHERE book=? ORDER BY closed_at,id", (book.value,))
        trip_result = evaluate_ibkr_evidence(
            [float(row["net_pnl_usd"]) for row in rows], elapsed_days=elapsed,
            budget_usd=cfg.budget_usd, min_observations=30)
        return daily_result.ready and trip_result.ready

    async def _connect(self):
        if self._ib is not None and self._ib.isConnected():
            return
        async with self._lock:
            if self._ib is not None and self._ib.isConnected():
                return
            from ib_async import IB
            cfg = self._settings.ibkr
            self._ib = IB()
            await self._ib.connectAsync(
                cfg.host, cfg.live_port,
                clientId=cfg.multiasset_execution_client_id, readonly=False)

    async def _save(self, execution_ref: str, **values) -> None:
        assignments = ", ".join(f"{key}=?" for key in values)
        await self._db.execute(
            f"UPDATE ibkr_execution_orders SET {assignments}, updated_at=datetime('now') "
            "WHERE execution_ref=?", (*values.values(), execution_ref))
        await self._db.commit()

    async def acknowledge(self, execution_ref: str) -> None:
        await self._save(execution_ref, accounted=1)

    @staticmethod
    def _trade_fill(trade) -> tuple[str, float, float]:
        status = str(getattr(trade.orderStatus, "status", "") or "")
        quantity = float(getattr(trade.orderStatus, "filled", 0) or 0)
        price = float(getattr(trade.orderStatus, "avgFillPrice", 0) or 0)
        return status, quantity, price

    async def place(self, spec: InstrumentSpec, side: str,
                    quantity: float) -> IBKRExecutionResult:
        reason = self.gate_reason(spec.book)
        if reason:
            return IBKRExecutionResult(False, reason=reason)
        if quantity <= 0:
            return IBKRExecutionResult(False, reason="quantity must be positive")
        if not await self.graduated(spec.book):
            return IBKRExecutionResult(False, reason="book has not passed IBKR evidence contract")
        side = side.upper()
        prior = await self._db.fetchone(
            """SELECT * FROM ibkr_execution_orders
                 WHERE book=? AND instrument_key=? AND side=? AND accounted=0
                 ORDER BY id DESC LIMIT 1""", (spec.book.value, spec.key, side))
        if prior is not None:
            if prior["status"] in {"filled", "partial"} and float(prior["filled_quantity"]) > 0:
                return IBKRExecutionResult(
                    True, str(prior["order_id"]), execution_ref=str(prior["execution_ref"]),
                    filled_quantity=float(prior["filled_quantity"]),
                    filled_price=float(prior["avg_fill_price"]))
            if prior["status"] in {"error", "cancelled", "apicancelled", "inactive"}:
                await self.acknowledge(str(prior["execution_ref"]))
            else:
                return IBKRExecutionResult(
                    False, str(prior["order_id"]),
                    f"unreconciled prior order: {prior['status']}",
                    execution_ref=str(prior["execution_ref"]))
        execution_ref = f"ibkr-live-{uuid4().hex}"
        await self._db.execute(
            """INSERT INTO ibkr_execution_orders
               (execution_ref,book,instrument_key,side,requested_quantity)
               VALUES (?,?,?,?,?)""",
            (execution_ref, spec.book.value, spec.key, side, quantity))
        await self._db.commit()
        try:
            await self._connect()
            from ib_async import MarketOrder
            contract = await self._resolver.resolve(spec)
            order = MarketOrder(side, quantity)
            order.orderRef = execution_ref
            trade = self._ib.placeOrder(contract, order)
            order_id = str(trade.order.orderId)
            await self._save(execution_ref, order_id=order_id, status="submitted")
            log.critical("ibkr_multiasset.order.live", book=spec.book.value,
                         key=spec.key, side=side, quantity=quantity,
                         order_id=order_id)
            deadline = asyncio.get_running_loop().time() + float(
                self._settings.ibkr.multiasset_execution_fill_timeout_seconds)
            terminal = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}
            while asyncio.get_running_loop().time() < deadline:
                status, filled, price = self._trade_fill(trade)
                if status in terminal:
                    break
                await asyncio.sleep(0.1)
            else:
                self._ib.cancelOrder(trade.order)
                cancel_deadline = asyncio.get_running_loop().time() + 5.0
                while asyncio.get_running_loop().time() < cancel_deadline:
                    status, filled, price = self._trade_fill(trade)
                    if status in terminal:
                        break
                    await asyncio.sleep(0.1)
            status, filled, price = self._trade_fill(trade)
            if filled > 0:
                state = "filled" if filled >= quantity else "partial"
                await self._save(execution_ref, status=state,
                                 filled_quantity=filled, avg_fill_price=price)
                return IBKRExecutionResult(
                    True, order_id, execution_ref=execution_ref,
                    filled_quantity=filled, filled_price=price)
            reason = f"broker order ended {status or 'without terminal status'} with no fill"
            await self._save(execution_ref, status=status.lower() or "unknown", error=reason)
            return IBKRExecutionResult(False, order_id, reason, execution_ref)
        except Exception as exc:  # noqa: BLE001
            await self._save(execution_ref, status="error", error=str(exc)[:200])
            log.error("ibkr_multiasset.order.error", book=spec.book.value,
                      key=spec.key, error=str(exc)[:200])
            return IBKRExecutionResult(False, reason=str(exc)[:200],
                                       execution_ref=execution_ref)

    async def close(self) -> None:
        if self._ib is not None:
            self._ib.disconnect()
        self._ib = None
