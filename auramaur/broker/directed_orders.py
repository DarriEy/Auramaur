"""Operator-directed IBKR orders: calibration probes and the beta deployment.

This path EXECUTES AN INSTRUCTION. It never derives one. No signal, no sizing
model, no strategy output reaches it — the caller says exactly what to send,
and this module's only job is to refuse it when a gate says no and to record
what happened either way.

Why it does not reuse `IBKRMultiAssetExecution`
-----------------------------------------------
That gate requires graduation evidence: 120 daily marks and 30 round trips
over 180 elapsed days. That is the correct contract for a STRATEGY claiming an
edge, and a category error for a calibration probe, which is not claiming an
edge — it is paying a known cost to measure one. Different purpose, so a
different gate, with tighter caps and an explicit instrument allowlist rather
than a book.

The gate chain is deliberately flat and ordered, and every refusal names
exactly one reason. Today's lesson (three separate cases of a flag not doing
what its name said) is that a gate you cannot read at a glance is a gate you
do not have.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import structlog

from auramaur.killswitch import kill_switch_present

log = structlog.get_logger()

_TERMINAL = {"Filled", "Cancelled", "ApiCancelled", "Inactive",
             "ValidationError"}
# Terminal states that mean the broker REFUSED the order. Recording these as
# 'submitted' claims an order is live and working when it was rejected
# outright — and 'submitted' consumes the daily cap, so a refusal would eat
# budget it never used. Observed 2026-07-30: IB Gateway in Read-Only API mode
# returned ValidationError, which was absent from _TERMINAL entirely, so the
# call also burned the full fill timeout before reporting the wrong state.
_REJECTED = {"ValidationError", "Cancelled", "ApiCancelled", "Inactive"}


@dataclass(frozen=True)
class DirectedOrder:
    """An explicit instruction. Every field is supplied by the operator."""

    symbol: str
    sec_type: str                 # STK | CASH
    currency: str
    exchange: str
    side: str                     # BUY | SELL
    quantity: float
    order_type: str = "LMT"       # LMT | MKT
    limit_price: float | None = None
    label: str = ""               # probe name, or "beta"
    dry_run: bool = True          # gate 3 — must be explicitly cleared

    @property
    def is_treasury(self) -> bool:
        """A currency conversion changes the denomination of cash already
        held. It adds no market exposure, so it is capped separately."""
        return self.sec_type.upper() == "CASH"

    def notional_usd(self, reference_price: float) -> float:
        """USD notional of this order.

        For an FX pair whose BASE is USD (USDCAD, USDJPY), the quantity is
        already denominated in USD and multiplying by the rate would yield the
        QUOTE-currency amount instead — 800 USD would read as 1,096, and be
        measured against a USD cap it never actually breached.
        """
        if self.is_treasury and self.symbol.upper().startswith("USD"):
            return abs(self.quantity)
        return abs(self.quantity) * abs(reference_price)


@dataclass(frozen=True)
class DirectedOrderResult:
    accepted: bool
    status: str
    reason: str = ""
    ib_order_id: str = ""
    filled_qty: float = 0.0
    filled_price: float | None = None
    row_id: int | None = None


class DirectedOrderExecutor:
    """Places operator-directed orders behind an explicit, auditable gate."""

    def __init__(self, settings, db) -> None:
        self._settings = settings
        self._db = db

    # ------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------
    async def gate_reason(self, order: DirectedOrder, notional_usd: float,
                          connected_account: str = "") -> str:
        """The single reason this order may not be sent, or "" to proceed.

        Ordered cheapest-and-most-absolute first, so a kill switch is never
        masked by a config detail.
        """
        cfg = self._settings.ibkr

        if kill_switch_present():
            return "kill switch"
        if not cfg.directed_orders_enabled:
            return "ibkr.directed_orders_enabled=false"
        if not cfg.directed_orders_confirm_live:
            return "ibkr.directed_orders_confirm_live=false"
        if not self._settings.is_live:
            return "global live gates are closed"

        # Fail closed: an unset expected account refuses everything. The
        # environment label cannot be trusted to say which account is at risk
        # (2026-07-29: `environment: paper` served a live U-prefixed account),
        # so the operator declares it and we verify against the connection.
        want = (cfg.directed_orders_account or "").strip()
        if not want:
            return "ibkr.directed_orders_account is unset (fail-closed)"
        if connected_account and connected_account != want:
            return (f"connected account {connected_account} != configured "
                    f"{want}")

        allow = {s.strip().upper() for s in cfg.directed_orders_allowlist}
        if order.symbol.upper() not in allow:
            return f"{order.symbol} not in ibkr.directed_orders_allowlist"

        if order.order_type.upper() == "LMT" and not order.limit_price:
            return "limit order without a limit price"
        if order.quantity <= 0:
            return "quantity must be positive"

        # Treasury (currency conversion) is capped separately from trading:
        # it changes the denomination of cash already held rather than taking
        # a position, so it neither borrows the trading allowance nor forces
        # the trading cap wider to accommodate a non-trade.
        treasury = order.is_treasury
        kind = "treasury" if treasury else "per-order"
        max_cap = (cfg.directed_orders_treasury_max_notional_usd if treasury
                   else cfg.directed_orders_max_notional_usd)
        daily_cap = (cfg.directed_orders_treasury_daily_notional_usd if treasury
                     else cfg.directed_orders_daily_notional_usd)

        if notional_usd > max_cap:
            return (f"notional ${notional_usd:,.2f} exceeds {kind} cap "
                    f"${max_cap:,.2f}")

        # Counts what was SUBMITTED today, not what filled: an unfilled order
        # is still committed exposure until it is cancelled. Scoped to the same
        # kind, so a conversion and a trade do not consume each other's budget.
        row = await self._db.fetchone(
            f"""SELECT COALESCE(SUM(notional_usd), 0) AS n FROM directed_orders
                 WHERE date(submitted_at) = date('now')
                   AND status IN ('submitted','filled')
                   AND sec_type {'=' if treasury else '!='} 'CASH'""")
        used = float(row["n"]) if row else 0.0
        if used + notional_usd > daily_cap:
            return (f"{kind} daily cap: ${used:,.2f} used + "
                    f"${notional_usd:,.2f} exceeds ${daily_cap:,.2f}")
        return ""

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------
    async def place(self, order: DirectedOrder, *, ib,
                    reference_price: float) -> DirectedOrderResult:
        """Record the attempt, gate it, and send it only if every gate passes.

        The audit row is written BEFORE anything is sent, so an order that is
        refused, errors, or never returns is still answerable after the fact.
        """
        notional = order.notional_usd(reference_price)
        accounts = list(getattr(ib, "managedAccounts", lambda: [])() or [])
        connected = accounts[0] if accounts else ""

        reason = await self.gate_reason(order, notional, connected)
        status = "refused" if reason else ("dry_run" if order.dry_run else "submitted")

        cur = await self._db.execute(
            """INSERT INTO directed_orders
                 (label, symbol, sec_type, currency, exchange, side, quantity,
                  order_type, limit_price, notional_usd, account, dry_run,
                  status, refuse_reason)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (order.label, order.symbol, order.sec_type, order.currency,
             order.exchange, order.side, order.quantity, order.order_type,
             order.limit_price, notional, connected, int(order.dry_run),
             status, reason))
        await self._db.commit()
        row_id = getattr(cur, "lastrowid", None)

        if reason:
            log.warning("directed_order.refused", symbol=order.symbol,
                        label=order.label, notional=round(notional, 2),
                        reason=reason)
            return DirectedOrderResult(False, "refused", reason, row_id=row_id)

        if order.dry_run:
            log.info("directed_order.dry_run", symbol=order.symbol,
                     label=order.label, side=order.side,
                     quantity=order.quantity, order_type=order.order_type,
                     limit_price=order.limit_price,
                     notional=round(notional, 2), account=connected)
            return DirectedOrderResult(True, "dry_run", row_id=row_id)

        # Live from here. Logged at warning level on purpose — a real order is
        # the event you want to find in a log without searching for it.
        log.warning("directed_order.live", symbol=order.symbol,
                    label=order.label, side=order.side,
                    quantity=order.quantity, order_type=order.order_type,
                    limit_price=order.limit_price,
                    notional=round(notional, 2), account=connected)
        try:
            result = await self._send(ib, order)
        except Exception as exc:  # noqa: BLE001 — record, then surface
            await self._settle(row_id, "error", reason=str(exc)[:200])
            log.error("directed_order.error", symbol=order.symbol,
                      error=str(exc)[:200])
            return DirectedOrderResult(False, "error", str(exc)[:200],
                                       row_id=row_id)

        # `result.status` carries the broker's own terminal word. A refusal is
        # NOT a submission: recording it as one claims a live working order
        # and consumes the daily cap for exposure that never existed.
        broker_status = result.status or ""
        if result.filled_qty:
            status = "filled"
        elif broker_status in _REJECTED:
            status = "error"
        else:
            status = "submitted"

        await self._settle(row_id, status, ib_order_id=result.ib_order_id,
                           filled_qty=result.filled_qty,
                           filled_price=result.filled_price,
                           reason="" if status != "error" else
                           f"broker returned {broker_status}")
        log.info("directed_order.settled", symbol=order.symbol,
                 ib_order_id=result.ib_order_id, filled_qty=result.filled_qty,
                 filled_price=result.filled_price, status=status,
                 broker_status=broker_status)
        return DirectedOrderResult(
            status != "error", status,
            reason="" if status != "error" else f"broker returned {broker_status}",
            ib_order_id=result.ib_order_id, filled_qty=result.filled_qty,
            filled_price=result.filled_price, row_id=row_id)

    async def _send(self, ib, order: DirectedOrder) -> DirectedOrderResult:
        """The only place an order reaches IBKR. Split out for tests."""
        from ib_async import Forex, LimitOrder, MarketOrder, Stock

        if order.sec_type.upper() == "CASH":
            contract = Forex(order.symbol.upper())
        else:
            contract = Stock(order.symbol.upper(), order.exchange or "SMART",
                             order.currency or "USD")
        qualified = await ib.qualifyContractsAsync(contract)
        if not qualified:
            raise RuntimeError(f"could not qualify contract {order.symbol}")

        if order.order_type.upper() == "MKT":
            ib_order = MarketOrder(order.side.upper(), order.quantity)
        else:
            ib_order = LimitOrder(order.side.upper(), order.quantity,
                                  float(order.limit_price))
        ib_order.orderRef = order.label or "directed"

        trade = ib.placeOrder(qualified[0], ib_order)
        timeout = float(self._settings.ibkr.multiasset_execution_fill_timeout_seconds)
        waited = 0.0
        while waited < timeout:
            if trade.orderStatus.status in _TERMINAL:
                break
            await asyncio.sleep(0.5)
            waited += 0.5
        return DirectedOrderResult(
            True, trade.orderStatus.status,
            ib_order_id=str(trade.order.orderId),
            filled_qty=float(trade.orderStatus.filled or 0),
            filled_price=(float(trade.orderStatus.avgFillPrice)
                          if trade.orderStatus.avgFillPrice else None))

    async def _settle(self, row_id, status, *, ib_order_id="", filled_qty=0.0,
                      filled_price=None, reason="") -> None:
        if row_id is None:
            return
        await self._db.execute(
            """UPDATE directed_orders
                  SET status=?, ib_order_id=?, filled_qty=?, filled_price=?,
                      refuse_reason=CASE WHEN ?='' THEN refuse_reason ELSE ? END,
                      settled_at=datetime('now')
                WHERE id=?""",
            (status, ib_order_id, filled_qty, filled_price, reason, reason,
             row_id))
        await self._db.commit()
