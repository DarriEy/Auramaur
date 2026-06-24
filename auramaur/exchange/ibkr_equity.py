"""IBKR equity trading client.

Separate from the options client (ibkr.py): connects on its own clientId so the
two can coexist, and uses a non-read-only connection so it can place orders.
Same three-gate live model + kill switch + a hard per-order USD cap.

Serves the equity pillars (currently the odd-lot tender harvester). The
directional momentum book that originally owned this client was REMOVED
2026-06-09: the identical strategy shape went 0W/20L on Kraken and
backtested net-negative in every variant — pre-failed, never activated.
"""

from __future__ import annotations

import time
from pathlib import Path
from auramaur.killswitch import kill_switch_present

import structlog

from auramaur.exchange.models import OrderResult, OrderSide

log = structlog.get_logger()

# Hold-off between FX top-up conversions. The fill posts to the cash balance
# within seconds, so this only needs to outlast that latency; kept generous
# since funds settle T+1 and there's never a reason to re-convert quickly.
_FX_COOLDOWN_S = 1800


class IBKREquityClient:
    def __init__(self, settings):
        self._settings = settings
        self._ib = None
        self._connected = False
        self._cooldown_until = 0.0
        # After a conversion fires, hold off re-converting until the fill posts
        # to the cash balance, so a fast loop can't double-convert while T+1
        # settlement is pending.
        self._fx_cooldown_until = 0.0

    async def _ensure_connected(self) -> None:
        if self._connected and self._ib is not None:
            return
        if time.monotonic() < self._cooldown_until:
            raise ConnectionError("ibkr equity connect on cooldown")
        from ib_async import IB
        cfg = self._settings.ibkr
        port = cfg.paper_port if cfg.environment == "paper" else cfg.live_port
        try:
            self._ib = IB()
            await self._ib.connectAsync(
                host=cfg.host, port=port, clientId=cfg.equity_client_id,
                readonly=cfg.readonly,
            )
            self._ib.reqMarketDataType(cfg.market_data_type)
        except Exception:
            self._ib = None
            self._cooldown_until = time.monotonic() + 300
            raise
        self._connected = True
        log.info("ibkr_equity.connected", port=port, client_id=cfg.equity_client_id,
                 readonly=cfg.readonly)

    async def get_price(self, symbol: str) -> float | None:
        await self._ensure_connected()
        from ib_async import Stock
        try:
            stock = Stock(symbol, "SMART", "USD")
            await self._ib.qualifyContractsAsync(stock)
            [tk] = await self._ib.reqTickersAsync(stock)
            px = tk.marketPrice()
            if px and px > 0:
                return float(px)
            # delayed feeds often only populate close
            if tk.close and tk.close > 0:
                return float(tk.close)
            # No live/delayed entitlement (e.g. Error 10089 without a US-equity
            # market-data subscription) leaves every quote NaN. Fall back to the
            # most recent historical daily close, which uses the free HMDS feed
            # and works without a subscription — same source momentum() reads.
            bars = await self._ib.reqHistoricalDataAsync(
                stock, endDateTime="", durationStr="3 D",
                barSizeSetting="1 day", whatToShow="TRADES", useRTH=True,
            )
            if bars and bars[-1].close and bars[-1].close > 0:
                return float(bars[-1].close)
            return None
        except Exception as e:  # noqa: BLE001
            log.warning("ibkr_equity.price_error", symbol=symbol, error=str(e)[:100])
            return None

    async def place_order(
        self, symbol: str, side: OrderSide | str, usd_amount: float,
        dry_run: bool | None = None,
    ) -> OrderResult:
        """Open/add to a position with ~usd_amount of `symbol`, dollar-sized via a
        monetary-value (cashQty) order. IBKR REJECTS a fractional share quantity
        over the API (Error 10243 "use desktop version") but accepts a cash amount
        and sizes the fraction server-side — so we never send a fractional qty.
        Gated + capped. To exit a position use close_position()."""
        side_str = (side.value if isinstance(side, OrderSide) else str(side)).upper()

        if kill_switch_present():
            log.critical("kill_switch.active", action="ibkr_equity_order_blocked")
            return OrderResult(order_id="BLOCKED", market_id=symbol, status="rejected",
                               is_paper=True, error_message="kill switch")

        cap = self._settings.ibkr.equity_max_order_usd
        if usd_amount > cap:
            return OrderResult(order_id="BLOCKED", market_id=symbol, status="rejected",
                               is_paper=True,
                               error_message=f"${usd_amount:.2f} exceeds cap ${cap:.2f}")

        if dry_run is None:
            dry_run = not self._settings.is_live
        # Price is only for paper sizing + the recorded fill estimate; the live
        # order sizes itself from cashQty, so a missing quote never blocks it.
        price = await self.get_price(symbol)
        if dry_run:
            qty = round(usd_amount / price, 4) if price else 0.0
            log.info("ibkr_equity.order.paper", symbol=symbol, side=side_str,
                     usd=round(usd_amount, 2), qty=qty, price=price)
            return OrderResult(order_id="PAPER", market_id=symbol, status="paper",
                               filled_size=qty, filled_price=price or 0.0, is_paper=True)

        if self._settings.ibkr.readonly:
            return OrderResult(order_id="BLOCKED", market_id=symbol, status="rejected",
                               is_paper=False,
                               error_message="ibkr.readonly=true — cannot place orders")
        return await self._place_cash_order(symbol, side_str, usd_amount, price or 0.0)

    async def close_position(
        self, symbol: str, dry_run: bool | None = None,
    ) -> OrderResult | None:
        """Fully exit a held position. Selling a fractional *quantity* is also
        rejected over the API (Error 10243), so we close by cash VALUE: a SELL
        cashQty sized to the position's current market value. Returns None when
        nothing is held (caller should still drop it from its book).

        CAVEAT: a cashQty close is only as exact as the quote, so it may leave
        sub-cent dust — verify live that positions flatten cleanly before trusting
        this for hands-off exits."""
        if kill_switch_present():
            log.critical("kill_switch.active", action="ibkr_equity_close_blocked")
            return OrderResult(order_id="BLOCKED", market_id=symbol, status="rejected",
                               is_paper=True, error_message="kill switch")

        held = (await self.get_positions()).get(symbol, 0.0)
        if abs(held) < 1e-6:
            return None  # nothing to close

        if dry_run is None:
            dry_run = not self._settings.is_live
        price = await self.get_price(symbol)
        value = round(held * price, 2) if price else 0.0
        if dry_run:
            log.info("ibkr_equity.close.paper", symbol=symbol, qty=held,
                     value=value, price=price)
            return OrderResult(order_id="PAPER", market_id=symbol, status="paper",
                               filled_size=held, filled_price=price or 0.0, is_paper=True)

        if self._settings.ibkr.readonly:
            return OrderResult(order_id="BLOCKED", market_id=symbol, status="rejected",
                               is_paper=False,
                               error_message="ibkr.readonly=true — cannot place orders")
        if not value:
            return OrderResult(order_id="ERROR", market_id=symbol, status="rejected",
                               is_paper=False, error_message="no price to size close")
        return await self._place_cash_order(symbol, "SELL", value, price or 0.0)

    async def _place_cash_order(
        self, symbol: str, side_str: str, usd_amount: float, price: float,
    ) -> OrderResult:
        """Place a live market order denominated in cash (cashQty=usd_amount,
        totalQuantity=0) — the API path IBKR accepts for fractional equity sizing
        (a fractional totalQuantity is rejected with Error 10243)."""
        await self._ensure_connected()
        try:
            from ib_async import Stock, MarketOrder
            stock = Stock(symbol, "SMART", "USD")
            await self._ib.qualifyContractsAsync(stock)
            order = MarketOrder(side_str, 0)   # size comes from cashQty, not qty
            order.cashQty = round(usd_amount, 2)
            trade = self._ib.placeOrder(stock, order)
            log.warning("ibkr_equity.order.live", symbol=symbol, side=side_str,
                        usd=round(usd_amount, 2))
            return OrderResult(order_id=str(trade.order.orderId), market_id=symbol,
                               status="pending", filled_price=price, is_paper=False)
        except Exception as e:  # noqa: BLE001
            log.error("ibkr_equity.order.error", symbol=symbol, error=str(e)[:150])
            return OrderResult(order_id="ERROR", market_id=symbol, status="rejected",
                               is_paper=False, error_message=str(e)[:200])

    async def place_share_order(
        self, symbol: str, side: OrderSide | str, qty: int, limit_price: float,
        dry_run: bool | None = None,
    ) -> OrderResult:
        """Place a WHOLE-share limit order for an exact quantity.

        The odd-lot tender trade needs an exact sub-100 share count (odd-lot
        priority applies to holders of FEWER than 100 shares — a cashQty
        order could fill to 100+ and forfeit the priority). Whole-share
        quantities are accepted over the API; only fractional quantities are
        rejected (Error 10243). Gated + capped like every order.
        """
        side_str = (side.value if isinstance(side, OrderSide) else str(side)).upper()
        qty = int(qty)
        if qty <= 0:
            return OrderResult(order_id="ERROR", market_id=symbol, status="rejected",
                               is_paper=True, error_message="qty must be positive")

        if kill_switch_present():
            log.critical("kill_switch.active", action="ibkr_equity_order_blocked")
            return OrderResult(order_id="BLOCKED", market_id=symbol, status="rejected",
                               is_paper=True, error_message="kill switch")

        notional = qty * limit_price
        cap = self._settings.ibkr.equity_max_order_usd
        if notional > cap:
            return OrderResult(order_id="BLOCKED", market_id=symbol, status="rejected",
                               is_paper=True,
                               error_message=f"${notional:.2f} exceeds cap ${cap:.2f}")

        if dry_run is None:
            dry_run = not self._settings.is_live
        if dry_run:
            log.info("ibkr_equity.share_order.paper", symbol=symbol, side=side_str,
                     qty=qty, limit=limit_price)
            return OrderResult(order_id="PAPER", market_id=symbol, status="paper",
                               filled_size=float(qty), filled_price=limit_price,
                               is_paper=True)

        if self._settings.ibkr.readonly:
            return OrderResult(order_id="BLOCKED", market_id=symbol, status="rejected",
                               is_paper=False,
                               error_message="ibkr.readonly=true — cannot place orders")
        await self._ensure_connected()
        try:
            from ib_async import LimitOrder, Stock
            stock = Stock(symbol, "SMART", "USD")
            await self._ib.qualifyContractsAsync(stock)
            order = LimitOrder(side_str, qty, round(limit_price, 2))
            trade = self._ib.placeOrder(stock, order)
            log.warning("ibkr_equity.share_order.live", symbol=symbol,
                        side=side_str, qty=qty, limit=limit_price)
            return OrderResult(order_id=str(trade.order.orderId), market_id=symbol,
                               status="pending", filled_price=limit_price,
                               is_paper=False)
        except Exception as e:  # noqa: BLE001
            log.error("ibkr_equity.share_order.error", symbol=symbol,
                      error=str(e)[:150])
            return OrderResult(order_id="ERROR", market_id=symbol, status="rejected",
                               is_paper=False, error_message=str(e)[:200])

    async def get_positions(self) -> dict[str, float]:
        """Held equity positions {symbol: qty} from the account (for reconcile)."""
        try:
            await self._ensure_connected()
            return {p.contract.symbol: p.position
                    for p in self._ib.positions() if p.position != 0}
        except Exception as e:  # noqa: BLE001
            log.debug("ibkr_equity.positions_error", error=str(e)[:80])
            return {}

    async def usd_cash(self) -> float | None:
        """Total USD held in the account — the funding the (USD-priced) stock book
        draws on, and the re-conversion guard for ensure_usd_float. Returns None
        if the gateway is unreachable so the caller skips rather than misfires.

        IBKR holds non-base cash as a *virtual FX position* (symbol USD, secType
        CASH), so a freshly converted balance shows up in positions() — NOT in
        accountValues().CashBalance, which stays empty on this account. Reading
        the FX position is what lets us see the convert and avoid re-converting
        every cooldown. We deliberately count total (incl. unsettled T+1) USD:
        the question here is 'have we already funded?', and IBKR — not this
        method — enforces settlement at order time. Falls back to the CashBalance
        row if a position isn't present (e.g. after settlement folds into base)."""
        try:
            await self._ensure_connected()
            for p in self._ib.positions():
                c = p.contract
                if getattr(c, "symbol", None) == "USD" and getattr(c, "secType", None) == "CASH":
                    return float(p.position or 0.0)
            for v in self._ib.accountValues():
                if v.tag == "CashBalance" and v.currency == "USD":
                    return float(v.value or 0.0)
            return 0.0
        except Exception as e:  # noqa: BLE001
            log.debug("ibkr_equity.usd_cash_error", error=str(e)[:80])
            return None

    async def ensure_usd_float(
        self, *, target_usd: float, max_convert_usd: float,
        min_convert_usd: float, source_ccy: str = "CAD",
        dry_run: bool | None = None,
    ) -> OrderResult | None:
        """Top up settled USD toward `target_usd` by converting `source_ccy`->USD,
        so the USD-priced stock book has buying power. Capped per conversion and
        gated like every order. Returns None when no action is needed (already at
        target, can't read balance, or below the dust floor).

        Converting BUYs the USD.<ccy> forex pair (e.g. USDCAD): buying USD, paying
        the source currency. On a cash account the proceeds settle T+1, so this is
        a buffer-maintainer, not same-cycle funding."""
        if dry_run is None:
            dry_run = not self._settings.is_live

        if kill_switch_present():
            log.critical("kill_switch.active", action="ibkr_fx_convert_blocked")
            return OrderResult(order_id="BLOCKED", market_id=f"USD{source_ccy}",
                               status="rejected", is_paper=True,
                               error_message="kill switch")

        # Don't re-fire while a just-placed conversion is still posting/settling.
        if time.monotonic() < self._fx_cooldown_until:
            return None

        usd = await self.usd_cash()
        if usd is None:
            return None  # unreachable — skip quietly, sizing will degrade anyway
        if usd >= target_usd:
            return None  # already funded

        # Round to whole USD; small sizes auto-route as odd lots, so the cap can
        # sit far below the IDEALPRO 25k minimum.
        convert = float(round(min(target_usd - usd, max_convert_usd)))
        if convert < min_convert_usd:
            return None

        if dry_run:
            self._fx_cooldown_until = time.monotonic() + _FX_COOLDOWN_S
            log.info("ibkr_equity.fx_convert.paper", source=source_ccy,
                     usd=convert, have_usd=round(usd, 2), target=target_usd)
            return OrderResult(order_id="PAPER", market_id=f"USD{source_ccy}",
                               status="paper", filled_size=convert, is_paper=True)

        # === LIVE ===
        if self._settings.ibkr.readonly:
            return OrderResult(order_id="BLOCKED", market_id=f"USD{source_ccy}",
                               status="rejected", is_paper=False,
                               error_message="ibkr.readonly=true — cannot convert")
        try:
            await self._ensure_connected()
            from ib_async import Forex, MarketOrder
            contract = Forex(f"USD{source_ccy}")
            await self._ib.qualifyContractsAsync(contract)
            order = MarketOrder("BUY", convert)  # buy USD, sell source_ccy
            trade = self._ib.placeOrder(contract, order)
            self._fx_cooldown_until = time.monotonic() + _FX_COOLDOWN_S
            log.warning("ibkr_equity.fx_convert.live", source=source_ccy,
                        usd=convert, have_usd=round(usd, 2), target=target_usd)
            return OrderResult(order_id=str(trade.order.orderId),
                               market_id=f"USD{source_ccy}", status="pending",
                               filled_size=convert, is_paper=False)
        except Exception as e:  # noqa: BLE001
            log.error("ibkr_equity.fx_convert.error", source=source_ccy,
                      error=str(e)[:150])
            return OrderResult(order_id="ERROR", market_id=f"USD{source_ccy}",
                               status="rejected", is_paper=False,
                               error_message=str(e)[:200])

    async def close(self) -> None:
        if self._ib is not None and self._connected:
            self._ib.disconnect()
            self._connected = False
