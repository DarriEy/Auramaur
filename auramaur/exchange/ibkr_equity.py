"""IBKR equity trading client — directional stocks/ETFs (gated speculation).

Separate from the options client (ibkr.py): connects on its own clientId so the
two can coexist, and uses a non-read-only connection so it can place orders.
Same three-gate live model + kill switch + a hard per-order USD cap.

No validated edge — this is opt-in speculation, bounded by config.
"""

from __future__ import annotations

import time
from pathlib import Path

import structlog

from auramaur.exchange.models import OrderResult, OrderSide

log = structlog.get_logger()


class IBKREquityClient:
    def __init__(self, settings):
        self._settings = settings
        self._ib = None
        self._connected = False
        self._cooldown_until = 0.0

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
            return float(tk.close) if tk.close and tk.close > 0 else None
        except Exception as e:  # noqa: BLE001
            log.warning("ibkr_equity.price_error", symbol=symbol, error=str(e)[:100])
            return None

    async def place_order(
        self, symbol: str, side: OrderSide | str, usd_amount: float,
        dry_run: bool | None = None,
    ) -> OrderResult:
        """Place a market equity order sized to ~usd_amount. Gated + capped."""
        side_str = (side.value if isinstance(side, OrderSide) else str(side)).upper()

        if Path("KILL_SWITCH").exists():
            log.critical("kill_switch.active", action="ibkr_equity_order_blocked")
            return OrderResult(order_id="BLOCKED", market_id=symbol, status="rejected",
                               is_paper=True, error_message="kill switch")

        cap = self._settings.ibkr.equity_max_order_usd
        if usd_amount > cap:
            return OrderResult(order_id="BLOCKED", market_id=symbol, status="rejected",
                               is_paper=True,
                               error_message=f"${usd_amount:.2f} exceeds cap ${cap:.2f}")

        price = await self.get_price(symbol)
        if not price:
            return OrderResult(order_id="ERROR", market_id=symbol, status="rejected",
                               is_paper=True, error_message="no price")
        qty = int(usd_amount / price)
        if qty < 1:
            return OrderResult(order_id="BLOCKED", market_id=symbol, status="rejected",
                               is_paper=True, error_message=f"size <1 share at ${price:.2f}")

        if dry_run is None:
            dry_run = not self._settings.is_live
        # Paper: route through the simulated path (no IBKR order).
        if dry_run:
            log.info("ibkr_equity.order.paper", symbol=symbol, side=side_str, qty=qty,
                     price=price)
            return OrderResult(order_id="PAPER", market_id=symbol, status="paper",
                               filled_size=qty, filled_price=price, is_paper=True)

        # === LIVE ===
        if self._settings.ibkr.readonly:
            return OrderResult(order_id="BLOCKED", market_id=symbol, status="rejected",
                               is_paper=False,
                               error_message="ibkr.readonly=true — cannot place orders")
        await self._ensure_connected()
        try:
            from ib_async import Stock, MarketOrder
            stock = Stock(symbol, "SMART", "USD")
            await self._ib.qualifyContractsAsync(stock)
            order = MarketOrder(side_str, qty)
            trade = self._ib.placeOrder(stock, order)
            log.warning("ibkr_equity.order.live", symbol=symbol, side=side_str, qty=qty)
            return OrderResult(order_id=str(trade.order.orderId), market_id=symbol,
                               status="pending", filled_price=price, is_paper=False)
        except Exception as e:  # noqa: BLE001
            log.error("ibkr_equity.order.error", symbol=symbol, error=str(e)[:150])
            return OrderResult(order_id="ERROR", market_id=symbol, status="rejected",
                               is_paper=False, error_message=str(e)[:200])

    async def momentum(self, symbol: str) -> float | None:
        """Percent change over the lookback using historical daily bars."""
        await self._ensure_connected()
        from ib_async import Stock
        cfg = self._settings.ibkr
        try:
            stock = Stock(symbol, "SMART", "USD")
            await self._ib.qualifyContractsAsync(stock)
            bars = await self._ib.reqHistoricalDataAsync(
                stock, endDateTime="", durationStr=f"{cfg.directional_equity_lookback + 2} D",
                barSizeSetting="1 day", whatToShow="TRADES", useRTH=True,
            )
            if not bars or len(bars) < cfg.directional_equity_lookback + 1:
                return None
            old = bars[-1 - cfg.directional_equity_lookback].close
            cur = bars[-1].close
            return None if old <= 0 else (cur - old) / old * 100.0
        except Exception as e:  # noqa: BLE001
            log.warning("ibkr_equity.momentum_error", symbol=symbol, error=str(e)[:100])
            return None

    async def close(self) -> None:
        if self._ib is not None and self._connected:
            self._ib.disconnect()
            self._connected = False
