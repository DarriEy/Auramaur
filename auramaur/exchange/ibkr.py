"""Interactive Brokers exchange client — options trading via reframed binary questions.

Uses ib_async (actively maintained successor to ib_insync) to connect to
TWS / IB Gateway. Options are reframed as binary questions by the reframer,
and binary trading decisions are mapped back to option orders.

Requires the optional ``ib_async`` package.

Safety: Same three-gate model as other exchanges.
  1. AURAMAUR_LIVE=true env var
  2. execution.live=true in config
  3. dry_run=False on the order
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import structlog

from auramaur.exchange.models import (
    Market,
    Order,
    OrderBook,
    OrderBookLevel,
    OrderResult,
    OrderSide,
    Signal,
    TokenType,
)
from auramaur.exchange.paper import PaperTrader
from auramaur.nlp.reframer import (
    OptionContract,
    ReframedMarket,
    reframe_option_as_binary,
    select_interesting_strikes,
)

log = structlog.get_logger()


class IBKRClient:
    """Interactive Brokers client implementing MarketDiscovery + ExchangeClient.

    Discovers option chains, reframes them as binary questions, and maps
    binary trading decisions back to option orders.
    """

    def __init__(self, settings, paper_trader: PaperTrader):
        self._settings = settings
        self._paper = paper_trader
        self._ib = None  # Lazy init of ib_async.IB
        self._connected = False
        # Cache of reframed markets: market_id → ReframedMarket
        self._reframed: dict[str, ReframedMarket] = {}
        # Watchlist of underlying symbols to scan
        self._watchlist: list[str] = settings.ibkr.watchlist

    async def _ensure_connected(self) -> None:
        """Lazily connect to TWS / IB Gateway."""
        if self._connected and self._ib is not None:
            return

        from ib_async import IB

        cfg = self._settings.ibkr
        self._ib = IB()

        port = cfg.paper_port if cfg.environment == "paper" else cfg.live_port
        await self._ib.connectAsync(
            host=cfg.host,
            port=port,
            clientId=cfg.client_id,
            readonly=cfg.environment == "paper",
        )
        self._connected = True
        log.info(
            "ibkr.connected",
            host=cfg.host,
            port=port,
            environment=cfg.environment,
        )

    # ------------------------------------------------------------------
    # MarketDiscovery protocol
    # ------------------------------------------------------------------

    async def get_markets(self, active: bool = True, limit: int = 100) -> list[Market]:
        """Scan option chains for watchlist symbols and return reframed markets."""
        await self._ensure_connected()

        markets: list[Market] = []
        for symbol in self._watchlist:
            try:
                reframed = await self._scan_options_for_symbol(symbol)
                for rm in reframed:
                    self._reframed[rm.market.id] = rm
                    markets.append(rm.market)
                    if len(markets) >= limit:
                        break
            except Exception as e:
                log.error("ibkr.scan_error", symbol=symbol, error=str(e))
            if len(markets) >= limit:
                break

        log.info("ibkr.markets_fetched", count=len(markets), symbols=len(self._watchlist))
        return markets

    async def get_market(self, market_id: str) -> Market | None:
        """Look up a reframed market by ID."""
        if market_id in self._reframed:
            return self._reframed[market_id].market
        return None

    async def search_markets(self, query: str, limit: int = 50) -> list[Market]:
        """Search reframed markets by keyword."""
        results: list[Market] = []
        query_lower = query.lower()
        for rm in self._reframed.values():
            if query_lower in rm.market.question.lower() or query_lower in rm.option.symbol.lower():
                results.append(rm.market)
                if len(results) >= limit:
                    break
        return results

    async def _scan_options_for_symbol(self, symbol: str) -> list[ReframedMarket]:
        """Fetch option chain for a symbol and reframe interesting strikes."""
        from ib_async import Stock, Option

        # Get underlying stock
        stock = Stock(symbol, "SMART", "USD")
        await self._ib.qualifyContractsAsync(stock)
        [ticker] = await self._ib.reqTickersAsync(stock)
        underlying_price = ticker.marketPrice()

        if not underlying_price or underlying_price <= 0:
            log.warning("ibkr.no_price", symbol=symbol)
            return []

        # Get option chains
        chains = await self._ib.reqSecDefOptParamsAsync(
            stock.symbol, "", stock.secType, stock.conId,
        )
        if not chains:
            return []

        # Use SMART exchange chain
        chain = next((c for c in chains if c.exchange == "SMART"), chains[0])

        # Filter expiries: 7-90 days out
        now = datetime.now(timezone.utc)
        valid_expiries = []
        for exp_str in sorted(chain.expirations):
            exp_date = datetime.strptime(exp_str, "%Y%m%d").replace(tzinfo=timezone.utc)
            dte = (exp_date - now).days
            if 7 <= dte <= 90:
                valid_expiries.append(exp_str)
            if len(valid_expiries) >= 3:
                break

        if not valid_expiries:
            return []

        # Select strikes near the money
        strikes = sorted(chain.strikes)
        atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - underlying_price))
        near_strikes = strikes[max(0, atm_idx - 3):atm_idx + 4]

        # Build option contracts and request market data
        option_contracts = []
        for expiry in valid_expiries:
            for strike in near_strikes:
                for right in ("C", "P"):
                    opt = Option(symbol, expiry, strike, right, "SMART")
                    option_contracts.append(opt)

        # Qualify in batch
        qualified = await self._ib.qualifyContractsAsync(*option_contracts)
        qualified = [c for c in qualified if c.conId > 0]

        if not qualified:
            return []

        # Request market data
        tickers = await self._ib.reqTickersAsync(*qualified)

        # Convert to OptionContract objects
        options: list[OptionContract] = []
        for tick in tickers:
            contract = tick.contract
            greeks = tick.modelGreeks or tick.lastGreeks
            if greeks is None or greeks.delta is None:
                continue

            bid = tick.bid if tick.bid > 0 else 0.0
            ask = tick.ask if tick.ask > 0 else 0.0
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else tick.last or 0.0

            options.append(OptionContract(
                symbol=contract.symbol,
                strike=contract.strike,
                expiry=datetime.strptime(contract.lastTradeDateOrContractMonth, "%Y%m%d").replace(
                    tzinfo=timezone.utc
                ),
                right=contract.right,
                delta=greeks.delta,
                mid_price=mid,
                bid=bid,
                ask=ask,
                implied_vol=greeks.impliedVol or 0.0,
                volume=tick.volume or 0,
                open_interest=0,  # Requires separate request
                underlying_price=underlying_price,
                con_id=contract.conId,
            ))

        # Select the most interesting options
        selected = select_interesting_strikes(options, underlying_price, max_contracts=10)

        # Reframe as binary questions
        reframed: list[ReframedMarket] = []
        for opt in selected:
            rm = reframe_option_as_binary(opt)
            reframed.append(rm)

        log.info(
            "ibkr.options_scanned",
            symbol=symbol,
            total_options=len(options),
            selected=len(reframed),
        )
        return reframed

    # ------------------------------------------------------------------
    # ExchangeClient protocol
    # ------------------------------------------------------------------

    def prepare_order(
        self, signal: Signal, market: Market, position_size: float, is_live: bool,
    ) -> Order | None:
        """Build an IB option order from a binary signal.

        Maps the binary decision back to the underlying option contract:
        - BUY YES on "Will X > $Y?" → buy the call option
        - SELL YES on "Will X > $Y?" → buy the put option
        """
        rm = self._reframed.get(market.id)
        if rm is None:
            log.warning("ibkr.prepare_order.no_reframe", market_id=market.id)
            return None

        mapping = rm.trade_mapping
        option = rm.option

        if signal.recommended_side == OrderSide.BUY:
            action = mapping.buy_yes_action
        else:
            action = mapping.sell_yes_action

        # Determine order side and price
        if action in ("buy_call", "buy_put"):
            side = OrderSide.BUY
            exec_price = option.mid_price
        else:
            side = OrderSide.SELL
            exec_price = option.mid_price

        if exec_price <= 0:
            log.info("ibkr.prepare_order.no_price", market_id=market.id)
            return None

        # Size: position_size in dollars / (option price × multiplier)
        multiplier = mapping.contract_multiplier
        contract_cost = exec_price * multiplier
        num_contracts = int(position_size / contract_cost) if contract_cost > 0 else 0

        if num_contracts < 1:
            log.info("ibkr.prepare_order.too_small", position_size=position_size, cost=contract_cost)
            return None

        # Encode action and contract info in token_id for order routing
        token_id = f"{option.con_id}:{action}:{option.right}:{option.strike}:{option.expiry.strftime('%Y%m%d')}"

        return Order(
            market_id=market.id,
            exchange="ibkr",
            token_id=token_id,
            side=side,
            token=TokenType.YES if action.endswith("call") else TokenType.NO,
            size=float(num_contracts),
            price=exec_price,
            dry_run=not is_live,
        )

    async def place_order(self, order: Order) -> OrderResult:
        """Place an option order via IB."""
        # Kill switch
        if Path("KILL_SWITCH").exists():
            log.critical("kill_switch.active", action="order_blocked")
            return OrderResult(
                order_id="BLOCKED",
                market_id=order.market_id,
                status="rejected",
                is_paper=True,
            )

        # Paper trade if ANY gate is closed
        if order.dry_run or not self._settings.is_live:
            result = await self._paper.execute(order)
            log.info(
                "order.paper",
                exchange="ibkr",
                market_id=order.market_id,
                side=order.side.value,
                size=order.size,
                price=order.price,
            )
            return result

        # === LIVE ORDER PATH ===
        await self._ensure_connected()

        log.warning(
            "order.live",
            exchange="ibkr",
            market_id=order.market_id,
            side=order.side.value,
            size=order.size,
            price=order.price,
        )

        try:
            from ib_async import LimitOrder, Option as IBOption

            # Parse contract info from token_id
            parts = order.token_id.split(":")
            con_id = int(parts[0])
            action_str = parts[1]
            right = parts[2]
            strike = float(parts[3])
            expiry = parts[4]

            # Resolve symbol from market
            rm = self._reframed.get(order.market_id)
            symbol = rm.option.symbol if rm else order.market_id.split(":")[1]

            contract = IBOption(symbol, expiry, strike, right, "SMART")
            contract.conId = con_id
            await self._ib.qualifyContractsAsync(contract)

            ib_action = "BUY" if order.side == OrderSide.BUY else "SELL"
            ib_order = LimitOrder(ib_action, int(order.size), order.price)

            trade = self._ib.placeOrder(contract, ib_order)

            return OrderResult(
                order_id=str(trade.order.orderId),
                market_id=order.market_id,
                status="pending",
                filled_size=0,
                filled_price=order.price,
                is_paper=False,
            )
        except Exception as e:
            log.error("order.live_error", exchange="ibkr", error=str(e))
            return OrderResult(
                order_id="ERROR",
                market_id=order.market_id,
                status="rejected",
                is_paper=False,
            )

    async def get_order_book(self, market_id: str) -> OrderBook:
        """Get order book for an option (via reframed market)."""
        rm = self._reframed.get(market_id)
        if rm is None:
            return OrderBook()

        await self._ensure_connected()

        try:
            from ib_async import Option as IBOption

            opt = rm.option
            contract = IBOption(
                opt.symbol,
                opt.expiry.strftime("%Y%m%d"),
                opt.strike,
                opt.right,
                "SMART",
            )
            contract.conId = opt.con_id
            book = await self._ib.reqMktDepthAsync(contract, numRows=5)

            bids = [OrderBookLevel(price=d.price, size=float(d.size)) for d in book if d.side == 1]
            asks = [OrderBookLevel(price=d.price, size=float(d.size)) for d in book if d.side == 0]
            return OrderBook(bids=bids, asks=asks)
        except Exception as e:
            log.error("ibkr.orderbook_error", market_id=market_id, error=str(e))
            return OrderBook()

    async def get_order_status(self, order_id: str) -> OrderResult:
        """Get status of an IB order by orderId."""
        await self._ensure_connected()
        try:
            trades = self._ib.trades()
            for trade in trades:
                if str(trade.order.orderId) == order_id:
                    status_map = {
                        "Submitted": "pending",
                        "Filled": "filled",
                        "Cancelled": "cancelled",
                        "Inactive": "rejected",
                        "PendingSubmit": "pending",
                        "PendingCancel": "pending",
                        "PreSubmitted": "pending",
                        "ApiPending": "pending",
                        "ApiCancelled": "cancelled",
                    }
                    raw = trade.orderStatus.status
                    status = status_map.get(raw, "pending")

                    return OrderResult(
                        order_id=order_id,
                        market_id=str(trade.contract.conId),
                        status=status,  # type: ignore[arg-type]
                        filled_size=float(trade.orderStatus.filled),
                        filled_price=float(trade.orderStatus.avgFillPrice),
                        is_paper=False,
                    )

            raise ValueError(f"Order {order_id} not found")
        except Exception as e:
            log.error("ibkr.order_status_error", order_id=order_id, error=str(e))
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an IB order."""
        await self._ensure_connected()
        try:
            trades = self._ib.trades()
            for trade in trades:
                if str(trade.order.orderId) == order_id:
                    self._ib.cancelOrder(trade.order)
                    log.info("order.cancelled", exchange="ibkr", order_id=order_id)
                    return True
            return False
        except Exception as e:
            log.error("ibkr.cancel_error", order_id=order_id, error=str(e))
            return False

    async def close(self) -> None:
        """Disconnect from IB."""
        if self._ib is not None and self._connected:
            self._ib.disconnect()
            self._connected = False
            log.info("ibkr.disconnected")
