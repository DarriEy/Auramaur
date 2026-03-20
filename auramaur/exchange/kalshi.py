"""Kalshi exchange client — implements both MarketDiscovery and ExchangeClient."""

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

log = structlog.get_logger()

# Kalshi basic tier: 20 reads/sec
_RATE_LIMIT = 20


class KalshiClient:
    """Kalshi exchange client for market discovery and order execution.

    Requires the optional ``kalshi-python`` package.

    Safety: Same three-gate model as Polymarket.
      1. AURAMAUR_LIVE=true env var
      2. execution.live=true in config
      3. dry_run=False on the order
    """

    def __init__(self, settings, paper_trader: PaperTrader):
        self._settings = settings
        self._paper = paper_trader
        self._api = None  # Lazy init
        self._semaphore = asyncio.Semaphore(_RATE_LIMIT)

    def _init_api(self):
        """Lazily initialize the kalshi-python client."""
        if self._api is not None:
            return

        from kalshi import KalshiClient as _KalshiSDK

        cfg = self._settings.kalshi
        host = (
            "https://demo-api.kalshi.co"
            if cfg.environment == "demo"
            else "https://trading-api.kalshi.com"
        )

        self._api = _KalshiSDK(
            host=host,
            key_id=cfg.api_key or self._settings.kalshi_api_key,
            private_key_path=cfg.private_key_path or self._settings.kalshi_private_key_path,
        )
        log.info("kalshi.initialized", host=host, environment=cfg.environment)

    async def _call(self, fn, *args, **kwargs):
        """Run a synchronous SDK call in a thread with rate limiting."""
        async with self._semaphore:
            return await asyncio.to_thread(fn, *args, **kwargs)

    # ------------------------------------------------------------------
    # MarketDiscovery protocol
    # ------------------------------------------------------------------

    async def get_markets(self, active: bool = True, limit: int = 100) -> list[Market]:
        """Fetch markets from Kalshi API."""
        self._init_api()
        try:
            params = {"limit": limit, "status": "open" if active else "closed"}
            response = await self._call(self._api.get_events, **params)
            events = response.get("events", []) if isinstance(response, dict) else []

            markets: list[Market] = []
            for event in events:
                for m in event.get("markets", [event]):
                    parsed = self._parse_market(m)
                    if parsed:
                        markets.append(parsed)
                    if len(markets) >= limit:
                        break
                if len(markets) >= limit:
                    break

            log.info("kalshi.markets_fetched", count=len(markets))
            return markets
        except Exception as e:
            log.error("kalshi.fetch_error", error=str(e))
            return []

    async def get_market(self, market_id: str) -> Market | None:
        """Fetch a single market by ticker."""
        self._init_api()
        try:
            response = await self._call(self._api.get_market, market_id)
            market_data = response.get("market", response) if isinstance(response, dict) else response
            return self._parse_market(market_data)
        except Exception as e:
            log.error("kalshi.market_fetch_error", market_id=market_id, error=str(e))
            return None

    async def search_markets(self, query: str, limit: int = 50) -> list[Market]:
        """Search Kalshi markets by keyword."""
        self._init_api()
        try:
            response = await self._call(
                self._api.get_events, limit=limit, status="open",
            )
            events = response.get("events", []) if isinstance(response, dict) else []

            markets: list[Market] = []
            query_lower = query.lower()
            for event in events:
                for m in event.get("markets", [event]):
                    title = (m.get("title", "") or m.get("question", "")).lower()
                    if query_lower in title:
                        parsed = self._parse_market(m)
                        if parsed:
                            markets.append(parsed)
                        if len(markets) >= limit:
                            break
                if len(markets) >= limit:
                    break
            return markets
        except Exception as e:
            log.error("kalshi.search_error", query=query, error=str(e))
            return []

    # ------------------------------------------------------------------
    # ExchangeClient protocol
    # ------------------------------------------------------------------

    def prepare_order(
        self, signal: Signal, market: Market, position_size: float, is_live: bool,
    ) -> Order | None:
        """Build a Kalshi order from a signal.

        Kalshi supports direct BUY/SELL of YES/NO — no token swap needed.
        """
        if signal.recommended_side == OrderSide.BUY:
            side = OrderSide.BUY
            token = TokenType.YES
            exec_price = market.outcome_yes_price
        else:
            side = OrderSide.SELL
            token = TokenType.YES
            exec_price = market.outcome_yes_price

        exec_price = max(0.01, min(0.99, round(exec_price, 2)))

        # Kalshi contracts are $1 notional; position_size in dollars = number of contracts
        contract_count = round(position_size / exec_price, 2) if exec_price > 0 else 0
        if contract_count < 1:
            log.info("kalshi.prepare_order.too_small", contracts=contract_count)
            return None

        return Order(
            market_id=market.id,
            exchange="kalshi",
            token_id=market.ticker or market.id,
            side=side,
            token=token,
            size=contract_count,
            price=exec_price,
            dry_run=not is_live,
        )

    async def place_order(self, order: Order) -> OrderResult:
        """Place an order. Paper trades by default."""
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
                exchange="kalshi",
                market_id=order.market_id,
                side=order.side.value,
                size=order.size,
                price=order.price,
            )
            return result

        # === LIVE ORDER PATH ===
        self._init_api()
        log.warning(
            "order.live",
            exchange="kalshi",
            market_id=order.market_id,
            side=order.side.value,
            size=order.size,
            price=order.price,
        )

        try:
            kalshi_side = "yes" if order.side == OrderSide.BUY else "no"
            response = await self._call(
                self._api.create_order,
                ticker=order.token_id,
                side=kalshi_side,
                count=int(order.size),
                type="market",
                yes_price=int(order.price * 100),
            )

            order_data = response.get("order", response) if isinstance(response, dict) else {}
            order_id = str(order_data.get("order_id", "unknown"))

            return OrderResult(
                order_id=order_id,
                market_id=order.market_id,
                status="pending",
                filled_size=0,
                filled_price=order.price,
                is_paper=False,
            )
        except Exception as e:
            log.error("order.live_error", exchange="kalshi", error=str(e))
            return OrderResult(
                order_id="ERROR",
                market_id=order.market_id,
                status="rejected",
                is_paper=False,
            )

    async def get_order_book(self, market_id: str) -> OrderBook:
        """Get order book for a Kalshi market."""
        self._init_api()
        try:
            response = await self._call(self._api.get_orderbook, market_id)
            book = response.get("orderbook", response) if isinstance(response, dict) else {}

            bids = [
                OrderBookLevel(price=float(b[0]) / 100, size=float(b[1]))
                for b in book.get("yes", [])
            ]
            asks = [
                OrderBookLevel(price=float(a[0]) / 100, size=float(a[1]))
                for a in book.get("no", [])
            ]
            return OrderBook(bids=bids, asks=asks)
        except Exception as e:
            log.error("kalshi.orderbook_error", market_id=market_id, error=str(e))
            return OrderBook()

    async def get_order_status(self, order_id: str) -> OrderResult:
        """Query order status from Kalshi."""
        self._init_api()
        try:
            response = await self._call(self._api.get_order, order_id)
            order_data = response.get("order", response) if isinstance(response, dict) else {}

            status_map = {
                "resting": "pending",
                "canceled": "cancelled",
                "executed": "filled",
                "pending": "pending",
            }
            raw_status = str(order_data.get("status", "pending")).lower()
            status = status_map.get(raw_status, "pending")

            return OrderResult(
                order_id=order_id,
                market_id=order_data.get("ticker", ""),
                status=status,  # type: ignore[arg-type]
                filled_size=float(order_data.get("filled_count", 0)),
                filled_price=float(order_data.get("yes_price", 0)) / 100,
                is_paper=False,
            )
        except Exception as e:
            log.error("kalshi.order_status_error", order_id=order_id, error=str(e))
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a Kalshi order."""
        self._init_api()
        try:
            await self._call(self._api.cancel_order, order_id)
            log.info("order.cancelled", exchange="kalshi", order_id=order_id)
            return True
        except Exception as e:
            log.error("kalshi.cancel_error", order_id=order_id, error=str(e))
            return False

    async def close(self) -> None:
        """Clean up resources."""
        self._api = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse_market(self, data: dict) -> Market | None:
        """Parse a Kalshi API market into our Market model."""
        try:
            ticker = data.get("ticker", "")
            yes_price = float(data.get("yes_bid", data.get("last_price", 50))) / 100
            no_price = 1.0 - yes_price

            end_date = None
            close_time = data.get("close_time") or data.get("expiration_time")
            if close_time:
                try:
                    end_date = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    pass

            volume = float(data.get("volume", 0) or 0)
            liquidity = float(data.get("liquidity", 0) or 0)

            yes_bid = float(data.get("yes_bid", 0) or 0) / 100
            yes_ask = float(data.get("yes_ask", 0) or 0) / 100
            spread = yes_ask - yes_bid if yes_ask > yes_bid else 0.0

            return Market(
                id=ticker,
                exchange="kalshi",
                ticker=ticker,
                question=data.get("title", data.get("question", "")),
                description=data.get("subtitle", ""),
                category=data.get("category", ""),
                end_date=end_date,
                active=data.get("status", "").lower() in ("open", "active", ""),
                outcome_yes_price=yes_price,
                outcome_no_price=no_price,
                volume=volume,
                liquidity=liquidity,
                spread=spread,
            )
        except Exception as e:
            log.warning("kalshi.parse_error", error=str(e))
            return None
