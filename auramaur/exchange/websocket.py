"""WebSocket client for real-time Polymarket price updates."""

from __future__ import annotations

import asyncio
import json
from typing import Callable, Awaitable

import aiohttp
import structlog

log = structlog.get_logger()

POLYMARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


class PolymarketWebSocket:
    """WebSocket connection to Polymarket for real-time price events.

    Auto-reconnects on disconnect with exponential backoff.
    """

    def __init__(
        self,
        on_price_update: Callable[[str, float], Awaitable[None]] | None = None,
        max_reconnect_delay: float = 60.0,
    ) -> None:
        self._on_price_update = on_price_update
        self._on_trade: Callable[[str, str, float], Awaitable[None]] | None = None  # (market_id, side, size)
        self._max_reconnect_delay = max_reconnect_delay
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._running = False
        self._subscribed_markets: set[str] = set()
        self._reconnect_delay = 1.0

    async def connect(self) -> None:
        """Connect to the WebSocket endpoint."""
        self._session = aiohttp.ClientSession()
        self._running = True
        log.info("websocket.connecting")

    async def subscribe(self, market_ids: list[str]) -> None:
        """Subscribe to price updates for given market IDs."""
        self._subscribed_markets.update(market_ids)
        if self._ws and not self._ws.closed:
            for market_id in market_ids:
                await self._ws.send_json({
                    "type": "subscribe",
                    "market": market_id,
                })
        log.debug("websocket.subscribed", markets=len(market_ids))

    async def unsubscribe(self, market_ids: list[str]) -> None:
        """Unsubscribe from price updates."""
        for mid in market_ids:
            self._subscribed_markets.discard(mid)
        if self._ws and not self._ws.closed:
            for market_id in market_ids:
                await self._ws.send_json({
                    "type": "unsubscribe",
                    "market": market_id,
                })

    async def run(self) -> None:
        """Main event loop — connects, listens, auto-reconnects."""
        if not self._session:
            await self.connect()

        while self._running:
            try:
                self._ws = await self._session.ws_connect(POLYMARKET_WS_URL)
                self._reconnect_delay = 1.0
                log.info("websocket.connected")

                # Re-subscribe on reconnect
                for market_id in self._subscribed_markets:
                    await self._ws.send_json({
                        "type": "subscribe",
                        "market": market_id,
                    })

                async for msg in self._ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        await self._handle_message(msg.data)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        log.warning("websocket.error", error=self._ws.exception())
                        break
                    elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED):
                        break

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                log.warning("websocket.disconnected", error=str(e))
            except Exception as e:
                log.error("websocket.unexpected_error", error=str(e))

            if self._running:
                log.info("websocket.reconnecting", delay=self._reconnect_delay)
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, self._max_reconnect_delay
                )

    async def _handle_message(self, raw: str) -> None:
        """Parse and dispatch a WebSocket message."""
        try:
            data = json.loads(raw)

            # Extract price update
            event_type = data.get("event_type", data.get("type", ""))
            if event_type in ("price_change", "book", "trade", "last_trade_price"):
                market_id = data.get("market", data.get("asset_id", ""))
                price = data.get("price", data.get("last_trade_price"))

                if market_id and price is not None:
                    price = float(price)
                    if self._on_price_update:
                        await self._on_price_update(market_id, price)

            # Also emit trade events for order flow tracking
            if event_type in ("trade", "last_trade_price"):
                trade_side = data.get("side", "")
                trade_size = data.get("size", data.get("amount"))
                if market_id and trade_size is not None and self._on_trade:
                    await self._on_trade(market_id, trade_side, float(trade_size))

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            log.debug("websocket.parse_error", error=str(e))

    async def close(self) -> None:
        """Close the WebSocket connection."""
        self._running = False
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        log.info("websocket.closed")
