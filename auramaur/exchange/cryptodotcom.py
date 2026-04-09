"""Crypto.com exchange client — implements both MarketDiscovery and ExchangeClient.

Connects to the Crypto.com Exchange API for prediction/event market trading.
Works internationally (including Canada).

Safety: Same three-gate model as other exchanges.
  1. AURAMAUR_LIVE=true env var
  2. execution.live=true in config
  3. dry_run=False on the order
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
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

# Crypto.com rate limit: 3 requests per 100ms = ~30/sec (conservative)
_RATE_LIMIT = 20

# API endpoints
_SANDBOX_BASE = "https://uat-api.3ona.co/exchange/v1"
_PROD_BASE = "https://api.crypto.com/exchange/v1"


def _sign_request(api_key: str, api_secret: str, method: str, params: dict, nonce: int) -> str:
    """Generate HMAC-SHA256 signature for Crypto.com API."""
    # Sort params alphabetically and concatenate
    param_str = ""
    if params:
        sorted_keys = sorted(params.keys())
        param_str = "".join(f"{k}{params[k]}" for k in sorted_keys)

    payload = f"{method}{nonce}{api_key}{param_str}{nonce}"
    return hmac.new(
        api_secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


class CryptoComClient:
    """Crypto.com exchange client for prediction market discovery and order execution.

    Safety: Same three-gate model as Polymarket/Kalshi.
      1. AURAMAUR_LIVE=true env var
      2. execution.live=true in config
      3. dry_run=False on the order
    """

    def __init__(self, settings, paper_trader: PaperTrader):
        self._settings = settings
        self._paper = paper_trader
        self._session: aiohttp.ClientSession | None = None
        self._semaphore = asyncio.Semaphore(_RATE_LIMIT)
        self._base_url: str | None = None

    def _get_base_url(self) -> str:
        if self._base_url is None:
            cfg = self._settings.cryptodotcom
            self._base_url = (
                _SANDBOX_BASE if cfg.environment == "sandbox" else _PROD_BASE
            )
            log.info(
                "cryptodotcom.initialized",
                environment=cfg.environment,
                base_url=self._base_url,
            )
        return self._base_url

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=15),
            )
        return self._session

    async def _public_get(self, endpoint: str, params: dict | None = None) -> dict:
        """Make a public (unauthenticated) GET request."""
        async with self._semaphore:
            session = await self._get_session()
            url = f"{self._get_base_url()}/{endpoint}"
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    log.warning(
                        "cryptodotcom.api_error",
                        status=resp.status,
                        endpoint=endpoint,
                        body=text[:200],
                    )
                    return {}
                return await resp.json()

    async def _private_post(self, method: str, params: dict | None = None) -> dict:
        """Make an authenticated POST request."""
        cfg = self._settings.cryptodotcom
        if not cfg.api_key or not cfg.api_secret:
            log.error("cryptodotcom.no_credentials")
            return {}

        async with self._semaphore:
            session = await self._get_session()
            nonce = int(time.time() * 1000)
            params = params or {}

            sig = _sign_request(cfg.api_key, cfg.api_secret, method, params, nonce)

            body = {
                "id": nonce,
                "method": method,
                "api_key": cfg.api_key,
                "params": params,
                "sig": sig,
                "nonce": nonce,
            }

            url = f"{self._get_base_url()}/{method}"
            async with session.post(url, json=body) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    log.warning(
                        "cryptodotcom.api_error",
                        status=resp.status,
                        method=method,
                        body=text[:200],
                    )
                    return {}
                data = await resp.json()
                if data.get("code") != 0:
                    log.warning(
                        "cryptodotcom.api_response_error",
                        code=data.get("code"),
                        message=data.get("message"),
                    )
                    return {}
                return data.get("result", {})

    # ------------------------------------------------------------------
    # MarketDiscovery protocol
    # ------------------------------------------------------------------

    async def get_markets(self, active: bool = True, limit: int = 100) -> list[Market]:
        """Fetch prediction markets from Crypto.com."""
        try:
            data = await self._public_get("public/get-instruments")
            if not data:
                return []

            instruments = data.get("result", {}).get("data", [])
            if not instruments:
                # Alternate response shape
                instruments = data.get("result", {}).get("instruments", [])

            markets: list[Market] = []
            for inst in instruments:
                parsed = self._parse_market(inst)
                if parsed is None:
                    continue
                if active and not parsed.active:
                    continue
                markets.append(parsed)
                if len(markets) >= limit:
                    break

            log.info("cryptodotcom.markets_fetched", count=len(markets))
            return markets
        except Exception as e:
            log.error("cryptodotcom.fetch_error", error=str(e))
            return []

    async def get_market(self, market_id: str) -> Market | None:
        """Fetch a single market by instrument name."""
        try:
            data = await self._public_get(
                "public/get-instruments",
                params={"instrument_name": market_id},
            )
            if not data:
                return None

            instruments = data.get("result", {}).get("data", [])
            if not instruments:
                instruments = data.get("result", {}).get("instruments", [])

            for inst in instruments:
                if inst.get("instrument_name") == market_id or inst.get("symbol") == market_id:
                    return self._parse_market(inst)
            return None
        except Exception as e:
            log.error("cryptodotcom.market_fetch_error", market_id=market_id, error=str(e))
            return None

    async def search_markets(self, query: str, limit: int = 50) -> list[Market]:
        """Search prediction markets by keyword (client-side filtering)."""
        try:
            all_markets = await self.get_markets(active=True, limit=500)
            query_lower = query.lower()

            results: list[Market] = []
            for market in all_markets:
                if query_lower in market.question.lower() or query_lower in market.description.lower():
                    results.append(market)
                    if len(results) >= limit:
                        break
            return results
        except Exception as e:
            log.error("cryptodotcom.search_error", query=query, error=str(e))
            return []

    # ------------------------------------------------------------------
    # ExchangeClient protocol
    # ------------------------------------------------------------------

    def prepare_order(
        self, signal: Signal, market: Market, position_size: float, is_live: bool,
    ) -> Order | None:
        """Build a Crypto.com order from a signal.

        Crypto.com prediction markets use BUY/SELL semantics like Kalshi.
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

        # Position size in dollars = number of contracts (each contract resolves to $1)
        contract_count = round(position_size / exec_price, 2) if exec_price > 0 else 0
        if contract_count < 1:
            log.info("cryptodotcom.prepare_order.too_small", contracts=contract_count)
            return None

        return Order(
            market_id=market.id,
            exchange="cryptodotcom",
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
                exchange="cryptodotcom",
                market_id=order.market_id,
                side=order.side.value,
                size=order.size,
                price=order.price,
            )
            return result

        # === LIVE ORDER PATH ===
        log.warning(
            "order.live",
            exchange="cryptodotcom",
            market_id=order.market_id,
            side=order.side.value,
            size=order.size,
            price=order.price,
        )

        try:
            side_str = "BUY" if order.side == OrderSide.BUY else "SELL"
            params = {
                "instrument_name": order.token_id,
                "side": side_str,
                "type": "LIMIT",
                "price": str(order.price),
                "quantity": str(int(order.size)),
                "time_in_force": "GOOD_TILL_CANCEL",
            }

            result = await self._private_post("private/create-order", params=params)
            if not result:
                return OrderResult(
                    order_id="ERROR",
                    market_id=order.market_id,
                    status="rejected",
                    is_paper=False,
                )

            order_id = str(result.get("order_id", "unknown"))

            log.info(
                "order.live_placed",
                exchange="cryptodotcom",
                order_id=order_id,
                status=result.get("status"),
            )

            return OrderResult(
                order_id=order_id,
                market_id=order.market_id,
                status="pending",
                filled_size=0,
                filled_price=order.price,
                is_paper=False,
            )
        except Exception as e:
            log.error(
                "order.live_error",
                exchange="cryptodotcom",
                error=str(e),
                instrument=order.token_id,
            )
            return OrderResult(
                order_id="ERROR",
                market_id=order.market_id,
                status="rejected",
                is_paper=False,
            )

    async def get_order_book(self, market_id: str) -> OrderBook:
        """Get order book for a Crypto.com market."""
        try:
            data = await self._public_get(
                "public/get-book",
                params={"instrument_name": market_id, "depth": "10"},
            )
            if not data:
                return OrderBook()

            book_data = data.get("result", {}).get("data", [{}])
            if isinstance(book_data, list) and book_data:
                book_data = book_data[0]

            bids = [
                OrderBookLevel(price=float(b[0]), size=float(b[1]))
                for b in book_data.get("bids", [])
            ]
            asks = [
                OrderBookLevel(price=float(a[0]), size=float(a[1]))
                for a in book_data.get("asks", [])
            ]

            return OrderBook(bids=bids, asks=asks)
        except Exception as e:
            log.error("cryptodotcom.orderbook_error", market_id=market_id, error=str(e))
            return OrderBook()

    async def get_order_status(self, order_id: str) -> OrderResult:
        """Query order status from Crypto.com."""
        try:
            result = await self._private_post(
                "private/get-order-detail",
                params={"order_id": order_id},
            )

            if not result:
                raise ValueError(f"No result for order {order_id}")

            status_map = {
                "NEW": "pending",
                "PENDING": "pending",
                "ACTIVE": "pending",
                "FILLED": "filled",
                "CANCELED": "cancelled",
                "CANCELLED": "cancelled",
                "REJECTED": "rejected",
                "EXPIRED": "expired",
            }
            raw_status = str(result.get("status", "PENDING")).upper()
            status = status_map.get(raw_status, "pending")

            cumulative_qty = float(result.get("cumulative_quantity", 0))
            avg_price = float(result.get("avg_price", 0))

            return OrderResult(
                order_id=order_id,
                market_id=result.get("instrument_name", ""),
                status=status,  # type: ignore[arg-type]
                filled_size=cumulative_qty,
                filled_price=avg_price,
                is_paper=False,
            )
        except Exception as e:
            log.error("cryptodotcom.order_status_error", order_id=order_id, error=str(e))
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a Crypto.com order."""
        try:
            result = await self._private_post(
                "private/cancel-order",
                params={"order_id": order_id},
            )
            if result is not None:
                log.info("order.cancelled", exchange="cryptodotcom", order_id=order_id)
                return True
            return False
        except Exception as e:
            log.error("cryptodotcom.cancel_error", order_id=order_id, error=str(e))
            return False

    async def close(self) -> None:
        """Clean up resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse_market(self, data: dict) -> Market | None:
        """Parse a Crypto.com instrument into our Market model."""
        try:
            instrument_name = data.get("instrument_name", "") or data.get("symbol", "")
            if not instrument_name:
                return None

            # Crypto.com prediction markets use instrument_type = "PREDICTION"
            # or may be under a different category. Accept broadly for now.
            inst_type = (data.get("instrument_type", "") or "").upper()

            # Extract question/title from instrument metadata
            question = data.get("display_name", "") or data.get("instrument_name", "")
            description = data.get("description", "") or ""

            # Price data
            last_price = float(data.get("last_price", 0) or 0)
            best_bid = float(data.get("best_bid", 0) or 0)
            best_ask = float(data.get("best_ask", 0) or 0)

            # Normalize to 0-1 range if needed (some markets use cents)
            if last_price > 1:
                last_price = last_price / 100
            if best_bid > 1:
                best_bid = best_bid / 100
            if best_ask > 1:
                best_ask = best_ask / 100

            yes_price = best_bid if best_bid > 0 else last_price
            if yes_price == 0:
                yes_price = 0.5  # No data default
            no_price = 1.0 - yes_price

            volume = float(data.get("volume_24h", 0) or data.get("volume", 0) or 0)
            liquidity = float(data.get("liquidity", 0) or 0)
            spread = (best_ask - best_bid) if best_ask > best_bid else 0.0

            # Parse end date if available
            end_date = None
            expiry = data.get("expiry_timestamp_ms") or data.get("expiry_time")
            if expiry:
                try:
                    if isinstance(expiry, (int, float)):
                        end_date = datetime.fromtimestamp(expiry / 1000, tz=timezone.utc)
                    elif isinstance(expiry, str):
                        end_date = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
                except (ValueError, OSError):
                    pass

            is_active = str(data.get("status", "")).upper() in ("ACTIVE", "OPEN", "TRADING", "")
            category = data.get("category", "") or inst_type

            return Market(
                id=instrument_name,
                exchange="cryptodotcom",
                ticker=instrument_name,
                question=question,
                description=description,
                category=category,
                end_date=end_date,
                active=is_active,
                outcome_yes_price=yes_price,
                outcome_no_price=no_price,
                volume=volume,
                liquidity=liquidity,
                spread=spread,
            )
        except Exception as e:
            log.warning("cryptodotcom.parse_error", error=str(e))
            return None
