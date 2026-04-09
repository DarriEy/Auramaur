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
        self._client = None  # KalshiClient (api_client)
        self._events_api = None
        self._markets_api = None
        self._portfolio_api = None
        self._semaphore = asyncio.Semaphore(_RATE_LIMIT)

    def _init_api(self):
        """Lazily initialize the kalshi-python client."""
        if self._client is not None:
            return

        from kalshi_python import KalshiClient as _KalshiSDK
        from kalshi_python import Configuration, EventsApi, MarketsApi, PortfolioApi

        cfg = self._settings.kalshi
        host = (
            "https://demo-api.kalshi.co/trade-api/v2"
            if cfg.environment == "demo"
            else "https://api.elections.kalshi.com/trade-api/v2"
        )

        configuration = Configuration(host=host)
        self._client = _KalshiSDK(configuration=configuration)
        self._client.set_kalshi_auth(
            key_id=cfg.api_key or self._settings.kalshi_api_key,
            private_key_path=cfg.private_key_path or self._settings.kalshi_private_key_path,
        )

        self._events_api = EventsApi(self._client)
        self._markets_api = MarketsApi(self._client)
        self._portfolio_api = PortfolioApi(self._client)

        log.info("kalshi.initialized", host=host, environment=cfg.environment)

    async def _call(self, fn, *args, **kwargs):
        """Run a synchronous SDK call in a thread with rate limiting."""
        async with self._semaphore:
            return await asyncio.to_thread(fn, *args, **kwargs)

    # ------------------------------------------------------------------
    # MarketDiscovery protocol
    # ------------------------------------------------------------------

    async def _get_events_raw(self, **kwargs) -> list[dict]:
        """Fetch events and return raw dicts (bypasses SDK model validation)."""
        import json
        response = await self._call(
            self._events_api.get_events_without_preload_content, **kwargs,
        )
        data = json.loads(response.data)
        return data.get("events", [])

    async def _get_market_raw(self, ticker: str) -> dict | None:
        """Fetch a single market as raw dict."""
        import json
        response = await self._call(
            self._markets_api.get_market_without_preload_content, ticker,
        )
        data = json.loads(response.data)
        return data.get("market")

    async def get_markets(self, active: bool = True, limit: int = 100) -> list[Market]:
        """Fetch markets from Kalshi API."""
        self._init_api()
        try:
            # Kalshi API caps events at 200
            api_limit = min(limit, 200)
            events = await self._get_events_raw(
                limit=api_limit,
                status="open" if active else "closed",
                with_nested_markets=True,
            )

            markets: list[Market] = []
            for event in events:
                event_markets = event.get("markets", [])
                for m in event_markets:
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
            raw = await self._get_market_raw(market_id)
            if raw:
                return self._parse_market(raw)
            return None
        except Exception as e:
            log.error("kalshi.market_fetch_error", market_id=market_id, error=str(e))
            return None

    async def search_markets(self, query: str, limit: int = 50) -> list[Market]:
        """Search Kalshi markets by keyword."""
        self._init_api()
        try:
            events = await self._get_events_raw(
                limit=limit,
                status="open",
                with_nested_markets=True,
            )

            markets: list[Market] = []
            query_lower = query.lower()
            for event in events:
                event_markets = event.get("markets", [])
                for m in event_markets:
                    title = (m.get("title", "") or "").lower()
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
        Prices aggressively to cross the spread and get fills:
        - High edge (>10%): pay 2 ticks through the spread
        - Normal edge: pay 1 tick through the spread
        """
        edge_pct = abs(signal.edge)
        # How aggressively to cross the spread (in dollars)
        aggression = 0.02 if edge_pct > 10 else 0.01

        if signal.recommended_side == OrderSide.BUY:
            side = OrderSide.BUY
            token = TokenType.YES
            # Cross the spread: pay above the ask to guarantee fill
            exec_price = market.outcome_yes_price + market.spread / 2 + aggression
        elif signal.recommended_side == OrderSide.SELL:
            # Check if this is an exit (selling a held position) vs new short
            # If the signal has _exit_token set, sell that token type
            exit_token = getattr(signal, '_exit_token', None)
            if exit_token:
                # Selling a held position — SELL the token we own
                side = OrderSide.SELL
                token = exit_token
                if exit_token == TokenType.NO:
                    exec_price = market.outcome_no_price - market.spread / 2 - aggression
                else:
                    exec_price = market.outcome_yes_price - market.spread / 2 - aggression
            else:
                # New bearish position → BUY NO
                side = OrderSide.BUY
                token = TokenType.NO
                exec_price = market.outcome_no_price + market.spread / 2 + aggression

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

        # Guard: skip if we already have a resting order on the same market + side
        try:
            import json as _json
            existing = await self._call(
                self._portfolio_api.get_orders_without_preload_content,
            )
            existing_data = _json.loads(existing.data)
            kalshi_side = "yes" if order.token == TokenType.YES else "no"
            kalshi_action = "buy" if order.side == OrderSide.BUY else "sell"
            ticker = order.token_id
            for o in existing_data.get("orders", []):
                if (o.get("status") == "resting"
                        and o.get("ticker") == ticker
                        and o.get("side") == kalshi_side
                        and o.get("action") == kalshi_action):
                    log.info(
                        "order.skip_duplicate",
                        exchange="kalshi",
                        ticker=ticker,
                        side=kalshi_side,
                        action=kalshi_action,
                        reason="resting order already exists on same side",
                    )
                    return OrderResult(
                        order_id="SKIP_DUP",
                        market_id=order.market_id,
                        status="rejected",
                        is_paper=False,
                    )
        except Exception as e:
            log.debug("order.dup_check_error", error=str(e))

        log.warning(
            "order.live",
            exchange="kalshi",
            market_id=order.market_id,
            side=order.side.value,
            size=order.size,
            price=order.price,
        )

        try:
            import json
            from kalshi_python import CreateOrderRequest

            # Map token type to Kalshi side, and order side to action
            kalshi_side = "yes" if order.token == TokenType.YES else "no"
            action = "buy" if order.side == OrderSide.BUY else "sell"
            # Kalshi API always takes yes_price in cents (1-99)
            # When buying NO, convert: yes_price = 100 - no_price
            if order.token == TokenType.NO:
                yes_price_cents = max(1, min(99, 100 - int(order.price * 100)))
            else:
                yes_price_cents = max(1, min(99, int(order.price * 100)))

            req = CreateOrderRequest(
                ticker=order.token_id,
                side=kalshi_side,
                action=action,
                count=int(order.size),
                type="limit",
                yes_price=yes_price_cents,
            )

            log.info(
                "order.live_request",
                exchange="kalshi",
                ticker=order.token_id,
                side=kalshi_side,
                action=action,
                count=int(order.size),
                yes_price=yes_price_cents,
            )

            response = await self._call(
                self._portfolio_api.create_order_without_preload_content,
                create_order_request=req,
            )
            data = json.loads(response.data)
            order_data = data.get("order", {})
            order_id = str(order_data.get("order_id", "unknown"))

            log.info(
                "order.live_placed",
                exchange="kalshi",
                order_id=order_id,
                status=order_data.get("status"),
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
            log.error("order.live_error", exchange="kalshi", error=str(e), ticker=order.token_id)
            return OrderResult(
                order_id="ERROR",
                market_id=order.market_id,
                status="rejected",
                is_paper=False,
                error_message=str(e)[:200],
            )

    async def get_order_book(self, market_id: str) -> OrderBook:
        """Get order book for a Kalshi market."""
        self._init_api()
        try:
            import json
            response = await self._call(
                self._markets_api.get_market_orderbook_with_http_info, market_id,
            )
            data = json.loads(response.raw_data)
            # API returns orderbook_fp (dollar strings) or orderbook (cents)
            book = data.get("orderbook_fp", data.get("orderbook", {}))

            bids = []
            for level in (book.get("yes_dollars") or book.get("yes") or book.get("var_true") or []):
                if isinstance(level, list):
                    # [price_str, size_str] in dollars
                    bids.append(OrderBookLevel(price=float(level[0]), size=float(level[1])))
                elif isinstance(level, dict):
                    price = float(level.get("price", 0))
                    if price > 1:
                        price = price / 100
                    bids.append(OrderBookLevel(price=price, size=float(level.get("count", 0))))

            asks = []
            for level in (book.get("no_dollars") or book.get("no") or book.get("var_false") or []):
                if isinstance(level, list):
                    asks.append(OrderBookLevel(price=float(level[0]), size=float(level[1])))
                elif isinstance(level, dict):
                    price = float(level.get("price", 0))
                    if price > 1:
                        price = price / 100
                    asks.append(OrderBookLevel(price=price, size=float(level.get("count", 0))))

            return OrderBook(bids=bids, asks=asks)
        except Exception as e:
            log.error("kalshi.orderbook_error", market_id=market_id, error=str(e))
            return OrderBook()

    async def get_order_status(self, order_id: str) -> OrderResult:
        """Query order status from Kalshi."""
        self._init_api()
        try:
            response = await self._call(self._portfolio_api.get_order, order_id)
            order_data = response.order

            status_map = {
                "resting": "pending",
                "canceled": "cancelled",
                "executed": "filled",
                "pending": "pending",
            }
            raw_status = str(getattr(order_data, "status", "pending")).lower()
            status = status_map.get(raw_status, "pending")

            return OrderResult(
                order_id=order_id,
                market_id=getattr(order_data, "ticker", ""),
                status=status,  # type: ignore[arg-type]
                filled_size=float(
                    getattr(order_data, "count", 0) - getattr(order_data, "remaining_count", 0)
                ),
                filled_price=float(getattr(order_data, "yes_price", 0)) / 100,
                is_paper=False,
            )
        except Exception as e:
            log.error("kalshi.order_status_error", order_id=order_id, error=str(e))
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a Kalshi order."""
        self._init_api()
        try:
            await self._call(self._portfolio_api.cancel_order, order_id)
            log.info("order.cancelled", exchange="kalshi", order_id=order_id)
            return True
        except Exception as e:
            log.error("kalshi.cancel_error", order_id=order_id, error=str(e))
            return False

    async def get_balance(self) -> float:
        """Get account balance in dollars."""
        self._init_api()
        try:
            response = await self._call(self._portfolio_api.get_balance)
            # Balance is returned in cents
            return float(response.balance) / 100
        except Exception as e:
            log.error("kalshi.balance_error", error=str(e))
            return 0.0

    async def sync_positions(self, db) -> int:
        """Sync live Kalshi positions into the portfolio table.

        Pulls positions from the Kalshi API (ground truth) and upserts
        into the DB so the allocator and exit checker can see them.

        Returns the number of active positions synced.
        """
        import json as _json

        self._init_api()
        try:
            resp = await self._call(
                self._portfolio_api.get_positions_without_preload_content,
            )
            data = _json.loads(resp.data)
            positions = data.get("market_positions", [])

            synced = 0
            for p in positions:
                pos_fp = float(p.get("position_fp", 0))
                if pos_fp == 0:
                    continue

                ticker = p.get("ticker", "")
                exposure = float(p.get("market_exposure_dollars", 0))
                contracts = abs(pos_fp)
                token = "NO" if pos_fp < 0 else "YES"
                avg_price = exposure / contracts if contracts > 0 else 0

                # Get current market price for this position
                try:
                    market = await self.get_market(ticker)
                    if market and token == "NO":
                        current_price = market.outcome_no_price
                    elif market:
                        current_price = market.outcome_yes_price
                    else:
                        current_price = avg_price
                except Exception:
                    current_price = avg_price

                await db.execute(
                    """INSERT INTO portfolio
                       (market_id, side, size, avg_price, current_price, token, updated_at)
                       VALUES (?, 'BUY', ?, ?, ?, ?, datetime('now'))
                       ON CONFLICT(market_id) DO UPDATE SET
                           size = excluded.size,
                           avg_price = excluded.avg_price,
                           current_price = excluded.current_price,
                           token = excluded.token,
                           updated_at = excluded.updated_at""",
                    (ticker, contracts, round(avg_price, 4),
                     round(current_price, 4), token),
                )
                synced += 1

            await db.commit()
            if synced > 0:
                log.info("kalshi.positions_synced", count=synced)
            return synced

        except Exception as e:
            log.error("kalshi.sync_positions_error", error=str(e))
            return 0

    async def close(self) -> None:
        """Clean up resources."""
        self._client = None
        self._events_api = None
        self._markets_api = None
        self._portfolio_api = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse_market(self, data) -> Market | None:
        """Parse a Kalshi market (raw dict or SDK model) into our Market model."""
        try:
            # Support both dict and SDK model
            def _get(key: str, default=None):
                if isinstance(data, dict):
                    return data.get(key, default)
                return getattr(data, key, default)

            ticker = _get("ticker", "")

            # New API uses *_dollars fields (string dollar amounts)
            # Old API uses yes_bid/yes_ask (int cents)
            yes_bid_str = _get("yes_bid_dollars")
            yes_ask_str = _get("yes_ask_dollars")
            last_price_str = _get("last_price_dollars")
            volume_str = _get("volume_fp")
            liquidity_str = _get("liquidity_dollars")

            if yes_bid_str is not None:
                # New dollar-based API
                yes_bid = float(yes_bid_str or 0)
                yes_ask = float(yes_ask_str or 0)
                last_price = float(last_price_str or 0)
                volume = float(volume_str or 0)
                reported_liq = float(liquidity_str or 0)
            else:
                # Legacy cents-based API
                yes_bid = float(_get("yes_bid", 0) or 0) / 100
                yes_ask = float(_get("yes_ask", 0) or 0) / 100
                last_price = float(_get("last_price", 50) or 50) / 100
                volume = float(_get("volume", 0) or 0)
                reported_liq = float(_get("liquidity", 0) or 0)

            # Compute liquidity from orderbook data when reported liquidity is 0
            # Priority: top-of-book sizes > open interest > reported liquidity
            liquidity = reported_liq
            if liquidity == 0:
                # Top-of-book depth: contracts available at best bid + ask
                yes_bid_size = float(_get("yes_bid_size_fp", 0) or 0)
                yes_ask_size = float(_get("yes_ask_size_fp", 0) or 0)
                if yes_bid_size > 0 or yes_ask_size > 0:
                    # Dollar liquidity = bid_contracts * bid_price + ask_contracts * ask_price
                    liquidity = (yes_bid_size * yes_bid) + (yes_ask_size * yes_ask)

            if liquidity == 0:
                # Fallback: open interest as proxy (total contracts outstanding)
                # Each contract is worth $1 at resolution, so OI ≈ dollar liquidity
                open_interest = float(_get("open_interest_fp", 0) or 0)
                if open_interest > 0:
                    liquidity = open_interest

            # Use midpoint for fair price (bid for execution would bias SELL signals)
            if yes_bid > 0 and yes_ask > 0:
                yes_price = (yes_bid + yes_ask) / 2
            elif yes_bid > 0:
                yes_price = yes_bid
            else:
                yes_price = last_price
            no_price = 1.0 - yes_price

            end_date = None
            close_time = _get("close_time") or _get("expiration_time")
            if close_time:
                if isinstance(close_time, datetime):
                    end_date = close_time
                elif isinstance(close_time, str):
                    try:
                        end_date = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        pass

            spread = yes_ask - yes_bid if yes_ask > yes_bid else 0.0
            status = str(_get("status", "")).lower()

            return Market(
                id=ticker,
                exchange="kalshi",
                ticker=ticker,
                question=_get("title", "") or "",
                description=_get("subtitle", "") or "",
                category="",
                end_date=end_date,
                active=status in ("open", "active", ""),
                outcome_yes_price=yes_price,
                outcome_no_price=no_price,
                volume=volume,
                liquidity=liquidity,
                spread=spread,
            )
        except Exception as e:
            log.warning("kalshi.parse_error", error=str(e))
            return None
