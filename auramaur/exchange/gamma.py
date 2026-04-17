"""Gamma API client for Polymarket market discovery (unauthenticated)."""

from __future__ import annotations

import time
from datetime import datetime

import aiohttp
import structlog

from auramaur.exchange.models import Market

log = structlog.get_logger()

GAMMA_API_BASE = "https://gamma-api.polymarket.com"

# How long (seconds) to remember that a market_id returned 4xx, so we don't
# re-hit the API every cycle for dead/orphaned IDs.
_NEGATIVE_CACHE_TTL = 3600.0


class GammaClient:
    """Unauthenticated Gamma API for market discovery and filtering."""

    def __init__(self):
        self._session: aiohttp.ClientSession | None = None
        # market_id -> expiry timestamp for ids that returned 4xx
        self._negative_cache: dict[str, float] = {}

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_markets(
        self,
        active: bool = True,
        limit: int = 100,
        offset: int = 0,
        order: str = "volume",
        ascending: bool = False,
    ) -> list[Market]:
        """Fetch markets from Gamma API."""
        session = await self._ensure_session()
        params = {
            "limit": limit,
            "offset": offset,
            "order": order,
            "ascending": str(ascending).lower(),
            "active": str(active).lower(),
            "closed": "false",
        }

        try:
            async with session.get(f"{GAMMA_API_BASE}/markets", params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()

            markets = []
            for item in data:
                market = self._parse_market(item)
                if market:
                    markets.append(market)

            log.info("gamma.markets_fetched", count=len(markets))
            return markets

        except aiohttp.ClientError as e:
            log.error("gamma.fetch_error", error=str(e))
            return []

    async def get_market(self, market_id: str) -> Market | None:
        """Fetch a single market by ID."""
        if not market_id:
            return None

        # Gamma's /markets/{id} expects a numeric id. Hex condition-id stubs
        # (e.g. orphaned "0xfac8e82eb828a0" stored by the reconciler when it
        # couldn't map a condition_id to a numeric market) will always 422.
        # Skip them silently instead of hammering the API every cycle.
        if market_id.startswith("0x"):
            return None

        expiry = self._negative_cache.get(market_id)
        if expiry is not None:
            if expiry > time.monotonic():
                return None
            self._negative_cache.pop(market_id, None)

        session = await self._ensure_session()
        try:
            async with session.get(f"{GAMMA_API_BASE}/markets/{market_id}") as resp:
                if resp.status == 404 or resp.status == 422:
                    self._negative_cache[market_id] = time.monotonic() + _NEGATIVE_CACHE_TTL
                    log.debug(
                        "gamma.market_not_found",
                        market_id=market_id,
                        status=resp.status,
                    )
                    return None
                resp.raise_for_status()
                data = await resp.json()
                return self._parse_market(data)
        except aiohttp.ClientError as e:
            log.error("gamma.market_fetch_error", market_id=market_id, error=str(e))
            return None

    async def search_markets(self, query: str, limit: int = 50) -> list[Market]:
        """Search markets by keyword.

        Uses the /public-search endpoint, which actually honors the query
        parameter. /markets?query=... silently ignores the filter and returns
        the default top-of-volume list, so every headline ends up matching the
        same handful of meme markets — breaking the news reactor.
        """
        session = await self._ensure_session()
        params = {
            "q": query,
            "limit_per_type": min(limit, 20),
            "events_status": "active",
        }

        try:
            async with session.get(
                f"{GAMMA_API_BASE}/public-search", params=params
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

            markets: list[Market] = []
            seen: set[str] = set()
            for event in (data.get("events") or []):
                for item in (event.get("markets") or []):
                    if not item.get("active") or item.get("closed"):
                        continue
                    market = self._parse_market(item)
                    if market and market.id not in seen:
                        seen.add(market.id)
                        markets.append(market)
                    if len(markets) >= limit:
                        break
                if len(markets) >= limit:
                    break
            return markets

        except aiohttp.ClientError as e:
            log.error("gamma.search_error", query=query, error=str(e))
            return []

    def _parse_market(self, data: dict) -> Market | None:
        """Parse a Gamma API market response into our Market model."""
        try:
            # Extract prices from outcomes/tokens
            yes_price = 0.5
            no_price = 0.5
            tokens = data.get("tokens", []) or data.get("clobTokenIds", [])

            if isinstance(data.get("outcomePrices"), str):
                import json
                prices = json.loads(data["outcomePrices"])
                if len(prices) >= 2:
                    yes_price = float(prices[0])
                    no_price = float(prices[1])
            elif isinstance(data.get("outcomePrices"), list) and len(data["outcomePrices"]) >= 2:
                yes_price = float(data["outcomePrices"][0])
                no_price = float(data["outcomePrices"][1])
            elif data.get("bestBid") is not None:
                yes_price = float(data.get("bestBid", 0.5))
                no_price = 1.0 - yes_price

            end_date = None
            if data.get("endDate"):
                try:
                    end_date = datetime.fromisoformat(data["endDate"].replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    pass

            spread = 0.0
            if data.get("bestAsk") is not None and data.get("bestBid") is not None:
                spread = float(data["bestAsk"]) - float(data["bestBid"])

            # Extract CLOB token IDs for YES/NO outcomes
            clob_yes = ""
            clob_no = ""
            raw_clob = data.get("clobTokenIds", [])
            if isinstance(raw_clob, str):
                import json as _json
                try:
                    raw_clob = _json.loads(raw_clob)
                except (ValueError, TypeError):
                    raw_clob = []
            if isinstance(raw_clob, list) and len(raw_clob) >= 2:
                clob_yes = str(raw_clob[0])
                clob_no = str(raw_clob[1])

            return Market(
                id=str(data.get("id", data.get("conditionId", ""))),
                exchange="polymarket",
                condition_id=str(data.get("conditionId", "")),
                question=data.get("question", ""),
                description=data.get("description", ""),
                category=data.get("category", data.get("groupSlug", "")),
                end_date=end_date,
                active=data.get("active", True),
                outcome_yes_price=yes_price,
                outcome_no_price=no_price,
                volume=float(data.get("volume", 0) or 0),
                liquidity=float(data.get("liquidity", 0) or 0),
                spread=spread,
                clob_token_yes=clob_yes,
                clob_token_no=clob_no,
            )
        except Exception as e:
            log.warning("gamma.parse_error", error=str(e), data_keys=list(data.keys()))
            return None
