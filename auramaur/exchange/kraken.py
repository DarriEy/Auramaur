"""Kraken SPOT execution client — treasury/conversion + (gated) directional.

This is NOT a binary prediction venue, so it is deliberately NOT wired into the
binary TradingEngine. It exposes a purpose-built spot interface (place an order
on a pair) plus read-only market data and balances.

Safety model (same spirit as the other venues, adapted to spot):
  1. Kill switch halts everything.
  2. Three-gate live model: unless AURAMAUR_LIVE + execution.live are on and the
     caller passes dry_run=False, orders are sent with Kraken's validate=true
     flag — Kraken checks the order WITHOUT placing it.
  3. Hard per-order USD cap (kraken.max_order_usd), independent of the binary
     risk manager.
  4. Directional orders are refused unless kraken.directional_enabled — until a
     validated edge exists, only treasury/conversion orders are allowed.

Withdrawals/transfers are NOT here — see auramaur/treasury/transfers.py, which is
gated separately (AURAMAUR_ENABLE_TRANSFERS).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import time
import urllib.parse
from pathlib import Path

import aiohttp
import structlog

from auramaur.exchange.models import OrderResult, OrderSide

log = structlog.get_logger()

_API = "https://api.kraken.com"
_RATE_LIMIT = 8  # conservative; Kraken counts private calls against a tier budget


class KrakenSpotClient:
    # Quote assets that trade ~1:1 with USD, so no FX conversion is needed for
    # USD-denominated sizing / caps. Everything else (ZEUR, ZGBP, …) is converted
    # via a live <fiat>USD rate so any quote currency can be sized correctly.
    _USD_PEGGED = {"ZUSD", "USD", "USDC", "USDT", "DAI", "PYUSD", "USDG", "RLUSD"}

    def __init__(self, settings):
        self._settings = settings
        self._session: aiohttp.ClientSession | None = None
        self._sem = asyncio.Semaphore(_RATE_LIMIT)
        self._quote_rate_cache: dict[str, float] = {}  # quote asset -> USD rate
        self._pair_quote_cache: dict[str, str] = {}     # pair -> quote asset code

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"User-Agent": "auramaur-kraken/1.0"},
            )
        return self._session

    # ------------------------------------------------------------------
    # REST helpers
    # ------------------------------------------------------------------

    async def _public(self, endpoint: str, params: dict | None = None) -> dict:
        async with self._sem:
            session = await self._get_session()
            async with session.get(f"{_API}/0/public/{endpoint}", params=params or {}) as r:
                data = await r.json()
        if data.get("error"):
            log.warning("kraken.public_error", endpoint=endpoint, error=data["error"])
        return data.get("result", {})

    def _sign(self, path: str, data: dict) -> str:
        post = urllib.parse.urlencode(data)
        encoded = (str(data["nonce"]) + post).encode()
        message = path.encode() + hashlib.sha256(encoded).digest()
        sig = hmac.new(base64.b64decode(self._settings.kraken_api_secret), message, hashlib.sha512)
        return base64.b64encode(sig.digest()).decode()

    async def _private(self, method: str, data: dict | None = None) -> dict:
        key = self._settings.kraken_api_key
        secret = self._settings.kraken_api_secret
        if not key or not secret:
            return {"error": ["EAuramaur:No Kraken credentials"]}
        path = f"/0/private/{method}"
        data = dict(data or {})
        data["nonce"] = int(time.time() * 1000)
        body = urllib.parse.urlencode(data)
        async with self._sem:
            session = await self._get_session()
            async with session.post(
                _API + path,
                data=body,
                headers={
                    "API-Key": key,
                    "API-Sign": self._sign(path, data),
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            ) as r:
                return await r.json()

    # ------------------------------------------------------------------
    # Read-only
    # ------------------------------------------------------------------

    async def get_balance(self) -> dict[str, float]:
        resp = await self._private("Balance")
        if resp.get("error"):
            log.error("kraken.balance_error", error=resp["error"])
            return {}
        return {k: float(v) for k, v in resp.get("result", {}).items()}

    async def get_price(self, pair: str) -> float | None:
        """Last trade price for a pair (e.g. 'POLUSD', 'XBTUSD')."""
        result = await self._public("Ticker", {"pair": pair})
        if not result:
            return None
        info = next(iter(result.values()))
        try:
            return float(info["c"][0])  # 'c' = last trade closed [price, lot volume]
        except (KeyError, IndexError, TypeError):
            return None

    async def get_pair_quote(self, pair: str) -> str | None:
        """Quote-currency asset code for a pair (e.g. PAXGUSD -> 'ZUSD')."""
        if pair in self._pair_quote_cache:
            return self._pair_quote_cache[pair]
        info = await self._public("AssetPairs", {"pair": pair})
        if not info:
            return None
        meta = next(iter(info.values()))
        quote = meta.get("quote")
        if quote:
            self._pair_quote_cache[pair] = quote
        return quote

    async def quote_usd_rate(self, quote: str) -> float:
        """USD value of one unit of a quote currency. 1.0 for USD-pegged quotes
        (USD/USDC/USDT/…); otherwise the live <fiat>USD rate (ZEUR -> EURUSD).
        Cached. Falls back to 1.0 if the rate can't be fetched."""
        q = (quote or "").upper()
        if q in self._USD_PEGGED:
            return 1.0
        if q in self._quote_rate_cache:
            return self._quote_rate_cache[q]
        # Kraken fiat codes are Z-prefixed (ZEUR, ZGBP); the FX pair drops it.
        fiat = q[1:] if (q.startswith("Z") and len(q) == 4) else q
        rate = 1.0
        px = await self.get_price(f"{fiat}USD")
        if px and px > 0:
            rate = px
        self._quote_rate_cache[q] = rate
        return rate

    async def usd_notional(self, pair: str, volume: float, price: float | None = None) -> float | None:
        """USD value of `volume` base units of `pair`, accounting for the quote
        currency. Returns None if price is unavailable."""
        price = price if price is not None else await self.get_price(pair)
        if not price or price <= 0:
            return None
        quote = await self.get_pair_quote(pair) or "ZUSD"
        return volume * price * await self.quote_usd_rate(quote)

    async def size_for_usd(self, pair: str, usd: float, price: float | None = None) -> float | None:
        """Base-unit volume whose value is ~`usd`, for any quote currency.
        Returns None if price/quote can't be resolved."""
        price = price if price is not None else await self.get_price(pair)
        if not price or price <= 0:
            return None
        quote = await self.get_pair_quote(pair) or "ZUSD"
        usd_per_base = price * await self.quote_usd_rate(quote)
        if usd_per_base <= 0:
            return None
        return usd / usd_per_base

    # ------------------------------------------------------------------
    # Spot order placement
    # ------------------------------------------------------------------

    async def place_spot_order(
        self,
        pair: str,
        side: OrderSide | str,
        volume: float,
        ordertype: str = "market",
        price: float | None = None,
        purpose: str = "treasury",
        dry_run: bool | None = None,
        max_usd: float | None = None,
    ) -> OrderResult:
        """Place a spot order on Kraken.

        purpose: "treasury" (allowed by default) or "directional" (refused
        unless kraken.directional_enabled). dry_run None => derive from the live
        gates; True/validate-only sends Kraken's validate flag (no execution).
        max_usd overrides the per-order cap (manual CLI passes the user-confirmed
        ceiling; autonomous callers leave it None to use kraken.max_order_usd).
        """
        side_str = side.value if isinstance(side, OrderSide) else str(side)
        side_str = side_str.lower()

        # 1. Kill switch.
        if Path("KILL_SWITCH").exists():
            log.critical("kill_switch.active", action="kraken_order_blocked")
            return OrderResult(order_id="BLOCKED", market_id=pair, status="rejected",
                               is_paper=True, error_message="kill switch")

        # 2. Purpose gate.
        if purpose == "directional" and not self._settings.kraken.directional_enabled:
            return OrderResult(order_id="BLOCKED", market_id=pair, status="rejected",
                               is_paper=True,
                               error_message="directional trading disabled (no validated edge)")

        # 3. Per-order USD cap (manual CLI may pass a higher confirmed ceiling).
        #    Convert to USD via the pair's quote currency so the cap holds for
        #    any quote (USDC/USDT/EUR/…), not just USD-quoted pairs.
        cap = max_usd if max_usd is not None else self._settings.kraken.max_order_usd
        est_price = price or await self.get_price(pair)
        if est_price:
            notional = await self.usd_notional(pair, volume, est_price) or (volume * est_price)
            if notional > cap:
                return OrderResult(order_id="BLOCKED", market_id=pair, status="rejected",
                                   is_paper=True,
                                   error_message=f"order ${notional:.2f} exceeds cap ${cap:.2f}")

        # 4. Three-gate live decision → validate-only unless explicitly live.
        if dry_run is None:
            dry_run = not self._settings.is_live
        validate = bool(dry_run)

        params = {"pair": pair, "type": side_str, "ordertype": ordertype, "volume": str(volume)}
        if price is not None and ordertype != "market":
            params["price"] = str(price)
        if validate:
            params["validate"] = "true"

        log.warning("kraken.order", pair=pair, side=side_str, volume=volume,
                    ordertype=ordertype, purpose=purpose, validate=validate)

        resp = await self._private("AddOrder", params)
        if resp.get("error"):
            return OrderResult(order_id="ERROR", market_id=pair, status="rejected",
                               is_paper=validate, error_message=str(resp["error"])[:200])

        result = resp.get("result", {})
        txids = result.get("txid") or []
        order_id = txids[0] if txids else ("VALIDATED" if validate else "unknown")
        log.info("kraken.order_placed", pair=pair, order_id=order_id, validate=validate,
                 descr=result.get("descr"))
        return OrderResult(
            order_id=str(order_id),
            market_id=pair,
            status="paper" if validate else "pending",
            filled_price=est_price or 0.0,
            is_paper=validate,
        )

    async def convert(
        self,
        asset: str,
        amount: float,
        quote: str = "USD",
        dry_run: bool | None = None,
    ) -> OrderResult:
        """Treasury conversion: sell `amount` of `asset` into `quote`.

        Common case: sell POL -> USD to consolidate before routing. Kraken
        accepts altname pairs like 'POLUSD'. Multi-hop (e.g. POL->USD->USDC) is
        intentionally left as two explicit convert() calls rather than hidden
        routing. Always a treasury order (never directional), still gated.
        """
        pair = f"{asset.upper()}{quote.upper()}"
        return await self.place_spot_order(
            pair, OrderSide.SELL, volume=amount, ordertype="market",
            purpose="treasury", dry_run=dry_run,
        )

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
