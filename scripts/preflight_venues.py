"""Read-only venue connectivity preflight.

Reports the live-trading gate state and, for each enabled exchange, whether
credentials are present and whether read-only market discovery succeeds.

Places NO orders and touches no money — it only calls public/discovery
endpoints. Safe to run at any time, in any gate state.

    python scripts/preflight_venues.py
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
import urllib.parse
import urllib.request

from config.settings import Settings


def _mask(present: bool) -> str:
    return "set" if present else "MISSING"


def _kraken_balance(key: str, secret: str) -> str:
    """Read-only Kraken spot balance probe (Query Funds permission). Never raises."""
    path = "/0/private/Balance"
    data = {"nonce": int(time.time() * 1000)}
    post = urllib.parse.urlencode(data)
    msg = path.encode() + hashlib.sha256((str(data["nonce"]) + post).encode()).digest()
    sign = base64.b64encode(hmac.new(base64.b64decode(secret), msg, hashlib.sha512).digest()).decode()
    req = urllib.request.Request(
        "https://api.kraken.com" + path,
        data=post.encode(),
        headers={"API-Key": key, "API-Sign": sign, "User-Agent": "auramaur-preflight/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            body = json.load(r)
    except Exception as e:  # noqa: BLE001
        return f"FAILED — {type(e).__name__}: {str(e)[:100]}"
    if body.get("error"):
        return f"FAILED — {body['error']}"
    nonzero = sum(1 for v in body.get("result", {}).values() if float(v) != 0)
    return f"OK — wallet reachable, {nonzero} non-zero asset(s)"


async def _probe(name: str, client) -> str:
    """Deep read-only probe: discovery latency, book quality, signed access.

    Multi-line report; never raises. Grades each aspect independently so a
    closed venue or missing entitlement reads as WARN, not FAIL.
    """
    lines = []
    try:
        t0 = time.monotonic()
        markets = await client.get_markets(active=True, limit=25)
        ms = (time.monotonic() - t0) * 1000
        grade = "OK" if markets and ms < 5000 else "WARN"
        lines.append(f"{grade} — {len(markets)} active market(s) in {ms:.0f}ms")

        book_fn = getattr(client, "get_order_book", None)
        if book_fn is not None and markets:
            market = markets[0]
            market_id = getattr(market, "id", None) or getattr(market, "ticker", None)
            try:
                t0 = time.monotonic()
                book = await book_fn(market_id)
                ms = (time.monotonic() - t0) * 1000
                bids = getattr(book, "bids", None) or []
                asks = getattr(book, "asks", None) or []
                if bids and asks:
                    best_bid = max(float(getattr(b, "price", 0.0) or 0.0) for b in bids)
                    best_ask = min(float(getattr(a, "price", 1.0) or 1.0) for a in asks)
                    lines.append(
                        f"  book      : OK — {len(bids)}x{len(asks)} levels, "
                        f"bbo {best_bid:.2f}/{best_ask:.2f} in {ms:.0f}ms "
                        f"({market_id})")
                else:
                    lines.append(
                        f"  book      : WARN — empty book on {market_id} "
                        f"(venue quiet or closed)")
            except Exception as e:  # noqa: BLE001
                lines.append(f"  book      : WARN — {type(e).__name__}: {str(e)[:90]}")

        balance_fn = getattr(client, "get_balance", None)
        if balance_fn is not None:
            try:
                t0 = time.monotonic()
                balance = await balance_fn()
                ms = (time.monotonic() - t0) * 1000
                note = " (instant: local paper value, not a venue call)" if ms < 5 else ""
                lines.append(
                    f"  signed api: OK — balance ${float(balance):,.2f} in {ms:.0f}ms{note}")
            except Exception as e:  # noqa: BLE001
                lines.append(
                    f"  signed api: FAILED — {type(e).__name__}: {str(e)[:90]}")
        return "\n".join(lines)
    except Exception as e:  # noqa: BLE001 — preflight must report, not crash
        return f"FAILED — {type(e).__name__}: {str(e)[:120]}"
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:
                pass


async def main() -> None:
    s = Settings()

    print("=" * 60)
    print("LIVE-TRADING GATES")
    print("=" * 60)
    print(f"  AURAMAUR_LIVE env       : {s.auramaur_live}")
    print(f"  execution.live (config) : {s.execution.live}")
    print(f"  KILL_SWITCH present     : {s.kill_switch_active}")
    print(f"  --> is_live             : {s.is_live}")
    print(f"      ({'REAL ORDERS' if s.is_live else 'paper mode — orders intercepted'})")

    print()
    print("=" * 60)
    print("VENUES")
    print("=" * 60)

    # Polymarket — always available (no enable flag); discovery is public.
    from auramaur.exchange.gamma import GammaClient
    print("\npolymarket  [always on]")
    print(f"  POLYGON_PRIVATE_KEY : {_mask(bool(s.polygon_private_key))}")
    print(f"  POLYMARKET_API_KEY  : {_mask(bool(s.polymarket_api_key))}")
    print(f"  discovery           : {await _probe('polymarket', GammaClient())}")
    if s.polygon_private_key:
        try:
            from auramaur.exchange.client import PolymarketClient
            clob = PolymarketClient(settings=s, paper_trader=None)
            gamma = GammaClient()
            try:
                sample = await gamma.get_markets(active=True, limit=1)
                token = None
                if sample:
                    token = (getattr(sample[0], "clob_token_yes", None)
                             or getattr(sample[0], "clob_token_no", None))
                if token:
                    t0 = time.monotonic()
                    book = await clob.get_order_book(token)
                    ms = (time.monotonic() - t0) * 1000
                    depth = len(getattr(book, "bids", []) or [])
                    print(f"  signed clob         : OK — book fetch ({depth} bid levels) in {ms:.0f}ms")
                else:
                    print("  signed clob         : WARN — no token id on sampled market")
            finally:
                for c in (clob, gamma):
                    close = getattr(c, "close", None)
                    if close is not None:
                        try:
                            await close()
                        except Exception:
                            pass
        except Exception as e:  # noqa: BLE001
            print(f"  signed clob         : FAILED — {type(e).__name__}: {str(e)[:90]}")

    # Kalshi
    print(f"\nkalshi  [enabled={s.kalshi.enabled}, env={s.kalshi.environment}]")
    print(f"  KALSHI_API_KEY          : {_mask(bool(s.kalshi.api_key or s.kalshi_api_key))}")
    print(f"  KALSHI_PRIVATE_KEY_PATH : {_mask(bool(s.kalshi.private_key_path or s.kalshi_private_key_path))}")
    if s.kalshi.enabled:
        from auramaur.exchange.kalshi import KalshiClient
        print(f"  discovery               : {await _probe('kalshi', KalshiClient(settings=s, paper_trader=None))}")
    else:
        print("  discovery               : skipped (disabled)")

    # Crypto.com
    print(f"\ncryptodotcom  [enabled={s.cryptodotcom.enabled}, env={s.cryptodotcom.environment}]")
    print(f"  CRYPTODOTCOM_API_KEY    : {_mask(bool(s.cryptodotcom.api_key or s.cryptodotcom_api_key))}")
    print(f"  CRYPTODOTCOM_API_SECRET : {_mask(bool(s.cryptodotcom.api_secret or s.cryptodotcom_api_secret))}")
    if s.cryptodotcom.enabled:
        from auramaur.exchange.cryptodotcom import CryptoComClient
        print(f"  discovery               : {await _probe('cryptodotcom', CryptoComClient(settings=s, paper_trader=None))}")
    else:
        print("  discovery               : skipped (disabled)")

    # IBKR — connects to a local TWS/Gateway socket, no API key.
    print(f"\nibkr  [enabled={s.ibkr.enabled}, env={s.ibkr.environment}, "
          f"host={s.ibkr.host}, port={s.ibkr.paper_port if s.ibkr.environment == 'paper' else s.ibkr.live_port}]")
    if s.ibkr.enabled:
        from auramaur.exchange.ibkr import IBKRClient
        print(f"  discovery               : {await _probe('ibkr', IBKRClient(settings=s, paper_trader=None))}")
        print("  (a FAILED here usually means TWS / IB Gateway is not running)")
    else:
        print("  discovery               : skipped (disabled)")

    # Kraken — read-only wallet check only. NOT a trading venue (no adapter):
    # the bot cannot place Kraken orders. This just confirms the wallet key.
    print("\nkraken  [wallet read-only — no trading adapter]")
    print(f"  KRAKEN_API_KEY          : {_mask(bool(s.kraken_api_key))}")
    print(f"  KRAKEN_API_SECRET       : {_mask(bool(s.kraken_api_secret))}")
    if s.kraken_api_key and s.kraken_api_secret:
        print(f"  wallet                  : {_kraken_balance(s.kraken_api_key, s.kraken_api_secret)}")
    else:
        print("  wallet                  : skipped (no key)")

    print()


if __name__ == "__main__":
    asyncio.run(main())
