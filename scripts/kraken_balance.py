"""Connect to your Kraken spot wallet and print balances — READ ONLY.

This calls only the private ``Balance`` (and ``TradeBalance``) endpoints, which
require nothing beyond the "Query Funds" API permission. It places no orders and
cannot move funds. It's the safe way to confirm your Kraken key is wired up.

Setup:
  1. kraken.com -> Settings -> API -> Create API key
     - Permission: enable "Query Funds" ONLY. Leave "Withdraw Funds" OFF.
  2. Put the key + private key in .env:
       KRAKEN_API_KEY=...
       KRAKEN_API_SECRET=...      (the long base64 "Private Key")
  3. python scripts/kraken_balance.py

Auth follows Kraken's scheme: API-Sign = HMAC-SHA512(path + SHA256(nonce + body),
base64-decoded secret), base64-encoded.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import urllib.parse
import urllib.request

from config.settings import Settings

_API = "https://api.kraken.com"


def _sign(path: str, data: dict, secret: str) -> str:
    post = urllib.parse.urlencode(data)
    encoded = (str(data["nonce"]) + post).encode()
    message = path.encode() + hashlib.sha256(encoded).digest()
    sig = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
    return base64.b64encode(sig.digest()).decode()


def _private(path: str, key: str, secret: str) -> dict:
    data = {"nonce": int(time.time() * 1000)}
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(
        _API + path,
        data=body,
        headers={
            "API-Key": key,
            "API-Sign": _sign(path, data, secret),
            "User-Agent": "auramaur-kraken-balance/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main() -> None:
    s = Settings()
    key, secret = s.kraken_api_key, s.kraken_api_secret
    if not key or not secret:
        print("KRAKEN_API_KEY / KRAKEN_API_SECRET not set in .env — see this "
              "file's docstring for setup.")
        return

    print("Connecting to Kraken spot wallet (read-only)...\n")
    try:
        bal = _private("/0/private/Balance", key, secret)
    except Exception as e:  # noqa: BLE001
        print(f"Request failed: {type(e).__name__}: {str(e)[:160]}")
        return

    if bal.get("error"):
        # Common: 'EAPI:Invalid key', 'EGeneral:Permission denied',
        # 'EAPI:Invalid nonce' (system clock skew).
        print("Kraken returned an error:", bal["error"])
        return

    funds = {k: float(v) for k, v in bal.get("result", {}).items() if float(v) != 0}
    if not funds:
        print("Connected ✓ — wallet has no non-zero balances.")
    else:
        print("Connected ✓ — non-zero balances:")
        for asset, amt in sorted(funds.items()):
            print(f"  {asset:8} {amt:.8f}")

    # Equity summary (also read-only).
    try:
        tb = _private("/0/private/TradeBalance", key, secret)
        eq = tb.get("result", {}).get("eb")
        if eq is not None:
            print(f"\n  Total equivalent balance: {float(eq):.2f} (quote ccy)")
    except Exception:
        pass


if __name__ == "__main__":
    main()
