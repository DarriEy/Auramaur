"""Cross-exchange arbitrage edge validation — READ ONLY.

Polls the same assets' best bid/ask across multiple exchanges, computes the
best after-fee cross-venue spread (buy on the cheap venue, sell on the rich
one), and measures how OFTEN a profitable gap exists. This is the validate-first
step before committing to multi-venue execution — cross-exchange arb is HFT-
dominated, so the real question is whether any capturable edge survives fees at
REST latency.

Places no orders, needs no keys.

    python scripts/research/cross_exchange_scan.py
    python scripts/research/cross_exchange_scan.py --rounds 30 --delay 10

Caveat: USDT-quoted venues (Binance) aren't pure USD — a USDT depeg shows up as
fake spread. Treat USDT-vs-USD gaps with suspicion.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

# taker fee per venue (fraction)
VENUES = {
    "kraken":    {"fee": 0.0025, "quote": "USD"},
    "binance":   {"fee": 0.0010, "quote": "USDT"},
    "gemini":    {"fee": 0.0035, "quote": "USD"},
    "cryptocom": {"fee": 0.0008, "quote": "USD"},
    "bitstamp":  {"fee": 0.0030, "quote": "USD"},
}

# asset -> per-venue symbol
SYMBOLS = {
    "BTC": {"kraken": "XBTUSD", "binance": "BTCUSDT", "gemini": "btcusd",
            "cryptocom": "BTC_USD", "bitstamp": "btcusd"},
    "ETH": {"kraken": "ETHUSD", "binance": "ETHUSDT", "gemini": "ethusd",
            "cryptocom": "ETH_USD", "bitstamp": "ethusd"},
    "SOL": {"kraken": "SOLUSD", "binance": "SOLUSDT", "gemini": "solusd",
            "cryptocom": "SOL_USD", "bitstamp": "solusd"},
}


def _get(url: str) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "auramaur-scan/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.load(r)
    except Exception:
        return None


def _bid_ask(venue: str, sym: str) -> tuple[float, float] | None:
    """Return (bid, ask) for a venue/symbol, or None."""
    try:
        if venue == "kraken":
            d = _get(f"https://api.kraken.com/0/public/Ticker?pair={sym}")
            r = next(iter(d["result"].values()))
            return float(r["b"][0]), float(r["a"][0])
        if venue == "binance":
            d = _get(f"https://api.binance.com/api/v3/ticker/bookTicker?symbol={sym}")
            return float(d["bidPrice"]), float(d["askPrice"])
        if venue == "gemini":
            d = _get(f"https://api.gemini.com/v1/pubticker/{sym}")
            return float(d["bid"]), float(d["ask"])
        if venue == "cryptocom":
            d = _get(f"https://api.crypto.com/exchange/v1/public/get-tickers?instrument_name={sym}")
            t = d["result"]["data"][0]
            return float(t["b"]), float(t["k"])
        if venue == "bitstamp":
            d = _get(f"https://www.bitstamp.net/api/v2/ticker/{sym}/")
            return float(d["bid"]), float(d["ask"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return None


def best_arb(asset: str) -> dict | None:
    """Best after-fee cross-venue spread for an asset this instant."""
    quotes = {}
    for v in VENUES:
        ba = _bid_ask(v, SYMBOLS[asset][v])
        if ba and ba[0] > 0 and ba[1] > 0:
            quotes[v] = ba
    if len(quotes) < 2:
        return None
    best = None
    for buy_v, (_, ask) in quotes.items():
        for sell_v, (bid, _) in quotes.items():
            if buy_v == sell_v:
                continue
            cost = ask * (1 + VENUES[buy_v]["fee"])
            proceeds = bid * (1 - VENUES[sell_v]["fee"])
            net_pct = (proceeds - cost) / cost * 100
            if best is None or net_pct > best["net_pct"]:
                best = {"buy": buy_v, "sell": sell_v, "net_pct": net_pct,
                        "buy_q": VENUES[buy_v]["quote"], "sell_q": VENUES[sell_v]["quote"]}
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--delay", type=int, default=5)
    args = ap.parse_args()

    print("=" * 70)
    print("CROSS-EXCHANGE ARB SCAN (read-only) — after-fee best spread per asset")
    print(f"venues: {', '.join(VENUES)}  |  {args.rounds} rounds x {args.delay}s")
    print("=" * 70)

    profitable = {a: 0 for a in SYMBOLS}
    best_seen = {a: -99.0 for a in SYMBOLS}
    for i in range(args.rounds):
        line = []
        for asset in SYMBOLS:
            b = best_arb(asset)
            if not b:
                line.append(f"{asset}: n/a")
                continue
            best_seen[asset] = max(best_seen[asset], b["net_pct"])
            if b["net_pct"] > 0:
                profitable[asset] += 1
            flag = "  <== PROFIT" if b["net_pct"] > 0 else ""
            note = " *USDT" if "USDT" in (b["buy_q"], b["sell_q"]) else ""
            line.append(f"{asset}: {b['net_pct']:+.3f}% ({b['buy']}->{b['sell']}){note}{flag}")
        print(f"[{i+1:>2}] " + " | ".join(line))
        if i < args.rounds - 1:
            time.sleep(args.delay)

    print("\n" + "=" * 70)
    print("SUMMARY")
    for asset in SYMBOLS:
        print(f"  {asset}: best after-fee spread {best_seen[asset]:+.3f}% | "
              f"profitable in {profitable[asset]}/{args.rounds} rounds")
    any_prof = any(profitable.values())
    print(f"\n  VERDICT: {'capturable gaps seen — worth deeper study' if any_prof else 'NO after-fee edge at REST latency — cross-exchange arb not viable here'}")
    print("  (a real opportunity must also PERSIST long enough to execute both legs)")


if __name__ == "__main__":
    main()
