"""Backtest the Kraken directional pillar's EXACT momentum rule — read-only.

The pillar goes long when momentum over `lookback` bars >= threshold%, exits when
it drops to <= -threshold%. This backtests that rule on historical Kraken OHLC
for the configured pairs, net of taker fees, vs buy-and-hold — to find whether
the speculation has real edge before we deploy more into it.

    python scripts/research/momentum_backtest.py
    python scripts/research/momentum_backtest.py --lookback 12 --threshold 3 --interval 60
"""

from __future__ import annotations

import argparse
import json
import urllib.request

TAKER_FEE = 0.0025  # Kraken spot taker, per side

PAIRS = ["XBTUSDC", "ETHUSDC", "SOLUSDC", "XRPUSDC", "ADAUSDC", "AVAXUSDC",
         "DOTUSDC", "LINKUSDC", "LTCUSDC", "BCHUSDC", "ATOMUSDC", "ALGOUSDC",
         "TONUSDC", "XMRUSDC", "XTZUSDC", "XDGUSDC"]


def _ohlc(pair: str, interval: int) -> list[float]:
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"
    try:
        d = json.load(urllib.request.urlopen(url, timeout=15))
        series = next(v for k, v in d["result"].items() if k != "last")
        return [float(c[4]) for c in series]  # close prices
    except Exception:
        return []


def backtest(closes: list[float], lookback: int, threshold: float) -> dict | None:
    if len(closes) < lookback + 5:
        return None
    equity = 1.0          # strategy equity (multiplicative)
    long = False
    entry = 0.0
    trips, wins = 0, 0
    for i in range(lookback, len(closes)):
        mom = (closes[i] - closes[i - lookback]) / closes[i - lookback] * 100
        if not long and mom >= threshold:
            long = True
            entry = closes[i] * (1 + TAKER_FEE)   # pay fee on entry
        elif long and mom <= -threshold:
            exit_px = closes[i] * (1 - TAKER_FEE)  # pay fee on exit
            r = exit_px / entry
            equity *= r
            trips += 1
            wins += 1 if r > 1 else 0
            long = False
    if long:  # mark-to-market open position
        equity *= closes[-1] * (1 - TAKER_FEE) / entry
        trips += 1
        wins += 1 if closes[-1] * (1 - TAKER_FEE) > entry else 0
    bh = closes[-1] / closes[0]   # buy-and-hold over same window
    return {"strat": (equity - 1) * 100, "bh": (bh - 1) * 100,
            "trips": trips, "wins": wins}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=12)
    ap.add_argument("--threshold", type=float, default=3.0)
    ap.add_argument("--interval", type=int, default=60)  # minutes per bar
    args = ap.parse_args()

    print("=" * 70)
    print(f"KRAKEN MOMENTUM BACKTEST — lookback {args.lookback} bars, "
          f"threshold {args.threshold}%, {args.interval}m bars, fee {TAKER_FEE:.2%}/side")
    print("=" * 70)
    print(f"  {'pair':10} {'strat%':>8} {'buy&hold%':>10} {'trips':>6} {'win%':>6}  vs B&H")
    print("  " + "-" * 56)

    tot_strat, tot_bh, beat, n = 0.0, 0.0, 0, 0
    for p in PAIRS:
        closes = _ohlc(p, args.interval)
        r = backtest(closes, args.lookback, args.threshold)
        if not r:
            continue
        n += 1
        tot_strat += r["strat"]; tot_bh += r["bh"]
        beat += 1 if r["strat"] > r["bh"] else 0
        wr = f"{r['wins']/r['trips']*100:.0f}%" if r["trips"] else "—"
        flag = "WIN" if r["strat"] > r["bh"] else ""
        print(f"  {p:10} {r['strat']:>8.1f} {r['bh']:>10.1f} {r['trips']:>6} {wr:>6}  {flag}")

    if n:
        print("\n" + "=" * 70)
        print(f"  AVG strategy {tot_strat/n:+.1f}%  vs  buy&hold {tot_bh/n:+.1f}%  "
              f"over the window")
        print(f"  momentum beat buy&hold in {beat}/{n} pairs")
        edge = tot_strat / n > tot_bh / n and beat > n / 2
        print(f"\n  VERDICT: {'momentum shows edge vs B&H — worth tuning/deploying' if edge else 'NO edge vs buy-and-hold — momentum is not adding value'}")


if __name__ == "__main__":
    main()
