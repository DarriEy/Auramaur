"""Backtest the Kraken directional (spec) momentum signal on real OHLC.

Read-only: pulls 30 days of hourly candles from Kraken's PUBLIC market-data API
(no auth, no orders) and replays the live entry/exit logic from
kraken_pillar.py so we can compare the current signal against candidate
redesigns BEFORE touching anything live.

Net P&L per trip = gain% - round-trip fee%. "total" sums trip returns (each trip
deploys one max_order_usd slug, so summed trip% ~ return on the per-slot capital;
good enough for ranking signals). Usage: python scripts/kraken_directional_backtest.py
"""

from __future__ import annotations

import json
import time
import urllib.request
from statistics import mean

PAIRS = ["XBTUSDC", "ETHUSDC", "SOLUSDC", "XRPUSDC", "ADAUSDC", "AVAXUSDC",
         "DOTUSDC", "LINKUSDC", "LTCUSDC", "BCHUSDC", "ATOMUSDC", "TONUSDC",
         "XTZUSDC", "MANAUSDC", "APEUSDC", "VETUSDC", "XDGUSDC", "SHIBUSDC"]

FEE_PCT = 0.26  # per side (config directional_fee_pct)


def fetch_closes(pair: str) -> list[float] | None:
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval=60"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            d = json.loads(r.read())
    except Exception as e:
        print(f"  fetch fail {pair}: {e}")
        return None
    if d.get("error"):
        print(f"  api error {pair}: {d['error']}")
        return None
    for k, v in d.get("result", {}).items():
        if k != "last" and isinstance(v, list):
            return [float(c[4]) for c in v]  # close
    return None


def simulate(closes: list[float], p: dict) -> list[dict]:
    """Replay entry/exit over one pair's close series. Returns list of trips."""
    lb = p["lb"]; entry = p["entry"]; exit_thr = p["exit_thr"]
    tp = p["tp"]; trail = p["trail"]; stop = p["stop"]
    cooldown = p["cooldown_bars"]; min_hold = p.get("min_hold", 0)
    sma_n = p.get("sma_n", 0)
    rt_fee = 2 * FEE_PCT
    trips: list[dict] = []
    pos = None  # (entry_px, entry_idx, peak_pct)
    cooldown_until = -1

    side = p.get("side", "mom")  # 'mom' = buy strength; 'rev' = buy weakness
    start = max(lb, sma_n)
    for i in range(start, len(closes)):
        c = closes[i]
        if closes[i - lb] <= 0:
            continue
        mom = (c - closes[i - lb]) / closes[i - lb] * 100.0
        if pos is None:
            if i < cooldown_until:
                continue
            if side == "mom":
                if mom < entry:
                    continue
                if sma_n and c < mean(closes[i - sma_n:i]):  # uptrend filter
                    continue
            else:  # mean reversion: buy a dip
                if mom > -entry:          # require a down-move of >= entry%
                    continue
                if sma_n and c > mean(closes[i - sma_n:i]):  # only buy below trend
                    continue
            pos = (c, i, 0.0)
        else:
            entry_px, entry_idx, peak = pos
            gain = (c - entry_px) / entry_px * 100.0
            peak = max(peak, gain)
            held = i - entry_idx
            reason = None
            if stop > 0 and gain <= -stop:
                reason = "stop"
            elif tp > 0 and (gain - rt_fee) >= tp:
                reason = "tp"
            elif trail > 0 and peak > rt_fee and (peak - gain) >= trail:
                reason = "trail"
            elif held >= min_hold and (
                (side == "mom" and mom <= -exit_thr)      # trend rolled over
                or (side == "rev" and mom >= exit_thr)     # mean-reverted back up
            ):
                reason = "signal"
            if reason:
                trips.append({"net": gain - rt_fee, "reason": reason,
                              "peak": peak, "held": held})
                pos = None
                cooldown_until = i + cooldown
            else:
                pos = (entry_px, entry_idx, peak)
    return trips


def evaluate(all_closes: dict, p: dict) -> dict:
    trips: list[dict] = []
    for pair, closes in all_closes.items():
        trips.extend(simulate(closes, p))
    if not trips:
        return {"n": 0}
    nets = [t["net"] for t in trips]
    wins = [n for n in nets if n > 0]
    reasons: dict[str, int] = {}
    for t in trips:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
    return {
        "n": len(trips),
        "win_rate": round(len(wins) / len(trips) * 100, 1),
        "avg_net": round(mean(nets), 2),
        "total_net": round(sum(nets), 1),
        "best": round(max(nets), 1),
        "worst": round(min(nets), 1),
        "by_reason": reasons,
    }


CANDIDATES = {
    "BASELINE (live)":      dict(lb=12, entry=2, exit_thr=4, tp=4, trail=8, stop=12, cooldown_bars=1),
    "tighter_exits":        dict(lb=12, entry=2, exit_thr=3, tp=2, trail=1.5, stop=12, cooldown_bars=1),
    "stronger_entry":       dict(lb=24, entry=5, exit_thr=3, tp=3, trail=2, stop=12, cooldown_bars=2),
    "trend_filter":         dict(lb=24, entry=4, exit_thr=3, tp=3, trail=2, stop=12, cooldown_bars=2, sma_n=72, min_hold=2),
    "ride_winners":         dict(lb=12, entry=3, exit_thr=8, tp=0, trail=3, stop=10, cooldown_bars=1),
    "strong+trend+ride":    dict(lb=24, entry=5, exit_thr=6, tp=0, trail=3, stop=10, cooldown_bars=2, sma_n=72, min_hold=2),
    "breakout_strong":      dict(lb=48, entry=6, exit_thr=4, tp=5, trail=3, stop=12, cooldown_bars=2, sma_n=48),
    # --- mean-reversion: buy weakness, sell strength (opposite of momentum) ---
    "MR_basic":             dict(side="rev", lb=12, entry=3, exit_thr=2, tp=2, trail=2, stop=8, cooldown_bars=1),
    "MR_deeper_dip":        dict(side="rev", lb=24, entry=6, exit_thr=2, tp=3, trail=2, stop=10, cooldown_bars=2),
    "MR_dip_below_trend":   dict(side="rev", lb=12, entry=4, exit_thr=2, tp=3, trail=2, stop=10, cooldown_bars=2, sma_n=72),
    "MR_quick_scalp":       dict(side="rev", lb=6, entry=2, exit_thr=1.5, tp=1.5, trail=1, stop=6, cooldown_bars=1),
}


def main():
    print(f"Fetching 30d hourly OHLC for {len(PAIRS)} pairs...")
    all_closes: dict[str, list[float]] = {}
    for pair in PAIRS:
        closes = fetch_closes(pair)
        if closes and len(closes) > 80:
            all_closes[pair] = closes
        time.sleep(0.4)  # be gentle with the public API
    print(f"Got {len(all_closes)} pairs.\n")

    def report(universe: dict, title: str):
        print(f"\n=== {title} ({len(universe)} pairs) ===")
        print(f"{'config':22s} {'trips':>6} {'win%':>6} {'avg%':>7} {'total%':>8} {'best':>7} {'worst':>7}")
        print("-" * 78)
        for name, p in CANDIDATES.items():
            r = evaluate(universe, p)
            if r["n"] == 0:
                print(f"{name:22s} {'0':>6}  (no trips)")
                continue
            print(f"{name:22s} {r['n']:>6} {r['win_rate']:>6} {r['avg_net']:>7} "
                  f"{r['total_net']:>8} {r['best']:>7} {r['worst']:>7}")

    report(all_closes, "ALL PAIRS")
    MAJORS = {"XBTUSDC", "ETHUSDC", "SOLUSDC", "XRPUSDC", "ADAUSDC",
              "AVAXUSDC", "DOTUSDC", "LINKUSDC", "LTCUSDC", "BCHUSDC"}
    majors = {k: v for k, v in all_closes.items() if k in MAJORS}
    report(majors, "MAJORS ONLY")


if __name__ == "__main__":
    main()
