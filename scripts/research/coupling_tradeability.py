"""Tradeability backtest for the spot->prediction lead-lag coupling.

coupling_discovery.py found spot LEADS some crypto threshold markets. This tests
whether that's *tradeable*: at each point, if spot just moved, take the matching
side of the prediction market and hold for the lead time, then exit — net of a
round-trip cost. If it profits after cost across enough instances, the coupling
has real, capturable edge.

Read-only. Honest about sample size.

    python scripts/research/coupling_tradeability.py
    python scripts/research/coupling_tradeability.py --cost 0.02 --threshold 0.002
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import urllib.request
from datetime import datetime, timezone

import numpy as np

DB = "auramaur.db"
PAIR = {"Bitcoin": "XBTUSD", "BTC": "XBTUSD", "Ethereum": "ETHUSD", "ETH": "ETHUSD"}


def _ts(s):
    return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp())


def prob_series(c, mid):
    return [(_ts(r["recorded_at"]), float(r["price"])) for r in c.execute(
        "SELECT recorded_at, price FROM price_history WHERE market_id=? ORDER BY recorded_at", (mid,))
        if r["price"] is not None]


def spot_series(pair, since, interval):
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}&since={since-interval*60}"
    try:
        d = json.load(urllib.request.urlopen(url, timeout=15))
        s = next(v for k, v in d["result"].items() if k != "last")
        return [(int(b[0]), float(b[4])) for b in s]
    except Exception:
        return []


def resample(series, t0, t1, step):
    grid, out, j, last = list(range(t0, t1, step)), [], 0, series[0][1]
    for t in grid:
        while j < len(series) and series[j][0] <= t:
            last = series[j][1]; j += 1
        out.append(last)
    return np.array(out)


def best_lag(spot_ret, prob_chg, max_lag):
    best = (0, 0.0)
    for k in range(1, max_lag + 1):  # only spot-leads (positive) lags
        x, y = spot_ret[:len(spot_ret) - k], prob_chg[k:]
        n = min(len(x), len(y))
        if n < 6 or x[:n].std() == 0 or y[:n].std() == 0:
            continue
        r = float(np.corrcoef(x[:n], y[:n])[0, 1])
        if abs(r) > abs(best[1]):
            best = (k, r)
    return best


def backtest_market(c, mid, question, cost, threshold):
    asset = next((a for a in PAIR if a in question), None)
    if not asset:
        return None
    ps = prob_series(c, mid)
    if len(ps) < 30:
        return None
    t0, t1 = ps[0][0], ps[-1][0]
    interval, step, max_lag = (1, 300, 12) if (t1 - t0) / 3600 <= 12 else (15, 900, 8)
    ss = spot_series(PAIR[asset], t0, interval)
    if len(ss) < 10:
        return None
    prob = resample(ps, t0, t1, step)
    spot = resample(ss, t0, t1, step)
    if len(prob) < 12:
        return None
    spot_ret = np.diff(np.log(spot))
    prob_chg = np.diff(prob)
    lag, r = best_lag(spot_ret, prob_chg, max_lag)
    if lag == 0:
        return {"q": question, "lag": 0, "r": r, "pnl": 0.0, "trades": 0, "wins": 0}

    # Trade: at step i, if spot moved beyond threshold, take that side, exit at i+lag.
    pnl, trades, wins = 0.0, 0, 0
    for i in range(len(spot_ret) - lag):
        sig = spot_ret[i]
        if abs(sig) < threshold:
            continue
        entry, exit_ = prob[i + 1], prob[i + 1 + lag]   # prob aligns to returns at i+1
        gain = (exit_ - entry) if sig > 0 else (entry - exit_)
        net = gain - cost
        pnl += net
        trades += 1
        wins += 1 if net > 0 else 0
    return {"q": question, "lag": lag, "r": r, "pnl": pnl, "trades": trades, "wins": wins}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=0.02, help="round-trip cost in prob units")
    ap.add_argument("--threshold", type=float, default=0.002, help="min |spot return| to act")
    args = ap.parse_args()

    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    mkts = c.execute("""
        SELECT m.id, m.question, COUNT(*) n FROM markets m
        JOIN price_history ph ON ph.market_id=m.id
        WHERE m.category='crypto' AND (m.question LIKE '%Bitcoin%' OR m.question LIKE '%Ethereum%')
          AND (m.question LIKE '%above%' OR m.question LIKE '%below%')
        GROUP BY m.id HAVING n >= 40 ORDER BY n DESC LIMIT 12""").fetchall()

    print("=" * 76)
    print(f"COUPLING TRADEABILITY — spot-lead signal, cost {args.cost} / trade, thr {args.threshold}")
    print("=" * 76)
    print(f"  {'lag':>4} {'r':>6} {'$pnl':>8} {'trades':>7} {'win%':>6}  market")
    print("  " + "-" * 58)
    tot_pnl, tot_tr, tot_w, n = 0.0, 0, 0, 0
    for m in mkts:
        res = backtest_market(c, m["id"], m["question"], args.cost, args.threshold)
        if not res or res["trades"] == 0:
            continue
        n += 1
        tot_pnl += res["pnl"]; tot_tr += res["trades"]; tot_w += res["wins"]
        wr = f"{res['wins']/res['trades']*100:.0f}%"
        print(f"  {res['lag']:>4} {res['r']:>+6.2f} {res['pnl']:>+8.3f} "
              f"{res['trades']:>7} {wr:>6}  {res['q'][:34]}")
    c.close()
    if tot_tr:
        print("\n" + "=" * 76)
        print(f"  TOTAL net {tot_pnl:+.3f} (per $1 notional) over {tot_tr} trades, "
              f"{tot_w/tot_tr*100:.0f}% win, {n} markets")
        print(f"  VERDICT: {'positive after cost — worth more data + a real test' if tot_pnl > 0 else 'not profitable after cost on this sample'}")
    else:
        print("\n  No tradeable instances on current data — need more price_history.")
    print("\n  NOTE: tiny sample; cost assumption dominates. Re-run as data accrues.")


if __name__ == "__main__":
    main()
