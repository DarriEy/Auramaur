"""Third-differential exploration — does a HIGHER-ORDER spot signal lead the
prediction market better than the level/velocity?

  1st diff (velocity)     = spot return            -> already tested (coupling)
  2nd diff (acceleration) = change in return
  3rd diff (jerk)         = change in acceleration  <- the a-priori-bullish bet

For crypto threshold markets we align prob (price_history) with spot (Kraken OHLC)
on a uniform grid and cross-correlate each spot derivative against the prob change
at forward lags. If the 3rd derivative shows a stronger lead correlation than the
1st/2nd, higher-order dynamics carry the edge. (Caveat: higher derivatives amplify
noise hard, and the sample is small — read directionally.)

Read-only.
    python scripts/research/third_differential.py
"""

from __future__ import annotations

import json
import sqlite3
import urllib.request
from datetime import datetime, timezone

import numpy as np

DB = "auramaur.db"
PAIR = {"Bitcoin": "XBTUSD", "Ethereum": "ETHUSD"}


def _ts(s):
    return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp())


def _spot(pair, since, interval):
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}&since={since-interval*60}"
    try:
        d = json.load(urllib.request.urlopen(url, timeout=15))
        s = next(v for k, v in d["result"].items() if k != "last")
        return [(int(b[0]), float(b[4])) for b in s]
    except Exception:
        return []


def _resample(series, t0, t1, step):
    out, j, last = [], 0, series[0][1]
    for t in range(t0, t1, step):
        while j < len(series) and series[j][0] <= t:
            last = series[j][1]; j += 1
        out.append(last)
    return np.array(out)


def _best_lead(x, prob_chg, max_lag):
    best = 0.0
    for k in range(1, max_lag + 1):
        a, b = x[: len(x) - k], prob_chg[k:]
        n = min(len(a), len(b))
        if n < 6 or a[:n].std() == 0 or b[:n].std() == 0:
            continue
        r = abs(float(np.corrcoef(a[:n], b[:n])[0, 1]))
        best = max(best, r)
    return best


def main():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    mkts = c.execute("""
        SELECT m.id, m.question, COUNT(*) n FROM markets m
        JOIN price_history ph ON ph.market_id=m.id
        WHERE m.category='crypto' AND (m.question LIKE '%Bitcoin%' OR m.question LIKE '%Ethereum%')
          AND (m.question LIKE '%above%' OR m.question LIKE '%below%')
        GROUP BY m.id HAVING n >= 40 ORDER BY n DESC LIMIT 12""").fetchall()

    print("=" * 70)
    print("THIRD-DIFFERENTIAL — best |lead corr| of each spot derivative vs prob")
    print("=" * 70)
    print(f"  {'market':40} {'1st(v)':>7} {'2nd(a)':>7} {'3rd(jerk)':>9}")
    print("  " + "-" * 66)
    agg = {1: [], 2: [], 3: []}
    for m in mkts:
        asset = next((a for a in PAIR if a in m["question"]), None)
        if not asset:
            continue
        ps = [(_ts(r["recorded_at"]), float(r["price"])) for r in c.execute(
            "SELECT recorded_at, price FROM price_history WHERE market_id=? ORDER BY recorded_at",
            (m["id"],)) if r["price"] is not None]
        if len(ps) < 30:
            continue
        t0, t1 = ps[0][0], ps[-1][0]
        interval, step, ml = (1, 300, 12) if (t1 - t0) / 3600 <= 12 else (15, 900, 8)
        ss = _spot(PAIR[asset], t0, interval)
        if len(ss) < 10:
            continue
        prob = _resample(ps, t0, t1, step)
        spot = _resample(ss, t0, t1, step)
        if len(prob) < 16:
            continue
        prob_chg = np.diff(prob)
        v = np.diff(np.log(spot))            # 1st
        a = np.diff(v)                       # 2nd
        j = np.diff(a)                       # 3rd
        # align lengths to prob_chg by trimming the front (diffs shorten the array)
        r1 = _best_lead(v, prob_chg, ml)
        r2 = _best_lead(a, prob_chg[1:], ml)
        r3 = _best_lead(j, prob_chg[2:], ml)
        agg[1].append(r1); agg[2].append(r2); agg[3].append(r3)
        print(f"  {m['question'][:40]:40} {r1:>7.2f} {r2:>7.2f} {r3:>9.2f}")
    c.close()
    if agg[1]:
        print("\n" + "=" * 70)
        m1, m2, m3 = (np.mean(agg[o]) for o in (1, 2, 3))
        print(f"  AVG |lead corr|: 1st(velocity) {m1:.2f} | 2nd(accel) {m2:.2f} | 3rd(jerk) {m3:.2f}")
        winner = max((m1, "1st/velocity"), (m2, "2nd/accel"), (m3, "3rd/jerk"))[1]
        print(f"  strongest lead signal: {winner}")
        print(f"\n  VERDICT: {'higher-order carries more signal — the bullish prior holds' if m3 >= max(m1, m2) else 'the 3rd differential does NOT lead better than velocity here'}")
    print("\n  (small n + noisy higher derivatives — directional only)")


if __name__ == "__main__":
    main()
