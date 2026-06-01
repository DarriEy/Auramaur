"""Lead-lag coupling discovery (L1) — crypto spot vs crypto prediction markets.

For crypto threshold markets ("Will BTC be above $X"), the prob is a direct
function of spot — so any lag is pure lead-lag. We align each market's prob
series (DB price_history) with the underlying spot (Kraken OHLC) on a uniform
grid and cross-correlate spot returns vs prob changes at offsets. A positive
best-lag with meaningful correlation means SPOT LEADS the prediction market by
that lag — a window we could act in (we're not racing HFT, we're exploiting a
slow-to-reprice retail prediction market).

Read-only. Validate-first: if spot leads reliably, the coupling thesis has legs.

    python scripts/research/coupling_discovery.py
"""

from __future__ import annotations

import json
import sqlite3
import urllib.request
from datetime import datetime, timezone

import numpy as np

DB = "auramaur.db"
KRAKEN_PAIR = {"BTC": "XBTUSD", "Bitcoin": "XBTUSD", "ETH": "ETHUSD", "Ethereum": "ETHUSD"}


def _ts(s: str) -> int:
    return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp())


def prob_series(c, market_id: str) -> list[tuple[int, float]]:
    rows = c.execute(
        "SELECT recorded_at, price FROM price_history WHERE market_id=? ORDER BY recorded_at",
        (market_id,)).fetchall()
    out = []
    for r in rows:
        try:
            out.append((_ts(r["recorded_at"]), float(r["price"])))
        except (ValueError, TypeError):
            pass
    return out


def spot_series(pair: str, since: int, interval: int) -> list[tuple[int, float]]:
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}&since={since-interval*60}"
    try:
        d = json.load(urllib.request.urlopen(url, timeout=15))
        series = next(v for k, v in d["result"].items() if k != "last")
        return [(int(b[0]), float(b[4])) for b in series]
    except Exception:
        return []


def resample(series: list[tuple[int, float]], t0: int, t1: int, step: int) -> np.ndarray:
    """Forward-fill onto a uniform grid [t0, t1) at `step` seconds."""
    grid = list(range(t0, t1, step))
    out, j, last = [], 0, series[0][1]
    for t in grid:
        while j < len(series) and series[j][0] <= t:
            last = series[j][1]
            j += 1
        out.append(last)
    return np.array(out)


def xcorr(spot_ret: np.ndarray, prob_chg: np.ndarray, max_lag: int):
    """Best lag where corr(spot_ret[t], prob_chg[t+lag]) is largest. lag>0 = spot leads."""
    best = (0, 0.0)
    curve = {}
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            x, y = spot_ret[: len(spot_ret) - k] if k else spot_ret, prob_chg[k:]
        else:
            x, y = spot_ret[-k:], prob_chg[: len(prob_chg) + k]
        n = min(len(x), len(y))
        if n < 6 or x[:n].std() == 0 or y[:n].std() == 0:
            continue
        r = float(np.corrcoef(x[:n], y[:n])[0, 1])
        curve[k] = r
        if abs(r) > abs(best[1]):
            best = (k, r)
    return best, curve


def analyze(c, market_id: str, question: str) -> None:
    asset = next((a for a in KRAKEN_PAIR if a in question), None)
    if not asset:
        return
    ps = prob_series(c, market_id)
    if len(ps) < 30:
        return
    t0, t1 = ps[0][0], ps[-1][0]
    dur_h = (t1 - t0) / 3600
    interval, step, max_lag = (1, 300, 12) if dur_h <= 12 else (15, 900, 8)  # 5m/60m or 15m/2h
    ss = spot_series(KRAKEN_PAIR[asset], t0, interval)
    if len(ss) < 10:
        print(f"  [no spot data] {question[:50]}")
        return

    prob = resample(ps, t0, t1, step)
    spot = resample(ss, t0, t1, step)
    if len(prob) < 8:
        return
    prob_chg = np.diff(prob)
    spot_ret = np.diff(np.log(spot))
    (lag, r), _ = xcorr(spot_ret, prob_chg, max_lag)
    lead_min = lag * step / 60
    verdict = ("SPOT LEADS" if lag > 0 and abs(r) > 0.2 else
               ("synchronous" if abs(lag) <= 0 else "weak/none"))
    print(f"  {asset:8} r={r:+.2f} @ lag {lag:+d} ({lead_min:+.0f}min)  "
          f"[{verdict}]  {question[:42]} ({len(prob)} grid pts)")


def main() -> None:
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    mkts = c.execute("""
        SELECT m.id, m.question, COUNT(*) n FROM markets m
        JOIN price_history ph ON ph.market_id=m.id
        WHERE m.category='crypto'
          AND (m.question LIKE '%Bitcoin%' OR m.question LIKE '%Ethereum%')
          AND (m.question LIKE '%above%' OR m.question LIKE '%below%' OR m.question LIKE '%between%')
        GROUP BY m.id HAVING n >= 40 ORDER BY n DESC LIMIT 12""").fetchall()

    print("=" * 78)
    print("LEAD-LAG COUPLING — crypto spot vs crypto prediction prob")
    print("  lag>0 + |r|>0.2 = spot leads the prediction market (tradeable window)")
    print("=" * 78)
    for m in mkts:
        analyze(c, m["id"], m["question"])
    c.close()
    print("\n  (positive lag = spot moves first, prediction prob catches up that much later)")


if __name__ == "__main__":
    main()
