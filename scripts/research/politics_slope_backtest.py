"""Backtest the long-horizon underconfidence (calibration-slope) edge in POLITICS.

Task A from the 2026-06-29 winner reverse-engineering: the calibration paper
(arXiv:2602.19520) finds long-horizon underconfidence is STRONGEST in politics
(slope ~1.31). That is a STRUCTURAL edge (apply the slope to the market's own
price — no forecast), DISTINCT from the directional politics forecasting the bot
has no edge in. Politics is globally BLOCKED (fail-closed + risk-enforced), so we
CANNOT paper-trade it without bypassing a risk check. This script instead TESTS
the edge offline against the bot's own resolved-politics universe — zero trading,
no block bypass.

Question: at long horizon, are politics favorites UNDERPRICED (realized win rate
> implied price), and does buying them clear cost? And does the underconfidence
GROW with horizon (the paper's core claim)?

Method (mirrors favorite_longshot_backtest.py): ex-ante price = signals.market_prob
(what the bot observed), outcome = calibration.actual_outcome, horizon =
resolved_at - signal timestamp. ONE entry per market (first signal with the
favored side in band). Favored side, hold to resolution, $1/market, slippage
scenarios. Honest about small n.

    python scripts/research/politics_slope_backtest.py
"""

from __future__ import annotations

import math
import sqlite3

DB = "auramaur.db"
POLITICS = ("politics_us", "politics_intl")
BAND = (0.55, 0.92)              # matches the long_horizon pillar
SLIPPAGES = [0.0, 0.01, 0.02]
HORIZON_BUCKETS = [("<7d", 0, 7), ("7-30d", 7, 30), (">30d", 30, 1e9)]
PRICE_BINS = [(0.55, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 0.92)]


def calibrated_fair(price: float, slope: float) -> float:
    p = min(max(price, 1e-6), 1.0 - 1e-6)
    return 1.0 / (1.0 + math.exp(-slope * math.log(p / (1.0 - p))))


def load(db: str):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    rows = c.execute(
        """SELECT s.market_id, s.market_prob AS p, cal.actual_outcome AS outcome,
                  julianday(cal.resolved_at) - julianday(s.timestamp) AS horizon
           FROM signals s
           JOIN calibration cal ON cal.market_id = s.market_id
                AND cal.actual_outcome IS NOT NULL
           JOIN markets m ON m.id = s.market_id
           WHERE m.category IN (?, ?) AND s.market_prob IS NOT NULL
             AND s.timestamp IS NOT NULL AND cal.resolved_at IS NOT NULL
           ORDER BY s.timestamp""",
        POLITICS,
    ).fetchall()
    # ONE entry per market: first signal whose FAVORED side is in band.
    seen, entries = set(), []
    for r in rows:
        mid, p, outcome, horizon = r["market_id"], r["p"], int(r["outcome"]), r["horizon"]
        if mid in seen or horizon is None or horizon < 0 or not (0.0 < p < 1.0):
            continue
        fav_is_yes = p >= 0.5
        fav_price = p if fav_is_yes else 1.0 - p
        if not (BAND[0] <= fav_price <= BAND[1]):
            continue
        fav_won = (outcome == 1) if fav_is_yes else (outcome == 0)
        seen.add(mid)
        entries.append({"price": fav_price, "won": fav_won, "horizon": horizon})
    return entries


def fit_slope(entries):
    """Best-fit calibration slope: logit(realized) ~ slope * logit(price), via
    price-binned realized rates (the paper's cell-level method). Returns slope or
    None if too few populated bins."""
    pts = []
    for lo, hi in PRICE_BINS:
        grp = [e for e in entries if lo <= e["price"] < hi]
        if len(grp) < 3:
            continue
        rate = sum(e["won"] for e in grp) / len(grp)
        rate = min(max(rate, 1e-3), 1 - 1e-3)
        center = (lo + hi) / 2
        pts.append((math.log(center / (1 - center)), math.log(rate / (1 - rate))))
    if len(pts) < 2:
        return None, len(pts)
    n = len(pts); sx = sum(x for x, _ in pts); sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts); sxy = sum(x * y for x, y in pts)
    denom = n * sxx - sx * sx
    return ((n * sxy - sx * sy) / denom if denom else None), n


def report(entries):
    print(f"\nResolved POLITICS favorites in band {BAND}: n={len(entries)}\n")
    if not entries:
        print("  no data"); return

    print("=== Underconfidence by horizon (realized win-rate vs implied price) ===")
    print(f"{'horizon':<8} {'n':>4} {'implied':>8} {'realized':>9} {'gap':>7}  underpriced?")
    for label, lo, hi in HORIZON_BUCKETS:
        grp = [e for e in entries if lo <= e["horizon"] < hi]
        if not grp:
            print(f"{label:<8} {0:>4}      --        --      --"); continue
        imp = sum(e["price"] for e in grp) / len(grp)
        rea = sum(e["won"] for e in grp) / len(grp)
        gap = rea - imp
        flag = "YES (edge)" if gap > 0.02 else ("~flat" if abs(gap) <= 0.02 else "NO (overpriced)")
        print(f"{label:<8} {len(grp):>4} {imp:>8.3f} {rea:>9.3f} {gap:>+7.3f}  {flag}")

    print("\n=== Best-fit calibration slope (>1 = underconfident, paper says ~1.31) ===")
    for label, lo, hi in [("all", 0, 1e9)] + list(HORIZON_BUCKETS):
        grp = [e for e in entries if lo <= e["horizon"] < hi]
        slope, nb = fit_slope(grp)
        s = f"{slope:.2f}" if slope is not None else "n/a"
        print(f"  {label:<8} slope={s:<6} ({nb} populated price-bins, n={len(grp)})")

    print("\n=== Strategy P&L: buy long-horizon (>=30d) favorites, hold to resolution ===")
    lh = [e for e in entries if e["horizon"] >= 30]
    if lh:
        wr = sum(e["won"] for e in lh) / len(lh)
        print(f"  n={len(lh)}  win-rate={wr:.1%}")
        for slip in SLIPPAGES:
            rois = [((1.0 if e["won"] else 0.0) - (e["price"] + slip)) / (e["price"] + slip)
                    for e in lh]
            print(f"   slip {slip*100:>3.1f}c: avg ROI/$1 = {sum(rois)/len(rois):+.1%}")
    else:
        print("  no >=30d entries")
    print("\nCAVEATS: small n (the bot's own scanned universe, politics blocked from")
    print("trading); entry-cohort time clustering; ex-ante price = first in-band signal.")
    print("Directional read, not proof — but it tests the edge with ZERO trading.")


if __name__ == "__main__":
    report(load(DB))
