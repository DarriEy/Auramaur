"""Proper edge audit — does the model beat the MARKET, not just look calibrated?

The original edge_audit.py measured the model's Brier in isolation, which rewards
predicting the base rate (a 90%-NO category scores well by always saying NO). That
is NOT edge. Real edge means the model's prediction beats two baselines:

  1. the MARKET PRICE (market_prob) — if Brier_model >= Brier_market, trusting the
     market would have been at least as good: NO edge over the market.
  2. the category BASE RATE — Brier Skill Score vs always-predict-base-rate.

Per category we show both, so we can tell whether the reallocation (#36) tilted
toward real skill or toward base-rate-predictable categories (an artifact).

Read-only.
    python scripts/research/edge_vs_market.py
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict

DB = "auramaur.db"


def _brier(rs, key):
    return sum((r[key] - r["o"]) ** 2 for r in rs) / len(rs) if rs else float("nan")


def main():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    # latest market_prob per market from signals
    mkt = {}
    for r in c.execute("SELECT market_id, market_prob FROM signals "
                       "WHERE market_prob IS NOT NULL ORDER BY timestamp"):
        mkt[r["market_id"]] = r["market_prob"]
    rows = []
    for r in c.execute("SELECT market_id, predicted_prob, actual_outcome, category "
                       "FROM calibration WHERE actual_outcome IS NOT NULL "
                       "AND predicted_prob IS NOT NULL"):
        if r["market_id"] in mkt:
            rows.append({"m": r["predicted_prob"], "mkt": mkt[r["market_id"]],
                         "o": r["actual_outcome"], "cat": r["category"] or "none"})
    c.close()

    def block(rs):
        bm, bk = _brier(rs, "m"), _brier(rs, "mkt")
        base = sum(r["o"] for r in rs) / len(rs)
        bb = sum((base - r["o"]) ** 2 for r in rs) / len(rs)
        # % of disagreements (|model-market|>5%) where the model was the better side
        dis = [r for r in rs if abs(r["m"] - r["mkt"]) > 0.05]
        right = sum(1 for r in dis if (r["m"] - r["mkt"]) * (r["o"] - r["mkt"]) > 0)
        return bm, bk, bb, base, len(dis), (right / len(dis) if dis else float("nan"))

    print("=" * 86)
    print("EDGE vs MARKET — does the model beat the price? (Brier: lower=better)")
    print("  beats_market = Brier_market - Brier_model  (>0 = model better than the price)")
    print("=" * 86)
    bm, bk, bb, base, ndis, dwin = block(rows)
    print(f"  OVERALL n={len(rows)}: model {bm:.3f} | market {bk:.3f} | base-rate {bb:.3f}"
          f"  -> beats_market {bk-bm:+.3f}")
    print(f"           when model disagreed w/ market by >5% (n={ndis}), it was right {dwin*100:.0f}% of the time\n")
    cats = defaultdict(list)
    for r in rows:
        cats[r["cat"]].append(r)
    print(f"  {'category':14} {'n':>4} {'model':>6} {'market':>7} {'baserate':>8} "
          f"{'beats_mkt':>9} {'disagree-win%':>13}")
    print("  " + "-" * 70)
    for cat, rs in sorted(cats.items(), key=lambda kv: -( _brier(kv[1],'mkt') - _brier(kv[1],'m'))):
        if len(rs) < 3:
            continue
        bm, bk, bb, baserate, nd, dw = block(rs)
        flag = "BEATS MKT" if bk - bm > 0.005 else ("≈market" if abs(bk-bm) <= 0.005 else "WORSE")
        print(f"  {cat:14} {len(rs):>4} {bm:>6.3f} {bk:>7.3f} {bb:>8.3f} "
              f"{bk-bm:>+9.3f} {(f'{dw*100:.0f}% (n={nd})' if nd else '—'):>13}  {flag}")
    print("\n  READ: a category only has real edge if model < market (beats_mkt > 0) AND")
    print("  the model wins its disagreements with the market >50% of the time.")


if __name__ == "__main__":
    main()
