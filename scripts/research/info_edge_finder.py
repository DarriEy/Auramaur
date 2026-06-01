"""Find the information-edge POCKETS — where does the bot actually beat the market?

Whole categories have no edge (the market beats us everywhere, edge_vs_market.py).
But edge may live in specific CONDITIONS — thin/neglected markets, high-confidence
calls, second-opinion agreement, etc. This slices resolved predictions by those
conditions and reports beats_market per slice. A slice is a real edge pocket only
if model Brier < market Brier (beats_mkt>0) AND it wins its market-disagreements
>50% of the time, with enough n.

Read-only.
    python scripts/research/info_edge_finder.py
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict

DB = "auramaur.db"


def _stats(rs):
    if len(rs) < 8:
        return None
    bm = sum((r["c"] - r["o"]) ** 2 for r in rs) / len(rs)
    bk = sum((r["mkt"] - r["o"]) ** 2 for r in rs) / len(rs)
    dis = [r for r in rs if abs(r["c"] - r["mkt"]) > 0.05]
    win = sum(1 for r in dis if (r["c"] - r["mkt"]) * (r["o"] - r["mkt"]) > 0) / len(dis) if dis else float("nan")
    return {"n": len(rs), "bm": bm, "bk": bk, "beats": bk - bm, "dwin": win, "ndis": len(dis)}


def main():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    # latest signal per market (market_prob, confidence, second-opinion)
    sig = {}
    for r in c.execute("SELECT market_id, market_prob, claude_prob, claude_confidence, "
                       "second_opinion_prob FROM signals WHERE market_prob IS NOT NULL "
                       "ORDER BY timestamp"):
        sig[r["market_id"]] = r
    liq = {r["id"]: (r["liquidity"] or 0) for r in c.execute("SELECT id, liquidity FROM markets")}
    rows = []
    for r in c.execute("SELECT market_id, predicted_prob, actual_outcome FROM calibration "
                       "WHERE actual_outcome IS NOT NULL AND predicted_prob IS NOT NULL"):
        s = sig.get(r["market_id"])
        if not s:
            continue
        rows.append({
            "c": r["predicted_prob"], "mkt": s["market_prob"], "o": r["actual_outcome"],
            "conf": s["claude_confidence"], "liq": liq.get(r["market_id"], 0),
            "so": s["second_opinion_prob"]})
    c.close()

    def show(title, slices):
        print(f"\n  {title}")
        print(f"  {'slice':22} {'n':>4} {'model':>6} {'market':>7} {'beats':>7} {'dis-win%':>9}")
        print("  " + "-" * 60)
        for name, rs in slices:
            st = _stats(rs)
            if not st:
                continue
            flag = "EDGE" if st["beats"] > 0.005 and (st["dwin"] != st["dwin"] or st["dwin"] > 0.5) else ""
            dw = f"{st['dwin']*100:.0f}%(n{st['ndis']})" if st["ndis"] else "—"
            print(f"  {name:22} {st['n']:>4} {st['bm']:>6.3f} {st['bk']:>7.3f} "
                  f"{st['beats']:>+7.3f} {dw:>9}  {flag}")

    print("=" * 74)
    print(f"INFORMATION-EDGE FINDER (n={len(rows)}) — beats>0 AND dis-win>50% = real pocket")
    print("=" * 74)

    liqb = defaultdict(list)
    for r in rows:
        b = ("1. <$1k" if r["liq"] < 1000 else "2. $1-10k" if r["liq"] < 10000
             else "3. $10-50k" if r["liq"] < 50000 else "4. $50k+")
        liqb[b].append(r)
    show("BY LIQUIDITY (thin markets = where the price may be uninformed)",
         sorted(liqb.items()))

    cb = defaultdict(list)
    for r in rows:
        cb[r["conf"] or "none"].append(r)
    show("BY CONFIDENCE", sorted(cb.items()))

    so = defaultdict(list)
    for r in rows:
        if r["so"] is None:
            continue
        agree = abs(r["c"] - r["so"]) <= 0.10
        so["agree (2nd opinion)" if agree else "disagree (2nd opinion)"].append(r)
    show("BY SECOND-OPINION AGREEMENT", sorted(so.items()))

    # thin + high-confidence + agreement combined (the strongest a-priori pocket)
    combo = [r for r in rows if r["liq"] < 10000 and r["conf"] in ("HIGH", "MEDIUM_HIGH")
             and r["so"] is not None and abs(r["c"] - r["so"]) <= 0.10]
    show("COMBINED: thin + high-conf + 2nd-opinion-agreement", [("thin+conf+agree", combo)])
    print("\n  READ: any row flagged EDGE is a candidate to trade in / others to avoid.")


if __name__ == "__main__":
    main()
