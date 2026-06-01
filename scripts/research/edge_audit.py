"""Edge audit — where does the prediction engine actually have edge?

Realized per-trade P&L is NOT persisted (trades.pnl is null), so $ edge can't be
measured directly yet. Instead we measure PREDICTION QUALITY from the
`calibration` table (predicted_prob vs actual resolved outcome) — Brier score and
calibration by category — and juxtapose it with trade VOLUME, to reveal where
capital goes vs where the edge actually is.

    python scripts/research/edge_audit.py

Brier: mean((pred-actual)^2). 0 = perfect, 0.25 = coin flip. <0.25 = real edge.
"""

from __future__ import annotations

import sqlite3
import sys

DB = "auramaur.db"


def _brier(rs):
    return sum((r["p"] - r["o"]) ** 2 for r in rs) / len(rs) if rs else float("nan")


def _acc(rs):
    return sum(1 for r in rs if (r["p"] > 0.5) == (r["o"] == 1)) / len(rs) if rs else float("nan")


def _section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def main() -> None:
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row

    # Where capital/activity goes (volume), shown in the edge table's trades col.
    vol = {}
    for r in c.execute(
        "SELECT m.category k, COUNT(*) n FROM trades t "
        "LEFT JOIN markets m ON t.market_id=m.id GROUP BY m.category"):
        vol[r["k"] or "none"] = r["n"]

    # Edge from resolved predictions (predicted vs actual).
    rows = [dict(p=r["predicted_prob"], o=r["actual_outcome"], cat=r["category"] or "none")
            for r in c.execute(
                "SELECT predicted_prob, actual_outcome, category FROM calibration "
                "WHERE actual_outcome IS NOT NULL AND predicted_prob IS NOT NULL")]
    c.close()

    _section("PREDICTION EDGE BY CATEGORY (Brier <0.25 = real edge)")
    print(f"  resolved predictions: {len(rows)}  |  overall Brier {_brier(rows):.3f}  "
          f"acc {_acc(rows)*100:.0f}%")
    print(f"\n  {'category':14} {'resolved':>8} {'trades':>7} {'Brier':>7} {'acc':>5} {'baseYES':>7}  verdict")
    print("  " + "-" * 68)
    cats: dict[str, list] = {}
    for r in rows:
        cats.setdefault(r["cat"], []).append(r)
    for cat, rs in sorted(cats.items(), key=lambda kv: _brier(kv[1])):
        b = _brier(rs)
        verdict = "STRONG" if b < 0.15 else ("edge" if b < 0.22 else
                  ("~none" if b < 0.27 else "ANTI-EDGE"))
        print(f"  {cat:14} {len(rs):>8} {vol.get(cat, 0):>7} {b:>7.3f} "
              f"{_acc(rs)*100:>4.0f}% {sum(x['o'] for x in rs)/len(rs)*100:>6.0f}%  {verdict}")

    _section("CALIBRATION CURVE (predicted bucket -> actual YES rate)")
    for lo in range(0, 100, 10):
        b = [r for r in rows if lo <= r["p"] * 100 < lo + 10]
        if b:
            bar = "█" * round(sum(x["o"] for x in b) / len(b) * 20)
            print(f"  pred {lo:>2}-{lo+10:>3}%  n={len(b):>4}  actual "
                  f"{sum(x['o'] for x in b)/len(b)*100:>3.0f}%  {bar}")

    print("\n  NOTE: realized $ P&L is not persisted (trades.pnl null) — edge here")
    print("  is prediction QUALITY. Fixing per-trade P&L would let us measure $ edge.")


if __name__ == "__main__":
    main()
