"""Reconstruct realized $ P&L from fills + resolved outcomes — measure edge in $.

trades.pnl is never persisted, so this reconstructs realized dollar P&L from the
`fills` (actual executions w/ price+fee) and `calibration` (resolved outcomes):

  realized_pnl(market) = sell_proceeds - buy_cost - fees + resolution_payout

A position counts as realized when it's fully closed via sells OR the market
resolved (held token pays $1 if it won, else $0). Still-open positions are
skipped. Aggregated by category to show where the $ edge actually is.

    python scripts/research/dollar_edge.py
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict

DB = "auramaur.db"


def main() -> None:
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row

    # outcome per market (1 = YES resolved true)
    outcome = {r["market_id"]: r["actual_outcome"] for r in c.execute(
        "SELECT market_id, actual_outcome FROM calibration WHERE actual_outcome IS NOT NULL")}
    category = {r["id"]: (r["category"] or "none") for r in c.execute(
        "SELECT id, category FROM markets")}

    # gather fills per market
    fills = defaultdict(list)
    for r in c.execute("SELECT market_id, side, token, size, price, fee FROM fills WHERE is_paper=0"):
        fills[r["market_id"]].append(r)
    c.close()

    by_cat = defaultdict(lambda: {"pnl": 0.0, "n": 0, "wins": 0})
    total = {"pnl": 0.0, "n": 0, "wins": 0, "open": 0}

    for mid, fs in fills.items():
        buy_cost = sum(f["size"] * f["price"] for f in fs if f["side"] == "BUY")
        sell_proc = sum(f["size"] * f["price"] for f in fs if f["side"] == "SELL")
        fees = sum(f["fee"] or 0 for f in fs)
        # net token held = buys - sells per token
        net = defaultdict(float)
        for f in fs:
            net[f["token"]] += f["size"] if f["side"] == "BUY" else -f["size"]
        residual = sum(v for v in net.values() if v > 0.01)

        resolved = mid in outcome
        if residual > 0.01 and not resolved:
            total["open"] += 1
            continue  # still open — unrealized, skip

        payout = 0.0
        if resolved:
            o = outcome[mid]
            for tok, sz in net.items():
                if sz <= 0.01:
                    continue
                won = (tok == "YES" and o == 1) or (tok == "NO" and o == 0)
                payout += sz * (1.0 if won else 0.0)

        pnl = sell_proc - buy_cost - fees + payout
        cat = category.get(mid, "none")
        by_cat[cat]["pnl"] += pnl
        by_cat[cat]["n"] += 1
        by_cat[cat]["wins"] += 1 if pnl > 0 else 0
        total["pnl"] += pnl
        total["n"] += 1
        total["wins"] += 1 if pnl > 0 else 0

    print("=" * 64)
    print("REALIZED $ EDGE (reconstructed from fills + resolved outcomes)")
    print("=" * 64)
    print(f"  {'category':14} {'$P&L':>9} {'closed':>7} {'win%':>6} {'$/trade':>8}")
    print("  " + "-" * 48)
    for cat, d in sorted(by_cat.items(), key=lambda kv: -kv[1]["pnl"]):
        wr = f"{d['wins']/d['n']*100:.0f}%" if d["n"] else "—"
        print(f"  {cat:14} {d['pnl']:>9.2f} {d['n']:>7} {wr:>6} {d['pnl']/d['n']:>8.2f}")
    print("  " + "-" * 48)
    wr = f"{total['wins']/total['n']*100:.0f}%" if total["n"] else "—"
    print(f"  {'TOTAL':14} {total['pnl']:>9.2f} {total['n']:>7} {wr:>6}")
    print(f"\n  ({total['open']} positions still open — unrealized, excluded)")
    print("  NOTE: small resolved sample; treat as directional, not precise.")


if __name__ == "__main__":
    main()
