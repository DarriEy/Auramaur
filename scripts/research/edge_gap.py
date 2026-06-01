"""Why doesn't good calibration make money? — the calibration->$ gap.

For each resolved live market we reconstruct realized $ P&L and pair it with the
signal that drove the trade (predicted divergence from market = |claude_prob -
market_prob|, and entry slippage). If higher-divergence ("more edge") trades lose
MORE, that's ADVERSE SELECTION — the bot trades hardest exactly where it's wrong.
We also check slippage (did we pay up vs the price we saw?).

Read-only.

    python scripts/research/edge_gap.py
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict

DB = "auramaur.db"


def realized_pnl(fills, outcome):
    buy = sum(f["size"] * f["price"] for f in fills if f["side"] == "BUY")
    sell = sum(f["size"] * f["price"] for f in fills if f["side"] == "SELL")
    fees = sum(f["fee"] or 0 for f in fills)
    net = defaultdict(float)
    for f in fills:
        net[f["token"]] += f["size"] if f["side"] == "BUY" else -f["size"]
    payout = sum(sz for tok, sz in net.items() if sz > 0.01
                 and ((tok == "YES" and outcome == 1) or (tok == "NO" and outcome == 0)))
    return sell - buy - fees + payout


def main():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    outcome = {r["market_id"]: r["actual_outcome"] for r in c.execute(
        "SELECT market_id, actual_outcome FROM calibration WHERE actual_outcome IS NOT NULL")}
    # strongest signal per market (max |claude-market| divergence)
    sig = {}
    for r in c.execute("SELECT market_id, claude_prob, market_prob FROM signals "
                       "WHERE claude_prob IS NOT NULL AND market_prob IS NOT NULL"):
        d = abs(r["claude_prob"] - r["market_prob"])
        if r["market_id"] not in sig or d > sig[r["market_id"]][0]:
            sig[r["market_id"]] = (d, r["market_prob"])
    fills = defaultdict(list)
    for r in c.execute("SELECT market_id, side, token, size, price, fee FROM fills WHERE is_paper=0"):
        fills[r["market_id"]].append(r)
    c.close()

    rows = []
    for mid, fs in fills.items():
        if mid not in outcome or mid not in sig:
            continue
        pnl = realized_pnl(fs, outcome[mid])
        div, mkt = sig[mid]
        buys = [f for f in fs if f["side"] == "BUY"]
        slip = (sum(f["price"] for f in buys) / len(buys) - mkt) if buys else 0.0
        rows.append({"pnl": pnl, "div": div, "slip": slip})

    if not rows:
        print("No resolved markets with both fills and signals yet.")
        return

    print("=" * 64)
    print(f"CALIBRATION -> $ GAP  ({len(rows)} resolved markets w/ fills+signal)")
    print("=" * 64)
    print("\n  ADVERSE SELECTION — realized $ by predicted divergence bucket")
    print(f"  {'divergence':14} {'mkts':>5} {'$P&L':>9} {'$/mkt':>8} {'win%':>6}")
    print("  " + "-" * 46)
    buckets = [("1. 0-5%", 0, .05), ("2. 5-10%", .05, .10),
               ("3. 10-20%", .10, .20), ("4. 20%+", .20, 9)]
    for name, lo, hi in buckets:
        b = [r for r in rows if lo <= r["div"] < hi]
        if not b:
            continue
        p = sum(r["pnl"] for r in b)
        w = sum(1 for r in b if r["pnl"] > 0) / len(b) * 100
        print(f"  {name:14} {len(b):>5} {p:>9.2f} {p/len(b):>8.2f} {w:>5.0f}%")

    import statistics as st
    print(f"\n  avg entry slippage vs seen price: {st.mean(r['slip'] for r in rows):+.3f} "
          "(positive = paid above the market price we saw)")
    hi = [r for r in rows if r["div"] >= 0.10]
    lo = [r for r in rows if r["div"] < 0.10]
    print(f"  high-divergence (>=10%) $/mkt: {sum(r['pnl'] for r in hi)/len(hi):+.2f}  "
          f"vs low (<10%): {sum(r['pnl'] for r in lo)/len(lo):+.2f}" if hi and lo else "")
    print("\n  READ: if $/mkt DROPS as divergence rises -> adverse selection")
    print("  (the bot's biggest 'edges' are where it's most wrong). Slippage adds drag.")


if __name__ == "__main__":
    main()
