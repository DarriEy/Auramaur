"""Backtest the favorite-longshot bias on the bot's own scanned universe.

Motivation: joining signals.market_prob to resolved outcomes showed the crowd
systematically misprices tails in the markets Auramaur scans — implied 0-10%
resolved ~0% YES, implied 60-90% resolved 90-100% YES. If that survives
realistic entries, "buy the favored side, hold to resolution" is a base-load
strategy that needs no forecast at all.

Methodology (deliberately conservative):
  * Universe: markets with a recorded signal (ex-ante: the bot observed the
    price at analysis time — exactly when a deployed strategy would act) AND
    a resolved outcome in `calibration`. No abandoned-market survivorship:
    every signaled+inactive market has an outcome (checked 2026-06-09).
  * ONE entry per market: the FIRST signal whose favored-side price falls in
    the band. No averaging over repeated signals (overlap bias).
  * Entry price: observed market_prob for the favored side PLUS a slippage
    scenario (0c / 1c / 2c / 4.4c — the last is the bot's measured average
    adverse entry slippage).
  * $1 staked per market; favored side pays $1 if it wins, else stake lost.
  * Control: the mirror trade (buy the LONGSHOT side at the same moments).

Caveats this script reports rather than hides: small n per band, correlated
event clusters (Iran), entry-cohort concentration in time, and holding-period
capital lock-up (per-day returns where resolved_at is recorded).

    python scripts/research/favorite_longshot_backtest.py
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import datetime

DB = "auramaur.db"

BANDS = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 0.97)]
AGG_BAND = (0.60, 0.97)
SLIPPAGES = [0.0, 0.01, 0.02, 0.044]


def _parse_ts(s: str) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def load(db: str):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    outcomes = {
        r["market_id"]: (int(r["actual_outcome"]), r["resolved_at"])
        for r in c.execute(
            "SELECT market_id, actual_outcome, resolved_at FROM calibration "
            "WHERE actual_outcome IS NOT NULL"
        )
    }
    meta = {
        r["id"]: (r["category"] or "(none)", (r["question"] or "")[:60])
        for r in c.execute("SELECT id, category, question FROM markets")
    }
    sigs = defaultdict(list)
    for r in c.execute(
        "SELECT market_id, timestamp, market_prob FROM signals "
        "WHERE market_prob IS NOT NULL ORDER BY timestamp"
    ):
        if r["market_id"] in outcomes:
            sigs[r["market_id"]].append((r["timestamp"], float(r["market_prob"])))
    c.close()
    return outcomes, meta, sigs


def first_entry(sig_list, lo: float, hi: float):
    """First signal whose favored-side price is in [lo, hi)."""
    for ts, p in sig_list:
        p_fav = max(p, 1.0 - p)
        if lo <= p_fav < hi:
            side_yes = p >= 0.5
            return ts, p_fav, side_yes
    return None


def trade(p_entry: float, won: bool) -> float:
    """P&L of $1 staked at p_entry on the favored side."""
    if p_entry >= 1.0:
        return 0.0
    return (1.0 / p_entry) - 1.0 if won else -1.0


def run_band(outcomes, sigs, lo, hi, slip, mirror=False):
    rows = []
    for mid, sl in sigs.items():
        e = first_entry(sl, lo, hi)
        if e is None:
            continue
        ts, p_fav, side_yes = e
        outcome, resolved_at = outcomes[mid]
        if mirror:
            # control: buy the longshot side at its (slipped) price
            p = (1.0 - p_fav) + slip
            won = (outcome == 0) if side_yes else (outcome == 1)
        else:
            p = p_fav + slip
            won = (outcome == 1) if side_yes else (outcome == 0)
        if p >= 0.995:  # nothing left to win
            continue
        days = None
        t0, t1 = _parse_ts(ts), _parse_ts(resolved_at or "")
        if t0 and t1 and t1 > t0:
            days = (t1 - t0).total_seconds() / 86400.0
        rows.append({"mid": mid, "pnl": trade(p, won), "won": won,
                     "p": p, "days": days, "ts": ts})
    return rows


def summarize(rows):
    n = len(rows)
    if n == 0:
        return "      (no trades)"
    wins = sum(1 for r in rows if r["won"])
    pnl = sum(r["pnl"] for r in rows)
    avg_p = sum(r["p"] for r in rows) / n
    days = [r["days"] for r in rows if r["days"]]
    avg_days = sum(days) / len(days) if days else float("nan")
    # per-day return on locked capital, equal-weight across trades with timing
    daily = [r["pnl"] / r["days"] for r in rows if r["days"] and r["days"] > 0.04]
    per_day = sum(daily) / len(daily) if daily else float("nan")
    return (f"  n={n:4d}  win {wins/n*100:5.1f}% (breakeven {avg_p*100:5.1f}%)  "
            f"$P&L {pnl:+8.2f}  per-$1 {pnl/n:+.4f}  "
            f"hold {avg_days:5.1f}d  /day {per_day:+.4f}")


def main():
    outcomes, meta, sigs = load(DB)
    print(f"universe: {len(sigs)} resolved markets with ex-ante signal prices\n")

    print("=" * 78)
    print("FAVORITE SIDE — $1/market, first band-crossing entry, hold to resolution")
    print("=" * 78)
    for slip in SLIPPAGES:
        print(f"\n--- slippage {slip*100:.1f}c ---")
        for lo, hi in BANDS:
            rows = run_band(outcomes, sigs, lo, hi, slip)
            print(f"  {lo:.2f}-{hi:.2f}: {summarize(rows)}")
        rows = run_band(outcomes, sigs, *AGG_BAND, slip)
        print(f"  AGG {AGG_BAND[0]:.2f}-{AGG_BAND[1]:.2f}: {summarize(rows)}")

    print()
    print("=" * 78)
    print("CONTROL: LONGSHOT SIDE at the same entry moments (should be the bleeder)")
    print("=" * 78)
    for lo, hi in BANDS:
        rows = run_band(outcomes, sigs, lo, hi, 0.0, mirror=True)
        print(f"  fav {lo:.2f}-{hi:.2f}: {summarize(rows)}")

    # honesty sections -----------------------------------------------------
    slip = 0.02
    rows = run_band(outcomes, sigs, *AGG_BAND, slip)

    print()
    print("=" * 78)
    print(f"BY CATEGORY (agg band, {slip*100:.0f}c slippage)")
    print("=" * 78)
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[meta.get(r["mid"], ("(none)", ""))[0]].append(r)
    for cat, rs in sorted(by_cat.items(), key=lambda kv: -sum(r["pnl"] for r in kv[1])):
        print(f"  {cat:15}{summarize(rs)}")

    print()
    print("=" * 78)
    print(f"BY ENTRY MONTH (agg band, {slip*100:.0f}c slippage) — regime stability")
    print("=" * 78)
    by_month = defaultdict(list)
    for r in rows:
        by_month[r["ts"][:7]].append(r)
    for m, rs in sorted(by_month.items()):
        print(f"  {m:15}{summarize(rs)}")

    print()
    print("=" * 78)
    print("WORST 8 TRADES (agg band) — correlation / cluster check")
    print("=" * 78)
    for r in sorted(rows, key=lambda r: r["pnl"])[:8]:
        cat, q = meta.get(r["mid"], ("(none)", ""))
        print(f"  {r['pnl']:+6.2f}  p={r['p']:.2f}  {cat:13} {q}")

    n_iran = sum(1 for r in rows
                 if "iran" in meta.get(r["mid"], ("", ""))[1].lower()
                 or "hormuz" in meta.get(r["mid"], ("", ""))[1].lower())
    print(f"\n  Iran/Hormuz-cluster trades in agg band: {n_iran}/{len(rows)} "
          f"(correlated outcomes — effective n is lower than nominal)")


if __name__ == "__main__":
    main()
