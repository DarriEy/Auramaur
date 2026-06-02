"""Backfill cost-basis for stuck Kraken directional (spec) positions.

These positions were opened before fill/cost-basis persistence landed, so the
``cost_basis`` table has no row for them. On every cycle the pillar then
recovers no basis and anchors ``entry`` to the *current* price — which zeroes
gain% and silently DISABLES the stop-loss and trailing-stop (they measure
against entry). Only the rare momentum exit ever fires, so the book just
accumulates. See ``KrakenPillar._recover_entry`` / ``_reconcile_positions``.

This reconstructs the true fee-weighted entry basis from Kraken's own
``TradesHistory`` (replayed through the same BUY/SELL averaging the live
``PnLTracker`` uses) and writes a ``cost_basis`` row per currently-held
directional pair. Once written, the existing stop-loss / trailing-stop re-arm
and the pillar's normal exit path takes over — no forced liquidation.

DRY-RUN by default (prints what it would write, touches nothing). Pass
``--execute`` to write to the DB.

  Preview:  python scripts/unstick_kraken_spec.py
  Execute:  python scripts/unstick_kraken_spec.py --execute
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from datetime import datetime, timezone

from config.settings import Settings
from auramaur.exchange.kraken import KrakenSpotClient


async def _held_directional_bases(k: KrakenSpotClient, pairs: list[str]):
    """Find configured pairs we currently hold a non-dust balance of.

    Returns ``(held, bal, aliases)`` where ``held`` maps base-asset code -> pair
    altname, and ``aliases`` maps pair altname -> the set of identifiers Kraken
    might use for it in TradesHistory (altname, internal name, wsname) so trade
    matching doesn't silently miss when Kraken reports the internal name.
    """
    info = await k._public("AssetPairs")
    by_alt = {}
    aliases: dict[str, set[str]] = {}
    for key, m in info.items():
        alt = m.get("altname")
        if not alt:
            continue
        by_alt[alt] = m
        aliases[alt] = {alt, key}
        if m.get("wsname"):
            aliases[alt].add(m["wsname"])
    bal = await k.get_free_balance()
    held: dict[str, str] = {}
    for pair in pairs:
        meta = by_alt.get(pair)
        if not meta:
            continue
        base = meta.get("base")
        if base and bal.get(base, 0.0) > 0:
            held[base] = pair
    return held, bal, aliases


async def _fetch_all_trades(k: KrakenSpotClient) -> list[dict]:
    """Full TradesHistory, paginated.

    Kraken returns at most 50 trades per call; without paging the offset, a
    pair with older trades would get an understated basis. Loops via ``ofs``
    until the reported ``count`` is reached (bounded for safety).
    """
    trades: list[dict] = []
    ofs = 0
    for _ in range(200):  # safety bound: 200 * 50 = 10k trades
        resp = await k._private("TradesHistory", {"type": "all", "ofs": ofs})
        if resp.get("error"):
            raise RuntimeError(str(resp["error"]))
        result = resp.get("result", {}) or {}
        page = list((result.get("trades", {}) or {}).values())
        trades.extend(page)
        count = int(result.get("count", 0) or 0)
        if not page or len(trades) >= count:
            break
        ofs += len(page)
    return trades


def _replay(trades: list[dict]) -> tuple[float, float, float, float]:
    """Replay buys/sells chronologically -> (size, avg_cost, total_cost,
    realized_pnl), matching PnLTracker.record_fill: fee-exclusive cost, avg
    held on sells."""
    size = avg = total_cost = realized = 0.0
    for t in sorted(trades, key=lambda x: float(x["time"])):
        price = float(t["price"])
        vol = float(t["vol"])
        fee = float(t.get("fee", 0) or 0)
        cost = price * vol
        if t["type"] == "buy":
            size += vol
            total_cost += cost
            avg = total_cost / size if size > 0 else 0.0
        else:  # sell
            sell = min(vol, size)
            realized += (price - avg) * sell - fee
            size -= sell
            total_cost = avg * size
            if size <= 0:
                avg = total_cost = 0.0
    return size, avg, total_cost, realized


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="write to the DB (default: dry-run)")
    ap.add_argument("--db", default="auramaur.db", help="sqlite DB path (default: auramaur.db)")
    args = ap.parse_args()

    s = Settings()
    if not (s.kraken_api_key and s.kraken_api_secret):
        print("No Kraken credentials in env — aborting.")
        return
    k = KrakenSpotClient(s)
    pairs = list(s.kraken.directional_pairs)

    held, _bal, aliases = await _held_directional_bases(k, pairs)
    if not held:
        print("No held directional positions — nothing to backfill.")
        await k.close()
        return

    try:
        all_trades = await _fetch_all_trades(k)
    except RuntimeError as e:
        print("TradesHistory error:", e)
        await k.close()
        return
    print(f"Fetched {len(all_trades)} trades from TradesHistory.")

    # Group trades by pair, matching any identifier Kraken might report (altname,
    # internal name, wsname) so internal-name trades aren't silently skipped.
    rows = []
    for base, pair in held.items():
        ids = aliases.get(pair, {pair})
        ptrades = [t for t in all_trades if t.get("pair") in ids]
        if not ptrades:
            print(f"  {pair}: held but NO trade history found — cannot recover basis, skipping.")
            continue
        size, avg, total_cost, realized = _replay(ptrades)
        if size <= 0:
            print(f"  {pair}: replay netted zero size — skipping.")
            continue
        rows.append((pair, size, avg, total_cost, realized, len(ptrades)))

    print(f"\n{'pair':12s} {'size':>16s} {'avg_cost':>12s} {'total_cost':>12s} {'realized':>10s}  trades")
    for pair, size, avg, total_cost, realized, n in rows:
        print(f"{pair:12s} {size:16.8f} {avg:12.8f} {total_cost:12.4f} {realized:10.4f}  {n}")

    if not args.execute:
        print("\nDRY-RUN — nothing written. Re-run with --execute to persist these cost_basis rows.")
        await k.close()
        return

    # Raw sqlite3 (not the Database wrapper, which would re-run schema init /
    # migrations against the live DB). busy_timeout waits out the running bot's
    # write transactions instead of failing immediately with "database is locked".
    await k.close()
    now = datetime.now(timezone.utc).isoformat()
    con = sqlite3.connect(args.db, timeout=30.0)
    try:
        con.execute("PRAGMA busy_timeout=30000")
        con.executemany(
            """INSERT INTO cost_basis
               (market_id, token, token_id, size, avg_cost, total_cost, realized_pnl, is_paper, updated_at)
               VALUES (?, 'YES', ?, ?, ?, ?, ?, 0, ?)
               ON CONFLICT(market_id, is_paper) DO UPDATE SET
                   token_id = excluded.token_id, size = excluded.size,
                   avg_cost = excluded.avg_cost, total_cost = excluded.total_cost,
                   realized_pnl = excluded.realized_pnl, updated_at = excluded.updated_at""",
            [(pair, pair, size, avg, total_cost, realized, now)
             for pair, size, avg, total_cost, realized, _n in rows],
        )
        con.commit()
    finally:
        con.close()
    print(f"\nWrote {len(rows)} cost_basis rows (is_paper=0). Stops will re-arm on the next pillar cycle.")


if __name__ == "__main__":
    asyncio.run(main())
