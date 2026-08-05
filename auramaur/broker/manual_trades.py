"""Manual-trade sweep — book off-bot Polymarket exits into the P&L ledger.

The 2026-08-05 incident: the operator manually sold two bot-held positions at
the venue (markets 3206142/3206143). The position mirror correctly reconciled
HOLDINGS — the cost_basis/portfolio rows vanished on the next sync cycle and
the cash appeared — but no ``pnl_ledger`` 'sell' event was ever booked, so
+$37.70 of term_structure's realized P&L silently vanished from attribution
and had to be booked by hand. This module closes that gap: it polls the venue's
per-wallet trade history (``data-api.polymarket.com/trades``), identifies
trades the bot did NOT place, and books manual SELLs of bot-held positions as
``kind='sell'`` ledger events.

Bot-vs-manual predicate
-----------------------
A venue SELL is classified BOT-PLACED iff live ``fills`` rows exist for the
same (market, token, side='SELL') whose timestamps are within ±10 minutes of
the venue trade AND whose total size covers the venue trade's size within 2%
(``trade.size <= 1.02 * sum(in-window fill sizes)``). Summing — rather than
requiring a single 2%-matching fill — also covers the split-fill shape where
the venue reports one bot order as several partial trade rows (each partial is
contained by the order's one fills row) or one venue row against several bot
fills. The deliberate bias is conservative: any same-side bot activity in the
window large enough to explain the venue row claims it, because a false "bot"
classification loses one booking (recoverable by hand, exactly the pre-module
state) while a false "manual" would DOUBLE-BOOK realized P&L. As a second
independent guard, a venue SELL is also skipped when an existing
``pnl_ledger`` sell row with ``source_ref LIKE 'fill:%'`` matches the same
market/token with qty within 2% and ``realized_at`` within the same ±10-minute
window — catching a bot exit whose fills row was renamed/purged after booking.
BUYs get the same classification but are never booked either way: the
cost_basis mirror (bot._task_position_sync) absorbs venue buys into the
venue-netted avg price on the next cycle; manual buys are logged
(``manual_trades.buy_observed``) so they stay visible.

Holdings / prune interaction
----------------------------
This sweep books the LEDGER EVENT ONLY — it never drains cost_basis or
portfolio. Verified against bot.py's ``_task_position_sync`` (live branch):
the cost_basis mirror overwrites ``size``/``avg_cost`` with venue truth every
cycle (``ON CONFLICT ... SET size = excluded.size``) and DELETEs any live
Polymarket row whose token the venue no longer reports (the stale-remove
DELETE with ``token_id NOT IN (...)`` — this is what erased the incident's
rows within one cycle). Holdings are therefore fully mirror-owned; a drain
here would be redundant at best and would fight the mirror at worst. The
sweep must run BEFORE the reconcile/mirror in the cycle so the cost_basis row
still exists when its exit is booked.

P&L basis and precision
-----------------------
Preferred basis is the live cost_basis row for (market, token) — the
token-scoped average the mirror maintains. If the row is already pruned (the
sweep can run after the exit's mirror cycle, e.g. across a restart), the basis
is reconstructed from live entry fills for that market/token: a
weighted-average walk in timestamp order, BUYs blending the average, SELLs
reducing size (average unchanged) — the same accounting as
``ledger.backfill_ledger``. Precision limits, in both branches: entry fees are
not capitalized into the basis; positions acquired before fill history began
(or absorbed off-bot via the mirror) carry the venue's netted average rather
than a true entry ledger; earlier venue-trade bookings do not reduce the
walk's residual size, so a second manual sell in the same window can book
against a stale (over-stated) holdable qty — the per-unit P&L is still
correct because a weighted average is invariant under sells. If no basis
exists at all, the trade is logged (``manual_trades.no_basis``) and skipped —
never book fabricated P&L.

Cursor and idempotency
----------------------
The cursor (``manual_trade_state``, one row per venue, unix seconds) persists
the newest processed trade timestamp. On first run it initializes to NOW and
the run books nothing: NO historical backfill, because pre-existing off-bot
exits were hand-booked under ``manual-sell:*`` refs and a backfill would
double-book them under ``venue-trade:*`` refs. The cursor only ever advances
(MAX upsert) and only after a successful fetch — a request failure raises and
leaves it untouched. Booking is idempotent by construction: ``source_ref``
``venue-trade:<transactionHash>`` and the column is UNIQUE. If one on-chain
transaction carries several distinct trades (one hash, many rows), only the
first books — a conservative under-count, consistent with the false-"bot"
bias above. Per booked trade, the daily_stats accrual and the cursor advance
land in one ``manual_trades.sweep`` transaction span AFTER the ledger row's
own commit; a crash between the two leaves the ledger row as the idempotent
anchor, and the retry takes the skip path (the stats accrual is then lost,
the same documented trade-off as ResolutionTracker._settle_position).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import aiohttp
import structlog

from auramaur.broker.ledger import record_ledger_event
from auramaur.broker.reconciler import token_for_outcome
from auramaur.db.database import Database

log = structlog.get_logger()

POLYMARKET_TRADES_API = "https://data-api.polymarket.com/trades"

VENUE = "polymarket"
# Bot-match window and size tolerance for the bot-vs-manual predicate.
BOT_MATCH_WINDOW_SECONDS = 600.0
SIZE_TOLERANCE = 0.02


@dataclass
class VenueTrade:
    """One trade row from Polymarket's per-wallet trade history."""

    condition_id: str
    side: str            # "BUY" | "SELL"
    outcome: str
    outcome_index: int   # data-api outcomeIndex; -1 = absent
    size: float
    price: float
    timestamp: float     # unix seconds
    tx_hash: str


async def fetch_trades_since(
    proxy_address: str,
    cursor_ts: float,
    session: aiohttp.ClientSession | None = None,
) -> list[VenueTrade]:
    """Fetch wallet trades newer than *cursor_ts*, oldest first.

    Same house pattern as ``redeemer.fetch_current_positions``: offset
    pagination, optional caller-owned session, and hard-fail semantics — a
    request failure or malformed payload raises and therefore never advances
    the cursor or erases prior good state. The data-api returns newest-first;
    paging stops once a page reaches trades at/behind the cursor.
    """
    if not proxy_address:
        raise ValueError("proxy_address is required")
    page_size = 500
    max_offset = 10_000
    close_session = session is None
    if session is None:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20))
    items: list[dict] = []
    offset = 0
    try:
        while True:
            params = {
                "user": proxy_address,
                "limit": str(page_size), "offset": str(offset),
            }
            async with session.get(
                POLYMARKET_TRADES_API, params=params, timeout=15,
            ) as resp:
                resp.raise_for_status()
                page = await resp.json()
            if not isinstance(page, list) or any(not isinstance(row, dict) for row in page):
                raise ValueError("Polymarket trades response must be a list of objects")
            items.extend(page)
            if len(page) < page_size:
                break
            oldest = min(
                (float(row.get("timestamp") or 0) for row in page), default=0.0)
            if oldest <= cursor_ts:
                break
            offset += page_size
            if offset > max_offset:
                raise RuntimeError("Polymarket trades exceeded supported pagination range")
    finally:
        if close_session:
            await session.close()

    trades: list[VenueTrade] = []
    for item in items:
        ts = float(item.get("timestamp") or 0)
        if ts <= cursor_ts:
            continue
        tx_hash = str(item.get("transactionHash", ""))
        condition_id = str(item.get("conditionId", ""))
        if not tx_hash or not condition_id:
            raise ValueError("Polymarket trade is missing transactionHash or conditionId")
        try:
            raw_index = item.get("outcomeIndex")
            outcome_index = int(raw_index) if raw_index is not None else -1
        except (TypeError, ValueError):
            outcome_index = -1
        trades.append(VenueTrade(
            condition_id=condition_id,
            side=str(item.get("side", "")).upper(),
            outcome=str(item.get("outcome", "")),
            outcome_index=outcome_index,
            size=float(item.get("size", 0) or 0),
            price=float(item.get("price", 0) or 0),
            timestamp=ts,
            tx_hash=tx_hash,
        ))
    trades.sort(key=lambda t: t.timestamp)
    return trades


def _parse_ts(value: str | None) -> float | None:
    """ISO / ``datetime('now')`` DB timestamp -> unix seconds (naive = UTC)."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


async def _is_bot_trade(
    db: Database, market_id: str, token: str, side: str, trade: VenueTrade,
) -> bool:
    """True iff in-window live fills of the same (market, token, side) cover
    this venue trade's size within tolerance (see module docstring)."""
    rows = await db.fetchall(
        "SELECT size, timestamp FROM fills "
        "WHERE market_id = ? AND UPPER(token) = UPPER(?) AND side = ? AND is_paper = 0",
        (market_id, token, side),
    )
    window_total = 0.0
    for r in rows:
        ts = _parse_ts(r["timestamp"])
        if ts is None or abs(ts - trade.timestamp) > BOT_MATCH_WINDOW_SECONDS:
            continue
        window_total += float(r["size"])
    if window_total <= 0:
        return False
    return trade.size <= window_total * (1.0 + SIZE_TOLERANCE) + 1e-9


async def _booked_via_fill_ref(
    db: Database, market_id: str, token: str, trade: VenueTrade,
) -> bool:
    """Second double-book guard: an existing ``fill:%`` sell ledger row for the
    same market/token with qty within 2% and realized_at within the bot window
    (unparseable realized_at counts as a match — conservative toward bot)."""
    rows = await db.fetchall(
        """SELECT qty, realized_at FROM pnl_ledger
           WHERE market_id = ? AND UPPER(token) = UPPER(?)
             AND kind = 'sell' AND is_paper = 0
             AND source_ref LIKE 'fill:%'""",
        (market_id, token),
    )
    for r in rows:
        if abs(float(r["qty"]) - trade.size) > trade.size * SIZE_TOLERANCE + 1e-9:
            continue
        ts = _parse_ts(r["realized_at"])
        if ts is None or abs(ts - trade.timestamp) <= BOT_MATCH_WINDOW_SECONDS:
            return True
    return False


async def _basis_for(
    db: Database, market_id: str, token: str,
) -> tuple[float, float] | None:
    """``(avg_cost, held_size)`` for the live position, or None if unknowable.

    Prefers the live cost_basis row (the mirror's token-scoped average); falls
    back to a weighted-average walk over live fills — the backfill_ledger
    accounting — when the row was already pruned. Precision limits are
    documented in the module docstring.
    """
    row = await db.fetchone(
        "SELECT avg_cost, size FROM cost_basis "
        "WHERE market_id = ? AND is_paper = 0 AND UPPER(token) = UPPER(?) AND size > 0",
        (market_id, token),
    )
    if row is not None:
        return float(row["avg_cost"]), float(row["size"])
    fills = await db.fetchall(
        "SELECT side, size, price FROM fills "
        "WHERE market_id = ? AND UPPER(token) = UPPER(?) AND is_paper = 0 "
        "ORDER BY timestamp, id",
        (market_id, token),
    )
    size = 0.0
    avg = 0.0
    for f in fills:
        if f["side"] == "BUY":
            new_size = size + float(f["size"])
            if new_size > 0:
                avg = ((avg * size) + float(f["price"]) * float(f["size"])) / new_size
            size = new_size
        else:
            size = max(0.0, size - float(f["size"]))
            if size <= 0:
                avg = 0.0
    if size <= 0 or avg <= 0:
        return None
    return avg, size


async def _advance_cursor(db: Database, ts: float) -> None:
    """Monotonic cursor upsert — must run inside the caller's span."""
    await db.execute(
        """INSERT INTO manual_trade_state (venue, cursor_ts, updated_at)
           VALUES (?, ?, datetime('now'))
           ON CONFLICT(venue) DO UPDATE SET
               cursor_ts = MAX(cursor_ts, excluded.cursor_ts),
               updated_at = excluded.updated_at""",
        (VENUE, ts),
    )


async def _book_manual_sell(
    db: Database, market_id: str, token: str, trade: VenueTrade,
) -> bool:
    """Book one manual SELL; returns True iff a new ledger row landed."""
    source_ref = f"venue-trade:{trade.tx_hash}"
    if await db.fetchone(
            "SELECT 1 FROM pnl_ledger WHERE source_ref = ?", (source_ref,)):
        return False

    basis = await _basis_for(db, market_id, token)
    if basis is None:
        log.warning(
            "manual_trades.no_basis",
            market_id=market_id, token=token, size=round(trade.size, 4),
            price=trade.price, tx_hash=trade.tx_hash[:18],
        )
        return False
    avg_cost, held = basis
    qty = min(trade.size, held)
    if qty < trade.size:
        log.warning(
            "manual_trades.sell_exceeds_position",
            market_id=market_id, token=token,
            trade_size=round(trade.size, 4), held=round(held, 4),
        )
    pnl = (trade.price - avg_cost) * qty
    realized_at = datetime.fromtimestamp(
        trade.timestamp, tz=timezone.utc).isoformat()

    # The ledger row goes FIRST, outside any span (record_ledger_event manages
    # its own commit and resolves strategy attribution from entry ancestry).
    # It is the idempotent anchor: a crash before the stats/cursor span leaves
    # the ref booked, so the retry takes the skip path above.
    await record_ledger_event(
        db,
        market_id=market_id, kind="sell", token=token,
        qty=qty, pnl=pnl, fees=0.0, is_paper=False,
        source_ref=source_ref, realized_at=realized_at,
    )
    landed = await db.fetchone(
        "SELECT 1 FROM pnl_ledger WHERE source_ref = ?", (source_ref,))
    if landed is None:
        # record_ledger_event swallows write errors by contract; for the sweep
        # the ledger row IS the point, so surface the failure — the host loop
        # warns and the un-advanced cursor retries next cycle.
        raise RuntimeError(f"manual-trade ledger write did not land: {source_ref}")

    # Reporting accrual + cursor advance as one short, db-only span. Holdings
    # are NOT touched here — the position mirror owns them (module docstring).
    async with db.transaction(owner="manual_trades.sweep"):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        await db.execute(
            """INSERT INTO daily_stats (date, total_pnl, trades_count, wins, losses)
               VALUES (?, ?, 1, ?, ?)
               ON CONFLICT(date) DO UPDATE SET
                   total_pnl = total_pnl + excluded.total_pnl,
                   trades_count = trades_count + 1,
                   wins = wins + excluded.wins,
                   losses = losses + excluded.losses""",
            (today, pnl, 1 if pnl > 0 else 0, 1 if pnl < 0 else 0),
        )
        await _advance_cursor(db, trade.timestamp)

    log.info(
        "manual_trades.sell_booked",
        market_id=market_id, token=token, qty=round(qty, 4),
        price=trade.price, pnl=round(pnl, 4), source_ref=source_ref,
    )
    return True


async def sweep_manual_trades(
    db: Database,
    proxy_address: str,
    session: aiohttp.ClientSession | None = None,
) -> int:
    """Poll the venue trade history and book off-bot exits. Returns the
    number of newly booked ledger rows. See the module docstring for the
    classification predicate, basis rules and cursor semantics.
    """
    if not proxy_address:
        raise ValueError("proxy_address is required")

    row = await db.fetchone(
        "SELECT cursor_ts FROM manual_trade_state WHERE venue = ?", (VENUE,))
    if row is None:
        # First run: start the clock at NOW. Deliberately no backfill — the
        # operator hand-booked historical off-bot exits under manual-sell:*
        # refs, and a backfill would double-book them under venue-trade:*.
        now_ts = time.time()
        async with db.transaction(owner="manual_trades.sweep"):
            await db.execute(
                """INSERT OR IGNORE INTO manual_trade_state
                   (venue, cursor_ts, updated_at)
                   VALUES (?, ?, datetime('now'))""",
                (VENUE, now_ts),
            )
        log.info("manual_trades.cursor_initialized", cursor_ts=now_ts)
        return 0
    cursor_ts = float(row["cursor_ts"])

    # All network I/O happens here, before any DB write. A fetch failure
    # raises out of the sweep with the cursor untouched.
    trades = await fetch_trades_since(proxy_address, cursor_ts, session)
    if not trades:
        return 0

    booked = 0
    for trade in trades:
        market = await db.fetchone(
            "SELECT id FROM markets WHERE condition_id = ?",
            (trade.condition_id,))
        if market is None:
            # A market the bot never tracked cannot be a bot-held position;
            # nothing to attribute or realize against.
            log.info(
                "manual_trades.unknown_market",
                condition_id=trade.condition_id[:20], side=trade.side,
                size=round(trade.size, 4),
            )
            continue
        market_id = market["id"]
        token = token_for_outcome(trade.outcome_index, trade.outcome).value

        if trade.side != "SELL":
            # BUYs are never booked: the cost_basis mirror absorbs them into
            # the venue-netted avg price on the next cycle. Log the manual
            # ones so off-bot entries stay visible.
            if not await _is_bot_trade(db, market_id, token, "BUY", trade):
                log.info(
                    "manual_trades.buy_observed",
                    market_id=market_id, token=token,
                    size=round(trade.size, 4), price=trade.price,
                    tx_hash=trade.tx_hash[:18],
                )
            continue

        if await _is_bot_trade(db, market_id, token, "SELL", trade):
            log.debug(
                "manual_trades.bot_sell_skipped",
                market_id=market_id, token=token, size=round(trade.size, 4))
            continue
        if await _booked_via_fill_ref(db, market_id, token, trade):
            log.debug(
                "manual_trades.fill_ref_booked_skipped",
                market_id=market_id, token=token, size=round(trade.size, 4))
            continue

        if await _book_manual_sell(db, market_id, token, trade):
            booked += 1

    # Advance past the whole fetched batch (bot trades, buys and skips
    # included) so they are never re-fetched. Booked sells already advanced
    # incrementally inside their own spans.
    async with db.transaction(owner="manual_trades.sweep"):
        await _advance_cursor(db, max(t.timestamp for t in trades))

    if booked:
        log.info("manual_trades.sweep_complete", booked=booked,
                 scanned=len(trades))
    return booked
