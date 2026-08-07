"""Manual-trade sweep (auramaur/broker/manual_trades.py).

The 2026-08-05 incident: two operator sells at the venue (SELL NO 83.33 @
0.41236 and SELL NO 121.11 @ 0.21) reconciled HOLDINGS correctly but booked no
pnl_ledger 'sell' event — +$37.70 of term_structure's realized P&L vanished
from attribution. These tests pin the sweep's contract: exactly one ledger row
per manual sell with entry-ancestry attribution, idempotency by source_ref AND
by cursor, bot-placed trades and manual buys book nothing, the cursor
initializes to NOW (no historical backfill), fetch failures never move the
cursor, and the stats/cursor batch is atomic (crash-mid-batch, house pattern).
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone

import pytest

import auramaur.broker.manual_trades as manual_trades_mod
from auramaur.broker.manual_trades import VENUE, sweep_manual_trades
from auramaur.db.database import Database
from tests.txn_helpers import failing_on, span_owners, transaction_spy

PROXY = "0x" + "a" * 40
COND = "0x" + "c" * 64
MID = "3206142"
T0 = 1_754_000_000.0  # fixed sweep cursor for deterministic feeds

# Incident leg 1: SELL NO 83.33 @ 0.41236 against a 0.20 basis.
SIZE = 83.33
ENTRY = 0.20
EXIT = 0.41236
PNL = (EXIT - ENTRY) * SIZE


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _trade(tx: str, *, side: str = "SELL", outcome: str = "No", index: int = 1,
           size: float = SIZE, price: float = EXIT, ts: float = T0 + 600,
           cond: str = COND) -> dict:
    return {"conditionId": cond, "side": side, "outcome": outcome,
            "outcomeIndex": index, "size": size, "price": price,
            "timestamp": ts, "transactionHash": tx}


class _Response:
    def __init__(self, payload, status: int = 200):
        self.payload = payload
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    async def json(self):
        return self.payload


class _Session:
    """Fake aiohttp session, same idiom as tests/test_current_positions.py."""

    def __init__(self, pages, status: int = 200):
        self.pages = pages
        self.status = status
        self.params: list[dict] = []
        self.closed = False

    def get(self, _url, *, params, timeout):
        assert timeout == 15
        self.params.append(params)
        return _Response(self.pages.get(int(params["offset"]), []), self.status)

    async def close(self):
        self.closed = True


async def _setup_db(tmp_path, *, with_cost_basis: bool = True) -> Database:
    """Live DB with one bot-held NO position and its entry ancestry."""
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    await db.execute(
        """INSERT INTO markets (id, exchange, condition_id, question, active,
             last_updated)
           VALUES (?, 'polymarket', ?, 'Will Gemini ship?', 1,
                   datetime('now'))""",
        (MID, COND))
    # Entry ancestry: the trade that DECIDED the position + its paired fill.
    # record_ledger_event's token-scoped attribution resolves through this
    # (trades JOIN fills ON order_id, same token/mode).
    await db.execute(
        """INSERT INTO trades (market_id, side, size, price, is_paper,
             order_id, status, strategy_source, timestamp)
           VALUES (?, 'BUY', ?, ?, 0, 'ord-entry', 'filled',
                   'term_structure', ?)""",
        (MID, SIZE, ENTRY, _iso(T0 - 86400)))
    await db.execute(
        """INSERT INTO fills (order_id, market_id, token_id, side, token,
             size, price, fee, is_paper, timestamp)
           VALUES ('ord-entry', ?, 'tok-no', 'BUY', 'NO', ?, ?, 0, 0, ?)""",
        (MID, SIZE, ENTRY, _iso(T0 - 86400)))
    if with_cost_basis:
        await db.execute(
            """INSERT INTO cost_basis (market_id, token, token_id, size,
                 avg_cost, total_cost, is_paper, updated_at)
               VALUES (?, 'NO', 'tok-no', ?, ?, ?, 0, datetime('now'))""",
            (MID, SIZE, ENTRY, SIZE * ENTRY))
    await db.commit()
    return db


async def _seed_cursor(db: Database, ts: float) -> None:
    await db.execute(
        "INSERT INTO manual_trade_state (venue, cursor_ts) VALUES (?, ?)",
        (VENUE, ts))
    await db.commit()


async def _cursor(db: Database) -> float | None:
    row = await db.fetchone(
        "SELECT cursor_ts FROM manual_trade_state WHERE venue = ?", (VENUE,))
    return None if row is None else float(row["cursor_ts"])


async def _ledger_rows(db: Database) -> list[dict]:
    return [dict(r) for r in await db.fetchall(
        "SELECT * FROM pnl_ledger ORDER BY id")]


# ---------------------------------------------------------------------------
# (a) A manual sell of a bot-held position books exactly one 'sell' row with
#     book-basis P&L and entry-ancestry attribution.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_manual_sell_books_one_row_with_ancestry_attribution(tmp_path):
    db = await _setup_db(tmp_path)
    try:
        await _seed_cursor(db, T0)
        session = _Session({0: [_trade("0xhash1")]})
        booked = await sweep_manual_trades(db, PROXY, session=session)
        assert booked == 1

        rows = await _ledger_rows(db)
        assert len(rows) == 1
        row = rows[0]
        assert row["kind"] == "sell"
        assert row["market_id"] == MID
        assert row["token"] == "NO"
        assert row["is_paper"] == 0
        assert row["qty"] == pytest.approx(SIZE)
        assert row["pnl"] == pytest.approx(PNL)
        assert row["source_ref"] == "venue-trade:0xhash1"
        assert row["venue"] == "polymarket"
        # Attribution resolved from the entry fill/trade ancestry — the
        # incident's missing +$37.70 belonged to term_structure.
        assert row["strategy_source"] == "term_structure"
        assert row["realized_at"] == _iso(T0 + 600)

        ds = await db.fetchone(
            "SELECT total_pnl, trades_count, wins, losses FROM daily_stats")
        assert ds["total_pnl"] == pytest.approx(PNL)
        assert (ds["trades_count"], ds["wins"], ds["losses"]) == (1, 1, 0)
        assert await _cursor(db) == pytest.approx(T0 + 600)
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# (b) Idempotent: re-sweeping the same feed books nothing — both through the
#     cursor filter and, with the cursor rewound, through the UNIQUE ref.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_second_sweep_of_same_feed_books_nothing(tmp_path):
    db = await _setup_db(tmp_path)
    try:
        await _seed_cursor(db, T0)
        pages = {0: [_trade("0xhash1")]}
        assert await sweep_manual_trades(db, PROXY, session=_Session(pages)) == 1

        # Cursor layer: the trade is behind the cursor now.
        assert await sweep_manual_trades(db, PROXY, session=_Session(pages)) == 0

        # Ref layer: even with the cursor rewound, the UNIQUE source_ref holds.
        await db.execute(
            "UPDATE manual_trade_state SET cursor_ts = ? WHERE venue = ?",
            (T0, VENUE))
        await db.commit()
        assert await sweep_manual_trades(db, PROXY, session=_Session(pages)) == 0

        rows = await _ledger_rows(db)
        assert len(rows) == 1
        ds = await db.fetchone(
            "SELECT total_pnl, trades_count FROM daily_stats")
        assert ds["trades_count"] == 1
        assert ds["total_pnl"] == pytest.approx(PNL)
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# (c) Bot-placed: a venue trade covered by an in-window live fills row (size
#     within 2%, timestamp within ±10 min) books nothing.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bot_placed_trade_books_nothing(tmp_path):
    db = await _setup_db(tmp_path)
    try:
        await _seed_cursor(db, T0)
        # The bot's own exit fill: 83.0 vs the venue's 83.33 (within 2%),
        # 60 seconds before the venue trade.
        await db.execute(
            """INSERT INTO fills (order_id, market_id, token_id, side, token,
                 size, price, fee, is_paper, timestamp)
               VALUES ('ord-exit', ?, 'tok-no', 'SELL', 'NO', 83.0, ?, 0, 0,
                       ?)""",
            (MID, EXIT, _iso(T0 + 540)))
        await db.commit()

        session = _Session({0: [_trade("0xhash-bot")]})
        assert await sweep_manual_trades(db, PROXY, session=session) == 0
        assert await _ledger_rows(db) == []
        # The bot trade is still consumed — the cursor moves past it.
        assert await _cursor(db) == pytest.approx(T0 + 600)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_existing_fill_ref_ledger_row_blocks_booking(tmp_path):
    """The second guard: no fills row survives, but a fill:% sell ledger row
    with matching market/qty in-window proves the bot already booked it."""
    db = await _setup_db(tmp_path)
    try:
        await _seed_cursor(db, T0)
        await db.execute(
            """INSERT INTO pnl_ledger (market_id, venue, kind, token, qty, pnl,
                 is_paper, source_ref, realized_at)
               VALUES (?, 'polymarket', 'sell', 'NO', ?, 15.0, 0, 'fill:99',
                       ?)""",
            (MID, SIZE, _iso(T0 + 500)))
        await db.commit()

        session = _Session({0: [_trade("0xhash-dup")]})
        assert await sweep_manual_trades(db, PROXY, session=session) == 0
        rows = await _ledger_rows(db)
        assert len(rows) == 1 and rows[0]["source_ref"] == "fill:99"
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# (d) First run initializes the cursor to NOW; a historical trade older than
#     that cursor is never booked (no backfill — manual-sell:* refs exist).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cursor_initializes_to_now_and_never_backfills(tmp_path):
    db = await _setup_db(tmp_path)
    try:
        historical = _trade("0xhash-old", ts=time.time() - 3600)
        session = _Session({0: [historical]})

        before = time.time()
        assert await sweep_manual_trades(db, PROXY, session=session) == 0
        cursor = await _cursor(db)
        assert cursor is not None and cursor >= before
        # The init run must not even consume the feed.
        assert session.params == []

        # Second run fetches — and the historical trade stays unbooked.
        assert await sweep_manual_trades(db, PROXY, session=session) == 0
        assert await _ledger_rows(db) == []
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# (e) A fetch failure raises out of the sweep and never touches the cursor.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_http_error_raises_and_preserves_cursor(tmp_path):
    db = await _setup_db(tmp_path)
    try:
        await _seed_cursor(db, T0)
        session = _Session({0: [_trade("0xhash1")]}, status=500)
        with pytest.raises(RuntimeError, match="HTTP 500"):
            await sweep_manual_trades(db, PROXY, session=session)
        assert await _cursor(db) == pytest.approx(T0)
        assert await _ledger_rows(db) == []
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# (f) Crash-mid-batch: the stats+cursor span is atomic under its own owner;
#     the ledger row is the idempotent anchor and the retry cannot double-book.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_crash_mid_batch_is_atomic_and_retry_safe(tmp_path):
    db = await _setup_db(tmp_path)
    try:
        await _seed_cursor(db, T0)
        pages = {0: [_trade("0xhash1")]}
        events: list = []
        with transaction_spy(db, events), failing_on(db, "manual_trade_state"):
            with pytest.raises(sqlite3.OperationalError, match="injected"):
                await sweep_manual_trades(db, PROXY, session=_Session(pages))
        assert "manual_trades.sweep" in span_owners(events)

        # The ledger row landed (its own commit, before the span) …
        rows = await _ledger_rows(db)
        assert len(rows) == 1
        assert rows[0]["source_ref"] == "venue-trade:0xhash1"
        # … but NOTHING from the failed span survives: no stats accrual,
        # cursor unmoved.
        assert await db.fetchone("SELECT 1 FROM daily_stats") is None
        assert await _cursor(db) == pytest.approx(T0)

        # Retry: the ref dedupes, the cursor advances, no second row and no
        # stats double-accrual (the crashed trade's accrual is forfeited by
        # design — same trade-off as _settle_position).
        assert await sweep_manual_trades(db, PROXY, session=_Session(pages)) == 0
        assert len(await _ledger_rows(db)) == 1
        assert await _cursor(db) == pytest.approx(T0 + 600)
        assert await db.fetchone("SELECT 1 FROM daily_stats") is None
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# (g) A manual BUY books nothing but is logged for visibility.
# ---------------------------------------------------------------------------

class _LogRecorder:
    def __init__(self):
        self.events: list[tuple[str, str, dict]] = []

    def __getattr__(self, level):
        def _record(event, **kw):
            self.events.append((level, event, kw))
        return _record


@pytest.mark.asyncio
async def test_manual_buy_books_nothing_but_logs(tmp_path, monkeypatch):
    db = await _setup_db(tmp_path)
    try:
        await _seed_cursor(db, T0)
        recorder = _LogRecorder()
        monkeypatch.setattr(manual_trades_mod, "log", recorder)

        session = _Session(
            {0: [_trade("0xhash-buy", side="BUY", price=0.35, size=40.0)]})
        assert await sweep_manual_trades(db, PROXY, session=session) == 0
        assert await _ledger_rows(db) == []
        assert await db.fetchone("SELECT 1 FROM daily_stats") is None
        assert await _cursor(db) == pytest.approx(T0 + 600)

        buys = [e for e in recorder.events if e[1] == "manual_trades.buy_observed"]
        assert len(buys) == 1
        assert buys[0][2]["market_id"] == MID
        assert buys[0][2]["size"] == pytest.approx(40.0)
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Basis fallback: cost_basis already pruned (the sweep ran after the mirror)
# -> basis reconstructed from live entry fills minus previously realized qty.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_basis_falls_back_to_fills_when_cost_basis_pruned(tmp_path):
    db = await _setup_db(tmp_path, with_cost_basis=False)
    try:
        # Rebuild ancestry as the incident's second leg: bought 121.11 @ 0.30,
        # bot partially sold 21.11 long ago -> residual 100 @ 0.30 avg.
        await db.execute("DELETE FROM fills")
        await db.execute(
            """INSERT INTO fills (order_id, market_id, token_id, side, token,
                 size, price, fee, is_paper, timestamp)
               VALUES ('ord-entry', ?, 'tok-no', 'BUY', 'NO', 121.11, 0.30, 0,
                       0, ?)""",
            (MID, _iso(T0 - 86400)))
        await db.execute(
            """INSERT INTO fills (order_id, market_id, token_id, side, token,
                 size, price, fee, is_paper, timestamp)
               VALUES ('ord-part', ?, 'tok-no', 'SELL', 'NO', 21.11, 0.50, 0,
                       0, ?)""",
            (MID, _iso(T0 - 43200)))
        await db.commit()
        await _seed_cursor(db, T0)

        session = _Session(
            {0: [_trade("0xhash2", size=100.0, price=0.21)]})
        assert await sweep_manual_trades(db, PROXY, session=session) == 1
        rows = await _ledger_rows(db)
        assert len(rows) == 1
        assert rows[0]["qty"] == pytest.approx(100.0)
        assert rows[0]["pnl"] == pytest.approx((0.21 - 0.30) * 100.0)
        assert rows[0]["strategy_source"] == "term_structure"
    finally:
        await db.close()
