"""Web dashboard API tests — the read-only FastAPI service over bot state.

The contract under test: every data endpoint serves the broker envelope
``{ok, error, updated_at, state}`` and never 500s — including with a missing
or schemaless database (the service must explain itself and recover on its
own) — and the dashboard's DB handle is structurally unable to write.
"""

import time

import aiosqlite
import pytest

# The dashboard is an OPTIONAL extra (pyproject [web]); skip cleanly where it
# isn't installed. CI installs it (--extra web) so this runs for real there.
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from auramaur.db.database import Database  # noqa: E402
from auramaur.web.app import create_app  # noqa: E402
from auramaur.web.db import ReadOnlyDatabase  # noqa: E402
from auramaur.web.serialize import serialize_state  # noqa: E402
from config.settings import Settings  # noqa: E402

FAST_REFRESH = 0.1  # keep degraded-recovery tests quick


async def _seed_bot_db(db_path: str) -> None:
    """A minimal paper book: one open position, one fill, one signal."""
    db = Database(db_path)
    await db.connect()
    try:
        await db.execute(
            """INSERT INTO markets (id, question, category, last_updated)
               VALUES ('mkt-1', 'Will it rain tomorrow?', 'weather', datetime('now'))"""
        )
        await db.execute(
            """INSERT INTO portfolio
               (market_id, exchange, side, size, avg_price, current_price,
                category, token, token_id, is_paper)
               VALUES ('mkt-1', 'polymarket', 'BUY', 20.0, 0.40, 0.55,
                       'weather', 'YES', 'tok-1', 1)"""
        )
        await db.execute(
            """INSERT INTO cost_basis
               (market_id, token, token_id, size, avg_cost, total_cost,
                realized_pnl, is_paper)
               VALUES ('mkt-1', 'YES', 'tok-1', 20.0, 0.40, 8.0, 0.0, 1)"""
        )
        await db.execute(
            """INSERT INTO fills (order_id, market_id, side, token, size, price, is_paper)
               VALUES ('ord-1', 'mkt-1', 'BUY', 'YES', 20.0, 0.40, 1)"""
        )
        await db.execute(
            """INSERT INTO signals
               (market_id, claude_prob, claude_confidence, market_prob, edge, action)
               VALUES ('mkt-1', 0.62, 'MEDIUM', 0.40, 22.0, 'BUY')"""
        )
        await db.commit()
    finally:
        await db.close()


def _paper_settings(tmp_path) -> Settings:
    settings = Settings()
    # Deterministic and offline: paper mode, no venue balance calls, and a
    # log tail isolated from any real auramaur.log in the working tree.
    settings.auramaur_live = False
    settings.execution.live = False
    settings.kalshi.enabled = False
    settings.kraken.enabled = False
    settings.logging.file = str(tmp_path / "auramaur.log")
    return settings


def _client(tmp_path, db_path: str) -> TestClient:
    return TestClient(create_app(
        db_path=db_path,
        settings=_paper_settings(tmp_path),
        refresh_seconds=FAST_REFRESH,
    ))


@pytest.mark.asyncio
async def test_state_endpoint_serves_seeded_paper_book(tmp_path):
    db_path = str(tmp_path / "auramaur.db")
    await _seed_bot_db(db_path)
    settings = _paper_settings(tmp_path)

    with _client(tmp_path, db_path) as client:
        env = client.get("/api/state").json()
        assert env["ok"] is True
        assert env["error"] is None
        assert env["updated_at"] is not None
        assert env["bot_mode"] == "paper"
        s = env["books"]["paper"]

        assert s["is_live"] is False
        assert s["position_count"] == 1
        pos = s["positions"][0]
        assert pos["market_id"] == "mkt-1"
        assert pos["question"] == "Will it rain tomorrow?"
        assert pos["token"] == "YES"  # row identity — a market can hold YES and NO
        # Unrealized MTM off cost_basis.avg_cost: (0.55 - 0.40) * 20 = 3.0
        assert pos["pnl"] == pytest.approx(3.0)
        assert s["total_pnl"] == pytest.approx(3.0)
        assert s["trade_count"] == 1
        assert s["balance"] == pytest.approx(
            settings.execution.paper_initial_balance + 3.0)
        assert s["signals"][0]["edge"] == pytest.approx(22.0)
        # Pillars are present but silent (no log lines seeded).
        assert all(p["age_seconds"] is None for p in s["pillars"])

        # The other book is served too, and it's empty — the toggle is a
        # view choice, not a second service.
        live = env["books"]["live"]
        assert live["position_count"] == 0
        assert live["trade_count"] == 0


@pytest.mark.asyncio
async def test_health_endpoint(tmp_path):
    db_path = str(tmp_path / "auramaur.db")
    await _seed_bot_db(db_path)

    with _client(tmp_path, db_path) as client:
        h = client.get("/api/health").json()
        assert h["database"] == {"ok": True, "detail": "ok"}
        assert h["is_live"] is False


@pytest.mark.asyncio
async def test_stream_emits_envelope_events(tmp_path):
    db_path = str(tmp_path / "auramaur.db")
    await _seed_bot_db(db_path)

    with _client(tmp_path, db_path) as client:
        # limit=1 bounds the otherwise-infinite stream so the response (and
        # the TestClient teardown) completes.
        resp = client.get("/api/stream?limit=1")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.text.startswith("event: state\ndata: ")
        assert '"ok": true' in resp.text
        assert '"position_count": 1' in resp.text


@pytest.mark.asyncio
async def test_missing_database_degrades_instead_of_500(tmp_path):
    """The exact failure that shipped as an infinite 'Connecting…' spinner:
    no usable database. The service must come up, say why, and never 500."""
    with _client(tmp_path, str(tmp_path / "nope.db")) as client:
        resp = client.get("/api/state")
        assert resp.status_code == 200
        env = resp.json()
        assert env["ok"] is False
        assert "nope.db" in env["error"]
        assert env["books"] is None

        h = client.get("/api/health").json()
        assert h["database"]["ok"] is False


@pytest.mark.asyncio
async def test_schemaless_database_names_the_problem(tmp_path):
    """An empty non-bot .db file (the Windows-dev repro) must produce an
    actionable message, not 'no such table: portfolio' via a 500."""
    db_path = str(tmp_path / "empty.db")
    async with aiosqlite.connect(db_path):
        pass  # creates a 0-table SQLite file

    with _client(tmp_path, db_path) as client:
        env = client.get("/api/state").json()
        assert env["ok"] is False
        assert "no bot schema" in env["error"]
        assert "AURAMAUR_DB_PATH" in env["error"]


@pytest.mark.asyncio
async def test_recovers_when_database_appears(tmp_path, tmp_path_factory):
    """The bot creating the DB after the dashboard started must be enough —
    no restart. The broker reconnects on its own cadence."""
    db_path = str(tmp_path / "auramaur.db")

    with _client(tmp_path, db_path) as client:
        assert client.get("/api/state").json()["ok"] is False

        await _seed_bot_db(db_path)

        deadline = time.time() + 5
        env = None
        while time.time() < deadline:
            env = client.get("/api/state").json()
            if env["ok"]:
                break
            time.sleep(FAST_REFRESH)
        assert env is not None and env["ok"] is True, f"never recovered: {env}"
        assert env["books"]["paper"]["position_count"] == 1


@pytest.mark.asyncio
async def test_book_split_and_breakdowns(tmp_path):
    """Paper and live books stay separate, and each carries its own
    per-strategy realized P&L (pnl_ledger) and category exposure; the Kraken
    paper book (its own table) surfaces on the paper view only."""
    db_path = str(tmp_path / "auramaur.db")
    await _seed_bot_db(db_path)

    db = Database(db_path)
    await db.connect()
    try:
        # A LIVE position + fill for the same market (books must not blur).
        await db.execute(
            """INSERT INTO portfolio
               (market_id, exchange, side, size, avg_price, current_price,
                category, token, token_id, is_paper)
               VALUES ('mkt-1', 'kalshi', 'BUY', 10.0, 0.50, 0.60,
                       'weather', 'YES', 'tok-1', 0)"""
        )
        await db.execute(
            """INSERT INTO fills (order_id, market_id, side, token, size, price, is_paper)
               VALUES ('ord-2', 'mkt-1', 'BUY', 'YES', 10.0, 0.50, 0)"""
        )
        # Ledger rows: two strategies on paper, one on live.
        for strategy, pnl, is_paper in (("llm", 4.0, 1), ("arb", -1.5, 1), ("llm", 2.0, 0)):
            await db.execute(
                """INSERT INTO pnl_ledger
                   (market_id, venue, category, strategy_source, kind, token,
                    qty, pnl, fees, is_paper, source_ref)
                   VALUES ('mkt-1', 'polymarket', 'weather', ?, 'sell', 'YES',
                           1, ?, 0.1, ?, ?)""",
                (strategy, pnl, is_paper, f"ref-{strategy}-{is_paper}"),
            )
        await db.execute(
            """INSERT INTO kraken_paper_positions
               (strategy, pair, quantity, entry_price, peak_gain_pct)
               VALUES ('llm', 'SOLUSDC', 2.5, 180.0, 3.2)"""
        )
        await db.commit()
    finally:
        await db.close()

    with _client(tmp_path, db_path) as client:
        env = client.get("/api/state").json()
        paper, live = env["books"]["paper"], env["books"]["live"]

        assert paper["position_count"] == 1 and live["position_count"] == 1
        assert live["positions"][0]["exchange"] == "kalshi"

        paper_strats = {s["strategy"]: s for s in paper["strategies"]}
        assert paper_strats["llm"]["pnl"] == pytest.approx(4.0)
        assert paper_strats["arb"]["pnl"] == pytest.approx(-1.5)
        live_strats = {s["strategy"]: s for s in live["strategies"]}
        assert set(live_strats) == {"llm"}
        assert live_strats["llm"]["pnl"] == pytest.approx(2.0)

        assert paper["categories"][0]["category"] == "weather"
        assert live["categories"][0]["positions"] == 1

        assert paper["kraken_paper"][0]["pair"] == "SOLUSDC"
        assert "kraken_paper" not in live


@pytest.mark.asyncio
async def test_two_sided_position_not_duplicated_by_cost_basis_join(tmp_path):
    """A market holding BOTH YES and NO must yield exactly two position rows
    and a P&L summed once per side. The cost_basis join must match on token —
    before 2026-07-19 it fanned out per side, duplicating rows and
    double-counting unrealized P&L (in the TUI cockpit too)."""
    db_path = str(tmp_path / "auramaur.db")
    await _seed_bot_db(db_path)

    db = Database(db_path)
    await db.connect()
    try:
        await db.execute(
            """INSERT INTO portfolio
               (market_id, exchange, side, size, avg_price, current_price,
                category, token, token_id, is_paper)
               VALUES ('mkt-1', 'polymarket', 'BUY', 10.0, 0.30, 0.20,
                       'weather', 'NO', 'tok-2', 1)"""
        )
        await db.execute(
            """INSERT INTO cost_basis
               (market_id, token, token_id, size, avg_cost, total_cost,
                realized_pnl, is_paper)
               VALUES ('mkt-1', 'NO', 'tok-2', 10.0, 0.30, 3.0, 0.0, 1)"""
        )
        await db.commit()
    finally:
        await db.close()

    with _client(tmp_path, db_path) as client:
        s = client.get("/api/state").json()["books"]["paper"]
        assert s["position_count"] == 2  # one YES row + one NO row, no fan-out
        by_token = {p["token"]: p for p in s["positions"]}
        assert by_token["YES"]["pnl"] == pytest.approx(3.0)   # (0.55−0.40)×20
        assert by_token["NO"]["pnl"] == pytest.approx(-1.0)   # (0.20−0.30)×10
        assert s["total_pnl"] == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_venue_balances_come_from_db_rows_with_age(tmp_path):
    """Venue cash reaches the dashboard via the rows the bot's balance
    recorder writes — the web process never calls a venue API (it holds no
    credentials). Both book views serve the same rows, each with an age so a
    stopped recorder shows as staleness, not as a fresh-looking number."""
    from datetime import datetime, timedelta, timezone

    db_path = str(tmp_path / "auramaur.db")
    await _seed_bot_db(db_path)

    db = Database(db_path)
    await db.connect()
    try:
        now = datetime.now(timezone.utc)
        for venue, detail, age_s in (
            ("ibkr", "$1,000.00 avail | $2,000.00 net", 30),
            ("kraken", "$500 USDC + 100 CAD", 7200),
        ):
            await db.execute(
                "INSERT INTO venue_balances (venue, detail, fetched_at) VALUES (?, ?, ?)",
                (venue, detail, (now - timedelta(seconds=age_s)).isoformat()),
            )
        await db.commit()
    finally:
        await db.close()

    with _client(tmp_path, db_path) as client:
        env = client.get("/api/state").json()
        assert env["ok"] is True
        for book in ("paper", "live"):
            venues = env["books"][book]["venues"]
            assert venues["ibkr"]["detail"] == "$1,000.00 avail | $2,000.00 net"
            assert 30 <= venues["ibkr"]["age_seconds"] < 300
            assert venues["kraken"]["age_seconds"] >= 7200


@pytest.mark.asyncio
async def test_readonly_database_cannot_write(tmp_path):
    """The safety property phase-4 control work must not erode: the dashboard's
    DB handle rejects writes at the SQLite layer, not by convention."""
    db_path = str(tmp_path / "auramaur.db")
    await _seed_bot_db(db_path)

    ro = ReadOnlyDatabase(db_path)
    await ro.connect()
    try:
        row = await ro.fetchone("SELECT COUNT(*) AS c FROM portfolio")
        assert row["c"] == 1
        with pytest.raises(aiosqlite.OperationalError, match="readonly|attempt to write"):
            await ro.db.execute("INSERT INTO fills (order_id, market_id, side, size, price)"
                                " VALUES ('x', 'mkt-1', 'BUY', 1, 0.5)")
    finally:
        await ro.close()


def test_serialize_state_is_json_safe():
    from datetime import datetime, timezone

    now = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
    earlier = datetime(2026, 7, 19, 11, 59, 30, tzinfo=timezone.utc)
    s = serialize_state({
        "now": now, "is_live": False, "transfers_armed": False,
        "kill_switch": False, "venues": {}, "pillars": {"kalshi": earlier, "ibkr": None},
        "activity": [("11:59:30", "order.filled mkt-1")],
        "health": {"errors": 0, "warnings": 0, "top": []},
        "positions": [], "position_count": 0, "position_value": 0.0,
        "signals": [], "trade_count": 0, "total_pnl": 0.0, "drawdown": 0.0,
        "balance": 5000.0,
    })
    assert s["pillars"][0] == {
        "name": "kalshi", "last_seen": earlier.isoformat(), "age_seconds": 30.0}
    assert s["pillars"][1]["age_seconds"] is None
    assert s["activity"] == [{"time": "11:59:30", "text": "order.filled mkt-1"}]


@pytest.mark.asyncio
async def test_ibkr_paper_books_summary_and_envelope_passthrough(tmp_path):
    """The venues panel needs IBKR paper books: positions + latest mark."""
    from auramaur.db.database import Database
    from auramaur.web.db import ReadOnlyDatabase
    from auramaur.web import queries

    src = tmp_path / "bot.db"
    db = Database(str(src))
    await db.connect()
    await db.execute(
        """INSERT INTO ibkr_paper_positions
           (book, instrument_key, con_id, quantity, avg_cost, currency,
            fx_to_usd, unrealized_pnl_usd)
           VALUES ('fx', 'GBPUSD', 1, 1, 1.34, 'USD', 1.0, -3.32)""")
    await db.execute(
        """INSERT INTO ibkr_paper_daily_marks
           (book, mark_date, equity_usd, realized_cum_usd, unrealized_usd)
           VALUES ('fx', date('now'), -2.81, 0.51, -3.32)""")
    await db.commit()
    await db.close()

    ro = ReadOnlyDatabase(str(src))
    await ro.connect()
    books = await queries.ibkr_paper_books(ro)
    await ro.close()
    assert books == [{"book": "fx", "positions": 1,
                      "unrealized": -3.32, "equity": -2.81}]


@pytest.mark.asyncio
async def test_ibkr_paper_books_degrades_without_schema(tmp_path):
    import sqlite3 as sq
    src = tmp_path / "bare.db"
    sq.connect(src).close()
    from auramaur.web.db import ReadOnlyDatabase
    from auramaur.web import queries
    ro = ReadOnlyDatabase(str(src))
    await ro.connect()
    assert await queries.ibkr_paper_books(ro) == []
    await ro.close()
