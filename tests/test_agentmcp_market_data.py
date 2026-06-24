"""S1 read-tool tests: read-only market data over a bot-shaped database."""

import aiosqlite
import pytest

from auramaur.agentmcp.market_data import MarketData


async def _seed_db(path: str) -> None:
    """Build a minimal bot-shaped DB: markets + orderbook + lens_verdicts."""
    conn = await aiosqlite.connect(path)
    await conn.executescript(
        """
        CREATE TABLE markets (
            id TEXT PRIMARY KEY, exchange TEXT, condition_id TEXT, ticker TEXT,
            question TEXT, description TEXT, category TEXT, end_date TEXT,
            active INTEGER, outcome_yes_price REAL, outcome_no_price REAL,
            volume REAL, liquidity REAL, last_updated TEXT,
            clob_token_yes TEXT, clob_token_no TEXT
        );
        CREATE TABLE orderbook_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT, market_id TEXT, token_id TEXT,
            exchange TEXT, best_bid REAL, best_ask REAL, bid_size REAL,
            ask_size REAL, mid REAL, recorded_at TEXT
        );
        CREATE TABLE lens_verdicts (
            market_id TEXT PRIMARY KEY, fair_prob REAL, gap_score REAL,
            mechanism TEXT, reasoning TEXT, checked_at TEXT, verified INTEGER,
            grounded_fair REAL
        );
        CREATE TABLE news_items (
            id TEXT PRIMARY KEY, source TEXT, title TEXT, content TEXT, url TEXT,
            published_at TEXT, relevance_score REAL, market_ids TEXT
        );
        """
    )
    await conn.executemany(
        "INSERT INTO markets (id, exchange, question, category, active, "
        "outcome_yes_price, outcome_no_price, volume, liquidity, last_updated, "
        "clob_token_yes, clob_token_no) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            ("big", "polymarket", "Will BTC hit 200k?", "crypto", 1, 0.30, 0.70,
             9000, 5000, "now", "tokY", "tokN"),
            ("small", "polymarket", "Will X ship?", "tech", 1, 0.50, 0.50,
             100, 80, "now", "", ""),
            ("dead", "polymarket", "Resolved thing", "crypto", 0, 0.99, 0.01,
             9999, 9999, "now", "", ""),
        ],
    )
    await conn.execute(
        "INSERT INTO orderbook_snapshots (market_id, token_id, exchange, "
        "best_bid, best_ask, bid_size, ask_size, mid, recorded_at) "
        "VALUES ('big','tokY','polymarket',0.29,0.31,500,400,0.30,'2026-06-24T00:00:00')",
    )
    await conn.execute(
        "INSERT INTO lens_verdicts (market_id, fair_prob, gap_score, mechanism, "
        "reasoning, verified) VALUES ('big',0.22,8.0,'fine-print','criteria say X',1)",
    )
    await conn.commit()
    await conn.close()


@pytest.mark.asyncio
async def test_scan_filters_active_and_orders_by_volume(tmp_path):
    path = str(tmp_path / "auramaur.db")
    await _seed_db(path)
    md = MarketData(path)

    rows = await md.scan_markets()
    ids = [r["id"] for r in rows]
    assert "dead" not in ids          # active=0 excluded
    assert ids == ["big", "small"]    # volume desc

    crypto = await md.scan_markets(category="crypto")
    assert [r["id"] for r in crypto] == ["big"]

    liquid = await md.scan_markets(min_liquidity=1000)
    assert [r["id"] for r in liquid] == ["big"]


@pytest.mark.asyncio
async def test_get_quote_returns_prices_and_depth(tmp_path):
    path = str(tmp_path / "auramaur.db")
    await _seed_db(path)
    md = MarketData(path)

    q = await md.get_quote("big")
    assert q["found"] is True
    assert q["outcome_yes_price"] == pytest.approx(0.30)
    assert len(q["order_book"]) == 1
    assert q["order_book"][0]["best_ask"] == pytest.approx(0.31)

    assert (await md.get_quote("nope"))["found"] is False


@pytest.mark.asyncio
async def test_get_evidence_returns_lens_verdict(tmp_path):
    path = str(tmp_path / "auramaur.db")
    await _seed_db(path)
    md = MarketData(path)

    ev = await md.get_evidence("big")
    assert ev["lens_verdict"]["fair_prob"] == pytest.approx(0.22)
    assert ev["lens_verdict"]["mechanism"] == "fine-print"
    assert ev["market"]["category"] == "crypto"
    assert ev["news"] == []


@pytest.mark.asyncio
async def test_connection_is_read_only(tmp_path):
    """mode=ro must reject writes at the kernel level — the agent can never
    mutate the bot's database, by construction not by convention."""
    path = str(tmp_path / "auramaur.db")
    await _seed_db(path)
    md = MarketData(path)

    conn = await md._connect()
    try:
        with pytest.raises(aiosqlite.OperationalError):
            await conn.execute("INSERT INTO markets (id, question, active) "
                               "VALUES ('x','y',1)")
    finally:
        await conn.close()
