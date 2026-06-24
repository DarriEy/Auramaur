"""S1 write-tool tests: the agent's unconstrained paper book.

Verifies place_trade / close_position drive Auramaur's real accounting
(cost_basis + pnl_ledger + portfolio) on the isolated agent.db, so the agent's
book is directly comparable to the live bot's.
"""

import pytest

from auramaur.agentmcp.book import AgentBook
from auramaur.db.database import Database
from config.settings import Settings


@pytest.fixture(autouse=True)
def _force_paper(monkeypatch):
    # Deterministic paper mode regardless of ambient env.
    monkeypatch.setenv("AURAMAUR_LIVE", "false")


async def _book(tmp_path):
    db = Database(str(tmp_path / "agent.db"))
    await db.connect()
    settings = Settings()
    assert settings.is_live is False  # guard: never live in this process
    return db, AgentBook(db, settings)


@pytest.mark.asyncio
async def test_buy_opens_then_adds_averages_cost(tmp_path):
    db, book = await _book(tmp_path)
    try:
        r1 = await book.place_trade("m1", "YES", "BUY", 10, 0.40, category="crypto")
        assert r1["position"]["size"] == pytest.approx(10)
        assert r1["position"]["avg_price"] == pytest.approx(0.40)
        assert r1["realized_delta"] == pytest.approx(0.0)

        # Add 10 more at 0.60 -> avg 0.50 over 20 shares.
        r2 = await book.place_trade("m1", "YES", "BUY", 10, 0.60, category="crypto")
        assert r2["position"]["size"] == pytest.approx(20)
        assert r2["position"]["avg_price"] == pytest.approx(0.50)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_sell_realizes_pnl_and_reduces(tmp_path):
    db, book = await _book(tmp_path)
    try:
        await book.place_trade("m1", "YES", "BUY", 20, 0.40)
        # Sell half at 0.55 -> realized (0.55-0.40)*10 = 1.50.
        r = await book.place_trade("m1", "YES", "SELL", 10, 0.55)
        assert r["realized_delta"] == pytest.approx(1.50)
        assert r["position"]["size"] == pytest.approx(10)
        assert r["position"]["avg_price"] == pytest.approx(0.40)  # basis unchanged
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_close_position_sells_all_and_clears_row(tmp_path):
    db, book = await _book(tmp_path)
    try:
        await book.place_trade("m1", "YES", "BUY", 20, 0.40)
        out = await book.close_position("m1", "YES", 0.50)
        assert out["closed"] is True
        # (0.50-0.40)*20 = 2.00 realized; position fully gone.
        assert out["realized_delta"] == pytest.approx(2.00)
        assert out["position"]["size"] == pytest.approx(0.0)

        row = await db.fetchone(
            "SELECT count(*) c FROM portfolio WHERE market_id='m1' AND is_paper=1"
        )
        assert row["c"] == 0

        # Closing again is a clean no-op.
        again = await book.close_position("m1", "YES", 0.50)
        assert again == {"closed": False, "reason": "no_position",
                         "market_id": "m1", "token": "YES"}
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_rejects_nonpositive_size(tmp_path):
    db, book = await _book(tmp_path)
    try:
        with pytest.raises(ValueError):
            await book.place_trade("m1", "YES", "BUY", 0, 0.40)
    finally:
        await db.close()
