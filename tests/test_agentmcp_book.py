"""Agent paper-book tests: unconstrained SIZING, verified PRICES (S1 + S4).

Verifies place_trade / close_position drive Auramaur's real accounting
(cost_basis + pnl_ledger + portfolio) on the isolated agent.db, AND that every
fill executes at the bot DB's quote — never at a caller-supplied price. The S4
guard exists because of the 2026-07 incident: an agent driving this book
through direct scripts settled still-active markets at fantasy prices and
fabricated its scorecard. The guard lives at the book layer precisely so
direct imports can't bypass it.
"""

import pytest

from auramaur.agentmcp.book import AgentBook, UnverifiablePrice
from auramaur.db.database import Database
from config.settings import Settings


@pytest.fixture(autouse=True)
def _force_paper(monkeypatch):
    # Deterministic paper mode regardless of ambient env.
    monkeypatch.setenv("AURAMAUR_LIVE", "false")


class StubQuotes:
    """In-memory stand-in for MarketData: a dict of market_id -> (yes, no)."""

    def __init__(self, quotes: dict[str, tuple[float, float]]) -> None:
        self.quotes = quotes

    async def get_quote(self, market_id: str) -> dict:
        if market_id not in self.quotes:
            return {"market_id": market_id, "found": False}
        yes, no = self.quotes[market_id]
        return {
            "market_id": market_id,
            "found": True,
            "outcome_yes_price": yes,
            "outcome_no_price": no,
        }


async def _book(tmp_path, quotes: dict[str, tuple[float, float]] | None = None):
    db = Database(str(tmp_path / "agent.db"))
    await db.connect()
    settings = Settings()
    assert settings.is_live is False  # guard: never live in this process
    book = AgentBook(db, settings, market_data=StubQuotes(quotes or {}))
    return db, book


@pytest.mark.asyncio
async def test_buy_opens_then_adds_averages_cost(tmp_path):
    quotes = {"m1": (0.40, 0.60)}
    db, book = await _book(tmp_path, quotes)
    try:
        r1 = await book.place_trade("m1", "YES", "BUY", 10, 0.40, category="crypto")
        assert r1["position"]["size"] == pytest.approx(10)
        assert r1["position"]["avg_price"] == pytest.approx(0.40)
        assert r1["realized_delta"] == pytest.approx(0.0)

        # Quote moves to 0.60; a second buy averages 0.50 over 20 shares.
        quotes["m1"] = (0.60, 0.40)
        r2 = await book.place_trade("m1", "YES", "BUY", 10, 0.60, category="crypto")
        assert r2["position"]["size"] == pytest.approx(20)
        assert r2["position"]["avg_price"] == pytest.approx(0.50)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_sell_realizes_pnl_and_reduces(tmp_path):
    quotes = {"m1": (0.40, 0.60)}
    db, book = await _book(tmp_path, quotes)
    try:
        await book.place_trade("m1", "YES", "BUY", 20, 0.40)
        # Quote moves to 0.55; selling half realizes (0.55-0.40)*10 = 1.50.
        quotes["m1"] = (0.55, 0.45)
        r = await book.place_trade("m1", "YES", "SELL", 10, 0.55)
        assert r["realized_delta"] == pytest.approx(1.50)
        assert r["position"]["size"] == pytest.approx(10)
        assert r["position"]["avg_price"] == pytest.approx(0.40)  # basis unchanged
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_close_position_sells_all_and_clears_row(tmp_path):
    quotes = {"m1": (0.40, 0.60)}
    db, book = await _book(tmp_path, quotes)
    try:
        await book.place_trade("m1", "YES", "BUY", 20, 0.40)
        quotes["m1"] = (0.50, 0.50)
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
    db, book = await _book(tmp_path, {"m1": (0.40, 0.60)})
    try:
        with pytest.raises(ValueError):
            await book.place_trade("m1", "YES", "BUY", 0, 0.40)
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# S4 price-integrity guard — the fabrication class from the 2026-07 incident
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fill_executes_at_quote_not_requested_price(tmp_path):
    """The incident shape: the caller asks to SELL at 1.00 while the market
    trades at 0.125. The fill must execute at the quote — the requested price
    is echoed back but never books P&L."""
    quotes = {"m1": (0.125, 0.875)}
    db, book = await _book(tmp_path, quotes)
    try:
        await book.place_trade("m1", "YES", "BUY", 100, 0.125)
        r = await book.place_trade("m1", "YES", "SELL", 100, 1.00)
        assert r["filled"]["price"] == pytest.approx(0.125)
        assert r["filled"]["requested_price"] == pytest.approx(1.00)
        assert r["realized_delta"] == pytest.approx(0.0)  # no fantasy profit
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_unknown_market_is_rejected(tmp_path):
    """A market the bot DB doesn't know has no verifiable price — the A/B
    premise is 'same universe as the bot', so the trade is refused."""
    db, book = await _book(tmp_path, {})
    try:
        with pytest.raises(UnverifiablePrice):
            await book.place_trade("ghost", "YES", "BUY", 10, 0.50)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_sell_clamped_to_held_size(tmp_path):
    """Overselling (the double-settlement shape) is clamped to the held size;
    a sell with no position at all is rejected outright."""
    quotes = {"m1": (0.40, 0.60)}
    db, book = await _book(tmp_path, quotes)
    try:
        await book.place_trade("m1", "YES", "BUY", 10, 0.40)
        r = await book.place_trade("m1", "YES", "SELL", 9999, 0.40)
        assert r["filled"]["size"] == pytest.approx(10)
        assert r["position"]["size"] == pytest.approx(0.0)

        again = await book.place_trade("m1", "YES", "SELL", 10, 0.40)
        assert again.get("rejected") == "sell_without_position"
        assert again["filled"] is None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_no_token_price_derived_from_yes_when_no_quote_stale(tmp_path):
    """A zero/stale NO quote is derived from the YES side (1 - yes), the same
    fallback the exchange client uses."""
    db, book = await _book(tmp_path, {"m1": (0.80, 0.0)})
    try:
        r = await book.place_trade("m1", "NO", "BUY", 10, 0.99)
        assert r["filled"]["price"] == pytest.approx(0.20)
    finally:
        await db.close()
