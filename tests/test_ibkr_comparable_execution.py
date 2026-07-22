from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from auramaur.db.database import Database
from auramaur.exchange.ibkr_instruments import GLOBAL_ETFS, IBKRBook
from auramaur.exchange.ibkr_multiasset_execution import IBKRMultiAssetExecution
from auramaur.exchange.ibkr_market_data import MarketDataQuote
from auramaur.strategy.ibkr_multiasset_paper import IBKRMultiAssetPaperBook
from auramaur.web import queries
from auramaur.web.db import ReadOnlyDatabase
from config.settings import Settings


@pytest.mark.asyncio
async def test_global_etf_enters_common_comparison_contract(tmp_path):
    path = str(tmp_path / "db.sqlite")
    db = Database(path)
    await db.connect()
    await db.execute(
        """INSERT INTO ibkr_paper_fills
           (book,instrument_key,side,quantity,price,currency,commission_usd,fill_ref)
           VALUES ('global_etf','SPY','BUY',2,500,'USD',1.25,'fill-1')""")
    await db.execute(
        """INSERT INTO ibkr_paper_positions
           (book,instrument_key,quantity,avg_cost,currency,unrealized_pnl_usd)
           VALUES ('global_etf','SPY',2,500,'USD',12.50)""")
    await db.execute(
        """INSERT INTO ibkr_paper_ledger(book,kind,pnl_usd,source_ref)
           VALUES ('global_etf','trade',8.75,'trade-1')""")
    await db.execute(
        """INSERT INTO ibkr_paper_daily_marks
           (book,mark_date,equity_usd,realized_cum_usd,unrealized_usd)
           VALUES ('global_etf','2026-07-21',5000,0,0),
                  ('global_etf','2026-07-22',5012.5,0,12.5)""")
    await db.commit()
    await db.close()

    ro = ReadOnlyDatabase(path)
    await ro.connect()
    rows = {row["strategy"]: row for row in
            await queries.strategy_breakdown(ro, 1, Settings())}
    await ro.close()
    row = rows["ibkr_global_etf"]
    assert row["entries"] == 1
    assert row["fees"] == pytest.approx(1.25)
    assert row["pnl"] == pytest.approx(8.75)
    assert row["unrealized"] == pytest.approx(12.50)
    assert row["observations"] == 1
    assert row["venues"] == ["ibkr"]
    assert row["graduation_status"] == "collecting"


@pytest.mark.asyncio
async def test_ibkr_paper_heartbeat_joins_canonical_compare_row(tmp_path):
    path = str(tmp_path / "heartbeats.sqlite")
    db = Database(path)
    await db.connect()
    from auramaur.monitoring.heartbeat import beat
    await beat(db, "ibkr_global_etf_paper", interval_seconds=900)
    await db.close()
    ro = ReadOnlyDatabase(path)
    await ro.connect()
    rows = await queries.strategy_heartbeats(ro)
    await ro.close()
    assert "ibkr_global_etf" in rows
    assert "ibkr_global_etf_paper" not in rows


def test_execution_adapter_defaults_closed():
    adapter = IBKRMultiAssetExecution(Settings(), SimpleNamespace(), None)
    assert "execution_enabled=false" in adapter.gate_reason(IBKRBook.GLOBAL_ETF)
    assert "supported" in adapter.gate_reason(IBKRBook.FX)


@pytest.mark.asyncio
async def test_execution_requires_graduation_even_when_every_live_gate_is_open():
    settings = Settings()
    settings.auramaur_live = True
    settings.execution.live = True
    settings.ibkr.environment = "live"
    settings.ibkr.readonly = False
    settings.ibkr.paper_trade = False
    settings.ibkr.multiasset_execution_enabled = True
    settings.ibkr.multiasset_execution_confirm_live = True
    settings.ibkr.multiasset_execution_books = ["global_etf"]
    adapter = IBKRMultiAssetExecution(settings, SimpleNamespace(), None)
    adapter.graduated = AsyncMock(return_value=False)
    with patch("auramaur.exchange.ibkr_multiasset_execution.kill_switch_present",
               return_value=False):
        result = await adapter.place(GLOBAL_ETFS[0], "BUY", 1)
    assert result.accepted is False
    assert "evidence contract" in result.reason
    assert adapter._ib is None


@pytest.mark.asyncio
async def test_confirmed_fill_is_durable_and_replayed_without_duplicate(tmp_path):
    path = str(tmp_path / "execution.sqlite")
    db = Database(path)
    await db.connect()
    settings = Settings()
    settings.auramaur_live = True
    settings.execution.live = True
    settings.ibkr.environment = "live"
    settings.ibkr.multiasset_execution_enabled = True
    settings.ibkr.multiasset_execution_confirm_live = True
    settings.ibkr.multiasset_execution_books = ["global_etf"]
    resolver = SimpleNamespace(resolve=AsyncMock(return_value=SimpleNamespace()))
    trade = SimpleNamespace(
        order=SimpleNamespace(orderId=91),
        orderStatus=SimpleNamespace(status="Filled", filled=1.5, avgFillPrice=501.25))
    fake_ib = SimpleNamespace(
        isConnected=lambda: True,
        placeOrder=lambda contract, order: trade,
        cancelOrder=lambda order: None)
    adapter = IBKRMultiAssetExecution(settings, resolver, db)
    adapter._ib = fake_ib
    adapter.graduated = AsyncMock(return_value=True)
    with patch("auramaur.exchange.ibkr_multiasset_execution.kill_switch_present",
               return_value=False):
        first = await adapter.place(GLOBAL_ETFS[0], "BUY", 1.5)
        replay = await adapter.place(GLOBAL_ETFS[0], "BUY", 1.5)
    assert first.accepted and first.filled_quantity == 1.5
    assert first.filled_price == pytest.approx(501.25)
    assert replay.execution_ref == first.execution_ref
    assert resolver.resolve.await_count == 1
    row = await db.fetchone("SELECT * FROM ibkr_execution_orders")
    assert row["status"] == "filled" and row["accounted"] == 0
    await adapter.acknowledge(first.execution_ref)
    row = await db.fetchone("SELECT accounted FROM ibkr_execution_orders")
    assert row["accounted"] == 1
    await db.close()


@pytest.mark.asyncio
async def test_partial_exit_preserves_remaining_shadow_position(tmp_path):
    path = str(tmp_path / "partial.sqlite")
    db = Database(path)
    await db.connect()
    await db.execute(
        """INSERT INTO ibkr_paper_positions
           (book,instrument_key,quantity,avg_cost,currency,entry_commission_usd,
            entry_fill_ref)
           VALUES ('global_etf','SPY',2,500,'USD',2,'entry')""")
    await db.commit()
    executor = SimpleNamespace(
        gate_reason=lambda book: "",
        graduated=AsyncMock(return_value=True),
        place=AsyncMock(return_value=SimpleNamespace(
            accepted=True, filled_quantity=0.75, filled_price=510,
            execution_ref="live-partial", reason="")),
        acknowledge=AsyncMock())
    book = IBKRMultiAssetPaperBook(
        Settings(), SimpleNamespace(), db, IBKRBook.GLOBAL_ETF,
        executor=executor)
    quote = MarketDataQuote("SPY", 509, 510, 0, 1, "USD", 1)
    await book._fill(GLOBAL_ETFS[0], quote, "SELL", 2, 1.0, entry_price=500)
    position = await db.fetchone(
        "SELECT quantity,entry_commission_usd FROM ibkr_paper_positions")
    assert position["quantity"] == pytest.approx(1.25)
    assert position["entry_commission_usd"] == pytest.approx(1.25)
    fill = await db.fetchone("SELECT quantity,price FROM ibkr_paper_fills")
    assert fill["quantity"] == pytest.approx(0.75)
    assert fill["price"] == pytest.approx(510)
    executor.acknowledge.assert_awaited_once_with("live-partial")
    await db.close()
