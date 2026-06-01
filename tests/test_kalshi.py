"""Tests for Kalshi exchange client."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from auramaur.db.database import Database
from auramaur.broker.sync import KalshiPositionSyncer
from auramaur.exchange.kalshi import KalshiClient
from auramaur.exchange.models import Market, Order, OrderSide, Position, TokenType


class TestKalshiPositionSyncerBalance:

    def _settings(self, is_live: bool):
        s = MagicMock()
        s.is_live = is_live
        return s

    @pytest.mark.asyncio
    async def test_paper_mode_returns_paper_balance(self):
        paper = MagicMock()
        paper.balance = 111.0
        syncer = KalshiPositionSyncer(
            settings=self._settings(is_live=False),
            db=MagicMock(),
            exchange=MagicMock(),
            paper=paper,
        )
        assert await syncer.get_cash_balance() == 111.0

    @pytest.mark.asyncio
    async def test_live_mode_queries_exchange(self):
        exchange = MagicMock()
        exchange.get_balance = AsyncMock(return_value=500.0)
        syncer = KalshiPositionSyncer(
            settings=self._settings(is_live=True),
            db=MagicMock(),
            exchange=exchange,
            paper=MagicMock(),
        )
        assert await syncer.get_cash_balance() == 500.0
        exchange.get_balance.assert_called_once()

    @pytest.mark.asyncio
    async def test_paper_mode_without_paper_falls_back_to_exchange(self):
        exchange = MagicMock()
        exchange.get_balance = AsyncMock(return_value=42.0)
        syncer = KalshiPositionSyncer(
            settings=self._settings(is_live=False),
            db=MagicMock(),
            exchange=exchange,
            paper=None,
        )
        assert await syncer.get_cash_balance() == 42.0


class TestKalshiPositionSyncerPaperSync:

    def _settings(self, is_live: bool):
        s = MagicMock()
        s.is_live = is_live
        return s

    def _db(self):
        db = MagicMock()
        db.execute = AsyncMock()
        db.commit = AsyncMock()
        db.fetchall = AsyncMock(return_value=[])
        return db

    @pytest.mark.asyncio
    async def test_paper_mode_returns_paper_positions(self):
        pos = Position(
            market_id="KXTEST",
            side=OrderSide.BUY,
            size=10.0,
            avg_price=0.5,
            current_price=0.6,
            category="test",
            token=TokenType.YES,
            token_id="KXTEST",
        )
        paper = MagicMock()
        paper.positions = {"KXTEST": pos}

        syncer = KalshiPositionSyncer(
            settings=self._settings(is_live=False),
            db=self._db(),
            exchange=MagicMock(),
            paper=paper,
        )

        positions = await syncer.sync()
        assert len(positions) == 1
        assert positions[0].market_id == "KXTEST"
        assert positions[0].size == 10.0

    @pytest.mark.asyncio
    async def test_paper_sync_empty_clears_portfolio(self):
        paper = MagicMock()
        paper.positions = {}

        syncer = KalshiPositionSyncer(
            settings=self._settings(is_live=False),
            db=self._db(),
            exchange=MagicMock(),
            paper=paper,
        )

        positions = await syncer.sync()
        assert positions == []
        delete_calls = [c for c in syncer._db.execute.call_args_list if "DELETE" in str(c)]
        assert delete_calls


class TestKalshiLivePositionAccounting:
    @pytest.mark.asyncio
    async def test_sync_positions_writes_portfolio_and_cost_basis(self):
        db = Database(":memory:")
        await db.connect()
        try:
            client = KalshiClient.__new__(KalshiClient)
            client._init_api = MagicMock()
            client._portfolio_api = MagicMock()
            client._portfolio_api.get_positions_without_preload_content = MagicMock()
            client._call_raw = AsyncMock(return_value=json.dumps({
                "market_positions": [{
                    "ticker": "KXTEST",
                    "position_fp": 10,
                    "market_exposure_dollars": 4.2,
                }]
            }))
            client.get_market = AsyncMock(return_value=Market(
                id="KXTEST",
                exchange="kalshi",
                question="Will test happen?",
                category="test",
                outcome_yes_price=0.45,
                outcome_no_price=0.55,
            ))

            count = await client.sync_positions(db)

            portfolio = await db.fetchone(
                "SELECT exchange, size, avg_price, token, is_paper FROM portfolio WHERE market_id = ?",
                ("KXTEST",),
            )
            cost_basis = await db.fetchone(
                "SELECT size, avg_cost, total_cost, token, is_paper FROM cost_basis WHERE market_id = ?",
                ("KXTEST",),
            )
            market_row = await db.fetchone(
                "SELECT exchange, condition_id FROM markets WHERE id = ?",
                ("KXTEST",),
            )

            assert count == 1
            assert market_row["exchange"] == "kalshi"
            assert market_row["condition_id"] == "KXTEST"
            assert portfolio["exchange"] == "kalshi"
            assert portfolio["size"] == pytest.approx(10)
            assert portfolio["avg_price"] == pytest.approx(0.42)
            assert portfolio["token"] == "YES"
            assert portfolio["is_paper"] == 0
            assert cost_basis["size"] == pytest.approx(10)
            assert cost_basis["avg_cost"] == pytest.approx(0.42)
            assert cost_basis["total_cost"] == pytest.approx(4.2)
            assert cost_basis["is_paper"] == 0
        finally:
            await db.close()


class TestKalshiMarketParsing:
    def _make_client(self):
        client = KalshiClient.__new__(KalshiClient)
        return client

    def test_parse_market_basic(self):
        client = self._make_client()
        data = {
            "ticker": "KXFUT24-LSV",
            "title": "Will event happen?",
            "subtitle": "Some description",
            "category": "politics",
            "yes_bid": 65,
            "yes_ask": 68,
            "volume": 5000,
            "status": "open",
        }
        market = client._parse_market(data)
        assert market is not None
        assert market.exchange == "kalshi"
        assert market.ticker == "KXFUT24-LSV"
        assert market.id == "KXFUT24-LSV"
        assert market.question == "Will event happen?"
        # Midpoint of bid 0.65 and ask 0.68
        assert market.outcome_yes_price == pytest.approx(0.665, abs=0.01)
        assert market.active is True

    def test_parse_market_closed(self):
        client = self._make_client()
        data = {
            "ticker": "KXTEST",
            "title": "Test?",
            "status": "closed",
            "yes_bid": 50,
        }
        market = client._parse_market(data)
        assert market is not None
        assert market.active is False

    def test_parse_market_spread(self):
        client = self._make_client()
        data = {
            "ticker": "KXTEST",
            "title": "Test?",
            "yes_bid": 40,
            "yes_ask": 45,
            "status": "open",
        }
        market = client._parse_market(data)
        assert market is not None
        assert market.spread == pytest.approx(0.05, abs=0.01)


class TestKalshiPaperGate:
    def _make_client(self):
        client = KalshiClient.__new__(KalshiClient)
        return client

    @pytest.mark.asyncio
    async def test_paper_gate_routes_to_paper(self):
        """When dry_run=True, order should go through PaperTrader."""
        from unittest.mock import AsyncMock, MagicMock

        paper = MagicMock()
        paper.execute = AsyncMock(return_value=MagicMock(
            order_id="PAPER-123",
            market_id="KXTEST",
            status="paper",
            is_paper=True,
        ))

        client = self._make_client()
        client._paper = paper
        client._settings = MagicMock()
        client._settings.is_live = False

        order = Order(
            market_id="KXTEST",
            exchange="kalshi",
            side=OrderSide.BUY,
            token=TokenType.YES,
            size=10,
            price=0.50,
            dry_run=True,
        )
        result = await client.place_order(order)
        assert result.is_paper is True
        paper.execute.assert_called_once()


class TestKalshiPrepareOrderDirectSell:
    def test_sell_signal_becomes_buy_no(self):
        """Kalshi SELL signal should become BUY NO (can't sell what you don't own)."""
        client = KalshiClient.__new__(KalshiClient)
        from auramaur.exchange.models import Confidence, Signal
        signal = Signal(
            market_id="KXTEST",
            claude_prob=0.3,
            claude_confidence=Confidence.HIGH,
            market_prob=0.5,
            edge=20.0,
            recommended_side=OrderSide.SELL,
        )
        market = Market(
            id="KXTEST",
            exchange="kalshi",
            ticker="KXTEST",
            question="Test?",
            outcome_yes_price=0.50,
            outcome_no_price=0.50,
        )
        order = client.prepare_order(signal, market, 25.0, False)
        assert order is not None
        assert order.side == OrderSide.BUY
        assert order.token == TokenType.NO
        assert order.exchange == "kalshi"
