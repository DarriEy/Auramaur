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

    @pytest.mark.asyncio
    async def test_sync_positions_preserves_paper_kalshi_rows(self):
        """The live sync must NOT touch is_paper=1 Kalshi rows. A purge lived
        here while nothing traded Kalshi on paper; once the Kalshi paper
        strategies shipped (informed_flow, the Kalshi lens, econ_indicator) it
        silently shredded their books on every sync — fills accrued but
        positions and cost basis vanished, so nothing ever settled into the
        ledger and those cells' records were structurally impossible."""
        db = Database(":memory:")
        await db.connect()
        try:
            # Seed a PAPER Kalshi position + cost basis (a live pillar's book).
            await db.execute(
                """INSERT INTO portfolio
                   (market_id, exchange, side, size, avg_price, current_price,
                    unrealized_pnl, category, token, token_id, is_paper, updated_at)
                   VALUES ('KXPAPER', 'kalshi', 'BUY', 27, 0.37, 0.37,
                           0.0, 'economics', 'YES', 'KXPAPER', 1, datetime('now'))"""
            )
            await db.execute(
                """INSERT INTO markets (id, exchange, question, category, active,
                   outcome_yes_price, outcome_no_price, last_updated)
                   VALUES ('KXPAPER', 'kalshi', 'Paper?', 'economics', 1, 0.37, 0.63, datetime('now'))"""
            )
            await db.execute(
                """INSERT INTO cost_basis
                   (market_id, token, token_id, size, avg_cost, total_cost, is_paper, updated_at)
                   VALUES ('KXPAPER', 'YES', 'KXPAPER', 27, 0.37, 9.99, 1, datetime('now'))"""
            )
            await db.commit()

            client = KalshiClient.__new__(KalshiClient)
            client._init_api = MagicMock()
            client._portfolio_api = MagicMock()
            client._portfolio_api.get_positions_without_preload_content = MagicMock()
            # Live API returns one (different) live position.
            client._call_raw = AsyncMock(return_value=json.dumps({
                "market_positions": [{
                    "ticker": "KXLIVE", "position_fp": 10,
                    "market_exposure_dollars": 4.2,
                }]
            }))
            client.get_market = AsyncMock(return_value=Market(
                id="KXLIVE", exchange="kalshi", question="Live?",
                category="test", outcome_yes_price=0.45, outcome_no_price=0.55,
            ))

            await client.sync_positions(db)

            # Paper book untouched by the LIVE sync.
            assert await db.fetchone(
                "SELECT 1 FROM portfolio WHERE market_id='KXPAPER' AND is_paper=1") is not None
            assert await db.fetchone(
                "SELECT 1 FROM cost_basis WHERE market_id='KXPAPER' AND is_paper=1") is not None
            # Live position still synced.
            assert await db.fetchone(
                "SELECT 1 FROM portfolio WHERE market_id='KXLIVE' AND is_paper=0") is not None
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


class TestKalshiV2CreateOrder:
    """The v2 single-book create-order endpoint (/portfolio/events/orders)."""

    def _client(self, create_response: dict | None, *, raise_exc=None):
        """A KalshiClient wired so the live path hits a mocked call_api."""
        from unittest.mock import AsyncMock, MagicMock
        import asyncio as _asyncio

        client = KalshiClient.__new__(KalshiClient)
        client._init_api = MagicMock()
        client._portfolio_api = MagicMock()
        client._settings = MagicMock()
        client._settings.is_live = True
        client._live_pending = {}
        client._api_base = "https://api.elections.kalshi.com/trade-api/v2"
        client._semaphore = _asyncio.Semaphore(1)
        # dup-check (get_orders) → no resting orders.
        client._call_raw = AsyncMock(return_value=json.dumps({"orders": []}))
        # The v2 create goes through self._client.call_api(...).read().
        sdk = MagicMock()
        resp = MagicMock()
        if raise_exc is not None:
            sdk.call_api = MagicMock(side_effect=raise_exc)
        else:
            resp.read = MagicMock(return_value=json.dumps(create_response).encode())
            sdk.call_api = MagicMock(return_value=resp)
        client._client = sdk
        return client

    def _body_sent(self, client):
        return client._client.call_api.call_args.kwargs["body"]

    @pytest.mark.asyncio
    async def test_yes_sell_maps_to_ask_at_price(self):
        """Exit of a YES position (sell YES) → side 'ask' at the YES price."""
        client = self._client({"order_id": "ord-1", "remaining_count": "16.00"})
        order = Order(market_id="KXT", exchange="kalshi", side=OrderSide.SELL,
                      token=TokenType.YES, token_id="KXT", size=16, price=0.04,
                      dry_run=False)
        result = await client.place_order(order)
        assert result.status == "pending" and result.order_id == "ord-1"
        body = self._body_sent(client)
        assert body["side"] == "ask"
        assert body["price"] == "0.0400"
        assert body["count"] == "16.00"
        assert body["time_in_force"] == "good_till_canceled"

    @pytest.mark.asyncio
    async def test_no_sell_maps_to_bid_at_one_minus_price(self):
        """Exit of a NO position (sell NO at 0.97) is economically buy YES at
        0.03 → side 'bid' at (1 - price)."""
        client = self._client({"order_id": "ord-2", "remaining_count": "24.00"})
        order = Order(market_id="KXT", exchange="kalshi", side=OrderSide.SELL,
                      token=TokenType.NO, token_id="KXT", size=24, price=0.97,
                      dry_run=False)
        await client.place_order(order)
        body = self._body_sent(client)
        assert body["side"] == "bid"
        assert body["price"] == "0.0300"

    @pytest.mark.asyncio
    async def test_yes_buy_maps_to_bid(self):
        client = self._client({"order_id": "ord-3"})
        order = Order(market_id="KXT", exchange="kalshi", side=OrderSide.BUY,
                      token=TokenType.YES, token_id="KXT", size=10, price=0.40,
                      dry_run=False)
        await client.place_order(order)
        body = self._body_sent(client)
        assert body["side"] == "bid" and body["price"] == "0.4000"

    @pytest.mark.asyncio
    async def test_no_buy_maps_to_ask_at_one_minus_price(self):
        client = self._client({"order_id": "ord-4"})
        order = Order(market_id="KXT", exchange="kalshi", side=OrderSide.BUY,
                      token=TokenType.NO, token_id="KXT", size=10, price=0.30,
                      dry_run=False)
        await client.place_order(order)
        body = self._body_sent(client)
        assert body["side"] == "ask" and body["price"] == "0.7000"

    @pytest.mark.asyncio
    async def test_request_carries_uuid_client_order_id_and_v2_fields(self):
        import uuid as _uuid
        client = self._client({"order_id": "ord-5"})
        order = Order(market_id="KXT", exchange="kalshi", side=OrderSide.SELL,
                      token=TokenType.YES, token_id="KXT", size=16, price=0.04,
                      dry_run=False)
        await client.place_order(order)
        body = self._body_sent(client)
        _uuid.UUID(body["client_order_id"])  # raises if not a valid UUID
        assert body["self_trade_prevention_type"] == "taker_at_cross"
        # Posted to the v2 events/orders path.
        assert client._client.call_api.call_args.args[1].endswith(
            "/portfolio/events/orders")

    @pytest.mark.asyncio
    async def test_subpenny_and_fractional_values_survive_submission(self):
        """The v2 fixed-point migration must survive the final HTTP boundary."""
        client = self._client({"order_id": "ord-fp"})
        order = Order(market_id="KXFP", exchange="kalshi", side=OrderSide.BUY,
                      token=TokenType.YES, token_id="KXFP", size=0.25,
                      price=0.004, dry_run=False)
        result = await client.place_order(order)
        assert result.status == "pending"
        body = self._body_sent(client)
        assert body["price"] == "0.0040"
        assert body["count"] == "0.25"

    @pytest.mark.asyncio
    async def test_response_without_order_id_rejects(self):
        """A body lacking order_id (soft rejection) → rejected, no ghost order."""
        client = self._client({"error": "too_few_contracts"})
        order = Order(market_id="KXT", exchange="kalshi", side=OrderSide.SELL,
                      token=TokenType.YES, token_id="KXT", size=16, price=0.04,
                      dry_run=False)
        result = await client.place_order(order)
        assert result.status == "rejected"
        assert result.order_id == "KALSHI_NO_ORDER"
        assert "too_few_contracts" in result.error_message
        assert client._live_pending == {}

    @pytest.mark.asyncio
    async def test_dup_check_skips_on_matching_book_side(self):
        """A resting order on the same v2 book side (YES SELL → ask) blocks the
        new order. The legacy yes/no compare never matched v2 resting orders."""
        from unittest.mock import AsyncMock
        client = self._client({"order_id": "ord"})
        client._call_raw = AsyncMock(return_value=json.dumps({"orders": [
            {"status": "resting", "ticker": "KXT", "side": "yes",
             "book_side": "ask"}]}))
        order = Order(market_id="KXT", exchange="kalshi", side=OrderSide.SELL,
                      token=TokenType.YES, token_id="KXT", size=16, price=0.04,
                      dry_run=False)
        result = await client.place_order(order)
        assert result.order_id == "SKIP_DUP" and result.status == "rejected"
        client._client.call_api.assert_not_called()  # never reached placement

    @pytest.mark.asyncio
    async def test_dup_check_allows_different_book_side(self):
        """A resting order on the OTHER book side does not block (bid vs ask)."""
        from unittest.mock import AsyncMock
        client = self._client({"order_id": "ord"})
        client._call_raw = AsyncMock(return_value=json.dumps({"orders": [
            {"status": "resting", "ticker": "KXT", "side": "yes",
             "book_side": "bid"}]}))
        order = Order(market_id="KXT", exchange="kalshi", side=OrderSide.SELL,
                      token=TokenType.YES, token_id="KXT", size=16, price=0.04,
                      dry_run=False)
        result = await client.place_order(order)
        assert result.status == "pending"  # ask != bid → proceeded


class TestKalshiV2BookSideMapping:
    def test_all_four_cases(self):
        from auramaur.exchange.models import TokenType as T, OrderSide as S
        f = KalshiClient._v2_book_side
        assert f(T.YES, S.BUY) == "bid"
        assert f(T.YES, S.SELL) == "ask"
        assert f(T.NO, S.BUY) == "ask"   # buy NO == sell YES
        assert f(T.NO, S.SELL) == "bid"  # sell NO == buy YES


class TestKalshiCancelV2:
    def _client(self):
        from unittest.mock import MagicMock
        import asyncio as _asyncio
        c = KalshiClient.__new__(KalshiClient)
        c._init_api = MagicMock()
        c._api_base = "https://api.elections.kalshi.com/trade-api/v2"
        c._semaphore = _asyncio.Semaphore(1)
        c._client = MagicMock()
        c._portfolio_api = MagicMock()
        return c

    @pytest.mark.asyncio
    async def test_cancel_uses_v2_delete_endpoint(self):
        c = self._client()
        ok = await c.cancel_order("oid-1")
        assert ok is True
        method, url = c._client.call_api.call_args.args[0], c._client.call_api.call_args.args[1]
        assert method == "DELETE"
        assert url.endswith("/portfolio/events/orders/oid-1")
        c._portfolio_api.cancel_order.assert_not_called()  # legacy not needed

    @pytest.mark.asyncio
    async def test_cancel_falls_back_to_legacy_on_v2_error(self):
        from unittest.mock import MagicMock
        c = self._client()
        c._client.call_api = MagicMock(side_effect=Exception("v2 boom"))
        ok = await c.cancel_order("oid-2")
        assert ok is True
        c._portfolio_api.cancel_order.assert_called_once()  # legacy fallback used


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
