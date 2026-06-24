"""Tests for cross-exchange arbitrage and Metaculus data source."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from auramaur.components import Components
import pytest

from auramaur.bot import AuramaurBot
from auramaur.exchange.models import Market, Order, OrderResult, OrderSide, TokenType
from auramaur.strategy.arbitrage_scanner import ArbOpportunity, ArbitrageScanner


# ---------------------------------------------------------------------------
# Fee-aware profit calculation
# ---------------------------------------------------------------------------


def _make_market(exchange: str, yes_price: float, question: str = "Will X happen?",
                 category: str = "other") -> Market:
    return Market(
        id=f"{exchange}_123",
        exchange=exchange,
        question=question,
        category=category,
        outcome_yes_price=yes_price,
        outcome_no_price=round(1.0 - yes_price, 4),
        volume=10000,
        liquidity=5000,
    )


class TestFeeAwareProfit:
    """Tests that the ArbitrageScanner correctly accounts for exchange fees."""

    def _make_scanner(
        self,
        markets_a: list[Market],
        markets_b: list[Market],
        fees: dict[str, float] | None = None,
        min_profit: float = 1.5,
    ) -> ArbitrageScanner:
        disc_a = AsyncMock()
        disc_a.get_markets = AsyncMock(return_value=markets_a)
        disc_b = AsyncMock()
        disc_b.get_markets = AsyncMock(return_value=markets_b)
        return ArbitrageScanner(
            discoveries={"polymarket": disc_a, "kalshi": disc_b},
            exchange_fees=fees or {"polymarket": 0.0, "kalshi": 0.07},
            min_profit_after_fees_pct=min_profit,
        )

    @pytest.mark.asyncio
    async def test_cross_exchange_arb_detected_with_fees(self):
        """10% spread, both legs taker. Poly cheap@0.40 (cat 'other' -> 0.05):
        0.05*0.4*0.6=0.012. Kalshi exp@0.50: 0.07*0.25=0.0175. Net 0.10 - 0.012
        - 0.0175 = 0.0705 = 7.05%."""
        poly = _make_market("polymarket", 0.40, "Will X happen?")
        kalshi = _make_market("kalshi", 0.50, "Will X happen?")
        scanner = self._make_scanner([poly], [kalshi])

        opps = await scanner.scan()
        cross = [o for o in opps if o.arb_type == "cross_exchange"]
        assert len(cross) == 1

        opp = cross[0]
        assert opp.spread == pytest.approx(0.10, abs=0.001)
        assert opp.expected_profit_pct == pytest.approx(7.05, abs=0.1)

    @pytest.mark.asyncio
    async def test_moderate_spread_passes_min_profit(self):
        """A 10% spread nets ~7% after both legs' taker fees — clears 1.5% min."""
        poly = _make_market("polymarket", 0.50, "Will Y happen?")
        kalshi = _make_market("kalshi", 0.60, "Will Y happen?")
        scanner = self._make_scanner([poly], [kalshi])

        opps = await scanner.scan()
        cross = [o for o in opps if o.arb_type == "cross_exchange"]
        assert len(cross) == 1

    @pytest.mark.asyncio
    async def test_tiny_spread_below_min_profit(self):
        """A 3.1% spread with 7% fee = ~2.88%. With min_profit=3.0, should be filtered."""
        poly = _make_market("polymarket", 0.50, "Will Z happen?")
        kalshi = _make_market("kalshi", 0.531, "Will Z happen?")
        scanner = self._make_scanner([poly], [kalshi], min_profit=3.0)

        opps = await scanner.scan()
        cross = [o for o in opps if o.arb_type == "cross_exchange"]
        assert len(cross) == 0

    @pytest.mark.asyncio
    async def test_cross_exchange_executor_uses_full_package_edge(self):
        """Execution risk checks should see scanner net edge, not half of it."""
        poly = _make_market("polymarket", 0.40, "Will X happen?")
        kalshi = _make_market("kalshi", 0.44, "Will X happen?")
        opp = ArbOpportunity(
            market_a=poly,
            market_b=kalshi,
            exchange_a="polymarket",
            exchange_b="kalshi",
            price_a=0.40,
            price_b=0.44,
            spread=0.04,
            expected_profit_pct=3.72,
            question="Will X happen?",
        )

        class FakeRisk:
            def __init__(self):
                self.calls = []

            async def evaluate(self, signal, market, price_history=None, available_cash=None):
                self.calls.append((signal.edge, market.exchange, available_cash))
                return SimpleNamespace(approved=True, position_size=12.0, reason="ok", checks=[])

        class FakeExchange:
            def __init__(self, name):
                self.name = name
                self.orders = []

            def prepare_order(self, signal, market, position_size, is_live):
                token = TokenType.NO if signal.recommended_side == OrderSide.SELL else TokenType.YES
                price = market.outcome_no_price if token == TokenType.NO else market.outcome_yes_price
                order = Order(
                    market_id=market.id,
                    exchange=self.name,
                    token_id=f"{market.id}-{token.value}",
                    token=token,
                    side=OrderSide.BUY,
                    size=position_size,
                    price=price,
                    dry_run=not is_live,
                )
                self.orders.append(order)
                return order

            async def place_order(self, order):
                return OrderResult(
                    order_id=f"{self.name}-1",
                    market_id=order.market_id,
                    status="pending",
                    is_paper=order.dry_run,
                )

        class FakeEngine:
            def __init__(self, name, cash):
                self.exchange = FakeExchange(name)
                self._cash = cash

            async def _get_available_cash(self):
                return self._cash

        settings = MagicMock()
        settings.is_live = True
        settings.arbitrage.max_arb_size = 25.0
        bot = AuramaurBot(settings=settings)
        bot._components = Components({"alerts": SimpleNamespace(send=AsyncMock())})

        risk = FakeRisk()
        engines = {
            "polymarket": FakeEngine("polymarket", 101.0),
            "kalshi": FakeEngine("kalshi", 202.0),
        }

        await bot._execute_cross_exchange_arb(opp, risk, engines)

        assert [call[0] for call in risk.calls] == [pytest.approx(3.72), pytest.approx(3.72)]
        assert [call[2] for call in risk.calls] == [101.0, 202.0]
        assert len(engines["polymarket"].exchange.orders) == 1
        assert len(engines["kalshi"].exchange.orders) == 1

    @pytest.mark.asyncio
    async def test_zero_fees_full_profit(self):
        """With genuinely fee-free legs, profit = spread. Polymarket is fee-free
        only for the geopolitics category (makers aside); Kalshi via override."""
        poly = _make_market("polymarket", 0.40, "Will A happen?", category="geopolitics")
        kalshi = _make_market("kalshi", 0.50, "Will A happen?", category="geopolitics")
        scanner = self._make_scanner(
            [poly], [kalshi],
            fees={"polymarket": 0.0, "kalshi": 0.0},
        )

        opps = await scanner.scan()
        cross = [o for o in opps if o.arb_type == "cross_exchange"]
        assert len(cross) == 1
        assert cross[0].expected_profit_pct == pytest.approx(10.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_near_resolution_cross_exchange_arb_filtered_before_execution(self):
        poly = _make_market("polymarket", 0.10, "Will X happen?")
        kalshi = _make_market("kalshi", 0.20, "Will X happen?")
        poly.end_date = datetime.now(timezone.utc) + timedelta(hours=1)
        kalshi.end_date = datetime.now(timezone.utc) + timedelta(hours=1)
        scanner = self._make_scanner([poly], [kalshi])

        opps = await scanner.scan()

        assert [o for o in opps if o.arb_type == "cross_exchange"] == []

    @pytest.mark.asyncio
    async def test_no_match_different_questions(self):
        """Markets with different questions should not match."""
        poly = _make_market("polymarket", 0.40, "Will Trump win?")
        kalshi = _make_market("kalshi", 0.50, "Will Bitcoin reach 100k?")
        scanner = self._make_scanner([poly], [kalshi])

        opps = await scanner.scan()
        cross = [o for o in opps if o.arb_type == "cross_exchange"]
        assert len(cross) == 0


# ---------------------------------------------------------------------------
# MetaculusSource
# ---------------------------------------------------------------------------


class TestMetaculusSource:
    """Tests for the Metaculus data source."""

    @pytest.mark.asyncio
    async def test_fetch_returns_news_items(self):
        """MetaculusSource should parse API response into NewsItems."""
        from auramaur.data_sources.metaculus import MetaculusSource

        mock_response = {
            "results": [
                {
                    "id": 12345,
                    "title": "Will there be a US recession in 2026?",
                    "community_prediction": {
                        "full": {"q2": 0.35},
                    },
                    "number_of_forecasters": 150,
                    "resolve_time": "2026-12-31T00:00:00Z",
                    "created_time": "2025-01-01T00:00:00Z",
                },
                {
                    "id": 12346,
                    "title": "Will inflation exceed 5% in 2026?",
                    "community_prediction": {
                        "full": {"q2": 0.22},
                    },
                    "number_of_forecasters": 80,
                    "resolve_time": "2026-12-31T00:00:00Z",
                    "created_time": "2025-02-01T00:00:00Z",
                },
            ]
        }

        source = MetaculusSource()

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        source._session = mock_session

        items = await source.fetch("US recession economy")

        assert len(items) == 2
        assert items[0].source == "metaculus"
        assert "35%" in items[0].title
        assert "Metaculus" in items[0].title
        assert items[0].relevance_score == pytest.approx(min(3.0, 150 / 20), abs=0.1)
        assert "metaculus.com" in items[0].url

        await source.close()

    @pytest.mark.asyncio
    async def test_fetch_handles_empty_results(self):
        """Should return empty list on no results."""
        from auramaur.data_sources.metaculus import MetaculusSource

        source = MetaculusSource()

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"results": []})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        source._session = mock_session

        items = await source.fetch("something obscure")
        assert items == []

        await source.close()

    @pytest.mark.asyncio
    async def test_fetch_handles_api_error(self):
        """Should return empty list on API error."""
        from auramaur.data_sources.metaculus import MetaculusSource

        source = MetaculusSource()

        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        source._session = mock_session

        items = await source.fetch("test query")
        assert items == []

        await source.close()


# ---------------------------------------------------------------------------
# ArbitrageConfig
# ---------------------------------------------------------------------------


class TestArbitrageConfig:
    """Tests for ArbitrageConfig loading."""

    def test_defaults_load(self):
        from config.settings import ArbitrageConfig

        config = ArbitrageConfig()
        assert config.enabled is True
        assert config.min_profit_after_fees_pct == 1.5
        assert config.max_arb_size == 25.0
        assert config.cross_exchange_auto_execute is True
        assert config.exchange_fees["polymarket"] == 0.0
        assert config.exchange_fees["kalshi"] == 0.07

    def test_settings_integration(self):
        from config.settings import Settings

        s = Settings()
        assert hasattr(s, "arbitrage")
        assert s.arbitrage.enabled is True
