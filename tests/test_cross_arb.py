"""Tests for cross-exchange arbitrage and Metaculus data source."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from auramaur.exchange.models import Market
from auramaur.strategy.arbitrage_scanner import ArbitrageScanner


# ---------------------------------------------------------------------------
# Fee-aware profit calculation
# ---------------------------------------------------------------------------


def _make_market(exchange: str, yes_price: float, question: str = "Will X happen?") -> Market:
    return Market(
        id=f"{exchange}_123",
        exchange=exchange,
        question=question,
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
        """A 10% spread should produce ~9.3% profit after 7% Kalshi fee."""
        poly = _make_market("polymarket", 0.40, "Will X happen?")
        kalshi = _make_market("kalshi", 0.50, "Will X happen?")
        scanner = self._make_scanner([poly], [kalshi])

        opps = await scanner.scan()
        cross = [o for o in opps if o.arb_type == "cross_exchange"]
        assert len(cross) == 1

        opp = cross[0]
        # Spread = 0.10, after 7% fee on profit: 0.10 * 0.93 = 0.093 = 9.3%
        assert opp.spread == pytest.approx(0.10, abs=0.001)
        assert opp.expected_profit_pct == pytest.approx(9.3, abs=0.1)

    @pytest.mark.asyncio
    async def test_small_spread_filtered_by_min_profit(self):
        """A 3.5% spread with 7% fee = ~3.255% profit. Should pass default 1.5% min."""
        poly = _make_market("polymarket", 0.50, "Will Y happen?")
        kalshi = _make_market("kalshi", 0.535, "Will Y happen?")
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
    async def test_zero_fees_full_profit(self):
        """With no fees on either exchange, profit = spread."""
        poly = _make_market("polymarket", 0.40, "Will A happen?")
        kalshi = _make_market("kalshi", 0.50, "Will A happen?")
        scanner = self._make_scanner(
            [poly], [kalshi],
            fees={"polymarket": 0.0, "kalshi": 0.0},
        )

        opps = await scanner.scan()
        cross = [o for o in opps if o.arb_type == "cross_exchange"]
        assert len(cross) == 1
        assert cross[0].expected_profit_pct == pytest.approx(10.0, abs=0.1)

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
