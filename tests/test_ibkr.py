"""Tests for Interactive Brokers client."""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, AsyncMock

from auramaur.exchange.ibkr import IBKRClient
from auramaur.exchange.models import Confidence, Market, Order, OrderSide, Signal, TokenType
from auramaur.nlp.reframer import (
    OptionContract,
    ReframedMarket,
    TradeMapping,
    ReframeType,
    reframe_option_as_binary,
)


def _make_option(**kwargs) -> OptionContract:
    defaults = dict(
        symbol="AAPL",
        strike=200.0,
        expiry=datetime.now(timezone.utc) + timedelta(days=30),
        right="C",
        delta=0.45,
        mid_price=5.50,
        bid=5.40,
        ask=5.60,
        implied_vol=0.30,
        volume=500,
        open_interest=1000,
        underlying_price=195.0,
        con_id=12345,
    )
    defaults.update(kwargs)
    return OptionContract(**defaults)


def _make_signal(side: OrderSide = OrderSide.BUY) -> Signal:
    return Signal(
        market_id="IB:AAPL:200.0:20260418:C",
        claude_prob=0.60,
        claude_confidence=Confidence.HIGH,
        market_prob=0.45,
        edge=15.0,
        recommended_side=side,
    )


class TestIBKRPrepareOrder:
    def _make_client_with_reframe(self) -> IBKRClient:
        client = IBKRClient.__new__(IBKRClient)
        client._reframed = {}

        opt = _make_option()
        rm = reframe_option_as_binary(opt)
        client._reframed[rm.market.id] = rm
        return client

    def test_buy_yes_call(self):
        """BUY YES on 'Will AAPL > $200?' → buy the call."""
        client = self._make_client_with_reframe()
        market_id = list(client._reframed.keys())[0]

        signal = Signal(
            market_id=market_id,
            claude_prob=0.60,
            claude_confidence=Confidence.HIGH,
            market_prob=0.45,
            edge=15.0,
            recommended_side=OrderSide.BUY,
        )

        order = client.prepare_order(signal, client._reframed[market_id].market, 1000.0, False)
        assert order is not None
        assert order.side == OrderSide.BUY
        assert order.exchange == "ibkr"
        assert "buy_call" in order.token_id
        assert order.dry_run is True

    def test_sell_yes_put(self):
        """SELL YES on 'Will AAPL > $200?' → buy the put."""
        client = self._make_client_with_reframe()
        market_id = list(client._reframed.keys())[0]

        signal = Signal(
            market_id=market_id,
            claude_prob=0.30,
            claude_confidence=Confidence.HIGH,
            market_prob=0.45,
            edge=15.0,
            recommended_side=OrderSide.SELL,
        )

        order = client.prepare_order(signal, client._reframed[market_id].market, 1000.0, False)
        assert order is not None
        assert order.side == OrderSide.BUY  # Buying a put
        assert "buy_put" in order.token_id

    def test_too_small_position(self):
        """If position_size can't buy 1 contract, return None."""
        client = self._make_client_with_reframe()
        market_id = list(client._reframed.keys())[0]
        signal = _make_signal()
        signal.market_id = market_id

        # $5.50 * 100 multiplier = $550 per contract; $100 can't buy one
        order = client.prepare_order(signal, client._reframed[market_id].market, 100.0, False)
        assert order is None

    def test_no_reframe_returns_none(self):
        """If market_id not in reframed cache, return None."""
        client = IBKRClient.__new__(IBKRClient)
        client._reframed = {}

        market = Market(id="unknown", exchange="ibkr", question="?")
        order = client.prepare_order(_make_signal(), market, 1000.0, False)
        assert order is None


class TestIBKRPaperGate:
    @pytest.mark.asyncio
    async def test_paper_routes_to_paper_trader(self):
        """Dry-run orders go through PaperTrader."""
        paper = MagicMock()
        paper.execute = AsyncMock(return_value=MagicMock(
            order_id="PAPER-123",
            market_id="IB:AAPL:200:20260418:C",
            status="paper",
            is_paper=True,
        ))

        client = IBKRClient.__new__(IBKRClient)
        client._paper = paper
        client._settings = MagicMock()
        client._settings.is_live = False

        order = Order(
            market_id="IB:AAPL:200:20260418:C",
            exchange="ibkr",
            side=OrderSide.BUY,
            size=1,
            price=5.50,
            dry_run=True,
        )
        result = await client.place_order(order)
        assert result.is_paper is True
        paper.execute.assert_called_once()


class TestIBKRConfig:
    def test_ibkr_config_defaults(self):
        from config.settings import Settings
        s = Settings()
        assert s.ibkr.enabled is False
        assert s.ibkr.environment == "paper"
        assert s.ibkr.paper_port == 7497
        assert s.ibkr.live_port == 7496
        assert "SPY" in s.ibkr.watchlist

    def test_ibkr_yaml_defaults(self):
        import yaml
        from pathlib import Path

        defaults_path = Path(__file__).parent.parent / "config" / "defaults.yaml"
        with open(defaults_path) as f:
            raw = yaml.safe_load(f)

        assert raw["ibkr"]["enabled"] is False
        assert raw["ibkr"]["environment"] == "paper"
