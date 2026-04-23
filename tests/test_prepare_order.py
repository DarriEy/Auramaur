"""Tests for exchange-specific prepare_order logic."""

from auramaur.exchange.client import PolymarketClient
from auramaur.exchange.models import (
    Confidence, Market, OrderSide, Signal, TokenType,
)


def _make_signal(side: OrderSide = OrderSide.BUY) -> Signal:
    return Signal(
        market_id="test-market",
        claude_prob=0.7,
        claude_confidence=Confidence.HIGH,
        market_prob=0.5,
        edge=20.0,
        recommended_side=side,
    )


def _make_market(
    yes_price: float = 0.50,
    no_price: float | None = None,
    clob_yes: str = "tok-yes",
    clob_no: str = "tok-no",
) -> Market:
    return Market(
        id="test-market",
        question="Will X happen?",
        outcome_yes_price=yes_price,
        outcome_no_price=no_price if no_price is not None else (1.0 - yes_price),
        clob_token_yes=clob_yes,
        clob_token_no=clob_no,
    )


class TestPolymarketPrepareOrder:
    def _make_client(self):
        """Create a PolymarketClient without real dependencies."""
        return PolymarketClient.__new__(PolymarketClient)

    def test_buy_yes(self):
        client = self._make_client()
        order = client.prepare_order(_make_signal(OrderSide.BUY), _make_market(), 25.0, False)
        assert order is not None
        assert order.side == OrderSide.BUY
        assert order.token == TokenType.YES
        assert order.token_id == "tok-yes"
        assert order.exchange == "polymarket"

    def test_sell_becomes_buy_no(self):
        client = self._make_client()
        order = client.prepare_order(_make_signal(OrderSide.SELL), _make_market(), 25.0, False)
        assert order is not None
        assert order.side == OrderSide.BUY  # Always BUY on Polymarket
        assert order.token == TokenType.NO
        assert order.token_id == "tok-no"

    def test_min_size_rejected(self):
        client = self._make_client()
        # $0.50 at 0.50 = $0.25 notional (< $1 minimum)
        order = client.prepare_order(_make_signal(), _make_market(), 0.50, False)
        assert order is None

    def test_price_clamping(self):
        client = self._make_client()
        order = client.prepare_order(_make_signal(), _make_market(yes_price=0.001), 25.0, False)
        assert order is not None
        assert order.price >= 0.01
        assert order.price <= 0.99

    def test_dry_run_when_not_live(self):
        client = self._make_client()
        order = client.prepare_order(_make_signal(), _make_market(), 25.0, False)
        assert order is not None
        assert order.dry_run is True

    def test_live_flag_propagated(self):
        client = self._make_client()
        order = client.prepare_order(_make_signal(), _make_market(), 25.0, True)
        assert order is not None
        assert order.dry_run is False


class TestKalshiPrepareOrder:
    def _make_client(self):
        from auramaur.exchange.kalshi import KalshiClient
        client = KalshiClient.__new__(KalshiClient)
        return client

    def test_buy_yes(self):
        client = self._make_client()
        market = _make_market()
        market.exchange = "kalshi"
        market.ticker = "KXTEST"
        order = client.prepare_order(_make_signal(OrderSide.BUY), market, 25.0, False)
        assert order is not None
        assert order.side == OrderSide.BUY
        assert order.token == TokenType.YES
        assert order.exchange == "kalshi"

    def test_sell_signal_becomes_buy_no(self):
        """Kalshi SELL signal becomes BUY NO (can't sell what you don't own)."""
        client = self._make_client()
        market = _make_market()
        market.exchange = "kalshi"
        market.ticker = "KXTEST"
        order = client.prepare_order(_make_signal(OrderSide.SELL), market, 25.0, False)
        assert order is not None
        assert order.side == OrderSide.BUY
        assert order.token == TokenType.NO

    def test_sell_with_exit_token_exits_position(self):
        """SELL + exit_token=YES closes a held YES position (actual SELL)."""
        client = self._make_client()
        market = _make_market()
        market.exchange = "kalshi"
        market.ticker = "KXTEST"
        signal = Signal(
            market_id="test-market",
            claude_prob=0.5,
            claude_confidence=Confidence.MEDIUM,
            market_prob=0.5,
            edge=10.0,
            recommended_side=OrderSide.SELL,
            exit_token=TokenType.YES,
        )
        order = client.prepare_order(signal, market, 25.0, False)
        assert order is not None
        assert order.side == OrderSide.SELL
        assert order.token == TokenType.YES

    def test_sell_with_exit_token_no_exits_no_position(self):
        """SELL + exit_token=NO closes a held NO position."""
        client = self._make_client()
        market = _make_market()
        market.exchange = "kalshi"
        market.ticker = "KXTEST"
        signal = Signal(
            market_id="test-market",
            claude_prob=0.5,
            claude_confidence=Confidence.MEDIUM,
            market_prob=0.5,
            edge=10.0,
            recommended_side=OrderSide.SELL,
            exit_token=TokenType.NO,
        )
        order = client.prepare_order(signal, market, 25.0, False)
        assert order is not None
        assert order.side == OrderSide.SELL
        assert order.token == TokenType.NO
