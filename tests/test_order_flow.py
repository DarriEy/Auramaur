"""Tests for order flow analysis."""

from auramaur.exchange.models import OrderBook, OrderBookLevel, OrderSide
from auramaur.strategy.order_flow import OrderFlowTracker


def _make_book(bid_sizes: list[float], ask_sizes: list[float], bid_price: float = 0.50, ask_price: float = 0.52) -> OrderBook:
    bids = [OrderBookLevel(price=bid_price - i * 0.01, size=s) for i, s in enumerate(bid_sizes)]
    asks = [OrderBookLevel(price=ask_price + i * 0.01, size=s) for i, s in enumerate(ask_sizes)]
    return OrderBook(bids=bids, asks=asks)


class TestOrderFlowTracker:
    def test_empty_flow_returns_neutral(self):
        tracker = OrderFlowTracker()
        signal = tracker.get_flow_signal("mkt_1")
        assert signal.imbalance == 0.0
        assert signal.large_order_detected is False
        assert signal.flow_intensity == 0.0

    def test_buy_imbalance(self):
        """Heavy bid depth should show positive imbalance."""
        tracker = OrderFlowTracker()
        book = _make_book([100, 80, 60], [20, 15, 10])
        tracker.record_book_snapshot("mkt_1", book)
        signal = tracker.get_flow_signal("mkt_1")
        assert signal.imbalance > 0.3  # Strong buy pressure

    def test_sell_imbalance(self):
        """Heavy ask depth should show negative imbalance."""
        tracker = OrderFlowTracker()
        book = _make_book([10, 5], [100, 80, 60])
        tracker.record_book_snapshot("mkt_1", book)
        signal = tracker.get_flow_signal("mkt_1")
        assert signal.imbalance < -0.3  # Strong sell pressure

    def test_large_order_detection(self):
        """Orders 3x+ average should be flagged."""
        tracker = OrderFlowTracker()
        # Record normal trades
        for _ in range(10):
            tracker.record_trade("mkt_1", OrderSide.BUY, 10.0)
        # Record a whale trade
        tracker.record_trade("mkt_1", OrderSide.BUY, 50.0)
        signal = tracker.get_flow_signal("mkt_1")
        assert signal.large_order_detected is True

    def test_no_large_order_when_normal(self):
        tracker = OrderFlowTracker()
        for _ in range(10):
            tracker.record_trade("mkt_1", OrderSide.BUY, 10.0)
        signal = tracker.get_flow_signal("mkt_1")
        assert signal.large_order_detected is False

    def test_flow_intensity(self):
        """Intensity should increase with more trades."""
        tracker = OrderFlowTracker(lookback=20)
        for i in range(10):
            tracker.record_trade("mkt_1", OrderSide.BUY, 5.0)
        signal = tracker.get_flow_signal("mkt_1")
        assert signal.flow_intensity == 0.5  # 10/20

    def test_probability_nudge_buy_pressure(self):
        """Buy pressure should give positive nudge."""
        tracker = OrderFlowTracker()
        book = _make_book([100, 80], [20, 15])
        tracker.record_book_snapshot("mkt_1", book)
        # Add some trades for intensity
        for _ in range(25):
            tracker.record_trade("mkt_1", OrderSide.BUY, 10.0)
        nudge = tracker.get_probability_nudge("mkt_1")
        assert nudge > 0  # Positive nudge = buy pressure

    def test_probability_nudge_capped(self):
        """Nudge should never exceed ±3%."""
        tracker = OrderFlowTracker()
        book = _make_book([1000], [1])  # Extreme imbalance
        tracker.record_book_snapshot("mkt_1", book)
        for _ in range(50):
            tracker.record_trade("mkt_1", OrderSide.BUY, 100.0)
        nudge = tracker.get_probability_nudge("mkt_1")
        assert -0.03 <= nudge <= 0.03

    def test_nudge_zero_when_no_data(self):
        tracker = OrderFlowTracker()
        nudge = tracker.get_probability_nudge("unknown")
        assert nudge == 0.0
