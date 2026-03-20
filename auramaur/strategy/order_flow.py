"""Order flow analysis — detect informed trader activity from order book data."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone

import structlog

from auramaur.exchange.models import OrderBook, OrderSide

log = structlog.get_logger()


class OrderFlowTracker:
    """Tracks order flow to detect informed trader activity.

    Signals:
    - Large orders relative to recent average
    - Bid/ask imbalance (more buying pressure or selling pressure)
    - Size acceleration (flow increasing rapidly)
    """

    def __init__(self, lookback: int = 50) -> None:
        self._lookback = lookback
        # market_id -> deque of (timestamp, side, size) trades
        self._recent_trades: dict[str, deque] = {}
        # market_id -> last known order book snapshot
        self._last_books: dict[str, OrderBook] = {}

    def record_trade(self, market_id: str, side: OrderSide, size: float) -> None:
        """Record a trade event (from WebSocket or order book delta)."""
        if market_id not in self._recent_trades:
            self._recent_trades[market_id] = deque(maxlen=self._lookback)
        self._recent_trades[market_id].append(
            (datetime.now(timezone.utc), side, size)
        )

    def record_book_snapshot(self, market_id: str, book: OrderBook) -> None:
        """Record an order book snapshot for imbalance analysis."""
        self._last_books[market_id] = book

    def get_flow_signal(self, market_id: str) -> FlowSignal:
        """Compute order flow signals for a market.

        Returns a FlowSignal with:
        - imbalance: -1 to +1 (negative = sell pressure, positive = buy pressure)
        - large_order_detected: whether an unusually large order was seen
        - flow_intensity: 0 to 1 (how active the market is relative to its history)
        """
        imbalance = self._compute_book_imbalance(market_id)
        large_order = self._detect_large_orders(market_id)
        intensity = self._compute_flow_intensity(market_id)

        signal = FlowSignal(
            market_id=market_id,
            imbalance=imbalance,
            large_order_detected=large_order,
            flow_intensity=intensity,
        )

        if large_order or abs(imbalance) > 0.3:
            log.info(
                "order_flow.signal",
                market_id=market_id,
                imbalance=round(imbalance, 3),
                large_order=large_order,
                intensity=round(intensity, 3),
            )

        return signal

    def _compute_book_imbalance(self, market_id: str) -> float:
        """Compute bid/ask size imbalance from the order book.

        Returns value from -1 (all asks, sell pressure) to +1 (all bids, buy pressure).
        """
        book = self._last_books.get(market_id)
        if not book or (not book.bids and not book.asks):
            return 0.0

        # Sum top 5 levels of depth on each side
        bid_depth = sum(level.size for level in book.bids[:5])
        ask_depth = sum(level.size for level in book.asks[:5])
        total = bid_depth + ask_depth

        if total == 0:
            return 0.0

        return (bid_depth - ask_depth) / total

    def _detect_large_orders(self, market_id: str) -> bool:
        """Check if any recent trades are 3x+ the average size."""
        trades = self._recent_trades.get(market_id)
        if not trades or len(trades) < 5:
            return False

        sizes = [size for _, _, size in trades]
        avg_size = sum(sizes) / len(sizes)
        if avg_size == 0:
            return False

        # Check if the most recent trade is 3x the average
        latest_size = sizes[-1]
        return latest_size >= avg_size * 3.0

    def _compute_flow_intensity(self, market_id: str) -> float:
        """Compute how active trading is, normalized 0-1.

        Based on number of trades in the lookback window relative to capacity.
        """
        trades = self._recent_trades.get(market_id)
        if not trades:
            return 0.0
        return len(trades) / self._lookback

    def get_probability_nudge(self, market_id: str) -> float:
        """Get a probability adjustment based on order flow.

        Positive = flow suggests price should be higher (buy pressure).
        Negative = flow suggests price should be lower (sell pressure).

        Magnitude is capped at ±3% to avoid over-relying on flow alone.
        """
        signal = self.get_flow_signal(market_id)

        # Base nudge from imbalance
        nudge = signal.imbalance * 0.02  # Max ±2% from imbalance

        # Amplify if large order detected in the direction of imbalance
        if signal.large_order_detected and abs(signal.imbalance) > 0.1:
            nudge *= 1.5

        # Dampen if flow intensity is low (not enough data to be confident)
        nudge *= max(0.2, signal.flow_intensity)

        # Hard cap at ±3%
        return max(-0.03, min(0.03, nudge))


class FlowSignal:
    """Order flow analysis result."""

    __slots__ = ("market_id", "imbalance", "large_order_detected", "flow_intensity")

    def __init__(
        self,
        market_id: str,
        imbalance: float = 0.0,
        large_order_detected: bool = False,
        flow_intensity: float = 0.0,
    ) -> None:
        self.market_id = market_id
        self.imbalance = imbalance
        self.large_order_detected = large_order_detected
        self.flow_intensity = flow_intensity
