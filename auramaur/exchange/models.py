"""Pydantic models for markets, orders, and positions."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class MarketStatus(str, Enum):
    ACTIVE = "active"
    CLOSED = "closed"
    RESOLVED = "resolved"


class Confidence(str, Enum):
    LOW = "LOW"
    MEDIUM_LOW = "MEDIUM_LOW"
    MEDIUM = "MEDIUM"
    MEDIUM_HIGH = "MEDIUM_HIGH"
    HIGH = "HIGH"


class Market(BaseModel):
    id: str
    exchange: str = "polymarket"
    condition_id: str = ""
    ticker: str = ""
    question: str
    description: str = ""
    category: str = ""
    end_date: datetime | None = None
    active: bool = True
    outcome_yes_price: float = 0.5
    outcome_no_price: float = 0.5
    volume: float = 0
    liquidity: float = 0
    spread: float = 0
    clob_token_yes: str = ""
    clob_token_no: str = ""
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class TokenType(str, Enum):
    YES = "YES"
    NO = "NO"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class Order(BaseModel):
    market_id: str
    exchange: str = "polymarket"
    token_id: str = ""  # CLOB token ID for the outcome being traded
    side: OrderSide
    token: TokenType = TokenType.YES
    size: float
    price: float
    order_type: OrderType = OrderType.MARKET
    dry_run: bool = True  # Paper trade by default
    # Post-only: reject rather than cross the spread. Only meaningful for
    # LIMIT orders. Protects against turning a maker quote into a taker fill
    # when the book moves between order formation and submission, and keeps
    # us eligible for Polymarket's maker-reward tier.
    post_only: bool = False

    @property
    def notional(self) -> float:
        return self.size * self.price


class OrderResult(BaseModel):
    order_id: str
    market_id: str
    status: Literal["filled", "partial", "pending", "rejected", "paper", "cancelled", "expired"]
    filled_size: float = 0
    filled_price: float = 0
    is_paper: bool = True
    error_message: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Position(BaseModel):
    market_id: str
    side: OrderSide
    size: float
    avg_price: float
    current_price: float = 0
    category: str = ""
    token: TokenType = TokenType.YES
    token_id: str = ""

    @property
    def unrealized_pnl(self) -> float:
        if self.side == OrderSide.BUY:
            return (self.current_price - self.avg_price) * self.size
        return (self.avg_price - self.current_price) * self.size


class OrderBookLevel(BaseModel):
    price: float
    size: float


class OrderBook(BaseModel):
    bids: list[OrderBookLevel] = Field(default_factory=list)
    asks: list[OrderBookLevel] = Field(default_factory=list)

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def midpoint(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    @property
    def spread(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None


class ExitReason(str, Enum):
    STOP_LOSS = "STOP_LOSS"
    PROFIT_TARGET = "PROFIT_TARGET"
    EDGE_EROSION = "EDGE_EROSION"
    TIME_DECAY = "TIME_DECAY"


class Fill(BaseModel):
    """A single execution fill from the CLOB or paper trader."""

    order_id: str
    market_id: str
    token_id: str = ""
    side: OrderSide
    token: TokenType = TokenType.YES
    size: float
    price: float
    fee: float = 0.0
    is_paper: bool = True
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LivePosition(BaseModel):
    """Ground-truth position from on-chain / CLOB API."""

    market_id: str
    token_id: str = ""
    token: TokenType = TokenType.YES
    size: float
    avg_cost: float = 0.0
    current_price: float = 0.0
    market_question: str = ""
    category: str = ""

    @property
    def notional(self) -> float:
        return self.size * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.size * self.avg_cost

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.avg_cost) * self.size


class Signal(BaseModel):
    market_id: str
    market_question: str = ""
    claude_prob: float
    claude_confidence: Confidence
    market_prob: float
    edge: float
    second_opinion_prob: float | None = None
    divergence: float | None = None
    evidence_summary: str = ""
    recommended_side: OrderSide | None = None
    recommended_size: float = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
