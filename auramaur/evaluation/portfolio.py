"""Pure deterministic counterfactual portfolio domain logic.

No broker, exchange, risk, database, or live-order modules are imported here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math
from typing import Protocol


class ForecastLike(Protocol):
    action: str


class Side(str, Enum):
    YES = "yes"
    NO = "no"


@dataclass(frozen=True)
class MarketSnapshot:
    market_id: str
    category: str
    timestamp: datetime
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    yes_bid_depth: float = math.inf
    yes_ask_depth: float = math.inf
    no_bid_depth: float = math.inf
    no_ask_depth: float = math.inf


@dataclass(frozen=True)
class SimulationPolicy:
    initial_cash: float = 1_000.0
    order_notional: float = 100.0
    fee_rate: float = 0.0
    slippage_bps: float = 0.0
    max_positions: int = 10
    max_position_notional: float = 100.0
    max_total_exposure: float = 1_000.0
    max_category_exposure: float = 500.0

    def __post_init__(self) -> None:
        values = (self.initial_cash, self.order_notional, self.fee_rate,
                  self.slippage_bps, self.max_position_notional,
                  self.max_total_exposure, self.max_category_exposure)
        if any(not math.isfinite(v) or v < 0 for v in values) or self.max_positions < 0:
            raise ValueError("policy limits must be finite and non-negative")


@dataclass
class Position:
    market_id: str
    category: str
    side: Side
    quantity: float
    entry_price: float
    entry_fee: float
    opened_at: datetime
    mark_price: float

    @property
    def notional(self) -> float:
        return self.quantity * self.entry_price


@dataclass(frozen=True)
class Fill:
    market_id: str
    side: Side
    quantity: float
    price: float
    fee: float
    timestamp: datetime
    opening: bool


@dataclass(frozen=True)
class PortfolioMetrics:
    cash: float
    locked_capital: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    max_drawdown: float
    turnover: float
    requested_orders: int
    filled_orders: int
    fill_rate: float
    capital_hours: float
    return_on_capital_time: float


@dataclass
class VirtualPortfolio:
    """Independent arm state; there is intentionally no global market registry."""
    arm_id: str
    policy: SimulationPolicy
    cash: float = field(init=False)
    positions: dict[str, Position] = field(default_factory=dict, init=False)
    fills: list[Fill] = field(default_factory=list, init=False)
    realized_pnl: float = field(default=0.0, init=False)
    turnover: float = field(default=0.0, init=False)
    requested_orders: int = field(default=0, init=False)
    capital_hours: float = field(default=0.0, init=False)
    _last_time: datetime | None = field(default=None, init=False)
    _peak: float = field(init=False)
    _drawdown: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.cash = self.policy.initial_cash
        self._peak = self.cash

    @property
    def locked_capital(self) -> float:
        return sum(p.notional + p.entry_fee for p in self.positions.values())

    def _advance(self, timestamp: datetime) -> None:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if self._last_time is not None:
            hours = (timestamp - self._last_time).total_seconds() / 3600
            if hours < 0:
                raise ValueError("events must be chronological")
            self.capital_hours += self.locked_capital * hours
        self._last_time = timestamp

    @staticmethod
    def _valid(price: float, depth: float) -> bool:
        return math.isfinite(price) and 0 < price < 1 and depth > 0 and not math.isnan(depth)

    def enter(self, snapshot: MarketSnapshot, forecast: ForecastLike) -> Fill | None:
        self._advance(snapshot.timestamp)
        action = str(forecast.action).lower()
        if action == "abstain":
            return None
        self.requested_orders += 1
        if action not in ("yes", "no") or snapshot.market_id in self.positions:
            return None
        if len(self.positions) >= self.policy.max_positions:
            return None
        side = Side(action)
        ask, depth = ((snapshot.yes_ask, snapshot.yes_ask_depth) if side is Side.YES
                      else (snapshot.no_ask, snapshot.no_ask_depth))
        price = ask * (1 + self.policy.slippage_bps / 10_000)
        if not self._valid(price, depth):
            return None
        category_used = sum(p.notional for p in self.positions.values() if p.category == snapshot.category)
        total_used = sum(p.notional for p in self.positions.values())
        budget = min(self.policy.order_notional, self.policy.max_position_notional,
                     self.policy.max_total_exposure - total_used,
                     self.policy.max_category_exposure - category_used,
                     self.cash / (1 + self.policy.fee_rate))
        if budget <= 0:
            return None
        quantity = min(budget / price, depth)
        notional = quantity * price
        fee = notional * self.policy.fee_rate
        if quantity <= 0 or notional + fee > self.cash + 1e-9:
            return None
        self.cash -= notional + fee
        position = Position(snapshot.market_id, snapshot.category, side, quantity,
                            price, fee, snapshot.timestamp, price)
        self.positions[snapshot.market_id] = position
        fill = Fill(snapshot.market_id, side, quantity, price, fee, snapshot.timestamp, True)
        self.fills.append(fill)
        self.turnover += notional
        self._track_drawdown()
        return fill

    def mark(self, snapshot: MarketSnapshot) -> None:
        self._advance(snapshot.timestamp)
        p = self.positions.get(snapshot.market_id)
        if p:
            bid, depth = ((snapshot.yes_bid, snapshot.yes_bid_depth) if p.side is Side.YES
                          else (snapshot.no_bid, snapshot.no_bid_depth))
            if self._valid(bid, depth):
                p.mark_price = bid
        self._track_drawdown()

    def exit(self, snapshot: MarketSnapshot) -> Fill | None:
        self._advance(snapshot.timestamp)
        p = self.positions.get(snapshot.market_id)
        if not p:
            return None
        self.requested_orders += 1
        bid, depth = ((snapshot.yes_bid, snapshot.yes_bid_depth) if p.side is Side.YES
                      else (snapshot.no_bid, snapshot.no_bid_depth))
        price = bid * (1 - self.policy.slippage_bps / 10_000)
        if not self._valid(price, depth):
            return None
        old_qty = p.quantity
        qty = min(old_qty, depth)
        proceeds, fee = qty * price, qty * price * self.policy.fee_rate
        basis, entry_fee = qty * p.entry_price, p.entry_fee * qty / old_qty
        self.cash += proceeds - fee
        self.realized_pnl += proceeds - fee - basis - entry_fee
        self.turnover += proceeds
        p.quantity -= qty
        p.entry_fee -= entry_fee
        p.mark_price = price
        if p.quantity <= 1e-12:
            del self.positions[p.market_id]
        fill = Fill(snapshot.market_id, p.side, qty, price, fee, snapshot.timestamp, False)
        self.fills.append(fill)
        self._track_drawdown()
        return fill

    def settle(self, market_id: str, yes_won: bool, timestamp: datetime) -> float:
        self._advance(timestamp)
        p = self.positions.pop(market_id, None)
        if not p:
            return 0.0
        proceeds = p.quantity if ((p.side is Side.YES) == yes_won) else 0.0
        pnl = proceeds - p.notional - p.entry_fee
        self.cash += proceeds
        self.realized_pnl += pnl
        self._track_drawdown()
        return pnl

    def _equity(self) -> float:
        return self.cash + sum(p.quantity * p.mark_price for p in self.positions.values())

    def _track_drawdown(self) -> None:
        equity = self._equity()
        self._peak = max(self._peak, equity)
        self._drawdown = max(self._drawdown, self._peak - equity)

    def metrics(self) -> PortfolioMetrics:
        equity = self._equity()
        unrealized = sum(p.quantity * p.mark_price - p.notional - p.entry_fee
                         for p in self.positions.values())
        pnl = equity - self.policy.initial_cash
        return PortfolioMetrics(self.cash, self.locked_capital, equity, self.realized_pnl,
                                unrealized, pnl, self._drawdown, self.turnover,
                                self.requested_orders, len(self.fills),
                                len(self.fills) / self.requested_orders if self.requested_orders else 0.0,
                                self.capital_hours,
                                pnl / (self.capital_hours / 24) if self.capital_hours else 0.0)
