"""Paper trading simulator — no real money involved."""

from __future__ import annotations

import random
import uuid
from datetime import datetime

import structlog

from auramaur.db.database import Database
from auramaur.exchange.models import Order, OrderResult, OrderSide, OrderType, Position, TokenType

log = structlog.get_logger()


class PaperTrader:
    """Simulates order execution for paper trading."""

    def __init__(self, db: Database, initial_balance: float = 1000.0):
        self.db = db
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: dict[str, Position] = {}
        self.trade_count = 0
        self.pending_orders: list[tuple[Order, str]] = []  # (order, order_id)

    async def load_state(self) -> None:
        """Load paper trading state from database."""
        rows = await self.db.fetchall(
            "SELECT * FROM portfolio WHERE is_paper = 1"
        )
        for row in rows:
            token_str = row["token"] if "token" in row.keys() else "YES"
            token_id = row["token_id"] if "token_id" in row.keys() else ""
            self.positions[row["market_id"]] = Position(
                market_id=row["market_id"],
                side=OrderSide(row["side"]),
                size=row["size"],
                avg_price=row["avg_price"],
                current_price=row["current_price"] or row["avg_price"],
                category=row["category"] or "",
                token=TokenType(token_str) if token_str else TokenType.YES,
                token_id=token_id or "",
            )

        # Calculate balance from trade history
        row = await self.db.fetchone(
            "SELECT COALESCE(SUM(CASE WHEN side='BUY' THEN -size*price ELSE size*price END), 0) as net FROM trades WHERE is_paper=1"
        )
        if row:
            self.balance = self.initial_balance + float(row["net"])

        log.info("paper.state_loaded", balance=self.balance, positions=len(self.positions))

    async def execute(self, order: Order) -> OrderResult:
        """Simulate order execution."""
        order_id = f"PAPER-{uuid.uuid4().hex[:12]}"
        cost = order.size * order.price

        if order.side == OrderSide.BUY and cost > self.balance:
            log.warning(
                "paper.insufficient_balance",
                market_id=order.market_id,
                cost=round(cost, 2),
                balance=round(self.balance, 2),
                shortfall=round(cost - self.balance, 2),
            )
            return OrderResult(
                order_id=order_id,
                market_id=order.market_id,
                status="rejected",
                is_paper=True,
            )

        # Update balance
        if order.side == OrderSide.BUY:
            self.balance -= cost
        else:
            self.balance += cost

        # Update position
        if order.market_id in self.positions:
            pos = self.positions[order.market_id]
            if order.side == OrderSide.BUY:
                total_cost = pos.avg_price * pos.size + order.price * order.size
                pos.size += order.size
                pos.avg_price = total_cost / pos.size if pos.size > 0 else 0
            else:
                pos.size -= order.size
                if pos.size <= 0:
                    del self.positions[order.market_id]
        elif order.side == OrderSide.BUY:
            self.positions[order.market_id] = Position(
                market_id=order.market_id,
                side=order.side,
                size=order.size,
                avg_price=order.price,
                current_price=order.price,
                token=order.token if hasattr(order, 'token') else TokenType.YES,
                token_id=order.token_id if hasattr(order, 'token_id') else "",
            )

        # Trade row is written by TradingEngine for every fill (paper/live/limit)
        # so we don't mirror here — doing so would double-count in the `trades` table.

        self.trade_count += 1
        log.info("paper.trade", order_id=order_id, side=order.side.value,
                 size=order.size, price=order.price, balance=self.balance)

        return OrderResult(
            order_id=order_id,
            market_id=order.market_id,
            status="paper",
            filled_size=order.size,
            filled_price=order.price,
            is_paper=True,
        )


    async def check_fills(self, current_prices: dict[str, float]) -> list[OrderResult]:
        """Check pending limit orders for simulated fills.

        Fill probability depends on distance from midpoint:
        closer to midpoint = lower fill probability.
        """
        filled: list[OrderResult] = []
        remaining: list[tuple[Order, str]] = []

        for order, order_id in self.pending_orders:
            market_price = current_prices.get(order.market_id)
            if market_price is None:
                remaining.append((order, order_id))
                continue

            # Simulate fill: BUY fills if market price drops to our limit
            # SELL fills if market price rises to our limit
            should_fill = False
            if order.side == OrderSide.BUY and market_price <= order.price:
                should_fill = True
            elif order.side == OrderSide.SELL and market_price >= order.price:
                should_fill = True

            # Add probabilistic element: closer to midpoint = less likely
            if should_fill:
                # 70-100% fill probability based on price distance
                fill_prob = 0.7 + random.random() * 0.3
                if random.random() < fill_prob:
                    result = await self.execute(order)
                    result.order_id = order_id
                    result.status = "filled"
                    filled.append(result)
                    log.info("paper.limit_filled", order_id=order_id, price=order.price)
                    continue

            remaining.append((order, order_id))

        self.pending_orders = remaining
        return filled

    def submit_limit_order(self, order: Order) -> OrderResult:
        """Queue a limit order for later fill checking."""
        order_id = f"PAPER-LMT-{uuid.uuid4().hex[:12]}"
        self.pending_orders.append((order, order_id))
        log.info("paper.limit_queued", order_id=order_id, side=order.side.value,
                 price=order.price, size=order.size)
        return OrderResult(
            order_id=order_id,
            market_id=order.market_id,
            status="pending",
            is_paper=True,
        )

    async def cancel_expired(self, ttl_seconds: int = 300) -> int:
        """Cancel pending limit orders older than TTL. Returns count cancelled."""
        # For paper trading, just clear all pending orders (no timestamp tracking needed for v1)
        count = len(self.pending_orders)
        if count > 0:
            self.pending_orders.clear()
            log.info("paper.limit_expired", count=count)
        return count

    @property
    def total_value(self) -> float:
        position_value = sum(p.size * p.current_price for p in self.positions.values())
        return self.balance + position_value

    @property
    def pnl(self) -> float:
        return self.total_value - self.initial_balance
