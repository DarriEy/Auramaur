"""Paper trading simulator — no real money involved."""

from __future__ import annotations

import uuid

import structlog

from auramaur.db.database import Database
from auramaur.exchange.ibkr_intent import INSTRUMENT_ID_PREFIX
from auramaur.exchange.models import Order, OrderResult, OrderSide, Position, TokenType

log = structlog.get_logger()


class PaperTrader:
    """Simulates order execution for paper trading."""

    # Set False to restore immediate fills for non-marketable paper orders
    # (execution.paper_defer_resting_fills). Kept as an attribute rather than
    # read from settings here so PaperTrader stays constructible without them.
    _defer_resting = True

    def __init__(self, db: Database, initial_balance: float = 1000.0):
        self.db = db
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: dict[str, Position] = {}
        self.trade_count = 0
        self.pending_orders: list[tuple[Order, str]] = []  # (order, order_id)
        # market_id -> (required cost, observed spendable cash). A strategy can
        # revisit the same candidate every few seconds; once cash has rejected
        # it, suppress identical attempts until authoritative cash changes.
        self._cash_blocks: dict[str, tuple[float, float]] = {}

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

        self.balance = await self._compute_balance()
        log.info("paper.state_loaded", balance=self.balance, positions=len(self.positions))

    async def _compute_balance(self) -> float:
        """Spendable paper cash from AUTHORITATIVE state — NOT the old
        ``SUM(trades)`` (which debits every BUY but is never credited on
        resolution, so it drained to ~$0 over months and blocked all paper
        entries). Spendable = initial + realized paper P&L (pnl_ledger) - cost
        tied up in OPEN paper positions (cost_basis). This SELF-HEALS: when a
        position resolves, its cost leaves `open cost` and its payout lands in
        realized P&L, so the cash returns automatically — no resolution-path
        coupling, no `trades` pollution.

        The IBKR instrument namespace is EXCLUDED from both sums. Those books
        are funded by their own paper budgets (``ibkr.etf_paper_budget_usd``,
        per-book ``budget_usd``), not by this wallet, and they started writing
        cost_basis/pnl_ledger rows when they were routed through
        ExecutionGateway for graduation evidence. Counting them here would
        subtract up to several thousand dollars of broker exposure from the
        prediction-market book's spendable cash and silently stop it entering
        — a change to what trades, caused purely by a bookkeeping move.
        ``cost_basis`` has no venue column, so the market_id namespace minted
        by ``auramaur.exchange.ibkr_intent`` is the scope."""
        like = f"{INSTRUMENT_ID_PREFIX}%"
        pnl_row = await self.db.fetchone(
            "SELECT COALESCE(SUM(pnl), 0) AS p FROM pnl_ledger "
            "WHERE is_paper = 1 AND market_id NOT LIKE ?", (like,))
        cost_row = await self.db.fetchone(
            "SELECT COALESCE(SUM(size * avg_cost), 0) AS c FROM cost_basis "
            "WHERE is_paper = 1 AND size > 0 AND market_id NOT LIKE ?", (like,))
        realized = float(pnl_row["p"]) if pnl_row else 0.0
        open_cost = float(cost_row["c"]) if cost_row else 0.0
        return self.initial_balance + realized - open_cost

    async def execute(self, order: Order, *, force: bool = False) -> OrderResult:
        """Simulate order execution.

        A NON-MARKETABLE order is queued rather than filled. Crediting a
        resting bid instantly, at the bid, is a fill live would never have
        granted — and measured 2026-07-28 that was essentially every paper
        maker fill in the system (llm crossed on 14%, every other strategy on
        0%). The evidence layer compensated by stamping them 'synthetic',
        which is uncountable, so maker strategies could never graduate.

        ``force`` bypasses the check and is used by check_fills, which has
        already established trade-through and must not re-queue.
        """
        if (not force and self._defer_resting and order.marketable is False):
            log.info("paper.order_rests", market_id=order.market_id,
                     side=order.side.value, price=order.price,
                     best_bid=order.best_bid, best_ask=order.best_ask,
                     detail="not marketable; awaiting trade-through")
            return self.submit_limit_order(order)
        order_id = f"PAPER-{uuid.uuid4().hex[:12]}"
        cost = order.size * order.price

        # Gate on the freshly-computed spendable balance (self-healing), not the
        # stale incrementally-mutated cache.
        spendable = await self._compute_balance()
        if order.side == OrderSide.BUY and cost > spendable:
            block = (round(cost, 8), round(spendable, 8))
            repeated = self._cash_blocks.get(order.market_id) == block
            self._cash_blocks[order.market_id] = block
            message = (
                f"insufficient paper balance: ${spendable:.2f}, "
                f"requires ${cost:.2f}"
            )
            if repeated:
                log.debug("paper.insufficient_balance_suppressed",
                          market_id=order.market_id, cost=round(cost, 2),
                          balance=round(spendable, 2))
            else:
                log.warning(
                    "paper.insufficient_balance",
                    market_id=order.market_id,
                    cost=round(cost, 2),
                    balance=round(spendable, 2),
                    shortfall=round(cost - spendable, 2),
                )
            return OrderResult(
                order_id="SKIP_CASH" if repeated else order_id,
                market_id=order.market_id,
                status="rejected",
                is_paper=True,
                error_message=message,
            )

        # A deposit, settlement, or smaller resized order unblocks this market.
        self._cash_blocks.pop(order.market_id, None)

        # In-memory display balance (the authoritative gate above is
        # _compute_balance; this cache self-corrects on the next load_state).
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


    async def check_fills(self, current_prices: dict[str, float]) -> list[tuple]:
        """Fill maker orders only after strict trade-through evidence.

        Merely touching the limit is not evidence that our queue position
        executed, so no random fill credit is awarded.

        Returns (result, order) pairs: the caller has to book the fill and
        stamp its evidence, and needs the ORDER for the side/token the ledger
        requires. Returning bare results is why deferred fills used to be
        logged and then dropped.
        """
        filled: list[tuple[OrderResult, Order]] = []
        remaining: list[tuple[Order, str]] = []
        for order, order_id in self.pending_orders:
            market_price = current_prices.get(order.market_id)
            if market_price is None:
                remaining.append((order, order_id))
                continue
            should_fill = (
                order.side == OrderSide.BUY and market_price < order.price
            ) or (
                order.side == OrderSide.SELL and market_price > order.price
            )
            if should_fill:
                result = await self.execute(order, force=True)
                result.order_id = order_id
                # execute() can still REFUSE. ``force`` bypasses only the
                # marketability/resting check (line 102), not the balance gate
                # below it — so a trade-through on a book with insufficient
                # paper cash returns status="rejected", filled_size=0,
                # filled_price=0 and updates no position, balance or cost
                # basis. Stamping "filled" over that booked a size-0/price-0
                # fill AND marked the decision executable with "trade_through"
                # evidence, which graduation.credible_fill_evidence accepts —
                # so orders the paper trader refused counted toward promoting a
                # strategy to live capital.
                #
                # The refusal goes back on `remaining` so the order is not
                # silently dropped mid-pass — before this, the `continue`
                # skipped the re-queue below and the order neither filled nor
                # rested. It does NOT mean the order survives to try again:
                # bot_order_monitor calls `cancel_expired(ttl)` immediately
                # after check_fills in the SAME loop iteration, and
                # cancel_expired ignores its ttl argument entirely
                # (`self.pending_orders.clear()`), so the whole queue is
                # discarded either way. What this branch guarantees is only
                # the thing that matters here: a refusal never leaves as a
                # fill.
                if result.status == "rejected" or result.filled_size <= 0:
                    log.debug("paper.limit_fill_refused", order_id=order_id,
                              status=result.status,
                              reason=(result.error_message or "")[:120])
                    remaining.append((order, order_id))
                    continue
                result.status = "filled"
                filled.append((result, order))
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
