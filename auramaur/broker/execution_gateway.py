"""ExecutionGateway — the single mechanical path from an approved trade to a
recorded fill.

Phase 0 of the execution refactor (see the platform-refactor roadmap): the
canonical entry tail historically lived in ``TradingEngine._build_and_place_order``
and was reimplemented, divergently, inside every standalone strategy. This
module extracts that tail VERBATIM into one reusable component so later phases
can route the strategies (and exits) through it and delete the duplication.

The gateway changes *who* calls the money path, never the path itself: routing,
the ``place_order`` triple-gate (kill-switch / geoblock / live-enabled), the
paper fork, and ``record_fill`` all keep their existing behavior. Risk
evaluation stays with the caller in Phase 0 — the caller passes the already
risk-approved ``size_dollars`` and ``force_paper`` on the intent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import structlog

from auramaur.broker.router import SmartOrderRouter, UnmarketableSignal
from auramaur.db.database import Database
from auramaur.exchange.models import Fill, Market, OrderResult, OrderSide, Signal
from auramaur.exchange.protocols import ExchangeClient
from auramaur.monitoring.display import show_order, show_order_dropped
from config.settings import Settings

log = structlog.get_logger()


@dataclass
class TradeIntent:
    """A risk-approved decision to trade, handed to the gateway to execute.

    Phase 0 covers the entry path only; ``size_dollars`` is the caller's
    risk-approved position size and ``force_paper`` carries the graduation
    ladder's downgrade. Later phases extend this with an exit ``kind`` and a
    ``prebuilt_order`` for paths that skip routing.
    """

    signal: Signal
    market: Market
    size_dollars: float
    force_paper: bool = False
    kind: Literal["entry"] = "entry"


@dataclass
class ExecutionResult:
    """The outcome of a gateway submission, exposing every decision point.

    ``result`` is the underlying :class:`OrderResult` for any submitted order
    (including a rejected one) and ``None`` when the trade was skipped before
    submission (unmarketable / build failure) — preserving the legacy
    ``_build_and_place_order`` return contract (``None`` == not submitted).
    """

    status: Literal["filled", "paper", "partial", "pending", "rejected", "skipped"]
    order: object | None = None
    result: OrderResult | None = None
    fill: Fill | None = None
    reason: str = ""


class ExecutionGateway:
    """Routes an approved :class:`TradeIntent` through route → place → record."""

    def __init__(
        self,
        *,
        router: SmartOrderRouter | None,
        exchange: ExchangeClient,
        exchange_name: str,
        settings: Settings,
        db: Database,
        pnl_tracker,
    ) -> None:
        self.router = router
        self.exchange = exchange
        self.exchange_name = exchange_name
        self.settings = settings
        self.db = db
        self.pnl_tracker = pnl_tracker

    async def submit(self, intent: TradeIntent) -> ExecutionResult:
        """Build an order (via router or direct) and place it, recording the
        fill. Behavior-identical to the former
        ``TradingEngine._build_and_place_order``.
        """
        signal = intent.signal
        market = intent.market
        size_dollars = intent.size_dollars

        # ``force_paper`` (graduation ladder) downgrades this ENTRY to dry-run
        # regardless of global live mode — it can only restrict, never arm.
        is_live = self.settings.is_live and not intent.force_paper
        try:
            if self.router:
                order = await self.router.route(signal, market, size_dollars, is_live)
            else:
                order = self.exchange.prepare_order(signal, market, size_dollars, is_live)
        except UnmarketableSignal as skip:
            # No realizable edge at the book. This gate runs before the
            # paper/live split on purpose: paper fills simulate at the
            # reference price, so letting paper books keep these trades
            # would graduate strategies on fills live could never get.
            show_order_dropped(market.id, f"unmarketable: {skip}")
            log.info(
                "engine.entry_unmarketable",
                market_id=market.id,
                strategy=signal.strategy_source,
                reason=str(skip),
            )
            try:
                # Shorter block than build failures: the book can move.
                await self.db.execute(
                    """INSERT OR REPLACE INTO order_build_drops
                       (market_id, blocked_until, reason)
                       VALUES (?, datetime('now', '+30 minutes'), ?)""",
                    (market.id, f"unmarketable: {skip}"),
                )
                await self.db.commit()
            except Exception:
                pass
            return ExecutionResult(status="skipped", reason=f"unmarketable: {skip}")

        if order is None:
            show_order_dropped(market.id, f"order build failed (${size_dollars:.2f} — could not build a valid order)")
            log.warning(
                "engine.order_dropped",
                market_id=market.id,
                size_dollars=size_dollars,
                reason="prepare_order returned None (bad price/token or router rejection)",
            )
            # Sub-minimum sizing is now bumped up in prepare_order, so a None
            # order here is a genuine build failure (bad price/token). Block
            # only briefly so a transient issue can retry next cycle.
            try:
                await self.db.execute(
                    """INSERT OR REPLACE INTO order_build_drops
                       (market_id, blocked_until, reason)
                       VALUES (?, datetime('now', '+2 hours'), ?)""",
                    (market.id, f"order build failed at ${size_dollars:.2f}"),
                )
                await self.db.commit()
            except Exception:
                pass  # Table may not exist yet
            return ExecutionResult(status="skipped", reason="order build failed")

        if not order.source:
            order.source = signal.strategy_source or "llm"
        result = await self.exchange.place_order(order)
        show_order(result.status, result.order_id, order.side.value, order.size, order.price, result.is_paper, exchange=self.exchange_name, error_message=result.error_message, market_id=order.market_id)

        # Cooldown on API errors — retry in 30 min, not every cycle
        if result.status == "rejected" and result.order_id == "ERROR":
            try:
                await self.db.execute(
                    """INSERT OR REPLACE INTO order_build_drops
                       (market_id, blocked_until, reason)
                       VALUES (?, datetime('now', '+30 minutes'), ?)""",
                    (order.market_id, "place_order API error"),
                )
                await self.db.commit()
            except Exception:
                pass

        # Log slippage only for actual executions.  Live pending orders echo
        # the limit price but have not filled yet.
        if result.status in ("filled", "paper", "partial") and result.filled_price > 0:
            slippage_bps = (result.filled_price - order.price) / order.price * 10000
            if order.side == OrderSide.SELL:
                slippage_bps = -slippage_bps  # For sells, lower fill = worse
            try:
                await self.db.execute(
                    """INSERT INTO slippage_log (market_id, exchange, side, expected_price, filled_price, slippage_bps, size, order_type)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (order.market_id, order.exchange or self.exchange_name, order.side.value,
                     order.price, result.filled_price, round(slippage_bps, 2), order.size,
                     order.order_type.value if hasattr(order, 'order_type') else 'limit'),
                )
                await self.db.commit()
            except Exception:
                pass

        fill_size = result.filled_size if result.filled_size > 0 else order.size
        fill_price = result.filled_price if result.filled_price > 0 else order.price

        # Record P&L only for actual executions.  Pending live orders are
        # mirrored to trades below, then finalized by the order monitor.
        recorded_fill: Fill | None = None
        if result.status in ("filled", "paper", "partial") and result.filled_size > 0:
            fill = Fill(
                order_id=result.order_id,
                market_id=order.market_id,
                token_id=order.token_id,
                side=order.side,
                token=order.token,
                size=result.filled_size,
                price=fill_price,
                is_paper=result.is_paper,
            )
            if self.pnl_tracker:
                await self.pnl_tracker.record_fill(fill)
                recorded_fill = fill

        if result.status in ("filled", "paper", "partial", "pending"):
            # Mirror into legacy `trades` table so the CLI stats view,
            # order monitor, and holding-period lookups stay in sync.
            # PnLTracker writes authoritative execution rows to `fills`.
            try:
                trade_status = "filled" if result.status == "paper" else result.status
                await self.db.execute(
                    """INSERT INTO trades
                       (market_id, signal_id, side, size, price, is_paper,
                        order_id, status, kelly_fraction, exchange, strategy_source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        order.market_id,
                        getattr(signal, "id", None),
                        order.side.value,
                        fill_size,
                        fill_price,
                        1 if result.is_paper else 0,
                        result.order_id,
                        trade_status,
                        None,
                        order.exchange or self.exchange_name,
                        signal.strategy_source,
                    ),
                )
                await self.db.commit()
            except Exception as e:
                log.debug("engine.trade_mirror_error", error=str(e))

        return ExecutionResult(
            status=result.status, order=order, result=result, fill=recorded_fill,
            reason=result.error_message or "",
        )
