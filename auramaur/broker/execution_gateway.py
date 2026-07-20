"""ExecutionGateway — the single mechanical path from an approved trade to a
recorded fill.

The canonical entry tail historically lived in
``TradingEngine._build_and_place_order`` and was reimplemented, divergently,
inside every standalone strategy. This component extracts that tail so the
strategies (and exits) route through one place and the duplication — the source
of token/accounting bugs — is deleted.

The gateway changes *who* calls the money path, never the path itself: routing,
the ``place_order`` triple-gate (kill-switch / geoblock / live-enabled), the
paper fork, and ``record_fill`` all keep their existing behavior. Risk
evaluation stays with the caller — the caller passes the already risk-approved
``size_dollars`` and ``force_paper`` on the intent.

``submit`` runs the single-leg path. ``submit_paired`` runs a both-or-nothing
pair (the arb pillars): it builds BOTH orders before placing EITHER, places A
then B, and unwinds a live-pending A if B fails — preserving the atomicity a
per-leg ``submit`` could not.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

import structlog

from auramaur.broker.router import SmartOrderRouter, UnmarketableSignal
from auramaur.db.database import Database
from auramaur.exchange.models import Fill, Market, Order, OrderResult, OrderSide, Signal
from auramaur.exchange.protocols import ExchangeClient
from auramaur.monitoring.display import show_order, show_order_dropped
from auramaur.research.polymarket_strategies import DecisionTracker
from auramaur.strategy.signals import taker_fee_rate
from config.settings import Settings

log = structlog.get_logger()

# Statuses that count as "the order left the building" (recorded / tracked).
_OK_STATUSES = ("filled", "paper", "partial", "pending")
_FILLED_STATUSES = ("filled", "paper", "partial")


@dataclass
class TradeIntent:
    """A risk-approved decision to trade, handed to the gateway to execute.

    ``size_dollars`` is the caller's risk-approved position size and
    ``force_paper`` carries the graduation ladder's (and the strategy's own
    ``cfg.paper``) downgrade to dry-run.
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
    order: Order | None = None
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit(self, intent: TradeIntent) -> ExecutionResult:
        """Build an order (via router or direct) and place it, recording the
        fill. Behavior-identical to the former
        ``TradingEngine._build_and_place_order``.
        """
        is_live = self.settings.is_live and not intent.force_paper
        order = await self._build_order(
            intent.signal, intent.market, intent.size_dollars, is_live,
            exchange=self.exchange, router=self.router,
        )
        if order is None:
            return ExecutionResult(status="skipped", reason="not submitted")
        capped = await self._exceeds_market_cap(order, is_live=is_live)
        if capped is not None:
            return ExecutionResult(status="skipped", reason=capped)
        await self._capture_decision(intent, order)
        return await self._place_and_record(
            order, strategy_source=intent.signal.strategy_source,
            signal_id=getattr(intent.signal, "id", None),
            exchange=self.exchange, exchange_name=self.exchange_name)

    async def _exceeds_market_cap(self, order: Order, *, is_live: bool) -> str | None:
        """Aggregate per-(market, token) stake cap. Returns a skip-reason string
        when this BUY would push TOTAL exposure to a market SIDE past the
        documented ceiling, else None.

        The risk manager's max_stake check only sees a single order, so the bot
        could STACK sub-cap entries into an over-cap position (legacy directional
        favorites reached ~$90 across stacked orders, each individually under the
        limit). This guard runs at the layer where the YES/NO token is finally
        known, so it scopes by (market, token): a side's existing holdings plus
        this order can't exceed the cap. Scoping by token, not market, is
        deliberate — internal arb legitimately holds YES *and* NO of one market,
        so a market-wide sum would false-block the opposite leg.

        Only BUY entries are bounded (a SELL reduces exposure). Scoped to the
        order's own mode (paper vs live) so a paper add isn't blocked by a live
        holding and vice-versa.
        """
        if order.side != OrderSide.BUY:
            return None
        cap = getattr(self.settings.risk, "max_stake_abs_ceiling", 25.0)
        is_paper_flag = 0 if is_live else 1
        try:
            row = await self.db.fetchone(
                "SELECT COALESCE(SUM(size * avg_cost), 0) AS held "
                "FROM cost_basis WHERE market_id = ? AND UPPER(token) = UPPER(?) "
                "AND is_paper = ? AND size > 0",
                (order.market_id, order.token.value, is_paper_flag),
            )
        except Exception as e:
            # This is the only guard that sees aggregate exposure. The per-order
            # risk check cannot protect against stacked entries, so an unknown
            # aggregate must never become permission to place another live BUY.
            log.error("gateway.market_cap_read_failed",
                      market_id=order.market_id, token=order.token.value,
                      is_live=is_live, error=str(e))
            return "market_cap: aggregate exposure unavailable"
        held = float(row["held"]) if row else 0.0
        proposed = order.size * order.price
        if held + proposed > cap + 1e-9:
            reason = (
                f"market_cap: ${held:.2f} held + ${proposed:.2f} new on "
                f"{order.market_id}/{order.token.value} exceeds ${cap:.2f}"
            )
            log.info(
                "gateway.market_cap_block",
                market_id=order.market_id,
                token=order.token.value,
                held=round(held, 2),
                proposed=round(proposed, 2),
                cap=cap,
            )
            return reason
        return None

    async def submit_exit(
        self,
        order: Order,
        *,
        exchange: ExchangeClient,
        exchange_name: str,
        strategy_source: str = "exit",
    ) -> ExecutionResult:
        """Place a PREBUILT exit order and record it.

        Exits skip risk (the portfolio monitor already decided) and skip routing
        (the caller prices the SELL against the live bid). The gateway places it
        and writes the pending trades-mirror — so the order monitor, which
        finalizes a live exit's fill asynchronously, UPDATEs that row instead of
        inserting an 'order_monitor'-attributed one (and an immediate paper-mode
        fill is recorded here). No double-write: a live exit is pending at
        placement (filled_size 0), so the fill is recorded once, by the monitor.
        """
        if not order.source:
            order.source = strategy_source
        return await self._place_and_record(
            order, strategy_source=strategy_source, signal_id=None,
            exchange=exchange, exchange_name=exchange_name)

    async def _capture_decision(self, intent: TradeIntent, order: Order) -> None:
        """Persist the immutable executable decision before submission."""
        try:
            coefficient = 0.0 if order.post_only else taker_fee_rate(
                order.exchange or self.exchange_name, intent.market.category)
            fee = order.size * coefficient * order.price * (1.0 - order.price)
            await DecisionTracker(self.db).capture(
                market_id=order.market_id,
                strategy_source=intent.signal.strategy_source or "llm",
                signal_id=getattr(intent.signal, "id", None),
                side=order.side.value,
                fair_probability=intent.signal.claude_prob,
                reference_price=intent.signal.market_prob,
                executable_price=order.price,
                best_bid=None,
                best_ask=None,
                requested_size=order.size * order.price,
                fee_estimate=fee,
            )
        except Exception as exc:
            # Research telemetry may never block an approved trade.
            log.warning("gateway.decision_capture_failed",
                        market_id=order.market_id, error=str(exc))

    async def submit_paired(
        self,
        a: TradeIntent,
        b: TradeIntent,
        *,
        exchange_a: ExchangeClient,
        exchange_name_a: str,
        exchange_b: ExchangeClient,
        exchange_name_b: str,
    ) -> tuple[ExecutionResult, ExecutionResult]:
        """Both-or-nothing paired execution for the arb pillars.

        Builds BOTH orders before placing EITHER (a leg that can't be built
        never leaves the other naked), places A then B, and cancels a
        live-pending A if B fails. ``record_fill`` + the trades-mirror are owned
        here; the caller keeps its signals / portfolio / verdict writes and any
        partial-fill bookkeeping. Legs may live on different exchanges
        (cross-venue), so each carries its own exchange + name. Both intents
        should share the same ``force_paper`` (the pillars paper-force the pair
        together).
        """
        is_live_a = self.settings.is_live and not a.force_paper
        is_live_b = self.settings.is_live and not b.force_paper
        order_a = await self._build_order(
            a.signal, a.market, a.size_dollars, is_live_a,
            exchange=exchange_a, router=None)
        order_b = await self._build_order(
            b.signal, b.market, b.size_dollars, is_live_b,
            exchange=exchange_b, router=None)
        if order_a is None or order_b is None:
            skip = ExecutionResult(status="skipped", reason="leg build failed")
            return skip, ExecutionResult(status="skipped", reason="leg build failed")

        cap_a = await self._exceeds_market_cap(order_a, is_live=is_live_a)
        cap_b = await self._exceeds_market_cap(order_b, is_live=is_live_b)
        if cap_a is not None or cap_b is not None:
            return (
                ExecutionResult(status="skipped", order=order_a,
                                reason=cap_a or "paired leg blocked by market cap"),
                ExecutionResult(status="skipped", order=order_b,
                                reason=cap_b or "paired leg blocked by market cap"),
            )

        await self._capture_decision(a, order_a)
        await self._capture_decision(b, order_b)

        res_a = await self._place_and_record(
            order_a, strategy_source=a.signal.strategy_source,
            signal_id=getattr(a.signal, "id", None),
            exchange=exchange_a, exchange_name=exchange_name_a)
        if res_a.status not in _OK_STATUSES:
            # Leg A rejected — B is never placed, nothing to unwind.
            return res_a, ExecutionResult(status="skipped", reason="leg_a_not_ok")

        res_b = await self._place_and_record(
            order_b, strategy_source=b.signal.strategy_source,
            signal_id=getattr(b.signal, "id", None),
            exchange=exchange_b, exchange_name=exchange_name_b)
        if res_b.status not in _OK_STATUSES:
            # Leg risk: A is in, B failed. Cancel a live-pending A so we don't
            # sit on a naked directional leg (paper / already-filled A can't be
            # cancelled — that's the genuine single-leg the caller logs).
            if (res_a.result is not None and res_a.result.status == "pending"
                    and not res_a.result.is_paper):
                try:
                    await exchange_a.cancel_order(res_a.result.order_id)
                except Exception as e:  # noqa: BLE001 — best-effort unwind
                    log.warning("gateway.paired_leg_a_cancel_failed",
                                order_id=res_a.result.order_id, error=str(e))
        return res_a, res_b

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _build_order(
        self, signal: Signal, market: Market, size_dollars: float, is_live: bool,
        *, exchange: ExchangeClient, router: SmartOrderRouter | None,
    ) -> Order | None:
        """Route (or prepare) an order. Returns ``None`` — having recorded the
        appropriate ``order_build_drops`` cooldown — when the signal is
        unmarketable at the book or the order can't be built.
        """
        try:
            if router:
                order = await router.route(signal, market, size_dollars, is_live)
            elif callable(getattr(type(exchange), "prepare_executable_order", None)):
                order = await exchange.prepare_executable_order(
                    signal, market, size_dollars, is_live)
            else:
                order = exchange.prepare_order(signal, market, size_dollars, is_live)
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
            return None

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
            return None

        if not order.source:
            order.source = signal.strategy_source or "llm"
        return order

    async def _place_and_record(
        self, order: Order, *, strategy_source: str, signal_id,
        exchange: ExchangeClient, exchange_name: str,
    ) -> ExecutionResult:
        """Place a built order, then log slippage, record the fill, and mirror
        to ``trades``. Shared by the single-leg, paired, and exit paths.
        """
        result = await exchange.place_order(order)
        show_order(result.status, result.order_id, order.side.value, order.size, order.price, result.is_paper, exchange=exchange_name, error_message=result.error_message, market_id=order.market_id)
        return await self._record_result(
            order, result, strategy_source=strategy_source,
            signal_id=signal_id, exchange_name=exchange_name)

    async def record_external_fill(
        self, order: Order, result: OrderResult, *,
        strategy_source: str, exchange_name: str,
    ) -> ExecutionResult:
        """Record a fill for an order placed OUTSIDE the gateway.

        The arb scanner places its legs CONCURRENTLY (asyncio.gather) to minimize
        the leg-risk window, so it can't go through submit_*; this gives those
        already-placed legs the same recording invariant — slippage, record_fill
        (paper/filled), and the pending trades-mirror the order monitor later
        UPDATEs for live fills. No double-record: a live leg is pending at
        placement (filled_size 0), so its fill is recorded once, by the monitor.
        """
        return await self._record_result(
            order, result, strategy_source=strategy_source,
            signal_id=None, exchange_name=exchange_name)

    async def place_legs(
        self,
        legs: list[tuple[Order, ExchangeClient, str]],
        *,
        strategy_source: str,
        concurrent: bool = True,
        show: bool = False,
    ) -> list[tuple[OrderResult, ExecutionResult]]:
        """Place multiple already-built legs through the gateway, then record each.

        The single owned entry point for multi-leg flows that must place directly
        rather than via ``submit_paired``'s A-then-B atomicity: arb legs placed
        CONCURRENTLY (asyncio.gather, to minimize the leg-risk window) or
        SEQUENTIALLY for same-exchange pairs. Each placed leg is recorded with the
        same invariant as ``record_external_fill`` (slippage, record_fill, the
        pending trades-mirror the monitor finalizes for live fills). Returns the
        raw ``OrderResult`` alongside the ``ExecutionResult`` per leg so the caller
        keeps its own half-fill / rollback / inventory logic. ``legs`` items are
        ``(order, exchange_client, exchange_name)``.
        """
        # External multi-leg paths bypass ``submit`` but are still entries.
        # Check every BUY before any leg leaves the building. When two legs add
        # the same market/token in one batch, reserve earlier legs locally so
        # the batch cannot collectively exceed the cap.
        reservations: dict[tuple[str, str, int], float] = {}
        for order, _client, _exchange_name in legs:
            if order.side != OrderSide.BUY:
                continue
            is_live = not order.dry_run
            blocked = await self._exceeds_market_cap(order, is_live=is_live)
            key = (order.market_id, order.token.value, 0 if is_live else 1)
            proposed = order.size * order.price
            cap = getattr(self.settings.risk, "max_stake_abs_ceiling", 25.0)
            if blocked is None and reservations.get(key, 0.0) + proposed > cap + 1e-9:
                blocked = "market_cap: multi-leg batch exceeds aggregate ceiling"
            if blocked is not None:
                skipped = ExecutionResult(status="skipped", order=order, reason=blocked)
                rejected = OrderResult(order_id="MARKET_CAP", market_id=order.market_id,
                                       status="rejected", is_paper=order.dry_run,
                                       error_message=blocked)
                return [(rejected, skipped) for order, _client, _name in legs]
            reservations[key] = reservations.get(key, 0.0) + proposed

        if concurrent:
            results = await asyncio.gather(
                *(client.place_order(order) for order, client, _ in legs))
        else:
            results = [await client.place_order(order) for order, client, _ in legs]
        out: list[tuple[OrderResult, ExecutionResult]] = []
        for (order, _client, exchange_name), result in zip(legs, results):
            if show:
                show_order(result.status, result.order_id, order.side.value,
                           order.size, order.price, result.is_paper,
                           exchange=exchange_name,
                           error_message=result.error_message,
                           market_id=order.market_id)
            exec_res = await self.record_external_fill(
                order, result, strategy_source=strategy_source,
                exchange_name=exchange_name)
            out.append((result, exec_res))
        return out

    async def place_quote_pair(
        self, bid_order: Order, ask_order: Order, *,
        exchange: ExchangeClient,
    ) -> tuple[OrderResult, OrderResult]:
        """Place a two-sided maker quote (bid then ask) through the gateway.

        The market maker's owned placement entry point. The MM has already built
        fully-priced ``post_only`` orders (source stamped) and needs BOTH raw
        ``OrderResult``s back to run its own inventory / pending-order /
        partial-leg-cancel bookkeeping, so this places SEQUENTIALLY (matching the
        MM's order, NOT both-or-nothing — the MM owns one-legged cleanup) and
        returns the pair. Recording stays with the order monitor (orders carry
        ``source="market_maker"``); the gateway owns the placement so no strategy
        calls ``exchange.place_order`` directly.
        """
        bid_result = await exchange.place_order(bid_order)
        ask_result = await exchange.place_order(ask_order)
        return bid_result, ask_result

    async def _record_result(
        self, order: Order, result: OrderResult, *,
        strategy_source: str, signal_id, exchange_name: str,
    ) -> ExecutionResult:
        """Post-placement recording shared by _place_and_record (single-leg /
        paired / exit) and record_external_fill (the concurrently-placed arb
        legs): API-error cooldown, slippage, record_fill, and the trades-mirror.
        """
        # Cooldown on API errors — retry in 30 min, not every cycle.
        # Each best-effort write below runs in its own transaction(): a bare
        # commit() on the shared connection would land whatever half-open
        # transaction ANOTHER task has in flight (db-contention plan, Phase 5).
        if result.status == "rejected" and result.order_id == "ERROR":
            try:
                async with self.db.transaction():
                    await self.db.execute(
                        """INSERT OR REPLACE INTO order_build_drops
                           (market_id, blocked_until, reason)
                           VALUES (?, datetime('now', '+30 minutes'), ?)""",
                        (order.market_id, "place_order API error"),
                    )
            except Exception:
                pass

        # Log slippage only for actual executions.  Live pending orders echo
        # the limit price but have not filled yet.
        if result.status in _FILLED_STATUSES and result.filled_price > 0:
            slippage_bps = (result.filled_price - order.price) / order.price * 10000
            if order.side == OrderSide.SELL:
                slippage_bps = -slippage_bps  # For sells, lower fill = worse
            try:
                async with self.db.transaction():
                    await self.db.execute(
                        """INSERT INTO slippage_log (market_id, exchange, side, expected_price, filled_price, slippage_bps, size, order_type)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (order.market_id, order.exchange or exchange_name, order.side.value,
                         order.price, result.filled_price, round(slippage_bps, 2), order.size,
                         order.order_type.value if hasattr(order, 'order_type') else 'limit'),
                    )
            except Exception:
                pass

        fill_size = result.filled_size if result.filled_size > 0 else order.size
        fill_price = result.filled_price if result.filled_price > 0 else order.price

        # Record P&L only for actual executions.  Pending live orders are
        # mirrored to trades below, then finalized by the order monitor.
        recorded_fill: Fill | None = None
        if result.status in _FILLED_STATUSES and result.filled_size > 0:
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
                try:
                    await self.pnl_tracker.record_fill(fill)
                    recorded_fill = fill
                except Exception as e:
                    # The order already executed. Never turn a persistence fault
                    # into an apparent placement failure that callers may retry.
                    log.critical(
                        "gateway.fill_record_failed",
                        order_id=result.order_id,
                        market_id=order.market_id,
                        error=str(e),
                    )

        if result.status in _OK_STATUSES:
            # Mirror into legacy `trades` table so the CLI stats view,
            # order monitor, and holding-period lookups stay in sync.
            # PnLTracker writes authoritative execution rows to `fills`.
            try:
                trade_status = "filled" if result.status == "paper" else result.status
                async with self.db.transaction():
                    await self.db.execute(
                        """INSERT INTO trades
                           (market_id, signal_id, side, size, price, is_paper,
                            order_id, status, kelly_fraction, exchange, strategy_source)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            order.market_id,
                            signal_id,
                            order.side.value,
                            fill_size,
                            fill_price,
                            1 if result.is_paper else 0,
                            result.order_id,
                            trade_status,
                            None,
                            order.exchange or exchange_name,
                            strategy_source,
                        ),
                    )
            except Exception as e:
                log.debug("engine.trade_mirror_error", error=str(e))

        return ExecutionResult(
            status=result.status, order=order, result=result, fill=recorded_fill,
            reason=result.error_message or "",
        )
