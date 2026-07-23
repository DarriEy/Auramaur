"""Order-lifecycle services — extracted from AuramaurBot (Phase 5 split).

Pure structural move: the order monitor (pending-fill finalization + TTL/expiry),
orphaned-pending-trade reconciliation, and the resolution checker live here as
OrderMonitorMixin, mixed into AuramaurBot. Behavior is unchanged — the methods
operate on the bot's self; run() (in bot.py) spawns these loops via self.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

from auramaur.exchange.models import Fill
from auramaur.monitoring.display import console

if TYPE_CHECKING:
    from auramaur.exchange.client import PolymarketClient
    from auramaur.exchange.paper import PaperTrader
    from auramaur.exchange.protocols import ExchangeClient, MarketDiscovery
    from auramaur.strategy.resolution_tracker import ResolutionTracker

log = structlog.get_logger()


class OrderMonitorMixin:
    """Order-lifecycle + resolution services for AuramaurBot (see module docstring)."""

    async def _task_order_monitor(self) -> None:
        """Monitor pending limit orders for fills and expiry."""
        from datetime import datetime, timezone

        paper: PaperTrader = self._components.paper
        primary_exchange: PolymarketClient = self._components.exchange
        exchanges: dict[str, ExchangeClient] = self._components.get("exchanges", {})
        discovery: MarketDiscovery = self._components.discovery
        ttl = self.settings.execution.limit_order_ttl_seconds

        # Reconcile orphaned live orders into _live_pending so the TTL-cancel
        # below can reap them and release their locked collateral. Runs at
        # startup AND periodically: a one-time startup pass only recovers orphans
        # from a *prior* session, but orders are also orphaned mid-session — a
        # lost cancel during a network blip leaves a resting CLOB order untracked
        # (e.g. cancel-replace whose cancel never lands stacks duplicate BUYs),
        # which then locks collateral until the next restart. A periodic re-pull
        # self-heals those within one interval; reconcile skips already-tracked
        # ids, so it only captures genuine orphans.
        async def _reconcile_orphans() -> None:
            seen: set[int] = set()
            for client in [primary_exchange, *exchanges.values()]:
                if (
                    client is not None
                    and hasattr(client, "reconcile_open_orders")
                    and id(client) not in seen
                ):
                    seen.add(id(client))
                    try:
                        await client.reconcile_open_orders()
                    except Exception as e:
                        log.debug("order_monitor.reconcile_error", error=str(e))

        await _reconcile_orphans()
        reconcile_every = 10  # cycles; loop sleeps 30s => re-pull orphans ~5 min
        cycle = 0

        while self._running:
            try:
                # Periodic orphan re-pull (see _reconcile_orphans above).
                if cycle and cycle % reconcile_every == 0:
                    await _reconcile_orphans()
                cycle += 1
                # Paper order monitoring
                if paper.pending_orders:
                    prices: dict[str, float] = {}
                    for order, _ in paper.pending_orders:
                        market = await discovery.get_market(order.market_id)
                        if market:
                            prices[order.market_id] = market.outcome_yes_price

                    filled = await paper.check_fills(prices)
                    if filled:
                        log.info("order_monitor.fills", count=len(filled))

                    await paper.cancel_expired(ttl)

                # Live order monitoring
                live_clients: list[tuple[str, ExchangeClient]] = []
                seen_client_ids: set[int] = set()
                for name, client in exchanges.items():
                    if client is None or not hasattr(client, "_live_pending"):
                        continue
                    live_clients.append((name, client))
                    seen_client_ids.add(id(client))
                if (
                    primary_exchange is not None
                    and hasattr(primary_exchange, "_live_pending")
                    and id(primary_exchange) not in seen_client_ids
                ):
                    live_clients.append(("polymarket", primary_exchange))

                for exchange_name, live_exchange in live_clients:
                    pending = getattr(live_exchange, "_live_pending", {})
                    for order_id in list(pending.keys()):
                        try:
                            order = pending.get(order_id)
                            result = await live_exchange.get_order_status(order_id)
                            if result.status in ("filled", "cancelled", "expired", "rejected"):
                                if result.status == "filled" and order is not None and result.filled_size > 0:
                                    try:
                                        fill = Fill(
                                            order_id=order_id,
                                            market_id=order.market_id,
                                            token_id=order.token_id,
                                            side=order.side,
                                            token=order.token,
                                            size=result.filled_size,
                                            price=result.filled_price if result.filled_price > 0 else order.price,
                                            is_paper=False,
                                        )
                                        pnl_tracker = self._components.pnl_tracker
                                        if pnl_tracker:
                                            await pnl_tracker.record_fill(fill)
                                    except Exception as e:
                                        log.error(
                                            "order_monitor.fill_record_error",
                                            exchange=exchange_name,
                                            order_id=order_id,
                                            error=str(e),
                                        )

                                db = self._components.db
                                if db and order is not None:
                                    try:
                                        price = result.filled_price if result.filled_price > 0 else order.price
                                        size = result.filled_size if result.filled_size > 0 else order.size
                                        cur = await db.execute(
                                            """UPDATE trades
                                               SET status = ?, size = ?, price = ?
                                               WHERE order_id = ?""",
                                            (result.status, size, price, order_id),
                                        )
                                        # Only INSERT a fallback row for an actual
                                        # EXECUTION. The UPDATE above already marks a
                                        # gateway-placed order's pre-written row with
                                        # its terminal status (cancelled/expired/...).
                                        # When rowcount==0 the order had no pre-written
                                        # row (placed outside the gateway) — but a
                                        # cancelled/expired/rejected order never traded,
                                        # so inserting a row would fabricate a phantom
                                        # "BUY" at the full quoted size that never
                                        # filled. The market maker cancels/expires the
                                        # vast majority of its post-only quotes, so this
                                        # was flooding the trades table (~1.6k/wk) and
                                        # corrupting strategy attribution. A non-fill
                                        # terminal status with no pre-existing row has
                                        # nothing to record.
                                        if (
                                            getattr(cur, "rowcount", 0) == 0
                                            and result.status == "filled"
                                        ):
                                            await db.execute(
                                                """INSERT INTO trades
                                                   (market_id, side, size, price, is_paper,
                                                    order_id, status, exchange, strategy_source)
                                                   VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?)""",
                                                (
                                                    order.market_id,
                                                    order.side.value,
                                                    size,
                                                    price,
                                                    order_id,
                                                    result.status,
                                                    order.exchange or exchange_name,
                                                    # Attribute to the order's own source (market_maker,
                                                    # exit, ...) when known — only orders placed OUTSIDE
                                                    # the gateway reach this fallback INSERT (gateway
                                                    # paths pre-write a row the UPDATE above hits), so
                                                    # 'order_monitor' was masking the real strategy.
                                                    getattr(order, "source", None) or "order_monitor",
                                                ),
                                            )
                                        await db.commit()
                                    except Exception as e:
                                        log.debug(
                                            "order_monitor.trade_update_error",
                                            exchange=exchange_name,
                                            order_id=order_id,
                                            error=str(e),
                                        )

                                pending.pop(order_id, None)
                                self._clear_exit_suppression(exchange_name, order, result.status)
                                log.info(
                                    "order_monitor.live_terminal",
                                    exchange=exchange_name,
                                    order_id=order_id,
                                    status=result.status,
                                    filled_size=result.filled_size,
                                )
                            elif order is not None:
                                # Still resting. Live limit orders never auto-expire
                                # on-chain, so an unfilled GTC order locks its
                                # collateral indefinitely. Mirror the paper TTL: a
                                # live order older than limit_order_ttl_seconds is
                                # cancelled to release balance for fresh signals.
                                ts = order.created_at
                                if ts.tzinfo is None:
                                    ts = ts.replace(tzinfo=timezone.utc)
                                age = (datetime.now(timezone.utc) - ts).total_seconds()
                                if age > ttl:
                                    cancelled = await live_exchange.cancel_order(order_id)
                                    if not cancelled:
                                        # Cancel can race a fill — keep the order
                                        # tracked so the next status poll resolves
                                        # it instead of dropping it on the floor.
                                        log.warning(
                                            "order_monitor.ttl_cancel_failed",
                                            exchange=exchange_name,
                                            order_id=order_id,
                                            age_seconds=round(age),
                                        )
                                        continue
                                    pending.pop(order_id, None)
                                    self._clear_exit_suppression(exchange_name, order, "ttl_cancelled")
                                    # Without this status write the trades row
                                    # stays 'pending' forever even though the
                                    # collateral was released (#94).
                                    db = self._components.db
                                    if db is not None:
                                        try:
                                            await db.execute(
                                                "UPDATE trades SET status = 'cancelled' WHERE order_id = ?",
                                                (order_id,),
                                            )
                                            await db.commit()
                                        except Exception as e:
                                            log.debug(
                                                "order_monitor.ttl_db_error",
                                                order_id=order_id,
                                                error=str(e),
                                            )
                                    # Reconciled orphans carry the CLOB condition
                                    # hash as market_id — resolve it to the real
                                    # market id so the terminal line is readable.
                                    display_market = order.market_id
                                    if display_market.startswith("0x") and db is not None:
                                        try:
                                            row = await db.fetchone(
                                                "SELECT id FROM markets WHERE condition_id = ?",
                                                (display_market,),
                                            )
                                            if row:
                                                display_market = str(row["id"])
                                        except Exception:
                                            pass
                                    from auramaur.monitoring.display import show_order_unfilled
                                    show_order_unfilled(
                                        order.side.value, order.size, order.price,
                                        age, exchange=exchange_name,
                                        market_id=display_market,
                                        source=getattr(order, "source", ""),
                                    )
                                    log.info(
                                        "order_monitor.live_ttl_cancel",
                                        exchange=exchange_name,
                                        order_id=order_id,
                                        age_seconds=round(age),
                                        cancelled=cancelled,
                                    )
                        except Exception as e:
                            log.debug(
                                "order_monitor.live_poll_error",
                                exchange=exchange_name,
                                order_id=order_id,
                                error=str(e),
                            )
            except Exception as e:
                log.debug("order_monitor.error", error=str(e))

            try:
                await self._reconcile_orphaned_pending_trades(live_clients)
            except Exception as e:
                log.debug("order_monitor.orphan_sweep_error", error=str(e))

            # Heartbeat for the loop watchdog: this coroutine ticking proves
            # the event loop is alive. A blocking sync call anywhere in the
            # loop silences this beat, and the watchdog THREAD raises the
            # alarm (the 2026-06-10 freeze ran 84 minutes with zero output).
            if self._watchdog is not None:
                self._watchdog.beat()

            # Private Polymarket WS events wake reconciliation immediately;
            # timeout preserves the 30-second polling fallback.
            wake = getattr(primary_exchange, "_user_event", None)
            if wake is None:
                await asyncio.sleep(30)
            else:
                try:
                    await asyncio.wait_for(wake.wait(), timeout=30)
                except asyncio.TimeoutError:
                    pass
                wake.clear()

    async def _reconcile_orphaned_pending_trades(self, live_clients) -> None:
        """Resolve live trades rows stuck in status='pending'.

        A row goes orphaned when its order left ``_live_pending`` without a
        terminal DB write — a TTL cancel before that path wrote status back
        (#94), or a restart that dropped the in-memory tracking. The order is
        long gone on-chain (collateral released) but the DB still says
        pending, which poisons anything that reads order history. Ask the
        exchange for the true terminal status and write it back, a few rows
        per pass to keep API load flat.

        ``get_order_status`` maps unknown/aged-out orders to 'cancelled', so
        rows whose orders the CLOB no longer indexes resolve too.
        """
        db = self._components.db
        if db is None or not live_clients:
            return

        tracked: set[str] = set()
        clients_by_name: dict[str, object] = {}
        for name, client in live_clients:
            tracked.update(getattr(client, "_live_pending", {}).keys())
            clients_by_name[name] = client

        rows = await db.fetchall(
            """SELECT order_id, exchange FROM trades
               WHERE is_paper = 0 AND status = 'pending'
                 AND order_id IS NOT NULL AND order_id != ''
                 AND timestamp < datetime('now', '-10 minutes')
               ORDER BY timestamp ASC LIMIT 25"""
        )
        # Placeholder ids from failed/odd submissions — not real orders. Asking
        # the exchange about them 400s on every pass forever (the 'unknown'
        # kalshi row did exactly that); mark them terminal instead.
        sentinel_ids = {"unknown", "ERROR", "BLOCKED", "INSUFFICIENT_BALANCE",
                        "SKIP_DUP", "POST_ONLY_REJECTED"}
        # Resolve every terminal status over the network FIRST, then apply
        # the UPDATEs in one tight write span — get_order_status must never
        # be awaited while the shared connection holds an open transaction.
        updates: list[tuple[str, str]] = []  # (new_status, order_id)
        for row in rows:
            order_id = row["order_id"]
            if order_id in tracked or order_id.startswith("PAPER"):
                continue
            if order_id in sentinel_ids:
                updates.append(("error", order_id))
                continue
            client = clients_by_name.get(row["exchange"] or "polymarket")
            if client is None or not hasattr(client, "get_order_status"):
                continue
            try:
                result = await client.get_order_status(order_id)
            except Exception:
                continue
            if result.status in ("filled", "cancelled", "expired", "rejected"):
                updates.append((result.status, order_id))
            if len(updates) >= 5:
                break
        if updates:
            for new_status, order_id in updates:
                if new_status == "error":
                    await db.execute(
                        "UPDATE trades SET status = 'error' WHERE order_id = ? AND status = 'pending'",
                        (order_id,),
                    )
                else:
                    await db.execute(
                        "UPDATE trades SET status = ? WHERE order_id = ?",
                        (new_status, order_id),
                    )
            await db.commit()
            log.info("order_monitor.orphans_reconciled", count=len(updates))

    async def _task_resolution_checker(self) -> None:
        """Poll for resolved markets and record calibration outcomes.

        Delegates to ResolutionTracker which handles multi-exchange
        resolution detection, calibration updates, and position settlement.
        """
        tracker: ResolutionTracker = self._components.resolution_tracker

        while self._running:
            if await self._check_kill_switch():
                return
            try:
                resolved = await tracker.check_resolutions()

                # Kalshi settlements sweep: the venue's settlements feed is
                # the only reliable booking source there — the syncer drops
                # settled positions from `portfolio` before the Gamma-style
                # detection path can see them, which is how Kalshi realized
                # P&L went entirely unrecorded until 2026-06-12.
                kalshi = (self._components.exchanges or {}).get("kalshi")
                if kalshi is not None:
                    try:
                        from auramaur.broker.kalshi_settlements import (
                            sweep_kalshi_settlements,
                        )
                        booked = await sweep_kalshi_settlements(
                            self._components.db, kalshi)
                        booked_ok = [b for b in booked if b.get("booked")]
                        if booked_ok:
                            resolved += len(booked_ok)
                            log.info("kalshi_settlements.swept",
                                     booked=len(booked_ok))
                    except Exception as e:
                        log.warning("kalshi_settlements.sweep_error",
                                    error=str(e))

                if resolved > 0:
                    from datetime import datetime, timezone
                    now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
                    console.print(
                        f"  [dim]{now_str}[/] [bold green]RESOLVED[/] {resolved} market(s) — calibration updated"
                    )
                    # Also feed into attribution if available
                    attributor = self._components.attributor
                    if attributor is not None:
                        # Attribution is handled inside _settle_position via
                        # daily_stats; log for visibility only.
                        log.info("resolution.attribution_notified", count=resolved)
            except Exception as e:
                log.debug("resolution_checker.error", error=str(e))
            await asyncio.sleep(1800)  # Every 30 minutes

