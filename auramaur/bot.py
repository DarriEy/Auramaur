"""Main async orchestrator — runs concurrent tasks."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from datetime import datetime, timezone
from auramaur.killswitch import kill_switch_present
from typing import TYPE_CHECKING

import structlog

# Silence noisy py_clob_client_v2 HTTP error logging (403 geoblock, 404 missing
# order books). Our code already handles these via try/except.
logging.getLogger("py_clob_client_v2.http_helpers.helpers").setLevel(logging.CRITICAL)

from config.settings import Settings
from auramaur.components import Components
from auramaur.db.database import Database
from auramaur.exchange.models import OrderSide, TokenType

if TYPE_CHECKING:
    pass
from auramaur.exchange.protocols import ExchangeClient, MarketDiscovery
from auramaur.monitoring.alerts import AlertManager
from auramaur.monitoring.display import (
    console, show_banner, show_error, show_portfolio, show_startup,
)
from auramaur.monitoring.logger import setup_logging
from auramaur.nlp.cache import NLPCache
from auramaur.nlp.calibration import CalibrationTracker
from auramaur.risk.manager import RiskManager
from auramaur.risk.portfolio import PortfolioTracker
from auramaur.strategy.engine import TradingEngine
from auramaur.strategy.market_maker import MarketMaker
from auramaur.strategy.news_reactor import NewsReactor
from auramaur.bot_exits import ExitExecutionMixin
from auramaur.bot_strategy_tasks import StrategyTaskMixin
from auramaur.bot_arb import ArbExecutionMixin
from auramaur.bot_order_monitor import OrderMonitorMixin
from auramaur.monitoring.heartbeat import run_pillar_once

log = structlog.get_logger()


async def run_ibkr_etf_arms_once(pillars, db=None) -> None:
    """Run every intelligence arm independently; one failure cannot stop peers."""
    for pillar in pillars:
        try:
            if db is not None:
                await run_pillar_once(db, pillar)
            else:
                await pillar.run_once()
        except Exception as e:  # noqa: BLE001 — isolation is the contract
            log.error("ibkr_etf.arm_cycle_error", model_alias=pillar.model_alias,
                      error=str(e), exc_info=True)


class AuramaurBot(
    ExitExecutionMixin, StrategyTaskMixin, ArbExecutionMixin, OrderMonitorMixin
):
    """Main bot orchestrator running concurrent async tasks."""

    def __init__(
        self,
        settings: Settings | None = None,
        db_path: str | None = None,
        exchange_filter: str | None = None,
        hybrid: bool = False,
    ):
        self.settings = settings or Settings()
        self._hybrid = hybrid
        self._running = False
        self._components: Components = Components()
        self._db_path = db_path
        self._exchange_filter = exchange_filter  # If set, only run this exchange
        self._lock_file = None  # File handle kept open for duration
        self._rebalance_cooldowns: dict[str, float] = {}  # kept for reference, allocator handles concentration
        self._exit_failures: set[str] = set()  # Track failed exit sells to avoid spam
        self._exit_pending: set[str] = set()  # Track exits with resting orders
        self._exit_gateway_obj = None  # Lazily built; see _exit_gateway property
        # Track attempted cross-exchange arbs so the scanner doesn't re-execute
        # the same opportunity every 5-minute cycle. Maps arb-key -> expiry ts.
        self._arb_attempts: dict[str, float] = {}
        self._watchdog = None  # LoopWatchdog thread, started with the task set

    def _acquire_db_path(self) -> str:
        """Find an available database slot using file locks.

        If ``auramaur.db`` is already locked by another instance, tries
        ``auramaur_2.db``, ``auramaur_3.db``, etc.
        """
        import fcntl

        if self._db_path:
            # Explicit path — lock it or fail
            lock_path = f"{self._db_path}.lock"
            fh = open(lock_path, "w")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._lock_file = fh
                return self._db_path
            except OSError:
                fh.close()
                raise RuntimeError(f"Database {self._db_path} is already locked by another instance")

        # Auto-detect: try auramaur.db, auramaur_2.db, ...
        for i in range(1, 20):
            db_name = "auramaur.db" if i == 1 else f"auramaur_{i}.db"
            lock_path = f"{db_name}.lock"
            fh = open(lock_path, "w")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._lock_file = fh
                if i > 1:
                    log.info("bot.multi_instance", db=db_name, instance=i)
                return db_name
            except OSError:
                fh.close()
                continue

        raise RuntimeError("Too many Auramaur instances running (max 19)")

    async def _init_components(self) -> None:
        """Initialize all components via the composition root."""
        from auramaur.composition import assemble_components
        self._components = await assemble_components(
            settings=self.settings, db_path=self._acquire_db_path(),
            hybrid=self._hybrid, exchange_filter=self._exchange_filter,
            rebalance_cooldowns=self._rebalance_cooldowns,
        )
        # Name-the-gap gate: wire the mispricing auditor onto the risk manager
        # now that db + analyzer exist.
        rm = self._components.risk_manager
        an = self._components.analyzer
        if rm is not None and an is not None:
            from auramaur.nlp.gap_audit import GapAuditor
            rm.gap_auditor = GapAuditor(self._components.db, an, self.settings)

    def _get_schedule_mode(self) -> str:
        """Return current adaptive schedule mode."""
        from datetime import datetime, timezone

        cfg = self.settings.intervals
        if not cfg.adaptive_enabled:
            return ""

        if self._is_cash_starved():
            return "starved"

        hour_utc = datetime.now(timezone.utc).hour
        if hour_utc in cfg.quiet_hours_utc:
            return "quiet"
        if hour_utc not in cfg.peak_hours_utc:
            return "off_peak"
        return "peak"

    def _adaptive_interval(self, base_seconds: int) -> int:
        """Scale interval based on time of day and capital.

        Peak hours (US market open):  base interval (full speed)
        Off-peak (evening/morning):   base × off_peak_multiplier (default 4x)
        Quiet hours (deep night):     base × quiet_multiplier (default 8x)
        Cash-starved (<$5 total):     2x slowdown (starved cycle is already lean,
                                      but no need to run it as often)
        """
        cfg = self.settings.intervals
        mode = self._get_schedule_mode()

        if mode == "quiet":
            multiplier = cfg.quiet_multiplier
        elif mode == "off_peak":
            multiplier = cfg.off_peak_multiplier
        else:
            multiplier = 1.0

        if self._is_cash_starved():
            multiplier *= 5.0  # Price refresh only — no rush

        return int(base_seconds * multiplier)

    def _is_cash_starved(self) -> bool:
        """Check if both exchanges are too low on cash to open new positions."""
        # Default to starved (0) until portfolio monitor sets the real value
        cash = getattr(self, '_last_known_cash', 0.0)
        return cash < 5.0

    async def _check_kill_switch(self) -> bool:
        if kill_switch_present():
            show_error("KILL SWITCH ACTIVE — halting all trading")
            self._running = False
            alerts = self._components.alerts
            if alerts:
                await alerts.send("KILL SWITCH ACTIVATED — bot halted", level="critical")
            return True
        return False

    async def _task_market_scan(self, engine: TradingEngine, name: str = "") -> None:
        """Periodically scan and store markets."""
        while self._running:
            if await self._check_kill_switch():
                return
            try:
                await engine.scan_and_store_markets()
            except Exception as e:
                show_error(f"Market scan failed ({name}): {e}")
            await asyncio.sleep(self._adaptive_interval(self.settings.intervals.market_scan_seconds))

    async def _task_trading_cycle(self, engine: TradingEngine, name: str = "") -> None:
        """Periodically run trading analysis cycle."""
        # Wait for the portfolio monitor to set _last_known_cash before the
        # first cycle, otherwise we default to 0 and enter starved mode.
        for _ in range(15):
            if getattr(self, '_last_known_cash', 0.0) > 0:
                break
            await asyncio.sleep(2)

        while self._running:
            if await self._check_kill_switch():
                return

            try:
                cash = getattr(self, '_last_known_cash', 0.0)
                await engine.run_cycle(cash_available=cash)
            except Exception as e:
                show_error(f"Trading cycle failed ({name}): {e}")

            # Kalshi-only: concentrated-position rebalance (daily cap, intra-event trim).
            # Position sync + generic exit checks now run in _task_portfolio_monitor
            # for both exchanges — no Kalshi-specific bolt-on required here.
            if name == "kalshi":
                try:
                    await self._rebalance_concentrated_positions(engine)
                except Exception as e:
                    log.debug("kalshi_rebalance.error", error=str(e))

            await asyncio.sleep(self._adaptive_interval(self.settings.intervals.analysis_seconds))

    async def _task_orderbook_recorder(self, engine: TradingEngine, name: str = "") -> None:
        """Persist top-of-book bid/ask (plus one depth level) for active, liquid
        markets into orderbook_snapshots.

        price_history is mid-only, which can't answer "does an edge clear the
        spread"; this captures the real spread so cost-aware research and honest
        paper fills become possible. Read-only on the exchange (places no
        orders). Order books can't be backfilled, so it only captures from when
        it's enabled — hence it runs by default. Throttled and capped to stay
        well within CLOB rate limits and not starve the live trading calls.
        Disable via settings.intervals.orderbook_recorder_enabled = false; tune
        via orderbook_seconds / orderbook_min_liquidity / orderbook_max_markets.
        """
        if not hasattr(engine.exchange, "get_order_book"):
            return
        interval = int(getattr(self.settings.intervals, "orderbook_seconds", 300))
        min_liquidity = float(getattr(self.settings.intervals, "orderbook_min_liquidity", 1000.0))
        max_markets = int(getattr(self.settings.intervals, "orderbook_max_markets", 150))
        pause = float(getattr(self.settings.intervals, "orderbook_call_pause_seconds", 0.25))

        while self._running:
            if await self._check_kill_switch():
                return
            try:
                # Actionable-first targeting: raw liquidity DESC over a
                # stale markets table pinned the recorder to a handful of
                # dead long-horizon books (three 2028 election markets) while
                # the markets the bot actually trades got zero snapshots --
                # decision_marks had NO rows ever (2026-07-20 audit). Rank
                # held positions and recent decisions first, then fresh
                # liquid markets that have not already ended.
                rows = await engine.db.fetchall(
                    """SELECT id, clob_token_yes,
                              (id IN (SELECT market_id FROM portfolio WHERE size > 0)) AS held,
                              (id IN (SELECT market_id FROM decision_snapshots
                                      WHERE observed_at >= datetime('now', '-2 days'))) AS decided
                       FROM markets
                       WHERE active = 1 AND exchange = ? AND clob_token_yes != ''
                         AND (end_date IS NULL OR end_date > datetime('now'))
                         AND (
                             id IN (SELECT market_id FROM portfolio WHERE size > 0)
                             OR id IN (SELECT market_id FROM decision_snapshots
                                       WHERE observed_at >= datetime('now', '-2 days'))
                             OR (liquidity >= ? AND last_updated >= datetime('now', '-1 day'))
                         )
                       ORDER BY held DESC, decided DESC, liquidity DESC
                       LIMIT ?""",
                    (engine.exchange_name or "polymarket", min_liquidity, max_markets),
                )
                recorded = 0
                for row in rows:
                    if not self._running:
                        break
                    market_id, token_id = row[0], row[1]
                    try:
                        book = await engine.exchange.get_order_book(token_id)
                    except Exception as e:
                        log.debug("orderbook_recorder.book_failed", market_id=market_id, error=str(e))
                        await asyncio.sleep(pause)
                        continue
                    bid, ask = book.best_bid, book.best_ask
                    if bid is None or ask is None or ask <= bid:
                        await asyncio.sleep(pause)
                        continue
                    # CLOB levels are worst-first; sort explicitly, never index
                    # bids[0]/asks[0] (see OrderBook docstring).
                    bids = sorted(book.bids, key=lambda lv: lv.price, reverse=True)
                    asks = sorted(book.asks, key=lambda lv: lv.price)
                    await engine.db.execute(
                        "INSERT INTO orderbook_snapshots "
                        "(market_id, token_id, exchange, best_bid, best_ask, bid_size, "
                        "ask_size, mid, bid2, ask2, bid2_size, ask2_size) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            market_id, token_id, engine.exchange_name or "polymarket",
                            bid, ask,
                            bids[0].size if bids else None,
                            asks[0].size if asks else None,
                            (bid + ask) / 2.0,
                            bids[1].price if len(bids) > 1 else None,
                            asks[1].price if len(asks) > 1 else None,
                            bids[1].size if len(bids) > 1 else None,
                            asks[1].size if len(asks) > 1 else None,
                        ),
                    )
                    recorded += 1
                    await asyncio.sleep(pause)
                if recorded:
                    await engine.db.commit()
                    log.debug("orderbook_recorder.swept", exchange=name, recorded=recorded)
            except Exception as e:
                show_error(f"Order-book recorder failed ({name}): {e}")
            await asyncio.sleep(self._adaptive_interval(interval))

    async def _task_portfolio_monitor(self) -> None:
        """Monitor portfolio using the broker syncers for ground truth.

        Runs once per ``portfolio_check_seconds``:
        1. Sync each exchange's positions into the portfolio table.
        2. Aggregate cash across exchanges for adaptive throttling.
        3. Run ``check_exits`` per-exchange and route each exit through
           the correct exchange-specific executor.
        """
        from auramaur.broker.pnl import PnLTracker

        syncers: list = self._components.get("syncers", [])
        pnl_tracker: PnLTracker = self._components.pnl_tracker
        discoveries: dict[str, MarketDiscovery] = self._components.discoveries
        exchanges: dict[str, ExchangeClient] = self._components.get("exchanges", {})
        alerts: AlertManager = self._components.alerts
        portfolio_tracker: PortfolioTracker = self._components.risk_manager.portfolio
        interval = self.settings.intervals.portfolio_check_seconds
        first_tick = True

        while self._running:
            try:
                all_positions = []
                per_exchange_cash: dict[str, float] = {}
                sync_failures: list[str] = []

                for syncer in syncers:
                    name = getattr(syncer, "exchange_name", "polymarket")
                    try:
                        positions_list = await syncer.sync()
                        cash = await syncer.get_cash_balance()
                    except Exception as e:
                        log.debug("portfolio_monitor.sync_error", exchange=name, error=str(e))
                        sync_failures.append(name)
                        continue

                    per_exchange_cash[name] = cash

                    # Polymarket-only: enrich with reconciler-discovered manual buys
                    if name == "polymarket":
                        reconciler_comp = self._components.reconciler
                        if self.settings.is_live and reconciler_comp:
                            try:
                                reconciled = await reconciler_comp.reconcile()
                                repaired = await reconciler_comp.repair_orphaned_ids(reconciled)
                                if repaired:
                                    positions_list = await syncer.sync()
                                live_from_recon = reconciler_comp.to_live_positions(reconciled)
                                known_ids = {p.market_id for p in positions_list}
                                new_positions = [p for p in live_from_recon if p.market_id not in known_ids]
                                if new_positions:
                                    await syncer._merge_new_positions(new_positions)
                                    positions_list.extend(new_positions)
                                    log.info("reconciler.new_positions_merged", count=len(new_positions))
                            except Exception as e:
                                log.debug("reconciler.enrich_error", error=str(e))

                    all_positions.extend(positions_list)

                seen_ids: set[str] = set()
                deduped: list = []
                for pos in all_positions:
                    if pos.market_id not in seen_ids:
                        seen_ids.add(pos.market_id)
                        deduped.append(pos)
                all_positions = deduped

                total_cash = sum(per_exchange_cash.values())
                self._last_known_cash = total_cash
                total_pnl = await pnl_tracker.get_total_pnl(all_positions)
                poly_cash = per_exchange_cash.get("polymarket", total_cash)

                # Feed the drawdown gate: nothing ever wrote
                # daily_stats.peak_balance, so get_drawdown() returned 0.0
                # forever and check_max_drawdown / check_drawdown_heat could
                # never trip (2026-07-20 audit, confirmed critical).
                # Complete ticks only: with any venue missing, "equity" is just
                # the venues that answered, and those partial sums recorded
                # phantom 75-92% drawdowns in daily_stats on 2026-07-21/22.
                if sync_failures:
                    log.debug("drawdown.equity_tick_skipped",
                              failed_venues=sync_failures)
                else:
                    try:
                        equity = total_cash + sum(
                            p.size * p.current_price for p in all_positions)
                        await portfolio_tracker.note_equity(equity)
                    except Exception as e:
                        log.debug("drawdown.note_equity_error", error=str(e))

                if first_tick:
                    first_tick = False
                    try:
                        from auramaur.monitoring.display import (
                            build_category_stats_from_positions,
                            show_category_performance,
                        )
                        accuracy_map: dict[str, float | None] = {}
                        kelly_map: dict[str, float] = {}
                        category_lookup: dict[str, str] = {}
                        attributor = self._components.attributor
                        if attributor:
                            accuracy_map, kelly_map = await attributor.get_accuracy_and_kelly_maps()
                            category_lookup = await attributor.get_category_lookup()
                        cat_stats = build_category_stats_from_positions(
                            all_positions, accuracy_map, kelly_map, category_lookup,
                        )
                        if cat_stats:
                            show_category_performance(cat_stats)
                        # Strategy-books view: per-book open exposure +
                        # realized (live & paper) from the pnl_ledger.
                        from auramaur.monitoring.books import (
                            gather_books, render_books_table,
                        )
                        books = await gather_books(self._components.db)
                        if books:
                            console.print(render_books_table(books))
                    except Exception as e:
                        log.debug("attribution.initial_error", error=str(e))

                # Collateral reserved by resting BUY orders — shown alongside
                # free cash so it stops reading as money randomly vanishing
                # between ticks.
                reserved = 0.0
                for client in exchanges.values():
                    pending = getattr(client, "_live_pending", None)
                    if pending:
                        reserved += sum(
                            o.notional for o in pending.values()
                            if getattr(o, "side", None) == OrderSide.BUY
                        )
                show_portfolio(
                    poly_cash, total_pnl, len(all_positions), 0.0,
                    schedule_mode=self._get_schedule_mode(), reserved=reserved,
                )

                # Per-exchange exit checks + execution
                for name, discovery in discoveries.items():
                    exchange_client = exchanges.get(name)
                    if exchange_client is None:
                        continue
                    try:
                        exit_list = await portfolio_tracker.check_exits(
                            self.settings, discovery, exchange=name,
                        )
                    except Exception as e:
                        log.debug("exit_check.error", exchange=name, error=str(e))
                        continue

                    for pos, reason in exit_list:
                        exit_key = f"exit:{name}:{pos.market_id}"
                        if exit_key in self._exit_failures or exit_key in self._exit_pending:
                            continue
                        log.info(
                            "exit.triggered",
                            exchange=name,
                            market_id=pos.market_id,
                            reason=reason.value,
                            pnl=pos.unrealized_pnl,
                        )
                        try:
                            if name == "polymarket":
                                ok = await self._execute_poly_exit(pos, reason, discovery, exchange_client, alerts)
                            elif name == "kalshi":
                                ok = await self._execute_kalshi_exit(pos, reason, discovery, exchange_client, alerts)
                            elif name == "ibkr":
                                ok = await self._execute_ibkr_exit(pos, reason, discovery, exchange_client, alerts)
                            else:
                                ok = False
                        except Exception as e:
                            log.warning("exit.execute_error", exchange=name, market_id=pos.market_id, error=str(e))
                            ok = False
                        if ok:
                            self._exit_pending.add(exit_key)
                        else:
                            self._exit_failures.add(exit_key)
            except Exception as e:
                log.debug("portfolio_monitor_error", error=str(e))
            await asyncio.sleep(interval)

    async def _task_cache_cleanup(self) -> None:
        """Periodically clean expired NLP cache entries."""
        cache: NLPCache = self._components.cache

        while self._running:
            try:
                await cache.cleanup()
            except Exception:
                pass
            await asyncio.sleep(300)

    async def _task_redemption_check(self) -> None:
        """Periodically check for redeemable Polymarket positions.

        When the Polymarket data-api reports positions in resolved markets
        that can be converted to USDC, print a loud banner so the user
        knows to hit 'Redeem All' in the Polymarket UI. Runs hourly.
        """
        from auramaur.broker.onchain import OnChainRedeemer
        from auramaur.broker.redeemer import (
            fetch_redeemable_positions, summarize_redemptions,
        )
        from auramaur.monitoring.display import console

        proxy = self.settings.polymarket_proxy_address
        if not proxy:
            return  # no proxy configured, can't check

        db: Database = self._components.db
        alerts: AlertManager = self._components.alerts
        redeemer = OnChainRedeemer(self.settings, db)
        # Only auto-redeem winners worth claiming; the replay guard in the
        # redemptions table prevents re-submitting an already-sent condition.
        AUTO_REDEEM_MIN_PAYOUT = 0.50

        # Track the last-notified payout to avoid spamming on every cycle
        last_notified_payout: float = -1.0

        while self._running:
            try:
                positions = await fetch_redeemable_positions(proxy)
                summary = summarize_redemptions(positions)
                payout = summary["payout_now_usdc"]

                gates_open = redeemer._is_live_submission_allowed()

                if summary["redeemable_now"] > 0 and payout > 0:
                    if gates_open:
                        # Auto-submit: claim winners on-chain. The redeemer
                        # enforces the live gates again internally and records
                        # each attempt (replay guard), so this is idempotent.
                        await self._auto_redeem(
                            redeemer, positions, AUTO_REDEEM_MIN_PAYOUT, alerts
                        )
                    elif payout != last_notified_payout:
                        # Gates closed — fall back to the manual-action banner.
                        console.print()
                        console.print(
                            "[bold green]╔══ REDEMPTION AVAILABLE ══╗[/]"
                        )
                        console.print(
                            f"[bold green]║[/] [green]${payout:.2f} USDC[/] ready to "
                            f"redeem on Polymarket — {summary['winning_now']} winning "
                            f"position{'s' if summary['winning_now'] != 1 else ''} "
                            f"(net [green]${summary['net_pnl_now']:+.2f}[/])"
                        )
                        console.print(
                            "[bold green]║[/] [dim]→ https://polymarket.com/portfolio  "
                            "or run:[/] [cyan]auramaur redeem --submit[/]"
                        )
                        console.print(
                            "[bold green]╚══════════════════════════╝[/]"
                        )
                        console.print()
                        last_notified_payout = payout

                # Alert on redemptions that broadcast but never confirmed — these
                # are stuck in the mempool / under-priced and silently lose the
                # payout until manually re-sent.
                await self._alert_stuck_redemptions(db, alerts)
            except Exception as e:
                log.debug("redemption_check.error", error=str(e))

            await asyncio.sleep(3600)  # hourly

    async def _auto_redeem(
        self,
        redeemer,
        positions: list,
        min_payout: float,
        alerts: "AlertManager",
    ) -> None:
        """Submit on-chain redemptions for winning, redeemable positions.

        Assumes the caller has confirmed all live gates are open. Each redeem()
        is idempotent via the redemptions-table replay guard, so re-running
        hourly will not double-submit a condition already sent or confirmed.
        """
        ready = [
            p for p in positions
            if p.redeemable_now and p.is_winner and p.payout >= min_payout
        ]
        for pos in ready:
            try:
                result = await redeemer.redeem(pos, dry_run=False)
            except Exception as e:
                log.error(
                    "redemption.submit_error",
                    condition_id=pos.condition_id[:10],
                    error=str(e)[:200],
                )
                continue

            if result.status in ("submitted", "confirmed"):
                log.info(
                    "redemption.submitted",
                    condition_id=pos.condition_id[:10],
                    title=pos.title[:50],
                    payout=round(pos.payout, 2),
                    status=result.status,
                    tx_hash=result.tx_hash[:12],
                )
                await alerts.send(
                    f"Redemption {result.status}: ${pos.payout:.2f} — "
                    f"{pos.title[:50]} (tx {result.tx_hash[:12]})",
                    level="warning",
                )
            elif result.status not in ("skipped",):
                log.warning(
                    "redemption.not_submitted",
                    condition_id=pos.condition_id[:10],
                    status=result.status,
                    error=result.error[:200],
                )

    async def _alert_stuck_redemptions(self, db: "Database", alerts: "AlertManager") -> None:
        """Warn when a broadcast redemption has not confirmed within 24h."""
        try:
            rows = await db.fetchall(
                """SELECT condition_id, title, tx_hash, submitted_at
                   FROM redemptions
                   WHERE status = 'submitted'
                   AND submitted_at IS NOT NULL
                   AND datetime(submitted_at) < datetime('now', '-1 day')"""
            )
        except Exception as e:
            log.debug("redemption.stuck_query_error", error=str(e))
            return
        for row in rows:
            log.warning(
                "redemption.stuck",
                condition_id=str(row["condition_id"])[:10],
                tx_hash=str(row["tx_hash"])[:12],
                submitted_at=row["submitted_at"],
            )
            await alerts.send(
                f"Redemption stuck >24h (unconfirmed): {str(row['title'])[:50]} "
                f"tx {str(row['tx_hash'])[:12]}. May be under-priced/dropped — "
                f"re-send with `auramaur redeem --submit`.",
                level="critical",
            )

    async def _task_kill_switch_monitor(self) -> None:
        """Rapid kill switch polling."""
        while self._running:
            if await self._check_kill_switch():
                return
            await asyncio.sleep(1)

    async def _run_live_gate(self, *, startup: bool = False) -> None:
        """Evaluate the live-readiness preflight and move the risk manager's
        ``live_entries_blocked`` latch in BOTH directions.

        The latch was previously set once at startup and never cleared, so a
        cold-start BLOCK (equity feed not yet warm, IB Gateway awaiting its
        manual re-login) paper-forced live entries until the next restart —
        and every restart re-blocked.
        """
        from auramaur.monitoring.live_gate import preflight

        rm = self._components.risk_manager
        report = await preflight(
            self.settings, self._components.db,
            tracker=rm.portfolio if rm is not None else None)
        blocked_now = not report.live_allowed
        was_blocked = bool(rm.live_entries_blocked) if rm is not None else False
        if rm is not None:
            rm.live_entries_blocked = blocked_now
        alerts = self._components.alerts

        if blocked_now and (startup or not was_blocked):
            blocked = ", ".join(b.name for b in report.blocks)
            log.error("live_gate.entries_blocked",
                      blocks=[f"{b.name}: {b.detail}" for b in report.blocks])
            console.print(f"  [bold red]LIVE ENTRIES BLOCKED by preflight:[/] "
                          f"{blocked} — exits stay live")
            if alerts is not None:
                await alerts.send(
                    f"LIVE ENTRIES BLOCKED by preflight: {blocked}", level="critical")
        elif not blocked_now and was_blocked:
            log.info("live_gate.entries_unblocked", warnings=len(report.warnings))
            console.print("  [green]live entries unblocked — preflight passes[/]")
            if alerts is not None:
                await alerts.send(
                    "Live entries unblocked — preflight passes", level="warning")
        elif startup:
            log.info("live_gate.passed", warnings=len(report.warnings))
            if report.warnings:
                console.print(f"  [yellow]preflight OK, {len(report.warnings)} warning(s)[/]")

    async def _task_live_gate_monitor(self) -> None:
        """Periodic live-readiness re-check (armed-live runs only)."""
        interval = max(60, int(self.settings.intervals.live_gate_recheck_seconds))
        while self._running:
            await asyncio.sleep(interval)
            try:
                await self._run_live_gate()
            except Exception as e:
                log.error("live_gate.error", error=str(e))

    async def _task_kraken_pillar(self) -> None:
        """Dual-purpose Kraken pillar: treasury (always) + directional (gated).

        First-class loop citizen — syncs Kraken balances, auto-converts idle
        fiat -> USDC, and alerts to refill Polymarket when cash is low. Directional
        spot trading runs only when kraken.directional_enabled (off by default).
        """
        from auramaur.exchange.kraken import KrakenSpotClient
        from auramaur.treasury.kraken_pillar import KrakenPillar
        from auramaur.monitoring.display import console

        client = KrakenSpotClient(self.settings)
        coinbase_client = None
        comparator = None
        if self.settings.coinbase.paper_enabled:
            from auramaur.exchange.coinbase import CoinbasePublicClient
            from auramaur.treasury.coinbase_paper import CoinbasePaperBook
            coinbase_client = CoinbasePublicClient()
            comparator = CoinbasePaperBook(
                self.settings, coinbase_client, self._components.get("db"))
        pillar = KrakenPillar(
            self.settings, client, bot=self, console=console,
            paper_comparator=comparator)
        interval = self.settings.kraken.treasury_interval_seconds
        try:
            while self._running:
                if await self._check_kill_switch():
                    return
                await run_pillar_once(self._components.db, pillar, interval_seconds=interval)
                await asyncio.sleep(interval)
        finally:
            await client.close()
            if coinbase_client is not None:
                await coinbase_client.close()

    async def _task_ibkr_etf_paper(self) -> None:
        """OpenAI intelligence-cap A/B over the paper-only ETF mandate."""
        from auramaur.exchange.ibkr_equity import IBKREquityClient
        from auramaur.nlp.openai_etf import OpenAIETFAnalyzer
        from auramaur.strategy.ibkr_etf_controls import MomentumETFAnalyzer
        from auramaur.strategy.ibkr_etf_paper import IBKRETFPaperPillar

        client = IBKREquityClient(self.settings, force_paper_readonly=True)
        evidence_cache = {}
        analyzers = [OpenAIETFAnalyzer(
            self.settings.openai_api_key, spec.model, spec.effort,
            self.settings.ibkr.etf_openai_timeout_seconds,
            db=self._components.get("db"), model_alias=spec.alias,
            input_cost_per_million=spec.input_cost_per_million,
            output_cost_per_million=spec.output_cost_per_million)
            for spec in self.settings.ibkr.etf_models]
        pillars = [IBKRETFPaperPillar(
            self.settings, client, self._components.get("db"),
            self._components.get("aggregator"), analyzer,
            self._components.get("cache"), model_alias=spec.alias,
            evidence_cache=evidence_cache)
            for spec, analyzer in zip(self.settings.ibkr.etf_models, analyzers)]
        pillars.append(IBKRETFPaperPillar(
            self.settings, client, self._components.get("db"), None,
            MomentumETFAnalyzer(), self._components.get("cache"),
            model_alias="momentum_control"))
        try:
            while self._running:
                if await self._check_kill_switch():
                    return
                evidence_cache.clear()
                await run_ibkr_etf_arms_once(pillars, db=self._components.get("db"))
                await asyncio.sleep(self.settings.ibkr.etf_cycle_seconds)
        finally:
            await client.close()
            for analyzer in analyzers:
                await analyzer.close()

    async def _task_ibkr_multiasset_paper(self) -> None:
        """Six read-only quote feeds with isolated local paper accounting."""
        from auramaur.exchange.ibkr_instruments import IBKRBook
        from auramaur.exchange.ibkr_market_data import IBKRReadOnlyMarketData
        from auramaur.strategy.ibkr_multiasset_paper import IBKRMultiAssetPaperBook

        client = IBKRReadOnlyMarketData(self.settings)
        executor = None
        if self.settings.ibkr.multiasset_execution_enabled:
            from auramaur.exchange.ibkr_multiasset_execution import IBKRMultiAssetExecution
            executor = IBKRMultiAssetExecution(
                self.settings, client, self._components.get("db"))
        rates_provider = None
        if self.settings.fred_api_key:
            from auramaur.data_sources.fred import FREDSource
            from auramaur.research.fx_rates import FredRatesProvider
            rates_provider = FredRatesProvider(
                FREDSource(api_key=self.settings.fred_api_key))
        books = [IBKRMultiAssetPaperBook(
            self.settings, client, self._components.get("db"), book,
            rates_provider=rates_provider, executor=executor)
            for book in IBKRBook
            if self.settings.ibkr.multiasset_books[book.value].enabled]
        from auramaur.strategy.ibkr_multiasset_paper import warn_stranded_positions
        await warn_stranded_positions(
            self._components.get("db"), {b.book.value for b in books})
        try:
            while self._running:
                if await self._check_kill_switch():
                    return
                for book in books:
                    try:
                        await run_pillar_once(
                            self._components.get("db"), book,
                            interval_seconds=self.settings.ibkr.multiasset_cycle_seconds)
                    except Exception as exc:  # noqa: BLE001
                        log.error("ibkr_multiasset.book_cycle_error",
                                  book=book.book.value, error=str(exc),
                                  exc_info=True)
                await asyncio.sleep(self.settings.ibkr.multiasset_cycle_seconds)
        finally:
            await client.close()
            if executor is not None:
                await executor.close()

    async def _task_balance_recorder(self) -> None:
        """Record venue cash to the ``venue_balances`` table so read-only
        consumers (the web dashboard) can show balances without ever holding
        venue credentials. IBKR runs on a slower cadence: its fetch is a real
        gateway session, not a REST call."""
        from auramaur.monitoring.balances import record_venue_balances
        cycle = 0
        while self._running:
            try:
                await record_venue_balances(
                    self._components.db, self.settings,
                    include_ibkr=(cycle % 5 == 0))
            except Exception as e:  # noqa: BLE001 — monitoring must not die
                log.warning("balance_recorder.error", error=str(e))
            cycle += 1
            await asyncio.sleep(60)

    async def _task_momentum_coupling(self) -> None:
        """Fast path: spot->prediction momentum-coupling pillar (gated, detect-only).

        Separate fast cadence + momentum signal, distinct from the LLM loop.
        """
        from auramaur.strategy.momentum_coupling import MomentumCouplingPillar
        from auramaur.monitoring.display import console
        poly_engine = self._components.get("engines", {}).get("polymarket")
        poly_client = poly_engine.exchange if poly_engine else None
        risk_mgr = poly_engine.risk_manager if poly_engine else None
        pillar = MomentumCouplingPillar(self.settings, console=console,
                                        polymarket_client=poly_client,
                                        risk_manager=risk_mgr, bot=self,
                                        db=self._components.db,
                                        pnl_tracker=self._components.pnl_tracker)
        interval = self.settings.momentum_coupling.poll_seconds
        while self._running:
            if await self._check_kill_switch():
                return
            try:
                await run_pillar_once(self._components.db, pillar, interval_seconds=interval)
            except Exception as e:  # noqa: BLE001
                log.error("coupling.error", error=str(e))
            await asyncio.sleep(interval)

    async def _task_recalibrate(self) -> None:
        """Periodically refit Platt scaling calibration parameters."""
        calibration: CalibrationTracker = self._components.calibration
        interval = self.settings.calibration.refit_interval_hours * 3600

        while self._running:
            try:
                await calibration.refit_all()
            except Exception as e:
                log.error("recalibrate.error", error=str(e))
            await asyncio.sleep(interval)

    async def _task_attribution_update(self) -> None:
        """Periodically update performance attribution and Kelly multipliers."""
        attributor = self._components.attributor
        if attributor is None:
            return

        while self._running:
            await asyncio.sleep(3600)
            try:
                await attributor.compute_kelly_multipliers()
                stats = await attributor.get_category_summary(is_live=self.settings.is_live)
                if stats:
                    from auramaur.monitoring.display import show_category_performance
                    show_category_performance(stats)
            except Exception as e:
                log.error("attribution.error", error=str(e))

    async def _task_strategy_report(self) -> None:
        """Hourly per-pillar P&L report for hybrid mode."""
        attributor = self._components.attributor
        if attributor is None:
            return

        while self._running:
            await asyncio.sleep(3600)
            try:
                summary = await attributor.get_strategy_summary(is_live=self.settings.is_live)
                if summary:
                    log.info("hybrid.strategy_report", pillars=summary)
                    from auramaur.monitoring.display import console
                    console.print("\n[bold cyan]--- Hybrid Strategy Report ---[/]")
                    for s in summary:
                        source = s.get("strategy_source", "unknown")
                        trades = s.get("trade_count", 0)
                        pnl = s.get("total_pnl", 0)
                        wins = s.get("wins", 0)
                        win_rate = (wins / trades * 100) if trades > 0 else 0
                        color = "green" if pnl > 0 else "red"
                        console.print(
                            f"  [{color}]{source:15s}[/] "
                            f"trades={trades:3d}  P&L=${pnl:+.2f}  "
                            f"win={win_rate:.0f}%"
                        )
            except Exception as e:
                log.error("hybrid.strategy_report_error", error=str(e))

    async def _task_performance_feedback(self) -> None:
        """Periodically update per-category calibration stats and Kelly multipliers."""
        feedback = self._components.feedback
        if feedback is None:
            return

        while self._running:
            try:
                await feedback.update_from_resolutions()
                stats = await feedback.get_category_accuracy()
                if stats:
                    log.info(
                        "feedback.updated",
                        categories=len(stats),
                        avoid=sorted(await feedback.get_avoid_categories()),
                    )
            except Exception as e:
                log.error("feedback.error", error=str(e))
            await asyncio.sleep(3600)  # Every hour

    @staticmethod
    def _rotate_scan_window(universe, offset, window):
        """Return (slice, next_offset) for a rolling window over ``universe``.

        Advances a cursor across the full market list so relationship detection
        covers niche/low-volume markets over time instead of only the
        top-by-volume head. Wraps at the end. Empty/short universes are handled.
        """
        if not universe:
            return [], 0
        if offset >= len(universe):
            offset = 0
        return universe[offset:offset + window], offset + window

    async def _prune_resolved_relationships(self, db) -> int:
        """Delete market_relationships whose either leg has resolved/closed.

        market_relationships was never pruned, so it filled with resolved
        markets (entailment then found ~no live conditional pairs to evaluate).
        A leg with markets.active = 0 is settled on the venue; drop the pair.
        Returns the number of relationship rows removed.
        """
        try:
            cur = await db.execute(
                """DELETE FROM market_relationships
                   WHERE market_id_a IN (SELECT id FROM markets WHERE active = 0)
                      OR market_id_b IN (SELECT id FROM markets WHERE active = 0)"""
            )
            await db.commit()
            return int(getattr(cur, "rowcount", 0) or 0)
        except Exception as e:
            log.debug("correlation.prune_error", error=str(e))
            return 0

    async def _task_correlation_scan(self) -> None:
        """Scan for correlated markets and execute conditional / divergence arbs.

        Relationship detection calls Claude (and is TTL-cached per market), so
        it runs on a slow sub-cadence. Arb detection + execution is a cheap DB
        read off the stored relationships and runs every cycle, so opportunities
        are acted on before they decay (previously the whole loop ran every 4h,
        by which point conditional violations had usually closed).
        """
        correlator = self._components.correlator
        arb_executor = self._components.arb_executor
        if correlator is None or arb_executor is None:
            return
        discovery: MarketDiscovery = self._components.discovery
        risk_manager: RiskManager = self._components.risk_manager
        engines: dict[str, TradingEngine] = self._components.engines

        db: Database = self._components.db
        scan_interval = 120              # act on arbs every 2 minutes
        relationship_refresh_cycles = 15  # refresh LLM relationships ~every 30 min
        cycle = 0
        scan_offset = 0
        scan_window = 10  # markets analyzed per refresh (== detector batch_size)

        while self._running:
            if await self._check_kill_switch():
                return
            try:
                # Refresh semantic relationships infrequently to bound LLM cost;
                # the detector also TTL-caches per market internally.
                if cycle % relationship_refresh_cycles == 0:
                    # Rotate the analysis window across the BROAD universe rather
                    # than re-feeding the top-by-volume head every cycle. Conditional
                    # / entailment pairs live in niche, lower-volume markets that
                    # never entered a top-20 window — so that pool only ever decayed
                    # (newest conditional was 3 days stale, mostly resolved legs).
                    # The window advances each refresh and wraps; the 24h TTL-cache
                    # dedups, so over a full pass every market is analyzed ~once.
                    # Budget-neutral: same LLM batch size, just a different slice.
                    universe = await discovery.get_markets(limit=300)
                    window, scan_offset = self._rotate_scan_window(
                        universe, scan_offset, scan_window)
                    if window:
                        await correlator.detect_relationships(window, batch_size=len(window))
                    # Prune relationships whose markets have resolved/closed so
                    # entailment's conditional pool reflects only live markets.
                    pruned = await self._prune_resolved_relationships(db)
                    if pruned:
                        log.info("correlation.pruned_resolved", count=pruned)
                cycle += 1

                # Generate and execute arbitrage signals (cheap DB read).
                pairs = await arb_executor.generate_arb_signals()
                for buy_signal, sell_signal, opp in pairs:
                    try:
                        buy_market = await arb_executor._load_market(buy_signal.market_id)
                        sell_market = await arb_executor._load_market(sell_signal.market_id)
                        if not buy_market or not sell_market:
                            continue
                        # Same category policy as every other book: this
                        # path executes through the exempt 'arbitrage'
                        # strategy_source, so the gateway's category checks
                        # never see it (it quoted KBO baseball live
                        # 2026-06-12). Both legs must clear the gates.
                        from auramaur.strategy.classifier import ensure_category
                        risk_cfg = self.settings.risk
                        blocked = set(risk_cfg.blocked_categories)
                        allowed = (set(risk_cfg.allowed_categories_live)
                                   if self.settings.is_live else None)
                        leg_cats = [
                            ensure_category(m.question, m.description, m.category)
                            for m in (buy_market, sell_market)
                        ]
                        if any(c in blocked for c in leg_cats) or (
                                allowed is not None
                                and any(c not in allowed for c in leg_cats)):
                            log.debug("arbitrage.category_gated",
                                      categories=leg_cats)
                            continue
                        await self._execute_conditional_arb(
                            buy_signal, sell_signal, buy_market, sell_market,
                            opp, risk_manager, engines,
                        )
                    except Exception as e:
                        log.debug("arbitrage.execution_error", error=str(e))
            except Exception as e:
                log.error("correlation_scan.error", error=str(e))
            await asyncio.sleep(scan_interval)

    async def _task_market_maker(self) -> None:
        """Run market making cycles — post two-sided quotes on liquid markets.

        Uses the Gamma client to find liquid markets, then posts bid/ask
        quotes via the exchange client. Runs every refresh_seconds.
        """
        mm: MarketMaker | None = self._components.market_maker
        if mm is None:
            return

        discovery: MarketDiscovery = self._components.discovery
        alerts: AlertManager = self._components.alerts
        interval = self.settings.market_maker.refresh_seconds

        while self._running:
            if await self._check_kill_switch():
                return
            # Heartbeat at INFO every iteration. run_cycle logs only at debug when
            # there are no quotable markets, so an idle MM looked identical to a
            # dead one to the health monitor (false STALE) — and when it DID wedge
            # (2026-06-30, ~16 min), the last log before silence didn't pinpoint
            # where. This line makes "alive but idle" distinguishable from "hung",
            # and the first missing op after it localizes any future stall.
            log.info("market_maker.cycle_start")
            try:
                # Watchdog: the per-op timeout inside run_cycle only covers the
                # quote ops — the OUTER-loop Polymarket calls (market discovery,
                # fill polling) were still unbounded and hung the loop (2026-06-30,
                # silent ~16 min with NO op_timeout firing). Bound them too so no
                # single stuck request can stall the MM task.
                op_timeout = self.settings.market_maker.op_timeout_seconds

                # Fetch liquid markets sorted by volume
                try:
                    markets = await asyncio.wait_for(
                        discovery.get_markets(limit=50, order="liquidity"),
                        timeout=op_timeout)
                except asyncio.TimeoutError:
                    log.warning("market_maker.op_timeout", op="get_markets",
                                timeout=op_timeout)
                    await asyncio.sleep(interval)
                    continue

                # Run the MM cycle
                results = await mm.run_cycle(markets)

                # Check for fills on pending live orders
                try:
                    fills = await asyncio.wait_for(
                        mm.check_fills(), timeout=op_timeout)
                except asyncio.TimeoutError:
                    log.warning("market_maker.op_timeout", op="check_fills",
                                timeout=op_timeout)
                    fills = []

                if results:
                    total_profit = sum(r.get("expected_profit", 0) for r in results)
                    log.info(
                        "market_maker.task_cycle",
                        quotes_placed=len(results),
                        expected_profit=round(total_profit, 4),
                        fills=len(fills),
                    )

                # Alert on significant inventory accumulation
                inventory = mm.get_inventory_summary()
                for mid, net in inventory.items():
                    if abs(net) > self.settings.market_maker.max_inventory * 0.8:
                        await alerts.send(
                            f"MM inventory warning: {mid[:12]} net={net:.1f} tokens",
                            level="warning",
                        )

            except Exception as e:
                show_error(f"Market maker cycle failed: {e}")
            await asyncio.sleep(interval)

    async def _task_price_monitor(self) -> None:
        """Monitor real-time price changes via WebSocket."""
        ws = self._components.websocket
        if ws is None:
            return

        engines: dict[str, TradingEngine] = self._components.engines
        # Use primary (polymarket) engine for price-triggered re-analysis
        engine: TradingEngine = engines.get("polymarket", list(engines.values())[0])
        discovery: MarketDiscovery = self._components.discovery
        threshold = self.settings.ensemble.price_move_threshold_pct / 100.0

        # Track last-known prices for change detection
        last_prices: dict[str, float] = {}

        async def on_price_update(market_id: str, new_price: float) -> None:
            old_price = last_prices.get(market_id)
            last_prices[market_id] = new_price

            if old_price is not None:
                change = abs(new_price - old_price)
                if change >= threshold:
                    log.info(
                        "price_monitor.significant_move",
                        market_id=market_id,
                        old=old_price,
                        new=new_price,
                        change_pct=round(change * 100, 1),
                    )
                    # Re-analyze on significant move
                    try:
                        market = await discovery.get_market(market_id)
                        if market and market.active:
                            await engine.analyze_market(market)
                    except Exception as e:
                        log.debug("price_monitor.reanalysis_error", error=str(e))

        ws._on_price_update = on_price_update

        flow_tracker = self._components.flow_tracker

        if flow_tracker is not None:
            async def on_trade(market_id: str, side: str, size: float) -> None:
                try:
                    order_side = OrderSide.BUY if side.upper() in ("BUY", "B", "1") else OrderSide.SELL
                    flow_tracker.record_trade(market_id, order_side, size)
                except Exception:
                    pass

            ws._on_trade = on_trade

        try:
            # Subscribe to markets we're tracking
            markets = await discovery.get_markets(limit=20)
            await ws.subscribe([m.id for m in markets])
            await ws.run()
        except Exception as e:
            log.error("price_monitor.error", error=str(e))

    async def _task_decision_marks(self) -> None:
        """Attach executable 1m/5m/1h/24h marks to immutable decisions."""
        from auramaur.research.polymarket_strategies import DecisionTracker

        db = self._components.db
        tracker = DecisionTracker(db)
        horizons = (60, 300, 3600, 86400)
        while True:
            try:
                for horizon in horizons:
                    rows = await db.fetchall(
                        """SELECT d.id, d.market_id, o.best_bid, o.best_ask
                           FROM decision_snapshots d
                           JOIN orderbook_snapshots o ON o.market_id = d.market_id
                           WHERE unixepoch('now') - unixepoch(d.observed_at) >= ?
                             AND NOT EXISTS (
                                 SELECT 1 FROM decision_marks x
                                 WHERE x.decision_id = d.id
                                   AND x.horizon_seconds = ?)
                             AND o.recorded_at = (
                                 SELECT MIN(o2.recorded_at)
                                 FROM orderbook_snapshots o2
                                 WHERE o2.market_id = d.market_id
                                   AND unixepoch(o2.recorded_at) >=
                                       unixepoch(d.observed_at) + ?)
                           LIMIT 250""",
                        (horizon, horizon, horizon),
                    )
                    for row in rows:
                        await tracker.mark(
                            row["id"], horizon, bid=row["best_bid"],
                            ask=row["best_ask"])
            except Exception as exc:
                log.warning("decision_marks.error", error=str(exc))
            await asyncio.sleep(60)

    async def _task_user_websocket(self) -> None:
        """Run the authenticated lifecycle stream as a monitor wake-up lane."""
        stream = self._components.user_websocket
        if stream is None:
            return
        discovery = self._components.discovery
        try:
            markets = await discovery.get_markets(limit=100)
            stream.subscribe([m.condition_id for m in markets if m.condition_id])
            await stream.run()
        except Exception as exc:
            log.error("websocket.user_error", error=str(exc))

    async def _task_source_weights_update(self) -> None:
        """Periodically update ensemble source weights."""
        ensemble = self._components.ensemble
        if ensemble is None:
            return

        interval = self.settings.ensemble.source_weights_update_hours * 3600
        while self._running:
            try:
                await ensemble.update_source_weights()
            except Exception as e:
                log.error("source_weights.error", error=str(e))
            await asyncio.sleep(interval)

    async def _task_position_sync(self) -> None:
        """Periodically sync net current positions from Polymarket."""
        from auramaur.broker.reconciler import PositionReconciler
        from auramaur.broker.sync import PositionSyncer

        reconciler: PositionReconciler = self._components.reconciler
        syncer: PositionSyncer = self._components.syncer
        interval = self.settings.broker.sync_interval_seconds

        while self._running:
            try:
                if self.settings.is_live:
                    # The Data API's net current positions are venue truth.
                    reconciled = await reconciler.reconcile()
                    positions = reconciler.to_live_positions(reconciled)

                    # Update cost_basis from real fill prices (ground truth),
                    # then delete stale rows — one short, serialized
                    # transaction (contention plan, Phase 2): the network work
                    # (reconcile()) is already done, so this block is db-only.
                    # A successful empty response is authoritative; a failed
                    # fetch leaves last_fetch_ok false and never wipes
                    # the tables. Both tables matter: cost_basis feeds sync(),
                    # and portfolio feeds the risk-manager correlation/exposure
                    # checks.
                    if reconciler.last_fetch_ok:
                        # Skip already-settled legs if the venue still exposes
                        # them before redemption (148 rows / $918 phantom,
                        # 2026-07-20 audit). Excluding them from live_ids also
                        # lets the stale-delete below sweep the residue out.
                        try:
                            settled = await syncer._settled_keys(0)
                        except Exception:
                            settled = set()
                        unsettled = [
                            rp for rp in reconciled
                            if (rp.market_id, TokenType.from_str(rp.outcome).value)
                            not in settled
                        ]
                        positions = reconciler.to_live_positions(unsettled)
                        if len(unsettled) < len(reconciled):
                            log.debug("reconciler.settled_skipped",
                                      count=len(reconciled) - len(unsettled))
                        async with self._components.db.transaction():
                            # This loop only runs in live mode, so we explicitly
                            # write is_paper=0 and conflict on (market_id, is_paper).
                            for rp in unsettled:
                                await self._components.db.execute(
                                    """INSERT INTO cost_basis (market_id, token, token_id, size, avg_cost, total_cost, is_paper, updated_at)
                                       VALUES (?, ?, ?, ?, ?, ?, 0, datetime('now'))
                                       ON CONFLICT(market_id, is_paper, token) DO UPDATE SET
                                           token = excluded.token,
                                           token_id = excluded.token_id,
                                           size = excluded.size,
                                           avg_cost = excluded.avg_cost,
                                           total_cost = excluded.total_cost,
                                           updated_at = excluded.updated_at""",
                                    # Normalize the raw CLOB outcome ("Yes"/"No") to the
                                    # canonical TokenType value ("YES"/"NO"). Writing the raw
                                    # title-case outcome here was the one site that diverged
                                    # from the fill/Kalshi paths, splitting a single position
                                    # into duplicate (market, "No") + (market, "NO") rows.
                                    (rp.market_id, TokenType.from_str(rp.outcome).value,
                                     rp.token_id, rp.size,
                                     rp.avg_cost, rp.size * rp.avg_cost),
                                )

                            live_tokens = [rp.token_id for rp in unsettled]
                            placeholders = ",".join("?" * len(live_tokens))
                            exclusion = (f"AND token_id NOT IN ({placeholders})"
                                         if live_tokens else "")
                            # Live-mode reconciliation must only delete live rows;
                            # paper rows (is_paper=1) live in their own namespace.
                            # Scope deletion to Polymarket: without it this task
                            # previously deleted live Kalshi cost-basis rows.
                            # it deleted every live KALSHI cost_basis row each
                            # ~75s pass — Kalshi sells then found size=0 and
                            # booked $0.00 realized P&L into the ledger that
                            # feeds the live daily-loss gate (found 2026-07-20;
                            # the portfolio delete below always had the filter).
                            cb_cur = await self._components.db.execute(
                                f"""DELETE FROM cost_basis WHERE size > 0 AND is_paper = 0
                                    {exclusion}
                                    AND market_id IN (SELECT id FROM markets WHERE exchange = 'polymarket')""",
                                live_tokens,
                            )
                            pf_cur = await self._components.db.execute(
                                f"DELETE FROM portfolio WHERE exchange = 'polymarket' AND is_paper = 0 {exclusion}",
                                live_tokens,
                            )
                        log.info(
                            "reconciler.stale_removed",
                            cost_basis=cb_cur.rowcount if hasattr(cb_cur, "rowcount") else 0,
                            portfolio=pf_cur.rowcount if hasattr(pf_cur, "rowcount") else 0,
                        )
                else:
                    positions = await syncer.sync()

                cash = await syncer.get_cash_balance()
                total_value = sum(p.size * p.current_price for p in positions)
                unrealized = sum(p.unrealized_pnl for p in positions)
                if self.settings.is_live:
                    now = datetime.now(timezone.utc).isoformat()
                    equity = cash + total_value
                    async with self._components.db.transaction():
                        await self._components.db.execute(
                            """INSERT INTO venue_balances
                               (venue,detail,available,equity,fetched_at)
                               VALUES ('polymarket',?,?,?,?)
                               ON CONFLICT(venue) DO UPDATE SET detail=excluded.detail,
                               available=excluded.available,equity=excluded.equity,
                               fetched_at=excluded.fetched_at""",
                            (f"${cash:.2f} available | ${equity:.2f} equity", cash, equity, now))

                log.info(
                    "sync.portfolio",
                    positions=len(positions),
                    cash=round(cash, 2),
                    value=round(total_value, 2),
                    unrealized=round(unrealized, 2),
                )
            except Exception as e:
                log.error("position_sync.error", error=str(e))
            await asyncio.sleep(interval)

    async def _task_news_reactor(self) -> None:
        """Poll RSS feeds for breaking news and trigger fast analysis on matching markets."""
        reactor: NewsReactor = self._components.news_reactor

        while self._running:
            if await self._check_kill_switch():
                return
            try:
                results = await reactor.check_for_news()
                if results:
                    trades = [r for r in results if r.get("order")]
                    if trades:
                        alerts: AlertManager = self._components.alerts
                        await alerts.send(
                            f"News reactor triggered {len(trades)} trade(s) from {len(results)} analysis(es)",
                            level="info",
                        )
            except Exception as e:
                show_error(f"News reactor failed: {e}")
            interval = self.settings.hybrid.news_cycle_seconds if self._hybrid else 60
            await asyncio.sleep(interval)

    async def _task_build_guard(self) -> None:
        """Warn when the on-disk checkout moves past the running process.

        The 2026-06 category leak traded for a day on pre-fix code because
        the merged fix sat on disk while the long-lived process kept running
        the old build. Checks every 5 minutes; the guard rate-limits its own
        alerts (hourly per the quiet-feed rules).
        """
        from auramaur.monitoring.build_info import BuildStalenessGuard
        guard = BuildStalenessGuard(
            alert=lambda msg: console.print(f"[bold red]⚠ STALE BUILD: {msg}[/]")
        )
        while self._running:
            try:
                guard.check()
            except Exception as e:
                log.debug("build_guard.error", error=str(e))
            await asyncio.sleep(300)

    async def run(self) -> None:
        """Start the bot with all concurrent tasks."""
        # Use console renderer for terminal, JSON for log file
        setup_logging(
            level=self.settings.logging.level,
            json_format=False,  # Console-friendly output
            log_file=self.settings.logging.file,
            rotate_max_mb=self.settings.logging.rotate_max_mb,
            rotate_backups=self.settings.logging.rotate_backups,
        )

        mode = "LIVE" if self.settings.is_live else "PAPER"
        show_banner(mode, "0.1.0")

        # Announce which build this process runs. The 2026-06 category leak
        # traded for a day on pre-fix code because nothing said the running
        # process was older than the checkout.
        from auramaur.monitoring.build_info import STARTUP_SHA
        if STARTUP_SHA:
            console.print(f"  [dim]Build: {STARTUP_SHA}[/]")
            log.info("bot.build", sha=STARTUP_SHA, mode=mode)

        await self._init_components()
        self._running = True

        db_path = self._components.db.db_path
        from auramaur.runtime import db_path as runtime_db_path
        if Path(db_path).resolve() != runtime_db_path().resolve():
            console.print(f"  [yellow]Instance: {db_path}[/]")
        # Persistent Claude-budget counter lives in the same sqlite file —
        # point it at this instance's path (multi-instance runs use
        # auramaur_N.db and must not share a budget row file-side).
        from auramaur.nlp import call_budget
        call_budget.set_db_path(db_path)

        # Show real balance — use reconciler for live, paper for paper mode.
        # In live mode we never fall back to paper.balance — that would show
        # paper PnL in a live banner if the reconciler fails to respond.
        startup_balance = 0.0 if self.settings.is_live else self._components.paper.balance
        if self.settings.is_live:
            try:
                total_cash = 0.0
                total_position_value = 0.0
                total_markets = 0

                # Polymarket balance
                if self._exchange_filter is None or self._exchange_filter == "polymarket":
                    reconciler_comp = self._components.reconciler
                    if reconciler_comp:
                        reconciled = await reconciler_comp.reconcile()
                        position_value = sum(p.size * p.current_price for p in reconciled)
                        syncer_comp = self._components.syncer
                        poly_cash = await syncer_comp.get_cash_balance() if syncer_comp else 0
                        total_cash += poly_cash
                        total_position_value += position_value
                        total_markets += len(reconciled)

                # Kalshi balance
                engines = self._components.get("engines", {})
                if self._exchange_filter is None or self._exchange_filter == "kalshi":
                    kalshi_engine = engines.get("kalshi")
                    if kalshi_engine and hasattr(kalshi_engine.exchange, "get_balance"):
                        kalshi_cash = await kalshi_engine.exchange.get_balance()
                        total_cash += kalshi_cash
                        console.print(f"  Kalshi balance: [green]${kalshi_cash:.2f}[/]")

                startup_balance = total_cash + total_position_value
                console.print(
                    f"  Cash: [green]${total_cash:.2f}[/] | "
                    f"Positions: [cyan]${total_position_value:.2f}[/] ({total_markets} markets)"
                )
            except Exception as e:
                log.debug("startup.balance_error", error=str(e))

        # Kraken treasury/speculation pool (separate from prediction-market bankroll)
        if self.settings.kraken.enabled and self._exchange_filter in (None, "kraken"):
            try:
                from auramaur.exchange.kraken import KrakenSpotClient
                kclient = KrakenSpotClient(self.settings)
                kb = await kclient.get_balance()
                usdc = kb.get("USDC", 0.0)
                cad = kb.get("ZCAD", 0.0)
                cadpx = await kclient.get_price("USDCCAD") or 1.38
                cad_usd = cad / cadpx if cadpx else 0.0
                crypto = [a for a, v in kb.items() if v > 0 and a not in ("USDC", "ZCAD")]
                if not self.settings.kraken.directional_enabled:
                    spec = "spec off"
                elif self.settings.kraken.directional_budget_usd <= 0:
                    spec = "[yellow]spec WIND-DOWN[/] (exits only)"
                else:
                    spec = (f"[red]SPEC ON[/] ({len(self.settings.kraken.directional_pairs)} pairs, "
                            f"${self.settings.kraken.directional_budget_usd:.0f} budget)")
                console.print(
                    f"  Kraken: [green]${usdc:.2f}[/] USDC + [green]${cad_usd:.0f}[/] CAD"
                    f"{' + ' + ','.join(crypto) if crypto else ''} | {spec}"
                )
                await kclient.close()
            except Exception as e:
                log.debug("startup.kraken_balance_error", error=str(e))

        # IBKR status (independently-gated options scanner + equity speculation)
        if self.settings.ibkr.enabled and self._exchange_filter in (None, "ibkr"):
            opt = ("[red]options ON[/]" if self.settings.ibkr.options_enabled
                   else "options off")
            console.print(f"  IBKR: enabled ({self.settings.ibkr.environment}) | {opt}")

        show_startup(
            self._components.source_names,
            startup_balance,
        )

        # Strategy-books panel: every book with its TRUE mode, the gates,
        # graduation mode, and the ledger lifetime number.
        try:
            from auramaur.monitoring.books import render_books_panel
            row = await self._components.db.fetchone(
                "SELECT COALESCE(SUM(pnl), 0) AS v FROM pnl_ledger WHERE is_paper = 0")
            console.print(render_books_panel(
                self.settings, float(row["v"]) if row else None))
        except Exception as e:
            log.debug("startup.books_panel_error", error=str(e))

        exchange_filter = self._components.exchange_filter
        if exchange_filter:
            console.print(f"  [cyan]Exchange filter: {exchange_filter} only[/]")

        # Operational live-readiness preflight. If any BLOCK condition is present
        # while armed live, force new ENTRIES to paper (exits bypass the risk
        # manager, so held positions can still get out). Same checks as
        # `auramaur health`; a "refuse to fool itself" gate. Re-evaluated
        # periodically by _task_live_gate_monitor so the latch moves in BOTH
        # directions without a restart.
        if self.settings.is_live:
            try:
                await self._run_live_gate(startup=True)
            except Exception as e:
                log.error("live_gate.error", error=str(e))

        tasks = [
            asyncio.create_task(self._task_kill_switch_monitor(), name="kill_switch"),
            asyncio.create_task(self._task_cache_cleanup(), name="cache_cleanup"),
            asyncio.create_task(self._task_recalibrate(), name="recalibrate"),
        ]
        if self.settings.is_live:
            tasks.append(asyncio.create_task(
                self._task_live_gate_monitor(), name="live_gate"))

        # Portfolio monitor runs whenever any exchange syncer is present so
        # Kalshi-only runs still populate `_last_known_cash` and exits fire.
        # Position sync (CLOB reconciler) remains Polymarket-specific.
        if self._components.syncers:
            tasks.append(asyncio.create_task(self._task_portfolio_monitor(), name="portfolio"))
        if self._components.syncer:
            tasks.append(asyncio.create_task(self._task_position_sync(), name="position_sync"))

        # Resolution checker and order monitor work with any exchange
        tasks.append(asyncio.create_task(self._task_resolution_checker(), name="resolution_checker"))
        tasks.append(asyncio.create_task(self._task_order_monitor(), name="order_monitor"))

        # Balance recorder: feeds the read-only dashboard's venue panel.
        if (self.settings.kalshi.enabled or self.settings.kraken.enabled
                or self.settings.ibkr.enabled):
            tasks.append(asyncio.create_task(
                self._task_balance_recorder(), name="balance_recorder"))

        # Loop watchdog: a daemon thread that screams when the asyncio loop
        # stops beating (blocking sync call). Runs outside the loop so it can
        # still speak during a freeze.
        from auramaur.monitoring.feed import LoopWatchdog
        self._watchdog = LoopWatchdog(
            alert=lambda msg: console.print(f"[bold red]⚠ WATCHDOG: {msg}[/]")
        )
        self._watchdog.beat()
        self._watchdog.start()

        # Build-staleness guard: warns when the checkout on disk moves past
        # the running process (merged fix, no restart — the 2026-06 leak).
        tasks.append(asyncio.create_task(
            self._task_build_guard(), name="build_guard"))

        # Redemption check — only meaningful with a Polymarket proxy wallet
        if self.settings.polymarket_proxy_address:
            tasks.append(asyncio.create_task(self._task_redemption_check(), name="redemption_check"))

        # News reactor (if available)
        if self._components.news_reactor:
            tasks.append(asyncio.create_task(self._task_news_reactor(), name="news_reactor"))

        # Per-exchange scan + trade tasks
        engines: dict[str, TradingEngine] = self._components.engines
        for ex_name, engine in engines.items():
            tasks.append(asyncio.create_task(
                self._task_market_scan(engine, ex_name), name=f"scan_{ex_name}",
            ))
            tasks.append(asyncio.create_task(
                self._task_trading_cycle(engine, ex_name), name=f"trade_{ex_name}",
            ))
            # Capture real bid/ask depth for cost-aware research + honest paper
            # fills. Read-only; disable via intervals.orderbook_recorder_enabled.
            if getattr(self.settings.intervals, "orderbook_recorder_enabled", True):
                tasks.append(asyncio.create_task(
                    self._task_orderbook_recorder(engine, ex_name),
                    name=f"orderbook_{ex_name}",
                ))

        if self._components.attributor:
            tasks.append(asyncio.create_task(self._task_attribution_update(), name="attribution"))
        # Correlation-arb executes through the exempt 'arbitrage' source and
        # had no off switch at all; it shares the arbitrage book's flag.
        if self._components.correlator and self.settings.arbitrage.enabled:
            tasks.append(asyncio.create_task(self._task_correlation_scan(), name="correlation"))
        if self._components.websocket:
            tasks.append(asyncio.create_task(self._task_price_monitor(), name="price_monitor"))
        if self._components.user_websocket:
            tasks.append(asyncio.create_task(
                self._task_user_websocket(), name="user_websocket"))
        tasks.append(asyncio.create_task(
            self._task_decision_marks(), name="decision_marks"))
        if self._components.ensemble:
            tasks.append(asyncio.create_task(self._task_source_weights_update(), name="source_weights"))
        if self._components.feedback:
            tasks.append(asyncio.create_task(self._task_performance_feedback(), name="performance_feedback"))

        # Arb scanner — arbitrage.enabled was a dead flag (the task ran
        # unconditionally), discovered 2026-06-12 when disabling it via
        # config did nothing while the book locked a -3% cross-attempt pair.
        if self.settings.arbitrage.enabled:
            tasks.append(asyncio.create_task(self._task_arb_scanner(), name="arb_scanner"))
        tasks.append(asyncio.create_task(self._task_depth_research(), name="depth_research"))

        # Favorite-longshot bias harvest (paper-forced until proven)
        if self.settings.bias_harvest.enabled:
            tasks.append(asyncio.create_task(self._task_bias_harvest(), name="bias_harvest"))

        # Platform consensus follower (paper-forced until proven)
        if self.settings.platform_consensus.enabled:
            tasks.append(asyncio.create_task(self._task_platform_consensus(), name="platform_consensus"))

        # Long-horizon favorite underpricing (paper-forced; structural slope edge)
        if self.settings.long_horizon.enabled:
            tasks.append(asyncio.create_task(self._task_long_horizon(), name="long_horizon"))

        # Multi-model LLM day-trader (paper-forced; intelligence-cap A/B)
        if self.settings.agent_trader.enabled:
            tasks.append(asyncio.create_task(self._task_agent_trader(), name="agent_trader"))

        if (self.settings.intelligence_eval.enabled
                and self.settings.local_llm.enabled):
            tasks.append(asyncio.create_task(
                self._task_intelligence_eval(), name="intelligence_eval"))

        # Deadline-ladder curve reader (paper-forced; amortized reading edge)
        if self.settings.term_structure.enabled:
            tasks.append(asyncio.create_task(self._task_term_structure(), name="term_structure"))

        # Vol-anchored crypto threshold pricing (paper-forced; deterministic)
        if self.settings.vol_anchor.enabled:
            tasks.append(asyncio.create_task(self._task_vol_anchor(), name="vol_anchor"))

        # Informed-flow follower over Kalshi (paper-forced; abnormal-trade-size)
        if self.settings.informed_flow.enabled:
            tasks.append(asyncio.create_task(self._task_informed_flow(), name="informed_flow"))

        # Entailment arbitrage (paper-forced until proven)
        if self.settings.entailment_arb.enabled:
            tasks.append(asyncio.create_task(self._task_entailment_arb(), name="entailment_arb"))
        if self.settings.cross_venue_arb.enabled:
            tasks.append(asyncio.create_task(self._task_cross_venue_arb(), name="cross_venue_arb"))

        # Data-driven Kalshi econ-indicator pricing (paper-forced until proven)
        if self.settings.econ_indicator.enabled:
            tasks.append(asyncio.create_task(self._task_econ_indicator(), name="econ_indicator"))

        # Operator-proposed interim book management (ladder-evaluated; default off).
        if self.settings.interim_manager.enabled:
            tasks.append(asyncio.create_task(self._task_interim_manager(), name="interim_manager"))

        # Settlement-lag / known-outcome arb (FRED-first). Paper-forced; default off.
        if self.settings.settlement_arb.enabled:
            tasks.append(asyncio.create_task(self._task_settlement_arb(), name="settlement_arb"))

        # Open-Meteo ensemble city-temperature pricing (paper-forced spike)
        if self.settings.weather_temp.enabled:
            tasks.append(asyncio.create_task(self._task_weather_temp(), name="weather_temp"))

        # Hydrology-market watcher (alert-only; arms the compHydro moat)
        if self.settings.hydro_watch.enabled:
            tasks.append(asyncio.create_task(self._task_hydro_watch(), name="hydro_watch"))

        # Intraday-drift measurement spike (no trading; gates the intraday strat)
        if self.settings.intraday_drift.enabled:
            tasks.append(asyncio.create_task(self._task_intraday_drift(), name="intraday_drift"))

        # Resolution-language lens (paper-forced until proven)
        if self.settings.resolution_lens.enabled:
            tasks.append(asyncio.create_task(self._task_resolution_lens(), name="resolution_lens"))
            # Kalshi measurement spike — a second lens instance, paper-forced,
            # attributed separately. Default off; flip resolution_lens.kalshi_enabled.
            if self.settings.resolution_lens.kalshi_enabled:
                tasks.append(asyncio.create_task(
                    self._task_resolution_lens_kalshi(), name="resolution_lens_kalshi"))

        # Odd-lot tender harvester (detection always; entries paper-forced)
        if self.settings.oddlot_tender.enabled:
            tasks.append(asyncio.create_task(self._task_oddlot_tender(), name="oddlot_tender"))

        # Local-LLM evidence distiller (evidence-side only, never trades)
        if self.settings.local_llm.enabled and self.settings.local_llm.distiller.enabled:
            tasks.append(asyncio.create_task(self._task_evidence_distiller(), name="evidence_distiller"))

        # Kraken treasury/capital pillar (+ gated directional spot)
        if self.settings.kraken.enabled:
            tasks.append(asyncio.create_task(self._task_kraken_pillar(), name="kraken_treasury"))

        if self.settings.ibkr.enabled and self.settings.ibkr.etf_paper_enabled:
            tasks.append(asyncio.create_task(
                self._task_ibkr_etf_paper(), name="ibkr_etf_paper"))
        if self.settings.ibkr.enabled and self.settings.ibkr.multiasset_paper_enabled:
            tasks.append(asyncio.create_task(
                self._task_ibkr_multiasset_paper(), name="ibkr_multiasset_paper"))

        # Fast path: momentum-coupling pillar (gated by momentum_coupling.enabled)
        if self.settings.momentum_coupling.enabled:
            tasks.append(asyncio.create_task(self._task_momentum_coupling(), name="momentum_coupling"))

        # Market maker (if enabled)
        if self._components.market_maker:
            tasks.append(asyncio.create_task(self._task_market_maker(), name="market_maker"))

        # Hybrid strategy report (hourly per-pillar P&L)
        if self._hybrid and self._components.attributor:
            tasks.append(asyncio.create_task(self._task_strategy_report(), name="strategy_report"))

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            pass
        except Exception as e:
            show_error(f"Bot error: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Gracefully shut down all components."""
        self._running = False
        if self._watchdog is not None:
            self._watchdog.stop()
        console.print("\n[dim]Shutting down...[/]")

        try:
            await self._cancel_resting_live_orders()
        except Exception as e:
            log.debug("shutdown.cancel_sweep_error", error=str(e))

        # Close the shared local-LLM client (singleton; safe if never opened)
        try:
            from auramaur.nlp import local_llm
            await local_llm.aclose()
        except Exception:
            pass

        for name, comp in self._components.items():
            if hasattr(comp, "close"):
                try:
                    await comp.close()
                except Exception:
                    pass

        # Release database lock
        if self._lock_file is not None:
            import fcntl
            try:
                fcntl.flock(self._lock_file, fcntl.LOCK_UN)
                self._lock_file.close()
            except Exception:
                pass
            self._lock_file = None

        console.print("[dim]Stopped.[/]")

    async def _cancel_resting_live_orders(self) -> None:
        """Cancel live orders still resting on the book before exit.

        GTC orders outlive the process, so every restart used to inherit the
        prior session's market-maker quotes and in-flight entries — collateral
        stayed locked until the startup reconciler and the TTL reaper mopped
        them up minutes into the next session. Cancel on the way out instead,
        and write the trades rows terminal so shutdown doesn't mint new
        orphaned 'pending' rows. Best-effort: a failed cancel is the next
        session's reconciler problem, exactly as before.
        """
        clients: dict[int, tuple[str, object]] = {}
        primary = self._components.exchange
        if primary is not None and hasattr(primary, "_live_pending"):
            clients[id(primary)] = ("polymarket", primary)
        for name, client in (self._components.exchanges or {}).items():
            if client is not None and hasattr(client, "_live_pending"):
                clients.setdefault(id(client), (name, client))

        db = self._components.db
        cancelled = 0
        for exchange_name, client in clients.values():
            for order_id in list(getattr(client, "_live_pending", {}).keys()):
                try:
                    ok = await client.cancel_order(order_id)
                except Exception as e:
                    log.debug(
                        "shutdown.cancel_error",
                        exchange=exchange_name,
                        order_id=order_id,
                        error=str(e),
                    )
                    continue
                if not ok:
                    continue
                cancelled += 1
                if db is not None:
                    try:
                        await db.execute(
                            "UPDATE trades SET status = 'cancelled' WHERE order_id = ?",
                            (order_id,),
                        )
                    except Exception:
                        pass
        if cancelled and db is not None:
            try:
                await db.commit()
            except Exception:
                pass
        if cancelled:
            console.print(f"[dim]Cancelled {cancelled} resting live orders[/]")
            log.info("shutdown.orders_cancelled", count=cancelled)
