"""Main async orchestrator — runs concurrent tasks."""

from __future__ import annotations

import asyncio
from pathlib import Path

import structlog

from config.settings import Settings
from auramaur.data_sources.aggregator import Aggregator
from auramaur.data_sources.newsapi import NewsAPISource
from auramaur.data_sources.reddit import RedditSource
from auramaur.data_sources.twitter import TwitterSource
from auramaur.data_sources.fred import FREDSource
from auramaur.data_sources.rss import RSSSource
from auramaur.data_sources.websearch import WebSearchSource
from auramaur.db.database import Database
from auramaur.exchange.client import PolymarketClient
from auramaur.exchange.gamma import GammaClient
from auramaur.exchange.models import ExitReason, Order, OrderSide, OrderType, TokenType
from auramaur.exchange.protocols import ExchangeClient, MarketDiscovery
from auramaur.exchange.paper import PaperTrader
from auramaur.monitoring.alerts import AlertManager
from auramaur.monitoring.display import (
    console, show_banner, show_error, show_portfolio, show_source_error, show_startup,
)
from auramaur.monitoring.logger import setup_logging
from auramaur.nlp.analyzer import ClaudeAnalyzer
from auramaur.nlp.cache import NLPCache
from auramaur.nlp.calibration import CalibrationTracker
from auramaur.risk.manager import RiskManager
from auramaur.risk.portfolio import PortfolioTracker
from auramaur.strategy.arbitrage_scanner import ArbOpportunity, ArbitrageScanner
from auramaur.strategy.engine import TradingEngine
from auramaur.strategy.market_maker import MarketMaker
from auramaur.strategy.news_reactor import NewsReactor
from auramaur.strategy.resolution_tracker import ResolutionTracker

log = structlog.get_logger()


class AuramaurBot:
    """Main bot orchestrator running concurrent async tasks."""

    def __init__(self, settings: Settings | None = None, db_path: str | None = None, exchange_filter: str | None = None):
        self.settings = settings or Settings()
        self._running = False
        self._components: dict = {}
        self._db_path = db_path
        self._exchange_filter = exchange_filter  # If set, only run this exchange
        self._lock_file = None  # File handle kept open for duration
        self._rebalance_cooldowns: dict[str, float] = {}  # kept for reference, allocator handles concentration
        self._exit_failures: set[str] = set()  # Track failed exit sells to avoid spam
        # Track attempted cross-exchange arbs so the scanner doesn't re-execute
        # the same opportunity every 5-minute cycle. Maps arb-key -> expiry ts.
        self._arb_attempts: dict[str, float] = {}

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
        """Initialize all components."""
        s = self.settings

        # Database — auto-detect available slot
        db_path = self._acquire_db_path()
        db = Database(db_path)
        await db.connect()

        # Data sources
        sources = []
        source_names = []
        if s.newsapi_key:
            sources.append(NewsAPISource(api_key=s.newsapi_key))
            source_names.append("NewsAPI")
        if s.reddit_client_id:
            sources.append(RedditSource(
                client_id=s.reddit_client_id,
                client_secret=s.reddit_client_secret,
                user_agent=s.reddit_user_agent,
            ))
            source_names.append("Reddit")
        if s.twitter_bearer_token:
            sources.append(TwitterSource(bearer_token=s.twitter_bearer_token))
            source_names.append("Twitter")
        if s.fred_api_key:
            sources.append(FREDSource(api_key=s.fred_api_key))
            source_names.append("FRED")
        sources.append(WebSearchSource())
        source_names.append("Web")
        sources.append(RSSSource())
        source_names.append("RSS")

        # Structured data sources (no API keys needed)
        from auramaur.data_sources.market_data import MarketDataSource
        from auramaur.data_sources.polymarket_context import PolymarketContextSource
        sources.append(MarketDataSource())
        source_names.append("Markets")
        sources.append(PolymarketContextSource())
        source_names.append("PolyCtx")
        from auramaur.data_sources.metaculus import MetaculusSource
        sources.append(MetaculusSource())
        source_names.append("Metaculus")
        from auramaur.data_sources.manifold import ManifoldSource
        sources.append(ManifoldSource())
        source_names.append("Manifold")

        # Domain-specific sources — category-gated so they only fire on
        # relevant markets (see DataSource.categories in data_sources/base.py).
        from auramaur.data_sources.usgs import USGSSource
        from auramaur.data_sources.coingecko import CoinGeckoSource
        from auramaur.data_sources.hackernews import HackerNewsSource
        from auramaur.data_sources.espn import ESPNSource
        sources.append(USGSSource())
        source_names.append("USGS")
        sources.append(CoinGeckoSource())
        source_names.append("CoinGecko")
        sources.append(HackerNewsSource())
        source_names.append("HN")
        sources.append(ESPNSource())
        source_names.append("ESPN")

        # Category-agnostic broad news (fires on every query).
        from auramaur.data_sources.gdelt import GDELTSource
        from auramaur.data_sources.google_trends import GoogleTrendsSource
        sources.append(GDELTSource())
        source_names.append("GDELT")
        sources.append(GoogleTrendsSource())
        source_names.append("Trends")
        from auramaur.data_sources.bluesky import BlueskySource
        sources.append(BlueskySource())
        source_names.append("Bluesky")

        aggregator = Aggregator(sources=sources)

        # Exchange
        paper = PaperTrader(db=db, initial_balance=s.execution.paper_initial_balance)
        await paper.load_state()

        # NLP
        analyzer = ClaudeAnalyzer(settings=s)
        cache = NLPCache(db=db)
        calibration = CalibrationTracker(
            db=db, min_samples=s.calibration.min_samples
        )

        # Risk
        risk_manager = RiskManager(settings=s, db=db)

        # Order flow (optional)
        flow_tracker = None
        try:
            from auramaur.strategy.order_flow import OrderFlowTracker
            flow_tracker = OrderFlowTracker()
        except ImportError:
            log.warning("optional.missing", component="OrderFlowTracker")

        # Broker layer
        from auramaur.broker.pnl import PnLTracker
        from auramaur.broker.allocator import CapitalAllocator

        pnl_tracker = PnLTracker(db=db, settings=s)
        allocator = CapitalAllocator(settings=s)

        # Strategic analyzer for breadth (batch analysis with world model)
        from auramaur.nlp.strategic import StrategicAnalyzer
        strategic = StrategicAnalyzer(settings=s, db=db)

        # Depth agent for deep research on high-potential markets
        from auramaur.strategy.agent_analyzer import AgentAnalyzer
        depth_agent = AgentAnalyzer(settings=s, db=db, calibration=calibration)

        log.info("bot.analyzer_mode", mode="strategic+depth_agent")

        # Strategy — per-exchange engines
        engines: dict[str, TradingEngine] = {}
        discoveries: dict[str, MarketDiscovery] = {}
        exchanges_map: dict[str, ExchangeClient] = {}
        syncers: list = []

        from auramaur.broker.sync import PositionSyncer, KalshiPositionSyncer
        from auramaur.broker.router import SmartOrderRouter
        from auramaur.broker.reconciler import PositionReconciler

        # Primary exchange (Polymarket) — only if not filtered to another exchange
        exchange = None
        gamma = None
        syncer = None
        reconciler = None
        router = None
        if self._exchange_filter is None or self._exchange_filter == "polymarket":
            gamma = GammaClient()
            exchange = PolymarketClient(settings=s, paper_trader=paper)
            discoveries["polymarket"] = gamma
            exchanges_map["polymarket"] = exchange

            syncer = PositionSyncer(settings=s, db=db, exchange=exchange, paper=paper, pnl=pnl_tracker)
            reconciler = PositionReconciler(exchange=exchange, db=db)
            router = SmartOrderRouter(settings=s, exchange=exchange)
            syncers.append(syncer)

            poly_engine = TradingEngine(
                settings=s, db=db, discovery=gamma, aggregator=aggregator,
                analyzer=analyzer, cache=cache, risk_manager=risk_manager,
                exchange=exchange, calibration=calibration,
                flow_tracker=flow_tracker,
                router=router, allocator=allocator,
            )
            poly_engine._components_pnl = pnl_tracker
            poly_engine._components_syncer = syncer
            poly_engine.strategic = strategic
            poly_engine.exchange_name = "polymarket"

            engines["polymarket"] = poly_engine

        # Kalshi (optional, guarded import) — same first-class wiring as Polymarket
        if s.kalshi.enabled and (self._exchange_filter is None or self._exchange_filter == "kalshi"):
            try:
                from auramaur.exchange.kalshi import KalshiClient
                kalshi = KalshiClient(settings=s, paper_trader=paper)
                discoveries["kalshi"] = kalshi
                exchanges_map["kalshi"] = kalshi

                kalshi_syncer = KalshiPositionSyncer(settings=s, db=db, exchange=kalshi)
                kalshi_router = SmartOrderRouter(settings=s, exchange=kalshi)
                syncers.append(kalshi_syncer)

                kalshi_engine = TradingEngine(
                    settings=s, db=db, discovery=kalshi, aggregator=aggregator,
                    analyzer=analyzer, cache=cache, risk_manager=risk_manager,
                    exchange=kalshi, calibration=calibration,
                    flow_tracker=flow_tracker,
                    router=kalshi_router, allocator=allocator,
                )
                kalshi_engine._components_pnl = pnl_tracker
                kalshi_engine._components_syncer = kalshi_syncer
                kalshi_engine.strategic = strategic
                kalshi_engine.exchange_name = "kalshi"
                kalshi_engine._rebalance_cooldowns = self._rebalance_cooldowns
                engines["kalshi"] = kalshi_engine
            except ImportError:
                log.warning("optional.missing", component="KalshiClient")

        # Crypto.com (optional, works internationally)
        if s.cryptodotcom.enabled and (self._exchange_filter is None or self._exchange_filter == "cryptodotcom"):
            try:
                from auramaur.exchange.cryptodotcom import CryptoComClient
                cryptodotcom = CryptoComClient(settings=s, paper_trader=paper)
                discoveries["cryptodotcom"] = cryptodotcom
                cdc_engine = TradingEngine(
                    settings=s, db=db, discovery=cryptodotcom, aggregator=aggregator,
                    analyzer=analyzer, cache=cache, risk_manager=risk_manager,
                    exchange=cryptodotcom, calibration=calibration,
                    flow_tracker=flow_tracker,
                    allocator=allocator,
                )
                cdc_engine.strategic = strategic
                cdc_engine.exchange_name = "cryptodotcom"
                engines["cryptodotcom"] = cdc_engine
            except ImportError:
                log.warning("optional.missing", component="CryptoComClient")

        # Interactive Brokers (optional, guarded import)
        if s.ibkr.enabled and (self._exchange_filter is None or self._exchange_filter == "ibkr"):
            try:
                from auramaur.exchange.ibkr import IBKRClient
                ibkr = IBKRClient(settings=s, paper_trader=paper)
                discoveries["ibkr"] = ibkr
                engines["ibkr"] = TradingEngine(
                    settings=s, db=db, discovery=ibkr, aggregator=aggregator,
                    analyzer=analyzer, cache=cache, risk_manager=risk_manager,
                    exchange=ibkr, calibration=calibration,
                    flow_tracker=flow_tracker,
                )
            except ImportError:
                log.warning("optional.missing", component="IBKRClient")

        # Resolution tracker — auto-detects when markets resolve and feeds
        # outcomes into the calibration loop for Platt scaling updates.
        resolution_tracker = ResolutionTracker(
            db=db,
            calibration=calibration,
            discoveries=discoveries,
        )

        # Cross-platform arbitrage scanner (fee-aware)
        arb_scanner = ArbitrageScanner(
            discoveries=discoveries,
            exchange_fees=s.arbitrage.exchange_fees,
            min_profit_after_fees_pct=s.arbitrage.min_profit_after_fees_pct,
        )

        # News reactor — monitors RSS for breaking news, triggers fast analysis
        # on every configured exchange (Polymarket + Kalshi).
        news_reactor = None
        if engines and discoveries:
            rss_source = next((s for s in sources if isinstance(s, RSSSource)), RSSSource())
            news_reactor = NewsReactor(
                rss_source=rss_source,
                discoveries=discoveries,
                engines=engines,
                db=db,
            )

        # Attribution (optional)
        attributor = None
        try:
            from auramaur.monitoring.attribution import PerformanceAttributor
            attributor = PerformanceAttributor(db=db)
        except ImportError:
            log.warning("optional.missing", component="PerformanceAttributor")

        # Performance feedback loop
        feedback = None
        try:
            from auramaur.broker.feedback import PerformanceFeedback
            feedback = PerformanceFeedback(db=db)
        except ImportError:
            log.warning("optional.missing", component="PerformanceFeedback")

        # Correlation & Arbitrage (optional)
        correlator = None
        arb_executor = None
        try:
            from auramaur.strategy.correlation import CorrelationDetector
            from auramaur.strategy.arbitrage import ArbitrageExecutor
            correlator = CorrelationDetector(db=db, model=s.nlp.model)
            arb_executor = ArbitrageExecutor(db=db, correlator=correlator)
        except ImportError:
            log.warning("optional.missing", component="CorrelationDetector/ArbitrageExecutor")

        # WebSocket & Ensemble (optional — only if ensemble enabled)
        ws = None
        ensemble = None
        if s.ensemble.enabled:
            try:
                from auramaur.exchange.websocket import PolymarketWebSocket
                ws = PolymarketWebSocket()
            except ImportError:
                log.warning("optional.missing", component="PolymarketWebSocket")
            try:
                from auramaur.nlp.ensemble import EnsembleEstimator
                ensemble = EnsembleEstimator(db=db)
            except ImportError:
                log.warning("optional.missing", component="EnsembleEstimator")

        # Market maker (optional, enabled by config — Polymarket only).
        # Intentionally not wired for Kalshi: the 7% fee on winnings eats
        # the maker spread, thin top-of-book liquidity makes adverse
        # selection worse, and Kalshi has no maker-rebate program
        # equivalent to Polymarket's. Revisit if Kalshi fees drop or a
        # rebate tier is introduced.
        market_maker = None
        if s.market_maker.enabled and exchange is not None:
            market_maker = MarketMaker(settings=s, exchange=exchange, db=db)

        # Alerts
        alerts = AlertManager(
            telegram_bot_token=s.telegram_bot_token,
            telegram_chat_id=s.telegram_chat_id,
            discord_webhook_url=s.discord_webhook_url,
        )

        # Use first available discovery/exchange as primary (for portfolio monitor etc.)
        primary_discovery = gamma if gamma else next(iter(discoveries.values()), None)
        primary_exchange = exchange if exchange else next(
            (engines[k]._exchange for k in engines if hasattr(engines[k], '_exchange')), None  # type: ignore[attr-defined]
        )

        self._components = {
            "db": db, "aggregator": aggregator, "discovery": primary_discovery,
            "discoveries": discoveries,
            "paper": paper, "exchange": primary_exchange, "analyzer": analyzer,
            "exchanges": exchanges_map,
            "cache": cache, "calibration": calibration,
            "risk_manager": risk_manager, "flow_tracker": flow_tracker,
            "engines": engines, "news_reactor": news_reactor,
            "pnl_tracker": pnl_tracker, "syncer": syncer, "syncers": syncers,
            "reconciler": reconciler,
            "router": router, "allocator": allocator,
            "attributor": attributor, "feedback": feedback,
            "correlator": correlator, "arb_executor": arb_executor,
            "arb_scanner": arb_scanner,
            "resolution_tracker": resolution_tracker,
            "depth_agent": depth_agent,
            "market_maker": market_maker,
            "alerts": alerts,
            "websocket": ws, "ensemble": ensemble,
            "source_names": source_names,
            "exchange_filter": self._exchange_filter,
        }

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
        if Path("KILL_SWITCH").exists():
            show_error("KILL SWITCH ACTIVE — halting all trading")
            self._running = False
            alerts = self._components.get("alerts")
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
        pnl_tracker: PnLTracker = self._components["pnl_tracker"]
        discoveries: dict[str, MarketDiscovery] = self._components["discoveries"]
        exchanges: dict[str, ExchangeClient] = self._components.get("exchanges", {})
        alerts: AlertManager = self._components["alerts"]
        portfolio_tracker: PortfolioTracker = self._components["risk_manager"].portfolio
        interval = self.settings.intervals.portfolio_check_seconds
        first_tick = True

        while self._running:
            try:
                all_positions = []
                per_exchange_cash: dict[str, float] = {}

                for syncer in syncers:
                    name = getattr(syncer, "exchange_name", "polymarket")
                    try:
                        positions_list = await syncer.sync()
                        cash = await syncer.get_cash_balance()
                    except Exception as e:
                        log.debug("portfolio_monitor.sync_error", exchange=name, error=str(e))
                        continue

                    per_exchange_cash[name] = cash

                    # Polymarket-only: enrich with reconciler-discovered manual buys
                    if name == "polymarket":
                        reconciler_comp = self._components.get("reconciler")
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
                        attributor = self._components.get("attributor")
                        if attributor:
                            accuracy_map, kelly_map = await attributor.get_accuracy_and_kelly_maps()
                            category_lookup = await attributor.get_category_lookup()
                        cat_stats = build_category_stats_from_positions(
                            all_positions, accuracy_map, kelly_map, category_lookup,
                        )
                        if cat_stats:
                            show_category_performance(cat_stats)
                    except Exception as e:
                        log.debug("attribution.initial_error", error=str(e))

                show_portfolio(poly_cash, total_pnl, len(all_positions), 0.0, schedule_mode=self._get_schedule_mode())

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
                        fail_key = f"exit_fail:{name}:{pos.market_id}"
                        if fail_key in self._exit_failures:
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
                            else:
                                ok = False
                        except Exception as e:
                            log.warning("exit.execute_error", exchange=name, market_id=pos.market_id, error=str(e))
                            ok = False
                        if not ok:
                            self._exit_failures.add(fail_key)
            except Exception as e:
                log.debug("portfolio_monitor_error", error=str(e))
            await asyncio.sleep(interval)

    async def _execute_poly_exit(
        self,
        pos,
        reason,
        discovery: MarketDiscovery,
        exchange: ExchangeClient,
        alerts: AlertManager,
    ) -> bool:
        """Execute an exit for a Polymarket position.

        Returns True if the sell was accepted (or a retry is worth attempting
        next cycle), False if we should stop retrying this position.
        """
        # Resolve the real token_id: reconciler → cost_basis → Gamma
        token_id = ""
        reconciler_comp = self._components.get("reconciler")
        if reconciler_comp and self.settings.is_live:
            try:
                for rp in await reconciler_comp.reconcile():
                    if rp.market_id == pos.market_id or rp.question == pos.market_id:
                        token_id = rp.token_id
                        log.info("exit.token_from_reconciler",
                                 market_id=pos.market_id, token_id=token_id[:20])
                        break
            except Exception as e:
                log.debug("exit.reconciler_error", error=str(e))

        if not token_id:
            try:
                row = await self._components["db"].fetchone(
                    "SELECT token_id FROM cost_basis WHERE market_id = ? AND size > 0",
                    (pos.market_id,),
                )
                if row and row["token_id"]:
                    token_id = row["token_id"]
            except Exception:
                pass

        if not token_id:
            market_data = await discovery.get_market(pos.market_id)
            if market_data:
                if pos.token == TokenType.NO:
                    token_id = market_data.clob_token_no or market_data.clob_token_yes
                else:
                    token_id = market_data.clob_token_yes or market_data.clob_token_no

        if not token_id:
            log.debug("exit.no_token", market_id=pos.market_id)
            return False

        # Cancel stale sell orders so balance is free
        if hasattr(exchange, "cancel_open_orders_for_token"):
            try:
                await exchange.cancel_open_orders_for_token(token_id)
            except Exception as e:
                log.debug("exit.pre_cancel_error", error=str(e))

        # On-chain balance ground truth
        sell_size = pos.size
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
            exchange._init_clob_client()
            bal = exchange._clob_client.get_balance_allowance(
                BalanceAllowanceParams(
                    asset_type=AssetType.CONDITIONAL,
                    token_id=token_id,
                    signature_type=2,
                )
            )
            onchain = int(bal.get("balance", 0)) / 1e6
            if onchain < sell_size:
                log.info("exit.size_adjusted",
                         market_id=pos.market_id,
                         db_size=sell_size, onchain=onchain)
                sell_size = onchain
        except Exception as e:
            log.debug("exit.balance_check_error", error=str(e))

        if sell_size < 5:
            log.debug("exit.too_small", market_id=pos.market_id, size=sell_size)
            return False
        if pos.current_price < 0.01:
            log.debug("exit.near_zero", market_id=pos.market_id, price=pos.current_price)
            return False

        sell_price = max(0.01, min(0.99, round(pos.current_price, 2)))
        sell_order = Order(
            market_id=pos.market_id,
            token_id=token_id,
            side=OrderSide.SELL,
            size=sell_size,
            price=sell_price,
            order_type=OrderType.LIMIT,
            dry_run=not self.settings.is_live,
        )
        result = await exchange.place_order(sell_order)
        if result.status == "rejected":
            log.warning("exit.sell_failed", market_id=pos.market_id)
            return False

        await alerts.send(
            f"Exit {reason.value} (poly): {pos.market_id[:12]} "
            f"size={pos.size:.2f} pnl={pos.unrealized_pnl:+.2f}",
            level="warning",
        )
        return True

    async def _execute_kalshi_exit(
        self,
        pos,
        reason,
        discovery: MarketDiscovery,
        exchange: ExchangeClient,
        alerts: AlertManager,
    ) -> bool:
        """Execute an exit for a Kalshi position.

        Kalshi is ticker-based with direct YES/NO sells, so we just build a
        SELL signal with ``exit_token`` set and let the exchange's
        ``prepare_order`` do the rest.
        """
        from auramaur.exchange.models import Confidence, Market, Signal

        market = await discovery.get_market(pos.market_id)
        if market is None:
            log.debug("exit.no_market", market_id=pos.market_id)
            return False

        exit_signal = Signal(
            market_id=pos.market_id,
            market_question=market.question,
            claude_prob=0.5,
            claude_confidence=Confidence.MEDIUM,
            market_prob=0.5,
            edge=10.0,
            evidence_summary=f"Exit: {reason.value}",
            recommended_side=OrderSide.SELL,
            exit_token=pos.token,
        )

        # Notional to feed into prepare_order sizing
        notional = pos.size * max(pos.current_price, 0.01)
        order = exchange.prepare_order(exit_signal, market, notional, self.settings.is_live)
        if order is None:
            return False

        # Never sell more than we hold
        order.size = min(order.size, pos.size)
        if order.size < 1:
            return False

        result = await exchange.place_order(order)
        from auramaur.monitoring.display import show_order
        show_order(
            result.status, result.order_id, "SELL", order.size, order.price,
            result.is_paper, exchange="kalshi", error_message=result.error_message,
        )
        if result.status == "rejected":
            return False

        await alerts.send(
            f"Exit {reason.value} (kalshi): {pos.market_id} "
            f"size={pos.size:.0f} pnl={pos.unrealized_pnl:+.2f}",
            level="warning",
        )
        return True

    async def _task_cache_cleanup(self) -> None:
        """Periodically clean expired NLP cache entries."""
        cache: NLPCache = self._components["cache"]

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
        from auramaur.broker.redeemer import (
            fetch_redeemable_positions, summarize_redemptions,
        )
        from auramaur.monitoring.display import console

        proxy = self.settings.polymarket_proxy_address
        if not proxy:
            return  # no proxy configured, can't check

        # Track the last-notified payout to avoid spamming on every cycle
        last_notified_payout: float = -1.0

        while self._running:
            try:
                positions = await fetch_redeemable_positions(proxy)
                summary = summarize_redemptions(positions)
                payout = summary["payout_now_usdc"]

                if summary["redeemable_now"] > 0 and payout > 0 and payout != last_notified_payout:
                    console.print()
                    console.print(
                        f"[bold green]╔══ REDEMPTION AVAILABLE ══╗[/]"
                    )
                    console.print(
                        f"[bold green]║[/] [green]${payout:.2f} USDC[/] ready to "
                        f"redeem on Polymarket — {summary['winning_now']} winning "
                        f"position{'s' if summary['winning_now'] != 1 else ''} "
                        f"(net [green]${summary['net_pnl_now']:+.2f}[/])"
                    )
                    console.print(
                        f"[bold green]║[/] [dim]→ https://polymarket.com/portfolio  "
                        f"or run:[/] [cyan]auramaur redeem-check[/]"
                    )
                    console.print(
                        f"[bold green]╚══════════════════════════╝[/]"
                    )
                    console.print()
                    last_notified_payout = payout
            except Exception as e:
                log.debug("redemption_check.error", error=str(e))

            await asyncio.sleep(3600)  # hourly

    async def _task_kill_switch_monitor(self) -> None:
        """Rapid kill switch polling."""
        while self._running:
            if await self._check_kill_switch():
                return
            await asyncio.sleep(1)

    async def _task_recalibrate(self) -> None:
        """Periodically refit Platt scaling calibration parameters."""
        calibration: CalibrationTracker = self._components["calibration"]
        interval = self.settings.calibration.refit_interval_hours * 3600

        while self._running:
            try:
                await calibration.refit_all()
            except Exception as e:
                log.error("recalibrate.error", error=str(e))
            await asyncio.sleep(interval)

    async def _task_attribution_update(self) -> None:
        """Periodically update performance attribution and Kelly multipliers."""
        attributor = self._components.get("attributor")
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

    async def _task_performance_feedback(self) -> None:
        """Periodically update per-category calibration stats and Kelly multipliers."""
        feedback = self._components.get("feedback")
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

    async def _task_correlation_scan(self) -> None:
        """Periodically scan for correlated markets and execute arbitrage."""
        correlator = self._components.get("correlator")
        arb_executor = self._components.get("arb_executor")
        if correlator is None or arb_executor is None:
            return
        discovery: MarketDiscovery = self._components["discovery"]
        risk_manager: RiskManager = self._components["risk_manager"]

        while self._running:
            if await self._check_kill_switch():
                return
            try:
                markets = await discovery.get_markets(limit=20)
                if markets:
                    await correlator.detect_relationships(markets)

                    # Generate and execute arbitrage signals
                    pairs = await arb_executor.generate_arb_signals()
                    for buy_signal, sell_signal, opp in pairs:
                        try:
                            # Load markets for risk evaluation
                            buy_market = await arb_executor._load_market(buy_signal.market_id)
                            sell_market = await arb_executor._load_market(sell_signal.market_id)
                            if not buy_market or not sell_market:
                                continue

                            # Run risk checks on both legs
                            buy_decision = await risk_manager.evaluate(buy_signal, buy_market)
                            sell_decision = await risk_manager.evaluate(sell_signal, sell_market)

                            if buy_decision.approved and sell_decision.approved:
                                log.info(
                                    "arbitrage.executing",
                                    buy_market=buy_signal.market_id,
                                    sell_market=sell_signal.market_id,
                                    type=opp.get("type"),
                                )
                                alerts: AlertManager = self._components["alerts"]
                                await alerts.send(
                                    f"Executing arbitrage: {opp.get('type')} "
                                    f"buy {buy_signal.market_id[:12]} / sell {sell_signal.market_id[:12]}",
                                    level="warning",
                                )
                                # Execute both legs
                                # Note: Uses analyze_market which goes through full order flow
                                # For now, just log — actual execution would need
                                # the engine to accept pre-computed signals
                            else:
                                log.debug(
                                    "arbitrage.risk_rejected",
                                    buy_approved=buy_decision.approved,
                                    sell_approved=sell_decision.approved,
                                )
                        except Exception as e:
                            log.debug("arbitrage.execution_error", error=str(e))
            except Exception as e:
                log.error("correlation_scan.error", error=str(e))
            await asyncio.sleep(14400)  # Every 4 hours

    async def _task_depth_research(self) -> None:
        """Run deep research on the most promising markets.

        Complements the strategic loop (breadth) with deep-dive analysis
        on markets where the strategic batch found potential edge but
        confidence was low or the market is high-value.
        """
        from auramaur.strategy.agent_analyzer import AgentAnalyzer

        depth_agent: AgentAnalyzer = self._components["depth_agent"]
        db: Database = self._components["db"]
        engines: dict[str, TradingEngine] = self._components["engines"]
        engine = engines.get("polymarket")
        if engine is None:
            return

        while self._running:
            if await self._check_kill_switch():
                return
            try:
                # Find markets with high edge but low confidence from recent signals.
                # CRITICAL: filter to the same exchange as the engine (polymarket).
                # Without this filter, Kalshi signals get routed through the
                # Polymarket CLOB client and fail with "No CLOB token_id".
                rows = await db.fetchall(
                    """SELECT s.market_id, m.question, m.description, m.category,
                              m.outcome_yes_price, m.outcome_no_price, m.end_date,
                              m.volume, m.liquidity,
                              s.edge, s.claude_confidence
                       FROM signals s
                       JOIN markets m ON s.market_id = m.id
                       WHERE s.timestamp > datetime('now', '-6 hours')
                         AND ABS(s.edge) >= 8
                         AND s.claude_confidence IN ('LOW', 'MEDIUM_LOW', 'MEDIUM')
                         AND m.active = 1
                         AND m.exchange = 'polymarket'
                       ORDER BY ABS(s.edge) DESC
                       LIMIT 3"""
                )

                for row in rows:
                    if await self._check_kill_switch():
                        return

                    # Build market from DB + enrich from Gamma
                    from auramaur.exchange.models import Market
                    market = Market(
                        id=row["market_id"],
                        question=row["question"] or "",
                        description=row["description"] or "",
                        category=row["category"] or "",
                        outcome_yes_price=row["outcome_yes_price"] or 0.5,
                        outcome_no_price=row["outcome_no_price"] or 0.5,
                        volume=row["volume"] or 0,
                        liquidity=row["liquidity"] or 0,
                    )
                    try:
                        end_str = row["end_date"]
                        if end_str:
                            from datetime import datetime
                            market.end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                    except Exception:
                        pass
                    # Enrich with Gamma data for CLOB tokens
                    try:
                        discovery = self._components["discovery"]
                        full_market = await discovery.get_market(market.id)
                        if full_market:
                            market.clob_token_yes = full_market.clob_token_yes
                            market.clob_token_no = full_market.clob_token_no
                            market.condition_id = full_market.condition_id
                            if full_market.description and len(full_market.description) > len(market.description):
                                market.description = full_market.description
                    except Exception:
                        pass

                    log.info(
                        "depth.researching",
                        market_id=market.id,
                        question=market.question[:60],
                        initial_edge=row["edge"],
                    )

                    candidate = await depth_agent.deep_research(market)
                    if candidate:
                        # Run through risk checks and execution
                        from auramaur.strategy.protocols import TradeCandidate
                        results = await engine._execute_candidates([candidate])
                        trades = [r for r in results if r.get("order")]
                        if trades:
                            log.info(
                                "depth.trade_placed",
                                market_id=market.id,
                                edge=round(candidate.signal.edge, 1),
                            )
                            alerts: AlertManager = self._components["alerts"]
                            await alerts.send(
                                f"Depth research trade: {market.question[:40]} "
                                f"edge={candidate.signal.edge:.1f}%",
                                level="info",
                            )

            except Exception as e:
                log.error("depth.error", error=str(e))
            await asyncio.sleep(1800)  # Every 30 minutes

    async def _task_arb_scanner(self) -> None:
        """Periodically scan all exchanges for arbitrage opportunities."""
        scanner: ArbitrageScanner = self._components["arb_scanner"]
        alerts: AlertManager = self._components["alerts"]
        risk_manager: RiskManager = self._components["risk_manager"]
        engines: dict[str, TradingEngine] = self._components["engines"]

        while self._running:
            if await self._check_kill_switch():
                return
            try:
                opportunities = await scanner.scan()

                for opp in opportunities:
                    # Log every opportunity
                    log.info(
                        "arb_scanner.opportunity",
                        arb_type=opp.arb_type,
                        question=opp.question[:80],
                        exchange_a=opp.exchange_a,
                        exchange_b=opp.exchange_b,
                        price_a=round(opp.price_a, 3),
                        price_b=round(opp.price_b, 3),
                        spread=round(opp.spread, 3),
                        profit_pct=round(opp.expected_profit_pct, 2),
                    )

                    # Alert on significant opportunities (> 5% expected profit)
                    if opp.expected_profit_pct > 5.0:
                        await alerts.send(
                            f"Arb opportunity ({opp.arb_type}): "
                            f"{opp.question[:60]} | "
                            f"{opp.exchange_a} {opp.price_a:.2f} vs "
                            f"{opp.exchange_b} {opp.price_b:.2f} | "
                            f"profit: {opp.expected_profit_pct:.1f}%",
                            level="warning",
                        )

                    # Auto-execute internal arbs (YES+NO < 0.97) if risk checks pass
                    if opp.arb_type == "internal":
                        await self._execute_internal_arb(opp, risk_manager, engines)
                    elif opp.arb_type == "cross_exchange" and self.settings.arbitrage.cross_exchange_auto_execute:
                        await self._execute_cross_exchange_arb(opp, risk_manager, engines)

            except Exception as e:
                log.error("arb_scanner.task_error", error=str(e))
            await asyncio.sleep(300)  # Every 5 minutes

    async def _execute_internal_arb(
        self,
        opp: ArbOpportunity,
        risk_manager: RiskManager,
        engines: dict[str, TradingEngine],
    ) -> None:
        """Execute an internal arb: buy both YES and NO when their sum < 0.97.

        Both legs must pass risk checks independently before execution.
        """
        from auramaur.exchange.models import Confidence, Signal

        market = opp.market_a  # Same market for both sides
        exchange_name = opp.exchange_a
        engine = engines.get(exchange_name)
        if engine is None:
            log.debug("arb_scanner.no_engine", exchange=exchange_name)
            return

        alerts: AlertManager = self._components["alerts"]

        # Build a synthetic signal for the YES leg
        edge_pct = opp.expected_profit_pct / 2  # Split edge across both legs
        yes_signal = Signal(
            market_id=market.id,
            market_question=market.question,
            claude_prob=market.outcome_yes_price + opp.spread / 2,
            claude_confidence=Confidence.HIGH,
            market_prob=market.outcome_yes_price,
            edge=edge_pct,
            evidence_summary=f"Internal arb: YES+NO={opp.price_a + opp.price_b:.3f}",
            recommended_side=OrderSide.BUY,
        )

        # Build a synthetic signal for the NO leg (buy NO)
        no_signal = Signal(
            market_id=market.id,
            market_question=market.question,
            claude_prob=market.outcome_no_price + opp.spread / 2,
            claude_confidence=Confidence.HIGH,
            market_prob=market.outcome_no_price,
            edge=edge_pct,
            evidence_summary=f"Internal arb: YES+NO={opp.price_a + opp.price_b:.3f}",
            recommended_side=OrderSide.BUY,
        )

        # Run risk checks on both legs
        yes_decision = await risk_manager.evaluate(yes_signal, market)
        no_decision = await risk_manager.evaluate(no_signal, market)

        if not yes_decision.approved or not no_decision.approved:
            log.debug(
                "arb_scanner.internal_risk_rejected",
                market_id=market.id,
                yes_approved=yes_decision.approved,
                no_approved=no_decision.approved,
                yes_reason=yes_decision.reason,
                no_reason=no_decision.reason,
            )
            return

        # Execute both legs -- use the smaller position size for balance
        position_size = min(yes_decision.position_size, no_decision.position_size)
        if position_size <= 0:
            return

        exchange_client = engine._exchange  # type: ignore[attr-defined]

        # YES leg
        yes_order = Order(
            market_id=market.id,
            exchange=exchange_name,
            token_id=market.clob_token_yes or market.id,
            side=OrderSide.BUY,
            size=position_size / market.outcome_yes_price if market.outcome_yes_price > 0 else 0,
            price=market.outcome_yes_price,
            dry_run=not self.settings.is_live,
        )

        # NO leg
        no_order = Order(
            market_id=market.id,
            exchange=exchange_name,
            token_id=market.clob_token_no or market.id,
            side=OrderSide.BUY,
            size=position_size / market.outcome_no_price if market.outcome_no_price > 0 else 0,
            price=market.outcome_no_price,
            dry_run=not self.settings.is_live,
        )

        if yes_order.size < 1 or no_order.size < 1:
            return

        try:
            yes_result = await exchange_client.place_order(yes_order)
            no_result = await exchange_client.place_order(no_order)

            log.info(
                "arb_scanner.internal_executed",
                market_id=market.id,
                question=market.question[:60],
                yes_status=yes_result.status,
                no_status=no_result.status,
                yes_size=round(yes_order.size, 2),
                no_size=round(no_order.size, 2),
                profit_pct=round(opp.expected_profit_pct, 2),
                is_paper=yes_order.dry_run,
            )

            mode = "PAPER" if yes_order.dry_run else "LIVE"
            await alerts.send(
                f"[{mode}] Internal arb executed: {market.question[:50]} | "
                f"YES@{opp.price_a:.2f} + NO@{opp.price_b:.2f} = "
                f"{opp.price_a + opp.price_b:.3f} | "
                f"profit: {opp.expected_profit_pct:.1f}%",
                level="warning",
            )
        except Exception as e:
            log.error(
                "arb_scanner.internal_execution_error",
                market_id=market.id,
                error=str(e),
            )

    async def _execute_cross_exchange_arb(
        self,
        opp: ArbOpportunity,
        risk_manager: RiskManager,
        engines: dict[str, TradingEngine],
    ) -> None:
        """Execute a cross-exchange arb: buy cheap YES on one exchange, buy
        equivalent NO on the other.

        Both legs go through each exchange's ``prepare_order`` so tick rounding,
        token-id resolution, and minimum-size checks match what the exchange
        accepts. Both legs must pass risk checks independently. Execution is
        concurrent; half-fills trigger a cancel with a critical alert if
        cancel can't confirm.
        """
        from auramaur.exchange.models import Confidence, Signal

        # Identify which side is cheap (lower YES price)
        if opp.price_a <= opp.price_b:
            cheap_market, expensive_market = opp.market_a, opp.market_b
            cheap_exchange, expensive_exchange = opp.exchange_a, opp.exchange_b
        else:
            cheap_market, expensive_market = opp.market_b, opp.market_a
            cheap_exchange, expensive_exchange = opp.exchange_b, opp.exchange_a

        engine_cheap = engines.get(cheap_exchange)
        engine_expensive = engines.get(expensive_exchange)
        if engine_cheap is None or engine_expensive is None:
            log.debug(
                "arb_scanner.cross_no_engine",
                cheap=cheap_exchange,
                expensive=expensive_exchange,
            )
            return

        alerts: AlertManager = self._components["alerts"]
        max_size = self.settings.arbitrage.max_arb_size

        # Synthetic signals for risk checks and prepare_order
        edge_pct = opp.expected_profit_pct / 2
        evidence = (
            f"Cross-exchange arb: {cheap_exchange} YES@{cheap_market.outcome_yes_price:.3f} "
            f"vs {expensive_exchange} YES@{expensive_market.outcome_yes_price:.3f}"
        )
        yes_signal = Signal(
            market_id=cheap_market.id,
            market_question=cheap_market.question,
            claude_prob=cheap_market.outcome_yes_price + opp.spread / 2,
            claude_confidence=Confidence.HIGH,
            market_prob=cheap_market.outcome_yes_price,
            edge=edge_pct,
            evidence_summary=evidence,
            recommended_side=OrderSide.BUY,
        )
        # SELL signal → each exchange's prepare_order produces a BUY NO
        # (Polymarket token-swap, Kalshi new-bearish-position branch).
        no_signal = Signal(
            market_id=expensive_market.id,
            market_question=expensive_market.question,
            claude_prob=expensive_market.outcome_no_price + opp.spread / 2,
            claude_confidence=Confidence.HIGH,
            market_prob=expensive_market.outcome_no_price,
            edge=edge_pct,
            evidence_summary=evidence,
            recommended_side=OrderSide.SELL,
        )

        # Risk checks on both legs
        yes_decision = await risk_manager.evaluate(yes_signal, cheap_market)
        no_decision = await risk_manager.evaluate(no_signal, expensive_market)
        if not yes_decision.approved or not no_decision.approved:
            log.debug(
                "arb_scanner.cross_risk_rejected",
                question=opp.question[:60],
                yes_approved=yes_decision.approved,
                no_approved=no_decision.approved,
            )
            return

        position_size = min(
            yes_decision.position_size,
            no_decision.position_size,
            max_size,
        )
        if position_size <= 0:
            return

        cheap_client = engine_cheap.exchange
        expensive_client = engine_expensive.exchange
        is_live = self.settings.is_live

        yes_order = cheap_client.prepare_order(yes_signal, cheap_market, position_size, is_live)
        no_order = expensive_client.prepare_order(no_signal, expensive_market, position_size, is_live)
        if yes_order is None or no_order is None:
            log.debug(
                "arb_scanner.cross_prepare_failed",
                question=opp.question[:60],
                yes_built=yes_order is not None,
                no_built=no_order is not None,
            )
            return

        # Dedup: 1-hour cooldown per (question, exchange-pair). Set before
        # placement so concurrent scans don't double-fire on the same arb.
        import time as _time
        arb_key = f"{opp.question[:80]}|{cheap_exchange}|{expensive_exchange}"
        now_ts = _time.monotonic()
        expiry = self._arb_attempts.get(arb_key)
        if expiry and expiry > now_ts:
            log.debug(
                "arb_scanner.cross_dedup_skip",
                question=opp.question[:60],
                cheap_exchange=cheap_exchange,
                expensive_exchange=expensive_exchange,
            )
            return
        self._arb_attempts[arb_key] = now_ts + 3600
        if len(self._arb_attempts) > 200:
            self._arb_attempts = {
                k: v for k, v in self._arb_attempts.items() if v > now_ts
            }

        try:
            yes_result, no_result = await asyncio.gather(
                cheap_client.place_order(yes_order),
                expensive_client.place_order(no_order),
            )

            yes_ok = yes_result.status not in ("rejected", "error")
            no_ok = no_result.status not in ("rejected", "error")
            if yes_ok != no_ok:
                rollback_client = cheap_client if yes_ok else expensive_client
                rollback_result = yes_result if yes_ok else no_result
                rollback_exchange = cheap_exchange if yes_ok else expensive_exchange
                log.warning(
                    "arb_scanner.cross_half_fill_rollback",
                    question=opp.question[:60],
                    yes_status=yes_result.status,
                    no_status=no_result.status,
                    rollback_exchange=rollback_exchange,
                    rollback_order_id=rollback_result.order_id,
                )
                cancelled = False
                # "unknown"/"ERROR" are sentinel IDs the exchange clients
                # return when they couldn't parse a real order_id — cancel
                # would no-op. Skip straight to the critical alert path.
                real_id = rollback_result.order_id not in ("", "unknown", "ERROR", "BLOCKED", "SKIP_DUP")
                if real_id:
                    try:
                        cancelled = bool(await rollback_client.cancel_order(rollback_result.order_id))
                    except Exception as e:
                        log.error(
                            "arb_scanner.cross_rollback_failed",
                            question=opp.question[:60],
                            error=str(e),
                        )
                if not cancelled:
                    # Unhedged directional exposure — operator must intervene.
                    await alerts.send(
                        f"[CRITICAL] Unhedged arb leg on {rollback_exchange}: "
                        f"{opp.question[:60]} order_id={rollback_result.order_id} "
                        f"size={rollback_result.filled_size or 'unknown'} — "
                        "cancel failed or id unknown. Manual flatten required.",
                        level="critical",
                    )

            log.info(
                "arb_scanner.cross_executed",
                question=opp.question[:60],
                cheap_exchange=cheap_exchange,
                expensive_exchange=expensive_exchange,
                yes_status=yes_result.status,
                no_status=no_result.status,
                yes_size=round(yes_order.size, 2),
                no_size=round(no_order.size, 2),
                spread=round(opp.spread, 3),
                profit_pct=round(opp.expected_profit_pct, 2),
                is_paper=yes_order.dry_run,
            )

            if yes_ok and no_ok:
                mode = "PAPER" if yes_order.dry_run else "LIVE"
                await alerts.send(
                    f"[{mode}] Cross-exchange arb executed: {opp.question[:50]} | "
                    f"BUY YES@{yes_order.price:.2f} on {cheap_exchange}, "
                    f"BUY NO@{no_order.price:.2f} on {expensive_exchange} | "
                    f"profit: {opp.expected_profit_pct:.1f}%",
                    level="warning",
                )
        except Exception as e:
            log.error(
                "arb_scanner.cross_execution_error",
                question=opp.question[:60],
                error=str(e),
            )

    async def _rebalance_concentrated_positions(self, engine: TradingEngine) -> None:
        """Trim oversized positions to free up capital for diversification.

        If any single event exceeds MAX_EVENT_PCT of total portfolio exposure,
        sell contracts to bring it down to TARGET_EVENT_PCT.  This prevents
        concentration risk from locking up all capital in one bet.
        """
        from auramaur.exchange.models import Confidence, Signal, TokenType as TT

        MAX_EVENT_PCT = 0.30   # trigger rebalance above 30%
        TARGET_EVENT_PCT = 0.20  # trim down to 20%

        db = engine.db
        rows = await db.fetchall(
            """SELECT p.market_id, p.token, p.size, p.avg_price, p.current_price,
                      m.outcome_yes_price, m.outcome_no_price, m.spread, m.question
               FROM portfolio p
               JOIN markets m ON p.market_id = m.id
               WHERE p.size > 0 AND m.exchange = 'kalshi'"""
        )

        if not rows:
            return

        # Group by event and compute exposure
        events: dict[str, list] = {}
        total_exposure = 0.0
        for r in rows:
            mid = r["market_id"]
            token = r["token"] or "YES"
            size = r["size"]
            price = r["current_price"] or r["avg_price"] or 0
            exposure = size * price

            # Extract event base from ticker
            if mid.count("-") >= 2:
                event_key = mid.rsplit("-", 1)[0]
            else:
                event_key = mid

            events.setdefault(event_key, []).append({
                "row": r, "exposure": exposure, "mid": mid,
                "token": token, "size": size, "price": price,
            })
            total_exposure += exposure

        if total_exposure <= 0:
            return

        # Check each event for overconcentration
        for event_key, positions in events.items():
            event_exposure = sum(p["exposure"] for p in positions)
            event_pct = event_exposure / total_exposure

            if event_pct <= MAX_EVENT_PCT:
                continue

            target_exposure = total_exposure * TARGET_EVENT_PCT
            excess = event_exposure - target_exposure

            log.info(
                "rebalance.triggered",
                event=event_key,
                event_pct=f"{event_pct:.0%}",
                exposure=round(event_exposure, 2),
                target=round(target_exposure, 2),
                excess=round(excess, 2),
            )
            from datetime import datetime, timezone
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            console.print(
                f"[dim]{ts}[/] [bold yellow]REBALANCE[/] {event_key} "
                f"at [red]{event_pct:.0%}[/] of portfolio — trimming to {TARGET_EVENT_PCT:.0%}"
            )

            # Sell from largest position in this event first
            sorted_positions = sorted(positions, key=lambda p: -p["exposure"])
            remaining_excess = excess

            for pos in sorted_positions:
                if remaining_excess <= 0:
                    break

                r = pos["row"]
                token = pos["token"]
                price = pos["price"]
                if price <= 0:
                    continue

                # How many contracts to sell
                contracts_to_sell = min(
                    int(remaining_excess / price),
                    int(pos["size"]) - 1,  # keep at least 1
                )
                if contracts_to_sell < 1:
                    continue

                exit_token = TT.NO if token == "NO" else TT.YES
                exit_signal = Signal(
                    market_id=pos["mid"],
                    market_question=r.get("question", ""),
                    claude_prob=0.5,
                    claude_confidence=Confidence.MEDIUM,
                    market_prob=0.5,
                    edge=5.0,
                    evidence_summary=f"Rebalance: {event_key} at {event_pct:.0%}",
                    recommended_side=OrderSide.SELL,
                    exit_token=exit_token,
                )

                from auramaur.exchange.models import Market
                market = Market(
                    id=pos["mid"],
                    exchange="kalshi",
                    ticker=pos["mid"],
                    question=r.get("question", ""),
                    outcome_yes_price=r["outcome_yes_price"] or 0.5,
                    outcome_no_price=r["outcome_no_price"] or 0.5,
                    spread=r["spread"] or 0,
                )

                order = engine.exchange.prepare_order(
                    exit_signal, market, contracts_to_sell * price, self.settings.is_live,
                )
                if order is None:
                    continue

                order.size = min(order.size, contracts_to_sell)
                if order.size < 1:
                    continue

                result = await engine.exchange.place_order(order)
                from auramaur.monitoring.display import show_order
                show_order(
                    result.status, result.order_id, "SELL", order.size,
                    order.price, result.is_paper, exchange="kalshi",
                    error_message=result.error_message,
                )

                if result.status not in ("rejected",):
                    sell_value = order.size * order.price
                    remaining_excess -= sell_value

                    # Block re-entry into this event for 24 hours (DB-persisted)
                    await engine.db.execute(
                        """INSERT OR REPLACE INTO rebalance_blocks
                           (event_key, blocked_until, reason)
                           VALUES (?, datetime('now', '+24 hours'), ?)""",
                        (event_key, f"rebalanced from {event_pct:.0%}"),
                    )
                    await engine.db.commit()

                    console.print(
                        f"         [yellow]Trimmed[/] {pos['mid']} "
                        f"—{int(order.size)} contracts (${sell_value:.2f}) "
                        f"[dim](blocked 24h)[/]"
                    )

                    alerts = self._components.get("alerts")
                    if alerts:
                        await alerts.send(
                            f"Rebalance: trimmed {pos['mid']} "
                            f"by {int(order.size)} contracts — "
                            f"event was {event_pct:.0%} of portfolio",
                            level="warning",
                        )

    async def _task_market_maker(self) -> None:
        """Run market making cycles — post two-sided quotes on liquid markets.

        Uses the Gamma client to find liquid markets, then posts bid/ask
        quotes via the exchange client. Runs every refresh_seconds.
        """
        mm: MarketMaker | None = self._components.get("market_maker")
        if mm is None:
            return

        discovery: MarketDiscovery = self._components["discovery"]
        alerts: AlertManager = self._components["alerts"]
        interval = self.settings.market_maker.refresh_seconds

        while self._running:
            if await self._check_kill_switch():
                return
            try:
                # Fetch liquid markets sorted by volume
                markets = await discovery.get_markets(limit=50, order="liquidity")

                # Run the MM cycle
                results = await mm.run_cycle(markets)

                # Check for fills on pending live orders
                fills = await mm.check_fills()

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
        ws = self._components.get("websocket")
        if ws is None:
            return

        engines: dict[str, TradingEngine] = self._components["engines"]
        # Use primary (polymarket) engine for price-triggered re-analysis
        engine: TradingEngine = engines.get("polymarket", list(engines.values())[0])
        discovery: MarketDiscovery = self._components["discovery"]
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

        flow_tracker = self._components.get("flow_tracker")

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

    async def _task_source_weights_update(self) -> None:
        """Periodically update ensemble source weights."""
        ensemble = self._components.get("ensemble")
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
        """Periodically sync positions from CLOB trade history (ground truth)."""
        from auramaur.broker.reconciler import PositionReconciler
        from auramaur.broker.pnl import PnLTracker
        from auramaur.broker.sync import PositionSyncer

        reconciler: PositionReconciler = self._components["reconciler"]
        syncer: PositionSyncer = self._components["syncer"]
        pnl: PnLTracker = self._components["pnl_tracker"]
        interval = self.settings.broker.sync_interval_seconds

        while self._running:
            try:
                if self.settings.is_live:
                    # Use reconciler for ground truth from CLOB trades
                    reconciled = await reconciler.reconcile()
                    positions = reconciler.to_live_positions(reconciled)

                    # Update cost_basis from real fill prices (ground truth)
                    for rp in reconciled:
                        await self._components["db"].execute(
                            """INSERT INTO cost_basis (market_id, token, token_id, size, avg_cost, total_cost, updated_at)
                               VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                               ON CONFLICT(market_id) DO UPDATE SET
                                   token_id = excluded.token_id,
                                   size = excluded.size,
                                   avg_cost = excluded.avg_cost,
                                   total_cost = excluded.total_cost,
                                   updated_at = excluded.updated_at""",
                            (rp.market_id, rp.outcome, rp.token_id, rp.size,
                             rp.avg_cost, rp.size * rp.avg_cost),
                        )

                    # Delete stale rows from cost_basis AND portfolio: positions no
                    # longer held on-chain. Only run this when the reconciler returned
                    # a non-empty result — an empty reconcile means the CLOB API call
                    # failed, not that we actually hold zero positions, so we must not
                    # wipe the tables. Both tables matter: cost_basis feeds sync(),
                    # and portfolio feeds the risk-manager correlation/exposure checks.
                    if reconciled:
                        live_ids = [rp.market_id for rp in reconciled]
                        placeholders = ",".join("?" * len(live_ids))
                        cb_cur = await self._components["db"].execute(
                            f"DELETE FROM cost_basis WHERE size > 0 AND market_id NOT IN ({placeholders})",
                            live_ids,
                        )
                        pf_cur = await self._components["db"].execute(
                            f"DELETE FROM portfolio WHERE market_id NOT IN ({placeholders})",
                            live_ids,
                        )
                        log.info(
                            "reconciler.stale_removed",
                            cost_basis=cb_cur.rowcount if hasattr(cb_cur, "rowcount") else 0,
                            portfolio=pf_cur.rowcount if hasattr(pf_cur, "rowcount") else 0,
                        )

                    await self._components["db"].commit()
                else:
                    positions = await syncer.sync()

                cash = await syncer.get_cash_balance()
                total_value = sum(p.size * p.current_price for p in positions)
                unrealized = sum(p.unrealized_pnl for p in positions)

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
        reactor: NewsReactor = self._components["news_reactor"]

        while self._running:
            if await self._check_kill_switch():
                return
            try:
                results = await reactor.check_for_news()
                if results:
                    trades = [r for r in results if r.get("order")]
                    if trades:
                        alerts: AlertManager = self._components["alerts"]
                        await alerts.send(
                            f"News reactor triggered {len(trades)} trade(s) from {len(results)} analysis(es)",
                            level="info",
                        )
            except Exception as e:
                show_error(f"News reactor failed: {e}")
            await asyncio.sleep(60)

    async def _task_order_monitor(self) -> None:
        """Monitor pending limit orders for fills and expiry."""
        paper: PaperTrader = self._components["paper"]
        exchange: PolymarketClient = self._components["exchange"]
        discovery: MarketDiscovery = self._components["discovery"]
        ttl = self.settings.execution.limit_order_ttl_seconds

        while self._running:
            try:
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
                for order_id in list(exchange._live_pending.keys()):
                    try:
                        result = await exchange.get_order_status(order_id)
                        if result.status in ("filled", "cancelled", "expired", "rejected"):
                            exchange._live_pending.pop(order_id, None)
                            log.info(
                                "order_monitor.live_terminal",
                                order_id=order_id,
                                status=result.status,
                                filled_size=result.filled_size,
                            )
                    except Exception as e:
                        log.debug("order_monitor.live_poll_error", order_id=order_id, error=str(e))
            except Exception as e:
                log.debug("order_monitor.error", error=str(e))
            await asyncio.sleep(30)

    async def _task_resolution_checker(self) -> None:
        """Poll for resolved markets and record calibration outcomes.

        Delegates to ResolutionTracker which handles multi-exchange
        resolution detection, calibration updates, and position settlement.
        """
        tracker: ResolutionTracker = self._components["resolution_tracker"]

        while self._running:
            if await self._check_kill_switch():
                return
            try:
                resolved = await tracker.check_resolutions()
                if resolved > 0:
                    from datetime import datetime, timezone
                    now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
                    console.print(
                        f"  [dim]{now_str}[/] [bold green]RESOLVED[/] {resolved} market(s) — calibration updated"
                    )
                    # Also feed into attribution if available
                    attributor = self._components.get("attributor")
                    if attributor is not None:
                        db: Database = self._components["db"]
                        # Attribution is handled inside _settle_position via
                        # daily_stats; log for visibility only.
                        log.info("resolution.attribution_notified", count=resolved)
            except Exception as e:
                log.debug("resolution_checker.error", error=str(e))
            await asyncio.sleep(1800)  # Every 30 minutes

    async def run(self) -> None:
        """Start the bot with all concurrent tasks."""
        # Use console renderer for terminal, JSON for log file
        setup_logging(
            level=self.settings.logging.level,
            json_format=False,  # Console-friendly output
            log_file=self.settings.logging.file,
        )

        mode = "LIVE" if self.settings.is_live else "PAPER"
        show_banner(mode, "0.1.0")

        await self._init_components()
        self._running = True

        db_path = self._components["db"].db_path
        if db_path != "auramaur.db":
            console.print(f"  [yellow]Instance: {db_path}[/]")

        # Show real balance — use reconciler for live, paper for paper mode.
        # In live mode we never fall back to paper.balance — that would show
        # paper PnL in a live banner if the reconciler fails to respond.
        startup_balance = 0.0 if self.settings.is_live else self._components["paper"].balance
        if self.settings.is_live:
            try:
                total_cash = 0.0
                total_position_value = 0.0
                total_markets = 0

                # Polymarket balance
                if self._exchange_filter is None or self._exchange_filter == "polymarket":
                    reconciler_comp = self._components.get("reconciler")
                    if reconciler_comp:
                        reconciled = await reconciler_comp.reconcile()
                        position_value = sum(p.size * p.current_price for p in reconciled)
                        syncer_comp = self._components.get("syncer")
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

        show_startup(
            self._components["source_names"],
            startup_balance,
        )

        exchange_filter = self._components.get("exchange_filter")
        if exchange_filter:
            console.print(f"  [cyan]Exchange filter: {exchange_filter} only[/]")

        tasks = [
            asyncio.create_task(self._task_kill_switch_monitor(), name="kill_switch"),
            asyncio.create_task(self._task_cache_cleanup(), name="cache_cleanup"),
            asyncio.create_task(self._task_recalibrate(), name="recalibrate"),
        ]

        # Portfolio monitor runs whenever any exchange syncer is present so
        # Kalshi-only runs still populate `_last_known_cash` and exits fire.
        # Position sync (CLOB reconciler) remains Polymarket-specific.
        if self._components.get("syncers"):
            tasks.append(asyncio.create_task(self._task_portfolio_monitor(), name="portfolio"))
        if self._components.get("syncer"):
            tasks.append(asyncio.create_task(self._task_position_sync(), name="position_sync"))

        # Resolution checker and order monitor work with any exchange
        tasks.append(asyncio.create_task(self._task_resolution_checker(), name="resolution_checker"))
        tasks.append(asyncio.create_task(self._task_order_monitor(), name="order_monitor"))

        # Redemption check — only meaningful with a Polymarket proxy wallet
        if self.settings.polymarket_proxy_address:
            tasks.append(asyncio.create_task(self._task_redemption_check(), name="redemption_check"))

        # News reactor (if available)
        if self._components.get("news_reactor"):
            tasks.append(asyncio.create_task(self._task_news_reactor(), name="news_reactor"))

        # Per-exchange scan + trade tasks
        engines: dict[str, TradingEngine] = self._components["engines"]
        for ex_name, engine in engines.items():
            tasks.append(asyncio.create_task(
                self._task_market_scan(engine, ex_name), name=f"scan_{ex_name}",
            ))
            tasks.append(asyncio.create_task(
                self._task_trading_cycle(engine, ex_name), name=f"trade_{ex_name}",
            ))

        if self._components.get("attributor"):
            tasks.append(asyncio.create_task(self._task_attribution_update(), name="attribution"))
        if self._components.get("correlator"):
            tasks.append(asyncio.create_task(self._task_correlation_scan(), name="correlation"))
        if self._components.get("websocket"):
            tasks.append(asyncio.create_task(self._task_price_monitor(), name="price_monitor"))
        if self._components.get("ensemble"):
            tasks.append(asyncio.create_task(self._task_source_weights_update(), name="source_weights"))
        if self._components.get("feedback"):
            tasks.append(asyncio.create_task(self._task_performance_feedback(), name="performance_feedback"))

        # Arb scanner always runs (handles single-exchange gracefully)
        tasks.append(asyncio.create_task(self._task_arb_scanner(), name="arb_scanner"))
        tasks.append(asyncio.create_task(self._task_depth_research(), name="depth_research"))

        # Market maker (if enabled)
        if self._components.get("market_maker"):
            tasks.append(asyncio.create_task(self._task_market_maker(), name="market_maker"))

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
        console.print("\n[dim]Shutting down...[/]")

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
