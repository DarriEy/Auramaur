"""Main async orchestrator — runs concurrent tasks."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

# Silence noisy py_clob_client_v2 HTTP error logging (403 geoblock, 404 missing
# order books). Our code already handles these via try/except.
logging.getLogger("py_clob_client_v2.http_helpers.helpers").setLevel(logging.CRITICAL)

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
from auramaur.exchange.models import Fill, Order, OrderSide, OrderType, TokenType

if TYPE_CHECKING:
    from auramaur.exchange.models import Market, Signal
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
from auramaur.nlp.errors import BudgetExhausted
from auramaur.risk.manager import RiskManager
from auramaur.risk.portfolio import PortfolioTracker
from auramaur.strategy.arbitrage_scanner import (
    ArbOpportunity,
    ArbitrageScanner,
    NegRiskArbOpportunity,
)
from auramaur.strategy.engine import TradingEngine
from auramaur.strategy.market_maker import MarketMaker
from auramaur.strategy.news_reactor import NewsReactor
from auramaur.strategy.resolution_tracker import ResolutionTracker
from auramaur.strategy.technical import TechnicalAnalyzer
from auramaur.bot_exits import ExitExecutionMixin
from auramaur.bot_strategy_tasks import StrategyTaskMixin
from auramaur.bot_arb import ArbExecutionMixin

log = structlog.get_logger()


class AuramaurBot(ExitExecutionMixin, StrategyTaskMixin, ArbExecutionMixin):
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
        self._components: dict = {}
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
        technical = TechnicalAnalyzer(settings=s)

        # Depth agent for deep research on high-potential markets
        from auramaur.strategy.agent_analyzer import AgentAnalyzer
        depth_agent = AgentAnalyzer(settings=s, db=db, calibration=calibration)

        log.info("bot.analyzer_mode", mode="strategic+depth_agent+technical")

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
                technical_analyzer=technical,
            )
            poly_engine._components_pnl = pnl_tracker
            poly_engine._components_syncer = syncer
            poly_engine.strategic = strategic
            poly_engine.exchange_name = "polymarket"
            poly_engine._hybrid = self._hybrid

            engines["polymarket"] = poly_engine

        # Kalshi (optional, guarded import) — same first-class wiring as Polymarket
        if s.kalshi.enabled and (self._exchange_filter is None or self._exchange_filter == "kalshi"):
            try:
                from auramaur.exchange.kalshi import KalshiClient
                kalshi = KalshiClient(settings=s, paper_trader=paper)
                discoveries["kalshi"] = kalshi
                exchanges_map["kalshi"] = kalshi

                kalshi_syncer = KalshiPositionSyncer(settings=s, db=db, exchange=kalshi, paper=paper)
                kalshi_router = SmartOrderRouter(settings=s, exchange=kalshi)
                syncers.append(kalshi_syncer)

                kalshi_engine = TradingEngine(
                    settings=s, db=db, discovery=kalshi, aggregator=aggregator,
                    analyzer=analyzer, cache=cache, risk_manager=risk_manager,
                    exchange=kalshi, calibration=calibration,
                    flow_tracker=flow_tracker,
                    router=kalshi_router, allocator=allocator,
                    technical_analyzer=technical,
                )
                kalshi_engine._components_pnl = pnl_tracker
                kalshi_engine._components_syncer = kalshi_syncer
                kalshi_engine.strategic = strategic
                kalshi_engine.exchange_name = "kalshi"
                kalshi_engine._hybrid = self._hybrid
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
                    technical_analyzer=technical,
                )
                cdc_engine.strategic = strategic
                cdc_engine.exchange_name = "cryptodotcom"
                cdc_engine._hybrid = self._hybrid
                engines["cryptodotcom"] = cdc_engine
            except ImportError:
                log.warning("optional.missing", component="CryptoComClient")

        # Interactive Brokers options scanner (optional, guarded import). Gated
        # by options_enabled so the equity book can run without waking the
        # OPRA-less option scanner (which otherwise spams Error 200/10091).
        if s.ibkr.enabled and s.ibkr.options_enabled and (self._exchange_filter is None or self._exchange_filter == "ibkr"):
            try:
                from auramaur.exchange.ibkr import IBKRClient
                ibkr = IBKRClient(settings=s, paper_trader=paper)
                discoveries["ibkr"] = ibkr
                exchanges_map["ibkr"] = ibkr
                engines["ibkr"] = TradingEngine(
                    settings=s, db=db, discovery=ibkr, aggregator=aggregator,
                    analyzer=analyzer, cache=cache, risk_manager=risk_manager,
                    exchange=ibkr, calibration=calibration,
                    flow_tracker=flow_tracker,
                    technical_analyzer=technical,
                )
            except ImportError:
                log.warning("optional.missing", component="IBKRClient")

        # Resolution tracker — auto-detects when markets resolve and feeds
        # outcomes into the calibration loop for Platt scaling updates.
        resolution_tracker = ResolutionTracker(
            db=db,
            calibration=calibration,
            discoveries=discoveries,
            # Enables the venue-truth sweep: settles positions whose markets
            # vanished from the Gamma API (the -100% phantom-mark legs).
            proxy_address=s.polymarket_proxy_address or "",
        )

        # Cross-platform arbitrage scanner (fee-aware)
        arb_scanner = ArbitrageScanner(
            discoveries=discoveries,
            analyzer=analyzer,
            exchange_fees=s.arbitrage.exchange_fees,
            min_profit_after_fees_pct=s.arbitrage.min_profit_after_fees_pct,
            # An arb is hedged only when BOTH legs fill — a single-leg fill
            # is directional inventory in a banned market (caught quoting
            # KBO baseball live 2026-06-12). Same gates as the MM.
            blocked_categories=s.risk.blocked_categories,
            allowed_categories_live=(s.risk.allowed_categories_live
                                     if s.is_live else None),
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
                fast_analysis=self._hybrid and self.settings.hybrid.news_fast_analysis,
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
            (engines[k].exchange for k in engines if getattr(engines[k], 'exchange', None) is not None), None
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

        # Name-the-gap gate: wire the post-hoc mispricing auditor into the
        # risk manager now that db + analyzer exist. Without this, the gate
        # (if enabled) blocks unexplained divergences without an LLM call.
        if risk_manager is not None and analyzer is not None:
            from auramaur.nlp.gap_audit import GapAuditor
            risk_manager.gap_auditor = GapAuditor(db, analyzer, self.settings)

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
                rows = await engine.db.fetchall(
                    "SELECT id, clob_token_yes FROM markets "
                    "WHERE active = 1 AND exchange = ? AND clob_token_yes != '' "
                    "AND liquidity >= ? ORDER BY liquidity DESC LIMIT ?",
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
                        # Strategy-books view: per-book open exposure +
                        # realized (live & paper) from the pnl_ledger.
                        from auramaur.monitoring.books import (
                            gather_books, render_books_table,
                        )
                        books = await gather_books(self._components["db"])
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
        from auramaur.broker.onchain import OnChainRedeemer
        from auramaur.broker.redeemer import (
            fetch_redeemable_positions, summarize_redemptions,
        )
        from auramaur.monitoring.display import console

        proxy = self.settings.polymarket_proxy_address
        if not proxy:
            return  # no proxy configured, can't check

        db: Database = self._components["db"]
        alerts: AlertManager = self._components["alerts"]
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
                            f"or run:[/] [cyan]auramaur redeem --submit[/]"
                        )
                        console.print(
                            f"[bold green]╚══════════════════════════╝[/]"
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
        pillar = KrakenPillar(self.settings, client, bot=self, console=console)
        interval = self.settings.kraken.treasury_interval_seconds
        try:
            while self._running:
                if await self._check_kill_switch():
                    return
                await pillar.run_once()
                await asyncio.sleep(interval)
        finally:
            await client.close()

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
                                        risk_manager=risk_mgr, bot=self)
        interval = self.settings.momentum_coupling.poll_seconds
        while self._running:
            if await self._check_kill_switch():
                return
            try:
                await pillar.run_once()
            except Exception as e:  # noqa: BLE001
                log.error("coupling.error", error=str(e))
            await asyncio.sleep(interval)

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

    async def _task_strategy_report(self) -> None:
        """Hourly per-pillar P&L report for hybrid mode."""
        attributor = self._components.get("attributor")
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
        correlator = self._components.get("correlator")
        arb_executor = self._components.get("arb_executor")
        if correlator is None or arb_executor is None:
            return
        discovery: MarketDiscovery = self._components["discovery"]
        risk_manager: RiskManager = self._components["risk_manager"]
        engines: dict[str, TradingEngine] = self._components["engines"]

        db: Database = self._components["db"]
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

                    # Update cost_basis from real fill prices (ground truth).
                    # This loop only runs in live mode, so we explicitly
                    # write is_paper=0 and conflict on (market_id, is_paper).
                    for rp in reconciled:
                        await self._components["db"].execute(
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

                    # Delete stale rows from cost_basis AND portfolio: positions no
                    # longer held on-chain. Only run this when the reconciler returned
                    # a non-empty result — an empty reconcile means the CLOB API call
                    # failed, not that we actually hold zero positions, so we must not
                    # wipe the tables. Both tables matter: cost_basis feeds sync(),
                    # and portfolio feeds the risk-manager correlation/exposure checks.
                    if reconciled:
                        live_ids = [rp.market_id for rp in reconciled]
                        placeholders = ",".join("?" * len(live_ids))
                        # Live-mode reconciliation must only delete live rows;
                        # paper rows (is_paper=1) live in their own namespace.
                        cb_cur = await self._components["db"].execute(
                            f"DELETE FROM cost_basis WHERE size > 0 AND is_paper = 0 AND market_id NOT IN ({placeholders})",
                            live_ids,
                        )
                        pf_cur = await self._components["db"].execute(
                            f"DELETE FROM portfolio WHERE exchange = 'polymarket' AND is_paper = 0 AND market_id NOT IN ({placeholders})",
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

    async def _task_order_monitor(self) -> None:
        """Monitor pending limit orders for fills and expiry."""
        from datetime import datetime, timezone

        paper: PaperTrader = self._components["paper"]
        primary_exchange: PolymarketClient = self._components["exchange"]
        exchanges: dict[str, ExchangeClient] = self._components.get("exchanges", {})
        discovery: MarketDiscovery = self._components["discovery"]
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
                                        pnl_tracker = self._components.get("pnl_tracker")
                                        if pnl_tracker:
                                            await pnl_tracker.record_fill(fill)
                                    except Exception as e:
                                        log.error(
                                            "order_monitor.fill_record_error",
                                            exchange=exchange_name,
                                            order_id=order_id,
                                            error=str(e),
                                        )

                                db = self._components.get("db")
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
                                        if getattr(cur, "rowcount", 0) == 0:
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
                                                    "order_monitor",
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
                                    db = self._components.get("db")
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

            await asyncio.sleep(30)

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
        db = self._components.get("db")
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
        fixed = 0
        for row in rows:
            order_id = row["order_id"]
            if order_id in tracked or order_id.startswith("PAPER"):
                continue
            if order_id in sentinel_ids:
                await db.execute(
                    "UPDATE trades SET status = 'error' WHERE order_id = ? AND status = 'pending'",
                    (order_id,),
                )
                fixed += 1
                continue
            client = clients_by_name.get(row["exchange"] or "polymarket")
            if client is None or not hasattr(client, "get_order_status"):
                continue
            try:
                result = await client.get_order_status(order_id)
            except Exception:
                continue
            if result.status in ("filled", "cancelled", "expired", "rejected"):
                await db.execute(
                    "UPDATE trades SET status = ? WHERE order_id = ?",
                    (result.status, order_id),
                )
                fixed += 1
            if fixed >= 5:
                break
        if fixed:
            await db.commit()
            log.info("order_monitor.orphans_reconciled", count=fixed)

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

                # Kalshi settlements sweep: the venue's settlements feed is
                # the only reliable booking source there — the syncer drops
                # settled positions from `portfolio` before the Gamma-style
                # detection path can see them, which is how Kalshi realized
                # P&L went entirely unrecorded until 2026-06-12.
                kalshi = (self._components.get("exchanges") or {}).get("kalshi")
                if kalshi is not None:
                    try:
                        from auramaur.broker.kalshi_settlements import (
                            sweep_kalshi_settlements,
                        )
                        booked = await sweep_kalshi_settlements(
                            self._components["db"], kalshi)
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

        # Announce which build this process runs. The 2026-06 category leak
        # traded for a day on pre-fix code because nothing said the running
        # process was older than the checkout.
        from auramaur.monitoring.build_info import STARTUP_SHA
        if STARTUP_SHA:
            console.print(f"  [dim]Build: {STARTUP_SHA}[/]")
            log.info("bot.build", sha=STARTUP_SHA, mode=mode)

        await self._init_components()
        self._running = True

        db_path = self._components["db"].db_path
        if db_path != "auramaur.db":
            console.print(f"  [yellow]Instance: {db_path}[/]")
        # Persistent Claude-budget counter lives in the same sqlite file —
        # point it at this instance's path (multi-instance runs use
        # auramaur_N.db and must not share a budget row file-side).
        from auramaur.nlp import call_budget
        call_budget.set_db_path(db_path)

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
            self._components["source_names"],
            startup_balance,
        )

        # Strategy-books panel: every book with its TRUE mode, the gates,
        # graduation mode, and the ledger lifetime number.
        try:
            from auramaur.monitoring.books import render_books_panel
            row = await self._components["db"].fetchone(
                "SELECT COALESCE(SUM(pnl), 0) AS v FROM pnl_ledger WHERE is_paper = 0")
            console.print(render_books_panel(
                self.settings, float(row["v"]) if row else None))
        except Exception as e:
            log.debug("startup.books_panel_error", error=str(e))

        exchange_filter = self._components.get("exchange_filter")
        if exchange_filter:
            console.print(f"  [cyan]Exchange filter: {exchange_filter} only[/]")

        # Operational live-readiness preflight. If any BLOCK condition is present
        # while armed live, force new ENTRIES to paper (exits bypass the risk
        # manager, so held positions can still get out). Same checks as
        # `auramaur health`; a "refuse to fool itself" gate.
        if self.settings.is_live:
            try:
                from auramaur.monitoring.live_gate import preflight
                report = await preflight(self.settings, self._components["db"])
                if not report.live_allowed:
                    rm = self._components.get("risk_manager")
                    if rm is not None:
                        rm.live_entries_blocked = True
                    blocked = ", ".join(b.name for b in report.blocks)
                    log.error("live_gate.entries_blocked",
                              blocks=[f"{b.name}: {b.detail}" for b in report.blocks])
                    console.print(f"  [bold red]LIVE ENTRIES BLOCKED by preflight:[/] "
                                  f"{blocked} — exits stay live")
                    alerts = self._components.get("alerts")
                    if alerts is not None:
                        await alerts.send(
                            f"LIVE ENTRIES BLOCKED by preflight: {blocked}", level="critical")
                else:
                    log.info("live_gate.passed", warnings=len(report.warnings))
                    if report.warnings:
                        console.print(f"  [yellow]preflight OK, {len(report.warnings)} warning(s)[/]")
            except Exception as e:
                log.error("live_gate.error", error=str(e))

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
            # Capture real bid/ask depth for cost-aware research + honest paper
            # fills. Read-only; disable via intervals.orderbook_recorder_enabled.
            if getattr(self.settings.intervals, "orderbook_recorder_enabled", True):
                tasks.append(asyncio.create_task(
                    self._task_orderbook_recorder(engine, ex_name),
                    name=f"orderbook_{ex_name}",
                ))

        if self._components.get("attributor"):
            tasks.append(asyncio.create_task(self._task_attribution_update(), name="attribution"))
        # Correlation-arb executes through the exempt 'arbitrage' source and
        # had no off switch at all; it shares the arbitrage book's flag.
        if self._components.get("correlator") and self.settings.arbitrage.enabled:
            tasks.append(asyncio.create_task(self._task_correlation_scan(), name="correlation"))
        if self._components.get("websocket"):
            tasks.append(asyncio.create_task(self._task_price_monitor(), name="price_monitor"))
        if self._components.get("ensemble"):
            tasks.append(asyncio.create_task(self._task_source_weights_update(), name="source_weights"))
        if self._components.get("feedback"):
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

        # Entailment arbitrage (paper-forced until proven)
        if self.settings.entailment_arb.enabled:
            tasks.append(asyncio.create_task(self._task_entailment_arb(), name="entailment_arb"))
        if self.settings.cross_venue_arb.enabled:
            tasks.append(asyncio.create_task(self._task_cross_venue_arb(), name="cross_venue_arb"))

        # Data-driven Kalshi econ-indicator pricing (paper-forced until proven)
        if self.settings.econ_indicator.enabled:
            tasks.append(asyncio.create_task(self._task_econ_indicator(), name="econ_indicator"))

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

        # Odd-lot tender harvester (detection always; entries paper-forced)
        if self.settings.oddlot_tender.enabled:
            tasks.append(asyncio.create_task(self._task_oddlot_tender(), name="oddlot_tender"))

        # Kraken treasury/capital pillar (+ gated directional spot)
        if self.settings.kraken.enabled:
            tasks.append(asyncio.create_task(self._task_kraken_pillar(), name="kraken_treasury"))

        # Fast path: momentum-coupling pillar (gated by momentum_coupling.enabled)
        if self.settings.momentum_coupling.enabled:
            tasks.append(asyncio.create_task(self._task_momentum_coupling(), name="momentum_coupling"))

        # Market maker (if enabled)
        if self._components.get("market_maker"):
            tasks.append(asyncio.create_task(self._task_market_maker(), name="market_maker"))

        # Hybrid strategy report (hourly per-pillar P&L)
        if self._hybrid and self._components.get("attributor"):
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
        primary = self._components.get("exchange")
        if primary is not None and hasattr(primary, "_live_pending"):
            clients[id(primary)] = ("polymarket", primary)
        for name, client in (self._components.get("exchanges") or {}).items():
            if client is not None and hasattr(client, "_live_pending"):
                clients.setdefault(id(client), (name, client))

        db = self._components.get("db")
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
