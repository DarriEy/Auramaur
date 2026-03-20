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
from auramaur.exchange.models import Order, OrderSide, OrderType
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

log = structlog.get_logger()


class AuramaurBot:
    """Main bot orchestrator running concurrent async tasks."""

    def __init__(self, settings: Settings | None = None, db_path: str | None = None):
        self.settings = settings or Settings()
        self._running = False
        self._components: dict = {}
        self._db_path = db_path
        self._lock_file = None  # File handle kept open for duration
        self._exit_failures: set[str] = set()  # Track failed exit sells to avoid spam

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

        aggregator = Aggregator(sources=sources)

        # Exchange
        gamma = GammaClient()
        paper = PaperTrader(db=db, initial_balance=s.execution.paper_initial_balance)
        await paper.load_state()
        exchange = PolymarketClient(settings=s, paper_trader=paper)

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
        from auramaur.broker.sync import PositionSyncer
        from auramaur.broker.router import SmartOrderRouter
        from auramaur.broker.allocator import CapitalAllocator
        from auramaur.broker.reconciler import PositionReconciler

        pnl_tracker = PnLTracker(db=db)
        syncer = PositionSyncer(settings=s, db=db, exchange=exchange, paper=paper, pnl=pnl_tracker)
        reconciler = PositionReconciler(exchange=exchange, db=db)
        router = SmartOrderRouter(settings=s, exchange=exchange)
        allocator = CapitalAllocator(settings=s)

        # Build the swappable market analyzer based on config
        from auramaur.strategy.protocols import MarketAnalyzer as _MAProto

        market_analyzer: _MAProto | None = None

        if s.analysis.mode == "agent":
            from auramaur.strategy.agent_analyzer import AgentAnalyzer
            market_analyzer = AgentAnalyzer(settings=s, db=db)
            log.info("bot.analyzer_mode", mode="agent")
        elif s.analysis.mode == "strategic":
            from auramaur.nlp.strategic import StrategicAnalyzer
            from auramaur.strategy.pipeline_analyzer import PipelineAnalyzer
            strategic = StrategicAnalyzer(settings=s, db=db)
            market_analyzer = PipelineAnalyzer(
                settings=s, db=db, aggregator=aggregator,
                analyzer=analyzer, cache=cache, exchange=exchange,
                calibration=calibration, flow_tracker=flow_tracker,
                strategic=strategic,
            )
            log.info("bot.analyzer_mode", mode="strategic")
        elif s.analysis.mode == "pipeline":
            from auramaur.strategy.pipeline_analyzer import PipelineAnalyzer
            market_analyzer = PipelineAnalyzer(
                settings=s, db=db, aggregator=aggregator,
                analyzer=analyzer, cache=cache, exchange=exchange,
                calibration=calibration, flow_tracker=flow_tracker,
            )
            log.info("bot.analyzer_mode", mode="pipeline")

        # Strategy — per-exchange engines
        engines: dict[str, TradingEngine] = {}
        discoveries: dict[str, MarketDiscovery] = {"polymarket": gamma}

        poly_engine = TradingEngine(
            settings=s, db=db, discovery=gamma, aggregator=aggregator,
            analyzer=analyzer, cache=cache, risk_manager=risk_manager,
            exchange=exchange, calibration=calibration,
            flow_tracker=flow_tracker,
            router=router, allocator=allocator,
            market_analyzer=market_analyzer,
        )
        poly_engine._components_pnl = pnl_tracker
        poly_engine._components_syncer = syncer

        # Set strategic on engine for backward compat (used by legacy paths)
        if s.analysis.mode == "strategic":
            poly_engine.strategic = market_analyzer.strategic if hasattr(market_analyzer, 'strategic') else None
        elif s.analysis.mode != "agent":
            from auramaur.nlp.strategic import StrategicAnalyzer
            poly_engine.strategic = StrategicAnalyzer(settings=s, db=db)

        engines["polymarket"] = poly_engine

        # Kalshi (optional, guarded import)
        if s.kalshi.enabled:
            try:
                from auramaur.exchange.kalshi import KalshiClient
                kalshi = KalshiClient(settings=s, paper_trader=paper)
                discoveries["kalshi"] = kalshi
                engines["kalshi"] = TradingEngine(
                    settings=s, db=db, discovery=kalshi, aggregator=aggregator,
                    analyzer=analyzer, cache=cache, risk_manager=risk_manager,
                    exchange=kalshi, calibration=calibration,
                    flow_tracker=flow_tracker,
                )
            except ImportError:
                log.warning("optional.missing", component="KalshiClient")

        # Interactive Brokers (optional, guarded import)
        if s.ibkr.enabled:
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

        # Cross-platform arbitrage scanner
        arb_scanner = ArbitrageScanner(discoveries=discoveries)

        # News reactor — monitors RSS for breaking news, triggers fast analysis
        rss_source = next((s for s in sources if isinstance(s, RSSSource)), RSSSource())
        news_reactor = NewsReactor(
            rss_source=rss_source,
            discovery=gamma,
            engine=engines["polymarket"],
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

        # Market maker (optional, enabled by config)
        market_maker = None
        if s.market_maker.enabled:
            market_maker = MarketMaker(settings=s, exchange=exchange, db=db)

        # Alerts
        alerts = AlertManager(
            telegram_bot_token=s.telegram_bot_token,
            telegram_chat_id=s.telegram_chat_id,
            discord_webhook_url=s.discord_webhook_url,
        )

        self._components = {
            "db": db, "aggregator": aggregator, "discovery": gamma,
            "discoveries": discoveries,
            "paper": paper, "exchange": exchange, "analyzer": analyzer,
            "cache": cache, "calibration": calibration,
            "risk_manager": risk_manager, "flow_tracker": flow_tracker,
            "engines": engines, "news_reactor": news_reactor,
            "pnl_tracker": pnl_tracker, "syncer": syncer, "reconciler": reconciler,
            "router": router, "allocator": allocator,
            "attributor": attributor, "feedback": feedback,
            "correlator": correlator, "arb_executor": arb_executor,
            "arb_scanner": arb_scanner,
            "market_maker": market_maker,
            "alerts": alerts,
            "websocket": ws, "ensemble": ensemble,
            "source_names": source_names,
        }

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
        interval = self.settings.intervals.market_scan_seconds

        while self._running:
            if await self._check_kill_switch():
                return
            try:
                await engine.scan_and_store_markets()
            except Exception as e:
                show_error(f"Market scan failed ({name}): {e}")
            await asyncio.sleep(interval)

    async def _task_trading_cycle(self, engine: TradingEngine, name: str = "") -> None:
        """Periodically run trading analysis cycle."""
        interval = self.settings.intervals.analysis_seconds

        while self._running:
            if await self._check_kill_switch():
                return

            # No cash gate — the allocator handles capital limits.
            # The previous gate was blocking trades due to CLOB balance
            # returning on-chain USDC (low) vs Polymarket balance (higher).

            try:
                await engine.run_cycle()
            except Exception as e:
                show_error(f"Trading cycle failed ({name}): {e}")
            await asyncio.sleep(interval)

    async def _task_portfolio_monitor(self) -> None:
        """Monitor portfolio using the broker syncer for ground truth."""
        from auramaur.broker.sync import PositionSyncer
        from auramaur.broker.pnl import PnLTracker

        syncer: PositionSyncer = self._components["syncer"]
        pnl_tracker: PnLTracker = self._components["pnl_tracker"]
        interval = self.settings.intervals.portfolio_check_seconds

        # Exit failures are tracked per-session — sells will be retried once
        # per restart with proper token approval now in place.

        while self._running:
            try:
                # Use reconciler for accurate position data when live
                reconciler_comp = self._components.get("reconciler")
                if self.settings.is_live and reconciler_comp:
                    try:
                        reconciled = await reconciler_comp.reconcile()
                        position_value = sum(p.size * p.current_price for p in reconciled)
                        cash = await syncer.get_cash_balance()
                        total_value = cash + position_value
                        # Calculate PnL from cost basis
                        total_cost = sum(p.size * p.avg_cost for p in reconciled)
                        total_pnl = position_value - total_cost
                        show_portfolio(total_value, total_pnl, len(reconciled), 0.0)
                        positions = syncer._paper.positions  # For exit checking below
                    except Exception:
                        positions = await syncer.sync()
                        show_portfolio(0, 0, len(positions), 0.0)
                else:
                    positions_list = await syncer.sync()
                    cash = await syncer.get_cash_balance()
                    total_pnl = await pnl_tracker.get_total_pnl(positions_list)
                    show_portfolio(cash, total_pnl, len(positions_list), 0.0)

                # Check exits and attempt sells (with dedup to avoid spam)
                discovery: MarketDiscovery = self._components["discovery"]
                exchange: ExchangeClient = self._components["exchange"]
                alerts: AlertManager = self._components["alerts"]
                portfolio_tracker: PortfolioTracker = self._components["risk_manager"].portfolio
                try:
                    exit_list = await portfolio_tracker.check_exits(self.settings, discovery)
                    for pos, reason in exit_list:
                        fail_key = f"exit_fail:{pos.market_id}"
                        if fail_key in self._exit_failures:
                            continue

                        log.info(
                            "exit.triggered",
                            market_id=pos.market_id,
                            reason=reason.value,
                            pnl=pos.unrealized_pnl,
                        )
                        # Look up the actual CLOB asset_id from trade history
                        from auramaur.exchange.client import PolymarketClient
                        poly_exchange = self._components["exchange"]

                        # First, fetch market data to register token mappings
                        market_data = await discovery.get_market(pos.market_id)
                        if market_data and isinstance(poly_exchange, PolymarketClient):
                            poly_exchange.register_market_tokens(
                                pos.market_id,
                                market_data.clob_token_yes,
                                market_data.clob_token_no,
                            )

                        # Now look up the sellable token from real positions
                        token_id = ""
                        if isinstance(poly_exchange, PolymarketClient) and self.settings.is_live:
                            token_id = poly_exchange.get_sellable_token_id(pos.market_id) or ""
                        if not token_id and market_data:
                            # Fallback to Gamma token IDs directly
                            token_id = market_data.clob_token_yes or market_data.clob_token_no
                        if not token_id:
                            self._exit_failures.add(fail_key)
                            log.debug("exit.no_token", market_id=pos.market_id)
                            continue

                        sell_order = Order(
                            market_id=pos.market_id,
                            token_id=token_id,
                            side=OrderSide.SELL,
                            size=pos.size,
                            price=pos.current_price,
                            order_type=OrderType.MARKET,
                            dry_run=not self.settings.is_live,
                        )
                        result = await exchange.place_order(sell_order)
                        if result.status == "rejected":
                            self._exit_failures.add(fail_key)
                            log.warning("exit.sell_failed", market_id=pos.market_id)
                        else:
                            await alerts.send(
                                f"Exit {reason.value}: {pos.market_id[:12]} "
                                f"size={pos.size:.2f} pnl={pos.unrealized_pnl:+.2f}",
                                level="warning",
                            )
                except Exception as e:
                    log.debug("exit_check.error", error=str(e))
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
            try:
                await attributor.compute_kelly_multipliers()
                stats = await attributor.get_category_stats()
                if stats:
                    from auramaur.monitoring.display import show_category_performance
                    show_category_performance(stats)
            except Exception as e:
                log.error("attribution.error", error=str(e))
            await asyncio.sleep(3600)  # Every hour

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
        """Poll for resolved markets and record calibration outcomes."""
        discovery: MarketDiscovery = self._components["discovery"]
        calibration: CalibrationTracker = self._components["calibration"]
        db: Database = self._components["db"]

        while self._running:
            try:
                # Find calibration entries without resolutions
                rows = await db.fetchall(
                    """
                    SELECT DISTINCT c.market_id
                    FROM calibration c
                    WHERE c.actual_outcome IS NULL
                    """
                )
                for row in rows:
                    market_id = row["market_id"]
                    try:
                        market = await discovery.get_market(market_id)
                        if market and not market.active and market.outcome_yes_price is not None:
                            # Market resolved: price at 1.0 means YES, 0.0 means NO
                            resolved_yes = market.outcome_yes_price >= 0.99
                            resolved_no = market.outcome_yes_price <= 0.01
                            if resolved_yes or resolved_no:
                                await calibration.record_resolution(
                                    market_id, resolved_yes
                                )

                                # Record PnL for attribution
                                attributor = self._components.get("attributor")
                                try:
                                    pos_row = await db.fetchone(
                                        "SELECT * FROM portfolio WHERE market_id = ?",
                                        (market_id,),
                                    )
                                    if pos_row:
                                        entry_price = pos_row["avg_price"]
                                        exit_price = 1.0 if resolved_yes else 0.0
                                        size = pos_row["size"]
                                        side = pos_row["side"]

                                        if side == "BUY":
                                            pnl = (exit_price - entry_price) * size
                                        else:
                                            pnl = (entry_price - exit_price) * size

                                        category = pos_row["category"] or ""

                                        sig_row = await db.fetchone(
                                            "SELECT edge FROM signals WHERE market_id = ? ORDER BY timestamp DESC LIMIT 1",
                                            (market_id,),
                                        )
                                        edge = float(sig_row["edge"]) if sig_row else 0.0

                                        if attributor is not None:
                                            await attributor.record_trade_result(category, pnl, edge)

                                        await db.execute(
                                            "DELETE FROM portfolio WHERE market_id = ?",
                                            (market_id,),
                                        )
                                        await db.commit()

                                        log.info(
                                            "attribution.trade_resolved",
                                            market_id=market_id,
                                            category=category,
                                            pnl=round(pnl, 2),
                                            side=side,
                                        )
                                except Exception as e:
                                    log.debug("attribution.resolve_error", market_id=market_id, error=str(e))
                    except Exception as e:
                        log.debug(
                            "resolution_check.market_error",
                            market_id=market_id,
                            error=str(e),
                        )
            except Exception as e:
                log.error("resolution_checker.error", error=str(e))
            await asyncio.sleep(300)  # Every 5 minutes

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

        # Show real balance — use reconciler for live, paper for paper mode
        startup_balance = self._components["paper"].balance
        if self.settings.is_live:
            try:
                reconciler_comp = self._components.get("reconciler")
                if reconciler_comp:
                    reconciled = await reconciler_comp.reconcile()
                    position_value = sum(p.size * p.current_price for p in reconciled)
                    syncer_comp = self._components.get("syncer")
                    cash = await syncer_comp.get_cash_balance() if syncer_comp else 0
                    startup_balance = cash + position_value
                    console.print(
                        f"  Cash: [green]${cash:.2f}[/] | "
                        f"Positions: [cyan]${position_value:.2f}[/] ({len(reconciled)} markets)"
                    )
            except Exception as e:
                log.debug("startup.balance_error", error=str(e))

        show_startup(
            self._components["source_names"],
            startup_balance,
        )

        tasks = [
            asyncio.create_task(self._task_kill_switch_monitor(), name="kill_switch"),
            asyncio.create_task(self._task_portfolio_monitor(), name="portfolio"),
            asyncio.create_task(self._task_cache_cleanup(), name="cache_cleanup"),
            asyncio.create_task(self._task_recalibrate(), name="recalibrate"),
            asyncio.create_task(self._task_resolution_checker(), name="resolution_checker"),
            asyncio.create_task(self._task_order_monitor(), name="order_monitor"),
            asyncio.create_task(self._task_news_reactor(), name="news_reactor"),
            asyncio.create_task(self._task_position_sync(), name="position_sync"),
        ]

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
