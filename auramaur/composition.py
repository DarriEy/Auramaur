"""Per-venue composition root for AuramaurBot.

``AuramaurBot._init_components`` used to inline the wiring of every venue
(exchange client → discovery → engine → syncer → reconciler → router) in one
~360-line method. This module extracts the per-venue construction into named
builders so the bot is a thin orchestrator: it builds the shared GLOBAL services,
then asks each venue builder for its slice of the graph.

The per-venue ASYMMETRIES are the whole point of keeping this explicit and are
encoded on ``VenueComposition`` rather than homogenized:
  - polymarket has a separate ``gamma`` discovery object (≠ its exchange) and is
    the only venue exporting scalar ``syncer``/``reconciler``/``router``;
  - cryptodotcom contributes no ``exchanges_map`` entry (``export_exchange=False``);
  - ibkr/kalshi/cdc differ in which engine attributes they inject.

All construction imports are kept FUNCTION-LOCAL (mirroring the original method),
so this module imports nothing heavy at top level and can't introduce an import
cycle. Guarded optional venues return ``None`` on ImportError, exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from auramaur.exchange.protocols import ExchangeClient, MarketDiscovery
    from auramaur.strategy.engine import TradingEngine

log = structlog.get_logger()


@dataclass
class GlobalServices:
    """Shared, cross-venue services passed to every venue builder."""

    settings: Any
    db: Any
    paper: Any
    aggregator: Any
    analyzer: Any
    cache: Any
    calibration: Any
    risk_manager: Any
    flow_tracker: Any
    pnl_tracker: Any
    allocator: Any
    strategic: Any
    technical: Any
    hybrid: bool
    rebalance_cooldowns: dict


@dataclass
class VenueComposition:
    """One venue's slice of the service graph, plus how it folds into Components."""

    name: str
    discovery: MarketDiscovery
    engine: TradingEngine
    exchange: ExchangeClient | None = None
    syncer: Any | None = None
    reconciler: Any | None = None
    router: Any | None = None
    export_exchange: bool = True   # cdc=False (never populated exchanges_map)
    export_scalars: bool = False   # polymarket=True (scalar syncer/reconciler/router)


def build_polymarket(g: GlobalServices) -> VenueComposition:
    """Polymarket: separate Gamma discovery, scalar syncer/reconciler/router."""
    from auramaur.broker.reconciler import PositionReconciler
    from auramaur.broker.router import SmartOrderRouter
    from auramaur.broker.sync import PositionSyncer
    from auramaur.exchange.client import PolymarketClient
    from auramaur.exchange.gamma import GammaClient
    from auramaur.strategy.engine import TradingEngine

    s = g.settings
    gamma = GammaClient()
    exchange = PolymarketClient(settings=s, paper_trader=g.paper)
    syncer = PositionSyncer(settings=s, db=g.db, exchange=exchange, paper=g.paper, pnl=g.pnl_tracker)
    reconciler = PositionReconciler(exchange=exchange, db=g.db)
    router = SmartOrderRouter(settings=s, exchange=exchange)

    engine = TradingEngine(
        settings=s, db=g.db, discovery=gamma, aggregator=g.aggregator,
        analyzer=g.analyzer, cache=g.cache, risk_manager=g.risk_manager,
        exchange=exchange, calibration=g.calibration, flow_tracker=g.flow_tracker,
        router=router, allocator=g.allocator, technical_analyzer=g.technical,
    )
    engine._components_pnl = g.pnl_tracker
    engine._components_syncer = syncer
    engine.strategic = g.strategic
    engine.exchange_name = "polymarket"
    engine._hybrid = g.hybrid

    return VenueComposition(
        name="polymarket", discovery=gamma, exchange=exchange, engine=engine,
        syncer=syncer, reconciler=reconciler, router=router,
        export_exchange=True, export_scalars=True)


def build_kalshi(g: GlobalServices) -> VenueComposition | None:
    """Kalshi: first-class wiring; syncer goes to the list only, router is local."""
    try:
        from auramaur.broker.router import SmartOrderRouter
        from auramaur.broker.sync import KalshiPositionSyncer
        from auramaur.exchange.kalshi import KalshiClient
        from auramaur.strategy.engine import TradingEngine
    except ImportError:
        log.warning("optional.missing", component="KalshiClient")
        return None

    s = g.settings
    kalshi = KalshiClient(settings=s, paper_trader=g.paper)
    kalshi_syncer = KalshiPositionSyncer(settings=s, db=g.db, exchange=kalshi, paper=g.paper)
    kalshi_router = SmartOrderRouter(settings=s, exchange=kalshi)

    engine = TradingEngine(
        settings=s, db=g.db, discovery=kalshi, aggregator=g.aggregator,
        analyzer=g.analyzer, cache=g.cache, risk_manager=g.risk_manager,
        exchange=kalshi, calibration=g.calibration, flow_tracker=g.flow_tracker,
        router=kalshi_router, allocator=g.allocator, technical_analyzer=g.technical,
    )
    engine._components_pnl = g.pnl_tracker
    engine._components_syncer = kalshi_syncer
    engine.strategic = g.strategic
    engine.exchange_name = "kalshi"
    engine._hybrid = g.hybrid
    engine._rebalance_cooldowns = g.rebalance_cooldowns

    # syncer in the list only (not a scalar export); router is a local throwaway.
    return VenueComposition(
        name="kalshi", discovery=kalshi, exchange=kalshi, engine=engine,
        syncer=kalshi_syncer, export_exchange=True, export_scalars=False)


def build_cryptodotcom(g: GlobalServices) -> VenueComposition | None:
    """Crypto.com: discovery + engine only — does NOT populate exchanges_map."""
    try:
        from auramaur.exchange.cryptodotcom import CryptoComClient
        from auramaur.strategy.engine import TradingEngine
    except ImportError:
        log.warning("optional.missing", component="CryptoComClient")
        return None

    s = g.settings
    cryptodotcom = CryptoComClient(settings=s, paper_trader=g.paper)
    engine = TradingEngine(
        settings=s, db=g.db, discovery=cryptodotcom, aggregator=g.aggregator,
        analyzer=g.analyzer, cache=g.cache, risk_manager=g.risk_manager,
        exchange=cryptodotcom, calibration=g.calibration, flow_tracker=g.flow_tracker,
        allocator=g.allocator, technical_analyzer=g.technical,
    )
    engine.strategic = g.strategic
    engine.exchange_name = "cryptodotcom"
    engine._hybrid = g.hybrid

    return VenueComposition(
        name="cryptodotcom", discovery=cryptodotcom, exchange=cryptodotcom,
        engine=engine, export_exchange=False, export_scalars=False)


def build_ibkr(g: GlobalServices) -> VenueComposition | None:
    """IBKR options scanner: no syncer/router, no engine attribute injection."""
    try:
        from auramaur.exchange.ibkr import IBKRClient
        from auramaur.strategy.engine import TradingEngine
    except ImportError:
        log.warning("optional.missing", component="IBKRClient")
        return None

    s = g.settings
    ibkr = IBKRClient(settings=s, paper_trader=g.paper)
    engine = TradingEngine(
        settings=s, db=g.db, discovery=ibkr, aggregator=g.aggregator,
        analyzer=g.analyzer, cache=g.cache, risk_manager=g.risk_manager,
        exchange=ibkr, calibration=g.calibration, flow_tracker=g.flow_tracker,
        technical_analyzer=g.technical,
    )
    return VenueComposition(
        name="ibkr", discovery=ibkr, exchange=ibkr, engine=engine,
        export_exchange=True, export_scalars=False)


# venue name -> (builder, enabled predicate). The enabled gate mirrors the
# original per-venue config flags; polymarket has no extra gate.
_VENUE_BUILDERS = [
    ("polymarket", build_polymarket, lambda s: True),
    ("kalshi", build_kalshi, lambda s: s.kalshi.enabled),
    ("cryptodotcom", build_cryptodotcom, lambda s: s.cryptodotcom.enabled),
    ("ibkr", build_ibkr, lambda s: s.ibkr.enabled and s.ibkr.options_enabled),
]


def assemble_venues(g: GlobalServices, exchange_filter: str | None) -> dict:
    """Build every enabled venue (honoring ``exchange_filter``) and fold the
    results into the venue-keyed maps + polymarket's scalar exports. Returns the
    dict of locals the rest of _init_components consumes — byte-identical to the
    original inline assembly.
    """
    discoveries: dict = {}
    exchanges_map: dict = {}
    engines: dict = {}
    syncers: list = []
    syncer = reconciler = router = None

    for name, builder, enabled in _VENUE_BUILDERS:
        if exchange_filter is not None and exchange_filter != name:
            continue
        if not enabled(g.settings):
            continue
        comp = builder(g)
        if comp is None:
            continue
        discoveries[comp.name] = comp.discovery
        engines[comp.name] = comp.engine
        if comp.export_exchange and comp.exchange is not None:
            exchanges_map[comp.name] = comp.exchange
        if comp.syncer is not None:
            syncers.append(comp.syncer)
        if comp.export_scalars:
            syncer = comp.syncer
            reconciler = comp.reconciler
            router = comp.router

    return {
        "discoveries": discoveries, "exchanges_map": exchanges_map,
        "engines": engines, "syncers": syncers,
        "syncer": syncer, "reconciler": reconciler, "router": router,
    }


async def assemble_components(
    *, settings, db_path: str, hybrid: bool, exchange_filter: str | None,
    rebalance_cooldowns: dict,
):
    """Build the full runtime Components registry: global services, the per-venue
    graph (assemble_venues), and the post-loop globals that read the assembled
    maps. Returns a Components dict identical to the original inline assembly;
    AuramaurBot._init_components is now a thin shim over this.
    """
    from auramaur.components import Components
    from auramaur.data_sources.aggregator import Aggregator
    from auramaur.data_sources.fred import FREDSource
    from auramaur.data_sources.newsapi import NewsAPISource
    from auramaur.data_sources.reddit import RedditSource
    from auramaur.data_sources.rss import RSSSource
    from auramaur.data_sources.twitter import TwitterSource
    from auramaur.data_sources.websearch import WebSearchSource
    from auramaur.db.database import Database
    from auramaur.exchange.paper import PaperTrader
    from auramaur.monitoring.alerts import AlertManager
    from auramaur.nlp.analyzer import ClaudeAnalyzer
    from auramaur.nlp.cache import NLPCache
    from auramaur.nlp.calibration import CalibrationTracker
    from auramaur.risk.manager import RiskManager
    from auramaur.strategy.arbitrage_scanner import ArbitrageScanner
    from auramaur.strategy.market_maker import MarketMaker
    from auramaur.strategy.news_reactor import NewsReactor
    from auramaur.strategy.resolution_tracker import ResolutionTracker
    from auramaur.strategy.technical import TechnicalAnalyzer

    s = settings

    # Database — auto-detect available slot
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

    # Primary official lanes start shadow-only: observations are persisted but
    # withheld from forecasts until their information cells graduate.
    from auramaur.data_sources.official import BLSSource, NWSSource
    sources.append(NWSSource())
    source_names.append("NWS[shadow]")
    sources.append(BLSSource(api_key=s.bls_api_key))
    source_names.append("BLS[shadow]")
    if s.bea_api_key:
        from auramaur.data_sources.official import BEASource
        sources.append(BEASource(api_key=s.bea_api_key))
        source_names.append("BEA[shadow]")
    if s.congress_api_key:
        from auramaur.data_sources.official import CongressSource
        sources.append(CongressSource(api_key=s.congress_api_key))
        source_names.append("Congress[shadow]")
    if s.eia_api_key:
        from auramaur.data_sources.official import EIASource
        sources.append(EIASource(api_key=s.eia_api_key))
        source_names.append("EIA[shadow]")

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

    ig = s.information_graduation
    from auramaur.lineage_observer import LineageObserver
    lineage_observer = await LineageObserver.create(
        db_path, min_resolved=ig.min_resolved, min_paired=ig.min_paired,
        min_success_rate=ig.min_success_rate,
        probation_multiplier=ig.probation_multiplier,
    )
    for source, category, horizon, event_type in (
        ("nws", "weather", "0-6h", "severe_weather_transition"),
        ("bls", "economics", "0-30m", "scheduled_release"),
        ("bea", "economics", "0-30m", "scheduled_release"),
        ("congress", "politics_us", "1-14d", "procedural_milestone"),
        ("eia", "economics", "0-24h", "scheduled_release"),
        ("edgar", "economics", "1-7d", "corporate_filing"),
        ("social_bundle", "crypto", "1-3d", "unscheduled_narrative"),
    ):
        await lineage_observer.ladder.register(source, category, horizon, event_type)
    aggregator = Aggregator(sources=sources, observer=lineage_observer)

    # Exchange
    paper = PaperTrader(db=db, initial_balance=s.execution.paper_initial_balance)
    await paper.load_state()

    # NLP
    analyzer = ClaudeAnalyzer(settings=s)
    cache = NLPCache(db=db)
    calibration = CalibrationTracker(
        db=db, min_samples=s.calibration.min_samples,
        lineage_observer=lineage_observer,
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

    # Per-venue service graph (composition root, auramaur/composition.py).
    # Build the shared globals once, then ask each enabled venue builder for
    # its slice (exchange/discovery/engine/syncer/...). Replaces ~110 lines of
    # inline per-venue wiring; produces the same venue-keyed maps + scalars.
    _g = GlobalServices(
        settings=s, db=db, paper=paper, aggregator=aggregator, analyzer=analyzer,
        cache=cache, calibration=calibration, risk_manager=risk_manager,
        flow_tracker=flow_tracker, pnl_tracker=pnl_tracker, allocator=allocator,
        strategic=strategic, technical=technical, hybrid=hybrid,
        rebalance_cooldowns=rebalance_cooldowns,
    )
    _venues = assemble_venues(_g, exchange_filter)
    discoveries = _venues["discoveries"]
    exchanges_map = _venues["exchanges_map"]
    engines = _venues["engines"]
    syncers = _venues["syncers"]
    syncer = _venues["syncer"]
    reconciler = _venues["reconciler"]
    router = _venues["router"]
    # Polymarket's exchange/discovery feed the primary_* picks + market maker.
    gamma = discoveries.get("polymarket")
    exchange = exchanges_map.get("polymarket")

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
        llm_match_cache_seconds=s.arbitrage.llm_match_cache_seconds,
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
            fast_analysis=hybrid and settings.hybrid.news_fast_analysis,
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
    user_ws = None
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

    # Private lifecycle events wake the existing idempotent order monitor; the
    # stream never writes fills itself, so REST reconciliation remains truth.
    if (exchange is not None and s.is_live and s.polymarket_api_key
            and s.polymarket_api_secret and s.polymarket_passphrase):
        from auramaur.exchange.websocket import PolymarketUserWebSocket
        user_ws = PolymarketUserWebSocket(
            api_key=s.polymarket_api_key,
            api_secret=s.polymarket_api_secret,
            passphrase=s.polymarket_passphrase,
            on_event=exchange.notify_user_event,
        )

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

    return Components({
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
        "websocket": ws, "user_websocket": user_ws, "ensemble": ensemble,
        "source_names": source_names,
        "exchange_filter": exchange_filter,
    })
