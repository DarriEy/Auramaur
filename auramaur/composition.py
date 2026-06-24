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
