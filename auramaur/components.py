"""Typed runtime component registry for AuramaurBot.

``AuramaurBot`` wires ~30 long-lived services (db, risk manager, pnl ledger,
syncers, engines, …) into a single registry that the bot and its mixins read
from. That registry used to be a bare ``dict[str, object]`` accessed by string
key — 85 untyped ``self._components["…"]`` lookups with no autocomplete, no
type-checking, and a KeyError-on-a-live-path failure mode for a typo.

``Components`` keeps that dict (it literally IS a ``dict``, so every existing
subscript, ``.get()``, ``.items()`` teardown, and external consumer keeps
working unchanged) while adding typed attribute accessors that read straight
from it. Call sites migrate ``["db"]`` → ``.db`` incrementally; nothing is ever
half-broken. The component-type imports live under ``TYPE_CHECKING`` (with
``from __future__ import annotations`` the property return types are strings),
so this module imports nothing at runtime and cannot introduce an import cycle.

Always-present services use a subscript accessor (raises ``KeyError`` if truly
absent — identical to the old ``self._components["db"]``); optional services use
``.get()`` and return ``T | None``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from auramaur.broker.allocator import CapitalAllocator
    from auramaur.broker.feedback import PerformanceFeedback
    from auramaur.broker.pnl import PnLTracker
    from auramaur.broker.reconciler import PositionReconciler
    from auramaur.broker.router import SmartOrderRouter
    from auramaur.broker.sync import PositionSyncer
    from auramaur.data_sources.aggregator import Aggregator
    from auramaur.db.database import Database
    from auramaur.exchange.paper import PaperTrader
    from auramaur.exchange.protocols import ExchangeClient, MarketDiscovery
    from auramaur.exchange.websocket import PolymarketWebSocket
    from auramaur.monitoring.alerts import AlertManager
    from auramaur.monitoring.attribution import PerformanceAttributor
    from auramaur.nlp.analyzer import ClaudeAnalyzer
    from auramaur.nlp.cache import NLPCache
    from auramaur.nlp.calibration import CalibrationTracker
    from auramaur.nlp.ensemble import EnsembleEstimator
    from auramaur.risk.manager import RiskManager
    from auramaur.strategy.agent_analyzer import AgentAnalyzer
    from auramaur.strategy.arbitrage import ArbitrageExecutor
    from auramaur.strategy.arbitrage_scanner import ArbitrageScanner
    from auramaur.strategy.correlation import CorrelationDetector
    from auramaur.strategy.engine import TradingEngine
    from auramaur.strategy.market_maker import MarketMaker
    from auramaur.strategy.news_reactor import NewsReactor
    from auramaur.strategy.order_flow import OrderFlowTracker
    from auramaur.strategy.resolution_tracker import ResolutionTracker


class Components(dict):
    """Typed view over the runtime component registry (see module docstring).

    IS a ``dict`` so all existing subscript / ``.get()`` / ``.items()`` usage and
    external consumers (cli, kraken pillar) keep working; the typed properties
    below just read from it.
    """

    # -- always present (subscript; KeyError if truly missing) -------------
    @property
    def db(self) -> Database: return self["db"]

    @property
    def aggregator(self) -> Aggregator: return self["aggregator"]

    @property
    def paper(self) -> PaperTrader: return self["paper"]

    @property
    def analyzer(self) -> ClaudeAnalyzer: return self["analyzer"]

    @property
    def cache(self) -> NLPCache: return self["cache"]

    @property
    def calibration(self) -> CalibrationTracker: return self["calibration"]

    @property
    def risk_manager(self) -> RiskManager: return self["risk_manager"]

    @property
    def engines(self) -> dict[str, TradingEngine]: return self["engines"]

    @property
    def discoveries(self) -> dict[str, MarketDiscovery]: return self["discoveries"]

    @property
    def exchanges(self) -> dict[str, ExchangeClient]: return self["exchanges"]

    @property
    def pnl_tracker(self) -> PnLTracker: return self["pnl_tracker"]

    @property
    def syncers(self) -> list: return self["syncers"]

    @property
    def allocator(self) -> CapitalAllocator: return self["allocator"]

    @property
    def arb_scanner(self) -> ArbitrageScanner: return self["arb_scanner"]

    @property
    def resolution_tracker(self) -> ResolutionTracker: return self["resolution_tracker"]

    @property
    def depth_agent(self) -> AgentAnalyzer: return self["depth_agent"]

    @property
    def alerts(self) -> AlertManager: return self["alerts"]

    @property
    def source_names(self) -> list[str]: return self["source_names"]

    # -- optional (.get(); may be None) ------------------------------------
    @property
    def discovery(self) -> MarketDiscovery | None: return self.get("discovery")

    @property
    def exchange(self) -> ExchangeClient | None: return self.get("exchange")

    @property
    def flow_tracker(self) -> OrderFlowTracker | None: return self.get("flow_tracker")

    @property
    def news_reactor(self) -> NewsReactor | None: return self.get("news_reactor")

    @property
    def syncer(self) -> PositionSyncer | None: return self.get("syncer")

    @property
    def reconciler(self) -> PositionReconciler | None: return self.get("reconciler")

    @property
    def router(self) -> SmartOrderRouter | None: return self.get("router")

    @property
    def attributor(self) -> PerformanceAttributor | None: return self.get("attributor")

    @property
    def feedback(self) -> PerformanceFeedback | None: return self.get("feedback")

    @property
    def correlator(self) -> CorrelationDetector | None: return self.get("correlator")

    @property
    def arb_executor(self) -> ArbitrageExecutor | None: return self.get("arb_executor")

    @property
    def market_maker(self) -> MarketMaker | None: return self.get("market_maker")

    @property
    def websocket(self) -> PolymarketWebSocket | None: return self.get("websocket")

    @property
    def ensemble(self) -> EnsembleEstimator | None: return self.get("ensemble")

    @property
    def exchange_filter(self) -> str | None: return self.get("exchange_filter")
