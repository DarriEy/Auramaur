"""Authoritative catalog of bot-scheduled trading strategy pipelines.

Readiness tooling and tests can audit this declarative catalog without
constructing clients or touching venues. Bot task entrypoints construct the
runtime dependencies; this is the source of truth for eligibility and required
pipeline capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from auramaur.strategy.protocols import DeploymentMode, ExecutionMode, StrategyCapability

FULL_TRADE_PIPELINE = frozenset(
    {
        StrategyCapability.DISCOVERY,
        StrategyCapability.MARKET_DATA,
        StrategyCapability.SIGNAL,
        StrategyCapability.RISK,
        StrategyCapability.EXECUTION,
        StrategyCapability.RECORDING,
        StrategyCapability.RECONCILIATION,
        StrategyCapability.SETTLEMENT,
    }
)


@dataclass(frozen=True)
class StrategySpec:
    """Static contract for one scheduled strategy family or venue lane."""

    key: str
    module: str
    class_name: str
    task: str
    runner: str
    execution_mode: ExecutionMode
    deployment: DeploymentMode
    venues: frozenset[str]
    capabilities: frozenset[StrategyCapability]

    def resolve_class(self) -> type:
        return getattr(import_module(self.module), self.class_name)


def _spec(
    key: str,
    module: str,
    class_name: str,
    task: str,
    execution_mode: ExecutionMode,
    deployment: DeploymentMode,
    venues: set[str],
    *,
    runner: str = "run_once",
    evidence: bool = False,
    analysis: bool = False,
) -> StrategySpec:
    capabilities = set(FULL_TRADE_PIPELINE)
    if evidence:
        capabilities.add(StrategyCapability.EVIDENCE)
    if analysis:
        capabilities.add(StrategyCapability.ANALYSIS)
    return StrategySpec(
        key,
        module,
        class_name,
        task,
        runner,
        execution_mode,
        deployment,
        frozenset(venues),
        frozenset(capabilities),
    )


G = ExecutionMode.GATEWAY_SINGLE
P = ExecutionMode.GATEWAY_PAIRED
GRAD = DeploymentMode.GRADUATABLE

STRATEGY_SPECS: tuple[StrategySpec, ...] = (
    _spec(
        "core_trading",
        "auramaur.strategy.engine",
        "TradingEngine",
        "_task_trading_cycle",
        G,
        GRAD,
        {"polymarket", "kalshi"},
        evidence=True,
        analysis=True,
        runner="run_cycle",
    ),
    _spec(
        "news_reactor",
        "auramaur.strategy.news_reactor",
        "NewsReactor",
        "_task_news_reactor",
        G,
        GRAD,
        {"polymarket", "kalshi"},
        evidence=True,
        analysis=True,
        runner="check_for_news",
    ),
    _spec(
        "kraken",
        "auramaur.treasury.kraken_pillar",
        "KrakenPillar",
        "_task_kraken_pillar",
        ExecutionMode.DIRECT_SPOT,
        GRAD,
        {"kraken"},
        evidence=True,
        analysis=True,
    ),
    _spec(
        "arbitrage",
        "auramaur.strategy.arbitrage_scanner",
        "ArbitrageScanner",
        "_task_arb_scanner",
        ExecutionMode.GATEWAY_EXTERNAL,
        DeploymentMode.STRUCTURAL_LIVE,
        {"polymarket", "kalshi"},
        runner="scan",
    ),
    _spec(
        "bias_harvest",
        "auramaur.strategy.bias_harvest",
        "BiasHarvestPillar",
        "_task_bias_harvest",
        G,
        GRAD,
        {"polymarket"},
    ),
    _spec(
        "platform_consensus",
        "auramaur.strategy.platform_consensus",
        "PlatformConsensusPillar",
        "_task_platform_consensus",
        G,
        GRAD,
        {"polymarket"},
        evidence=True,
    ),
    _spec(
        "long_horizon",
        "auramaur.strategy.long_horizon",
        "LongHorizonPillar",
        "_task_long_horizon",
        G,
        GRAD,
        {"polymarket", "kalshi"},
    ),
    _spec(
        "agent_trader",
        "auramaur.strategy.agent_trader",
        "AgentTraderPillar",
        "_task_agent_trader",
        G,
        GRAD,
        {"polymarket"},
        analysis=True,
    ),
    _spec(
        "agent_trader_kalshi",
        "auramaur.strategy.agent_trader",
        "AgentTraderPillar",
        "_task_agent_trader_kalshi",
        G,
        GRAD,
        {"kalshi"},
        analysis=True,
    ),
    _spec(
        "term_structure",
        "auramaur.strategy.term_structure",
        "TermStructurePillar",
        "_task_term_structure",
        G,
        GRAD,
        {"polymarket"},
        analysis=True,
    ),
    _spec(
        "vol_anchor",
        "auramaur.strategy.vol_anchor",
        "VolAnchorPillar",
        "_task_vol_anchor",
        G,
        GRAD,
        {"polymarket"},
    ),
    _spec(
        "informed_flow",
        "auramaur.strategy.informed_flow_pillar",
        "InformedFlowPillar",
        "_task_informed_flow",
        G,
        GRAD,
        {"kalshi"},
    ),
    _spec(
        "entailment_arb",
        "auramaur.strategy.entailment_arb",
        "EntailmentArbPillar",
        "_task_entailment_arb",
        P,
        GRAD,
        {"polymarket", "kalshi"},
        analysis=True,
    ),
    _spec(
        "cross_venue_arb",
        "auramaur.strategy.cross_venue_arb",
        "CrossVenueArbPillar",
        "_task_cross_venue_arb",
        P,
        GRAD,
        {"polymarket", "kalshi"},
        analysis=True,
    ),
    _spec(
        "econ_indicator",
        "auramaur.strategy.econ_indicator",
        "EconIndicatorPillar",
        "_task_econ_indicator",
        G,
        GRAD,
        {"kalshi"},
        evidence=True,
    ),
    _spec(
        "interim_manager",
        "auramaur.strategy.interim_manager",
        "InterimManagerPillar",
        "_task_interim_manager",
        G,
        GRAD,
        {"polymarket", "kalshi"},
    ),
    _spec(
        "settlement_arb",
        "auramaur.strategy.settlement_arb",
        "SettlementArbPillar",
        "_task_settlement_arb",
        G,
        GRAD,
        {"polymarket", "kalshi"},
        evidence=True,
    ),
    _spec(
        "weather_temp",
        "auramaur.strategy.weather_temp",
        "WeatherTempPillar",
        "_task_weather_temp",
        G,
        GRAD,
        {"polymarket"},
        evidence=True,
    ),
    _spec(
        "resolution_lens",
        "auramaur.strategy.resolution_lens",
        "ResolutionLensPillar",
        "_task_resolution_lens",
        G,
        GRAD,
        {"polymarket"},
        evidence=True,
        analysis=True,
    ),
    _spec(
        "resolution_lens_kalshi",
        "auramaur.strategy.resolution_lens",
        "ResolutionLensPillar",
        "_task_resolution_lens_kalshi",
        G,
        GRAD,
        {"kalshi"},
        evidence=True,
        analysis=True,
    ),
    _spec(
        "oddlot_tender",
        "auramaur.strategy.oddlot_tender",
        "OddLotTenderPillar",
        "_task_oddlot_tender",
        ExecutionMode.DIRECT_EQUITY,
        GRAD,
        {"ibkr"},
        evidence=True,
        analysis=True,
    ),
    _spec(
        "momentum_coupling",
        "auramaur.strategy.momentum_coupling",
        "MomentumCouplingPillar",
        "_task_momentum_coupling",
        G,
        GRAD,
        {"polymarket"},
    ),
    _spec(
        "market_maker",
        "auramaur.strategy.market_maker",
        "MarketMaker",
        "_task_market_maker",
        ExecutionMode.DIRECT_QUOTING,
        DeploymentMode.STRUCTURAL_LIVE,
        {"polymarket"},
        runner="run_cycle",
    ),
    _spec(
        "ibkr_etf_paper",
        "auramaur.strategy.ibkr_etf_paper",
        "IBKRETFPaperPillar",
        "_task_ibkr_etf_paper",
        ExecutionMode.PAPER_SIMULATED,
        DeploymentMode.PAPER_ONLY,
        {"ibkr"},
        evidence=True,
        analysis=True,
    ),
    _spec(
        "ibkr_multiasset",
        "auramaur.strategy.ibkr_multiasset_paper",
        "IBKRMultiAssetPaperBook",
        "_task_ibkr_multiasset_paper",
        ExecutionMode.PAPER_SIMULATED,
        GRAD,
        {"ibkr"},
    ),
)

STRATEGY_BY_KEY = {spec.key: spec for spec in STRATEGY_SPECS}
STRATEGY_BY_TASK = {spec.task: spec for spec in STRATEGY_SPECS}
# Every task defined by the bot's scheduling modules must be either a strategy
# above or explicitly classified here as operational/research-only. This closes
# the old failure mode where a new trading task silently escaped the registry.
NON_STRATEGY_TASKS = frozenset(
    {
        "_task_attribution_update",
        "_task_balance_recorder",
        "_task_build_guard",
        "_task_cache_cleanup",
        "_task_correlation_scan",
        "_task_decision_marks",
        "_task_depth_research",
        "_task_evidence_distiller",
        "_task_hydro_watch",
        "_task_intelligence_eval",
        "_task_intraday_drift",
        "_task_kill_switch_monitor",
        "_task_live_gate_monitor",
        "_task_market_scan",
        "_task_orderbook_recorder",
        "_task_performance_feedback",
        "_task_portfolio_monitor",
        "_task_position_sync",
        "_task_price_monitor",
        "_task_recalibrate",
        "_task_redemption_check",
        "_task_source_weights_update",
        "_task_strategy_report",
        "_task_user_websocket",
    }
)
