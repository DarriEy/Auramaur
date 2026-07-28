"""Exhaustive contracts for every application-level exposure mutation path."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum


class ExposureKind(str, Enum):
    ENTRY = "entry"
    STRUCTURAL = "structural"
    EXIT = "exit"
    TREASURY = "treasury"
    REDEMPTION = "redemption"
    PAPER_BOOKING = "paper_booking"
    EXECUTION_BOUNDARY = "execution_boundary"


@dataclass(frozen=True)
class ExposurePathSpec:
    key: str
    kind: ExposureKind
    decision_source: str
    data_services: tuple[str, ...]
    risk_authority: str
    execution_boundary: str
    modes: frozenset[str]
    booking: str
    monitoring: str
    exit_path: str
    reconciliation: str
    settlement: str
    attribution: str


def _path(
    key: str,
    kind: ExposureKind,
    decision: str,
    data: tuple[str, ...],
    risk: str,
    execution: str,
    modes: set[str],
    booking: str,
    monitoring: str,
    exit_path: str,
    reconciliation: str,
    settlement: str,
    attribution: str,
) -> ExposurePathSpec:
    return ExposurePathSpec(
        key,
        kind,
        decision,
        data,
        risk,
        execution,
        frozenset(modes),
        booking,
        monitoring,
        exit_path,
        reconciliation,
        settlement,
        attribution,
    )


EXPOSURE_PATHS = (
    _path(
        "prediction_directional",
        ExposureKind.ENTRY,
        "strategy or TradingEngine signal",
        ("venue discovery", "order book", "strategy evidence/analysis"),
        "RiskManager + graduation ladder",
        "ExecutionGateway.submit",
        {"paper", "live"},
        "ExecutionGateway/PnLTracker",
        "order monitor",
        "venue portfolio exit",
        "venue sync + order reconciliation",
        "resolution tracker/Kalshi settlement",
        "strategy_source ledger",
    ),
    _path(
        "prediction_paired",
        ExposureKind.STRUCTURAL,
        "arb scanner/pair strategy",
        ("multi-venue discovery", "two-sided books"),
        "pair risk contract",
        "ExecutionGateway.submit_paired/place_legs",
        {"paper", "live"},
        "gateway external-fill recording",
        "order monitor + unwind",
        "paired unwind/rebalance",
        "venue sync",
        "resolution tracker",
        "per-leg strategy_source ledger",
    ),
    _path(
        "prediction_quoting",
        ExposureKind.STRUCTURAL,
        "MarketMaker quote model",
        ("venue discovery", "order book depth", "inventory"),
        "market-maker reserve/exposure checks",
        "ExecutionGateway.place_quote_pair",
        {"paper", "live"},
        "gateway quote fill recording",
        "order monitor TTL",
        "cancel/requote",
        "venue sync",
        "resolution tracker",
        "market_maker ledger",
    ),
    _path(
        "prediction_exit",
        ExposureKind.EXIT,
        "portfolio monitor/strategy harvest",
        ("portfolio", "executable bid/ask"),
        "exit contract (entries only are graduated)",
        "ExecutionGateway.submit_exit",
        {"paper", "live"},
        "PnLTracker",
        "order monitor",
        "terminal",
        "venue sync",
        "resolution tracker",
        "originating strategy_source",
    ),
    _path(
        "ibkr_equity",
        ExposureKind.ENTRY,
        "odd-lot/portfolio exit",
        ("EDGAR", "IBKR quotes", "portfolio"),
        "IBKR equity gates + strategy limits",
        "IBKREquityClient",
        {"paper", "live"},
        "strategy/PnLTracker",
        "IBKR status",
        "IBKR equity exit",
        "IBKR portfolio sync",
        "broker cash/fill",
        "oddlot or originating strategy",
    ),
    _path(
        # A resting PAPER order that the market later traded through. The fill
        # is established by check_fills' strict trade-through test; this path
        # only books it and stamps the evidence. Before it existed, deferred
        # fills were logged and dropped -- no ledger row, no decision stamp --
        # so a resting strategy accrued nothing at all.
        "paper_deferred_fill",
        ExposureKind.PAPER_BOOKING,
        "market traded through a resting paper order",
        ("venue mid price", "the queued order"),
        "none -- the entry was already risk-approved when it was placed",
        "PaperTrader.check_fills (no venue call)",
        {"paper"},
        "PnLTracker + decision stamped trade_through",
        "order monitor",
        "originating strategy's exit path",
        "paper position tables",
        "resolution tracker",
        "originating strategy_source",
    ),
    _path(
        # ONE booking rail for both IBKR books. They had near-duplicate copies
        # and drifted into three defects: the multiasset side stored prices in
        # fair_probability, stamped every fill 'synthetic', and used a
        # per-instrument event family that caps a book at 28 families forever.
        # The ETF arms separately had no registered callsite at all (audit
        # finding #7, 2026-07-26) because they wrote fills by raw SQL.
        #
        # Runs AFTER the book's own venue-native accounting and cannot veto it.
        # The execution boundary is a LOCAL simulator that refuses any
        # non-dry_run order, and a LIVE multiasset fill is not booked at all,
        # so paper is the only mode this can ever reach.
        "ibkr_paper_booking",
        ExposureKind.PAPER_BOOKING,
        "already-executed IBKR fill (ETF arm forecast or multiasset rule)",
        ("IBKR/Alpaca quotes", "adjusted daily closes", "FX rate"),
        "none — cannot veto a fill the book already made",
        "ExecutionGateway.submit -> SimulatedInstrumentExchange",
        {"paper"},
        "ibkr_etf_ledger / ibkr_paper_ledger + ExecutionGateway/PnLTracker",
        "book cycle marks (ibkr_etf_positions / ibkr_paper_positions)",
        "book exit rules (stop/target/trailing/bearish)",
        "book-owned positions table",
        "no settlement — instruments do not resolve",
        "ibkr_<book or alias> ledger",
    ),
    _path(
        "ibkr_multiasset",
        ExposureKind.ENTRY,
        "graduated IBKR paper book",
        ("registered instruments", "IBKR quotes", "paper evidence"),
        "IBKR evidence contract + independent live allowlist",
        "IBKRMultiAssetExecution.place",
        {"paper", "live"},
        "IBKR paper/live ledgers",
        "IBKR execution orders",
        "book signal close",
        "execution acknowledgement",
        "broker fill",
        "IBKR book ledger",
    ),
    _path(
        "kraken_directional",
        ExposureKind.ENTRY,
        "Kraken directional signal",
        ("Kraken price/OHLC", "optional LLM view"),
        "Kraken directional gates/caps",
        "KrakenSpotClient.place_spot_order",
        {"paper", "live"},
        "PnLTracker",
        "Kraken fill query",
        "directional sell",
        "balance reconciliation",
        "spot fill",
        "kraken_directional ledger",
    ),
    _path(
        "kraken_treasury",
        ExposureKind.TREASURY,
        "reserve policy",
        ("Kraken balances", "FX/spot quote"),
        "treasury conversion gates/caps",
        "KrakenSpotClient.place_spot_order",
        {"paper", "live"},
        "treasury audit log",
        "Kraken fill query",
        "reverse conversion",
        "balance reconciliation",
        "spot fill",
        "kraken_treasury purpose",
    ),
    _path(
        "manual_kraken",
        ExposureKind.ENTRY,
        "typed CLI operator request",
        ("Kraken price", "Kraken balance"),
        "typed confirmation + adapter gates/caps",
        "KrakenSpotClient.place_spot_order",
        {"paper", "live"},
        "Kraken order result",
        "operator response/Kraken",
        "manual opposing order",
        "Kraken balance",
        "spot fill",
        "manual purpose",
    ),
    _path(
        "agent_paper",
        ExposureKind.PAPER_BOOKING,
        "MCP agent request",
        ("read-only bot DB quote",),
        "paper-only process guard + held-size clamp",
        "PnLTracker.record_fill",
        {"paper"},
        "PnLTracker + portfolio mirror",
        "synchronous paper fill",
        "agent close_position",
        "paper cost basis",
        "resolution tracker",
        "agent order id",
    ),
    _path(
        "coinbase_paper",
        ExposureKind.PAPER_BOOKING,
        "Kraken comparator signal",
        ("Coinbase public quote",),
        "paper-only comparator contract",
        "PnLTracker.record_fill",
        {"paper"},
        "PnLTracker",
        "synchronous paper fill",
        "comparator opposing fill",
        "paper cost basis",
        "paper resolution",
        "coinbase_paper ledger",
    ),
    _path(
        "redemption",
        ExposureKind.REDEMPTION,
        "auto/operator redemption",
        ("resolved positions", "on-chain balances"),
        "separate redemption live gate",
        "OnchainRedeemer.redeem",
        {"dry_run", "live"},
        "redemption records",
        "transaction receipt",
        "not applicable",
        "on-chain/DB reconciliation",
        "Safe redemption",
        "redemption audit",
    ),
    _path(
        "gateway_boundary",
        ExposureKind.EXECUTION_BOUNDARY,
        "registered callers",
        ("prepared order",),
        "kill switch + venue triple gate + caps",
        "venue ExchangeClient.place_order",
        {"paper", "live"},
        "fill + trades mirror",
        "pending-order monitor",
        "caller-owned",
        "venue sync",
        "caller-owned",
        "strategy_source passthrough",
    ),
)

EXPOSURE_BY_KEY = {path.key: path for path in EXPOSURE_PATHS}

# Exact application-level sensitive callsite multiset. Counts matter: quote pairs
# and concurrent legs intentionally contain multiple raw adapter calls.
REGISTERED_CALLSITES = Counter(
    {
        ("auramaur/agentmcp/book.py", "place_trade", "record_fill", "agent_paper"): 1,
        ("auramaur/bot.py", "_cancel_resting_live_orders", "cancel_order",
         "prediction_exit"): 1,
        ("auramaur/bot_arb_execute.py", "_execute_cross_exchange_arb", "cancel_order",
         "prediction_paired"): 1,
        ("auramaur/bot_order_monitor.py", "_task_order_monitor", "cancel_order",
         "gateway_boundary"): 1,
        ("auramaur/broker/execution_gateway.py", "submit_paired", "cancel_order",
         "prediction_paired"): 1,
        ("auramaur/strategy/market_maker.py", "_cancel_quote", "cancel_order",
         "prediction_quoting"): 2,
        ("auramaur/strategy/market_maker.py", "_place_two_sided", "cancel_order",
         "prediction_quoting"): 1,
        (
            "auramaur/bot_order_monitor.py",
            "_task_order_monitor",
            "record_fill",
            "gateway_boundary",
        ): 1,
        ("auramaur/bot.py", "_auto_redeem", "redeem", "redemption"): 1,
        ("auramaur/bot_arb.py", "_execute_conditional_arb", "place_legs", "prediction_paired"): 1,
        (
            "auramaur/bot_arb.py",
            "_rebalance_concentrated_positions",
            "place_legs",
            "prediction_paired",
        ): 1,
        (
            "auramaur/bot_arb_execute.py",
            "_execute_negrisk_arb",
            "place_legs",
            "prediction_paired",
        ): 1,
        (
            "auramaur/bot_arb_execute.py",
            "_execute_internal_arb",
            "place_legs",
            "prediction_paired",
        ): 1,
        (
            "auramaur/bot_arb_execute.py",
            "_execute_cross_exchange_arb",
            "place_legs",
            "prediction_paired",
        ): 1,
        ("auramaur/bot_exits.py", "_execute_poly_exit", "submit_exit", "prediction_exit"): 1,
        ("auramaur/bot_exits.py", "_execute_kalshi_exit", "submit_exit", "prediction_exit"): 1,
        ("auramaur/bot_exits.py", "_execute_ibkr_exit", "place_order", "ibkr_equity"): 1,
        (
            "auramaur/broker/execution_gateway.py",
            "_place_and_record",
            "place_order",
            "gateway_boundary",
        ): 1,
        (
            "auramaur/broker/execution_gateway.py",
            "_record_result",
            "record_fill",
            "gateway_boundary",
        ): 1,
        (
            "auramaur/broker/execution_gateway.py",
            "place_quote_pair",
            "place_order",
            "gateway_boundary",
        ): 2,
        (
            "auramaur/broker/execution_gateway.py",
            "place_legs",
            "place_order",
            "gateway_boundary",
        ): 2,
        ("auramaur/cli/kraken.py", "_kraken_spot", "place_spot_order", "manual_kraken"): 2,
        ("auramaur/cli/kraken.py", "_run", "place_spot_order", "manual_kraken"): 1,
        ("auramaur/cli/redeem.py", "run", "redeem", "redemption"): 1,
        ("auramaur/strategy/agent_trader.py", "_try_enter", "submit", "prediction_directional"): 1,
        ("auramaur/strategy/bias_harvest.py", "_try_enter", "submit", "prediction_directional"): 1,
        (
            "auramaur/strategy/cross_venue_arb.py",
            "_enter_pair",
            "submit_paired",
            "prediction_paired",
        ): 1,
        (
            "auramaur/strategy/cross_venue_arb.py",
            "_unwind_leg",
            "submit_exit",
            "prediction_exit",
        ): 1,
        (
            "auramaur/strategy/econ_indicator.py",
            "_enter_if_edge",
            "submit",
            "prediction_directional",
        ): 1,
        (
            "auramaur/strategy/engine.py",
            "_build_and_place_order",
            "submit",
            "prediction_directional",
        ): 1,
        (
            "auramaur/strategy/entailment_arb.py",
            "_enter_pair",
            "submit_paired",
            "prediction_paired",
        ): 1,
        ("auramaur/broker/instrument_booking.py", "book_instrument_fill", "submit",
         "ibkr_paper_booking"): 1,
        ("auramaur/bot_order_monitor.py", "_record_deferred_paper_fills",
         "record_fill", "paper_deferred_fill"): 1,
        ("auramaur/strategy/ibkr_multiasset_paper.py", "_fill", "place", "ibkr_multiasset"): 1,

        (
            "auramaur/exchange/ibkr_multiasset_execution.py",
            "place",
            "placeOrder",
            "ibkr_multiasset",
        ): 1,
        (
            "auramaur/strategy/informed_flow_pillar.py",
            "_try_enter",
            "submit",
            "prediction_directional",
        ): 1,
        (
            "auramaur/strategy/interim_manager.py",
            "_decide_one",
            "submit",
            "prediction_directional",
        ): 1,
        ("auramaur/strategy/long_horizon.py", "_try_enter", "submit", "prediction_directional"): 1,
        (
            "auramaur/strategy/long_horizon.py",
            "_harvest_decay",
            "submit_exit",
            "prediction_exit",
        ): 1,
        (
            "auramaur/strategy/market_maker.py",
            "_place_two_sided",
            "place_quote_pair",
            "prediction_quoting",
        ): 1,
        (
            "auramaur/strategy/momentum_coupling.py",
            "_execute",
            "submit",
            "prediction_directional",
        ): 1,
        (
            "auramaur/strategy/oddlot_tender.py",
            "_on_opportunity",
            "place_share_order",
            "ibkr_equity",
        ): 1,
        ("auramaur/strategy/oddlot_tender.py", "_record_entry", "record_fill", "ibkr_equity"): 1,
        (
            "auramaur/strategy/platform_consensus.py",
            "_try_enter",
            "submit",
            "prediction_directional",
        ): 1,
        ("auramaur/strategy/resolution_lens.py", "_enter", "submit", "prediction_directional"): 1,
        (
            "auramaur/strategy/settlement_arb.py",
            "_maybe_enter",
            "submit",
            "prediction_directional",
        ): 1,
        (
            "auramaur/strategy/term_structure.py",
            "_try_enter",
            "submit",
            "prediction_directional",
        ): 1,
        ("auramaur/strategy/vol_anchor.py", "_try_enter", "submit", "prediction_directional"): 1,
        (
            "auramaur/strategy/weather_temp.py",
            "_price_and_enter",
            "submit",
            "prediction_directional",
        ): 1,
        (
            "auramaur/treasury/coinbase_paper.py",
            "record_shadow_fill",
            "record_fill",
            "coinbase_paper",
        ): 1,
        (
            "auramaur/treasury/kraken_pillar.py",
            "_treasury",
            "place_spot_order",
            "kraken_treasury",
        ): 1,
        (
            "auramaur/treasury/kraken_pillar.py",
            "_record_directional_fill",
            "record_fill",
            "kraken_directional",
        ): 2,
        (
            "auramaur/treasury/kraken_pillar.py",
            "_directional",
            "place_spot_order",
            "kraken_directional",
        ): 2,
    }
)

SENSITIVE_METHODS = frozenset(
    {
        "submit",
        "submit_exit",
        "submit_paired",
        "place_legs",
        "place_quote_pair",
        "place_order",
        "place_spot_order",
        "place",
        "place_share_order",
        "placeOrder",
        "record_fill",
        "redeem",
        # Cancellation is half of a resting order's lifecycle. Omitting it left
        # the market maker's three off-gateway cancels — on the LIVE venue
        # client, not through the gateway — outside a perimeter this registry
        # claims to close, and left `prediction_quoting`'s declared
        # "cancel/requote" exit path unbacked by any registered callsite.
        "cancel_order",
    }
)
