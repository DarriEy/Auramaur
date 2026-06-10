"""Pydantic Settings for Auramaur configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


def _load_defaults() -> dict:
    defaults_path = Path(__file__).parent / "defaults.yaml"
    if defaults_path.exists():
        with open(defaults_path) as f:
            return yaml.safe_load(f)
    return {}


_DEFAULTS = _load_defaults()


class ExecutionConfig(BaseModel):
    live: bool = False
    paper_initial_balance: float = 1000.0
    limit_order_ttl_seconds: int = 120
    spread_capture_min_bps: int = 50
    stop_loss_pct: float = 30.0
    profit_target_pct: float = 50.0
    edge_erosion_min_pct: float = 2.0
    time_decay_hours: float = 12.0
    # Free capital from near-certain winners that are still far from resolution:
    # a position with <max_upside_pct left to gain but >min_hours until it
    # resolves ties up capital for little residual return. Sell it early to
    # redeploy into fresh edges. Only targets the winning side (tiny remaining
    # upside == price near the payout boundary).
    free_winners_enabled: bool = True
    free_winners_max_upside_pct: float = 3.0
    free_winners_min_hours: float = 48.0
    # Periodic dust sweep: close tiny stale positions to trim position count and
    # free locked slots. Age-guarded so freshly-opened small entries are never
    # swept (the bot opens $1-7 positions). Runs via the portfolio exit monitor.
    dust_sweep_enabled: bool = True
    dust_max_notional: float = 1.0
    dust_min_age_hours: float = 24.0


class RiskConfig(BaseModel):
    max_drawdown_pct: float = 15.0
    max_stake_per_market: float = 25.0
    daily_loss_limit: float = 200.0
    max_open_positions: int = 200
    min_edge_pct: float = 5.0
    min_liquidity: float = 1000.0
    # Kalshi reports thin top-of-book liquidity even on active markets, so its
    # candidate floor is lower than the Polymarket-tuned default (the engine
    # uses this when exchange_name == 'kalshi').
    kalshi_min_liquidity: float = 300.0
    max_spread_pct: float = 5.0
    confidence_floor: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"
    implied_prob_min: float = 0.03
    implied_prob_max: float = 0.97
    category_exposure_cap_pct: float = 30.0
    time_to_resolution_min_hours: int = 24
    time_to_resolution_max_days: int = 0  # 0 = no ceiling
    max_correlated_positions: int = 5
    second_opinion_divergence_max: float = 0.15
    # Divergence-aware filter (LLM signal). Edge-gap analysis found trades where
    # the LLM moderately disagrees with the market (|claude-market| in the band)
    # are adversely selected. When enabled, those need >= require_confidence.
    # OFF by default — A/B when resolution_pnl confirms the pattern at scale.
    divergence_filter_enabled: bool = False
    divergence_adverse_low: float = 0.05
    divergence_adverse_high: float = 0.20
    divergence_require_confidence: str = "HIGH"
    # sports: the LLM has no structural edge on game outcomes / spreads / O-U
    # (driven by live injury & lineup info we don't ingest), and resolved-market
    # performance bears that out. Blocked outright rather than left to the
    # feedback loop, which needs a sample sports keeps losing money to build.
    blocked_categories: list[str] = ["sports"]


class KellyConfig(BaseModel):
    fraction: float = 0.25


class IntervalsConfig(BaseModel):
    market_scan_seconds: int = 300
    news_poll_seconds: int = 120
    analysis_seconds: int = 180
    portfolio_check_seconds: int = 60
    dashboard_refresh_seconds: int = 5
    # Adaptive scheduling — scale intensity by market activity
    adaptive_enabled: bool = True
    peak_hours_utc: list[int] = Field(
        default_factory=lambda: [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    )
    off_peak_multiplier: float = 4.0
    quiet_multiplier: float = 8.0
    quiet_hours_utc: list[int] = Field(
        default_factory=lambda: [4, 5, 6, 7, 8, 9],
    )


_INTENSITY_PRESETS: dict[str, dict] = {
    "low": {
        "skip_second_opinion": True,
        "max_markets_per_cycle": 10,
        "evidence_per_source": 3,
        "daily_claude_call_budget": 50,
    },
    "medium": {
        "skip_second_opinion": False,
        "max_markets_per_cycle": 10,
        "evidence_per_source": 3,
        "daily_claude_call_budget": 100,
    },
    "full_blast": {
        "skip_second_opinion": False,
        "max_markets_per_cycle": 50,
        "evidence_per_source": 10,
        "daily_claude_call_budget": 0,  # 0 = unlimited
    },
}


class NLPConfig(BaseModel):
    cache_ttl_breaking_seconds: int = 900
    cache_ttl_slow_seconds: int = 7200
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    api_intensity: Literal["low", "medium", "full_blast"] = "medium"
    skip_second_opinion: bool = False
    max_markets_per_cycle: int = 10
    evidence_per_source: int = 3
    daily_claude_call_budget: int = 100

    # Tool-use analyzer — refines strategic-batch results on top-edge markets
    # by letting Claude Code drive its own web_search / web_fetch. "auto"
    # fires tool-use only when the strategic batch already showed a strong
    # edge signal; "tool_use" forces it for every batched market;
    # "strategic_batch" disables the refinement path entirely.
    analysis_mode: Literal["strategic_batch", "tool_use", "auto"] = "auto"
    # Lever 3: tool-use refinement is the single heaviest call (multi-turn web).
    # Tightened from 5.0/4 — only strong edges earn a web-research pass, and at
    # most 2 per cycle — which is where most of the realized token burn lived.
    tool_use_edge_threshold_pct: float = 8.0  # edge % above which tool-use fires in auto mode
    tool_use_max_budget_usd: float = 0.50  # per-market tool-use budget cap
    tool_use_max_markets_per_cycle: int = 2  # cap concurrent refinements per cycle
    tool_use_model: str = "claude-opus-4-7"  # can differ from strategic batch model

    # Lever 1: effort tiering. `--effort` scales thinking-token burn per call,
    # which is what eats the Max+ rate-limit window. Reserve `max` for the
    # primary estimate; cheaper tiers for challenge/secondary passes. These are
    # CLI effort levels (low|medium|high|max).
    effort_primary: str = "max"               # primary strategic / single-market estimate
    effort_adversarial: str = "medium"        # red-team second opinion (challenges, not re-derives)
    effort_ensemble_secondary: str = "high"   # ensemble's non-primary model(s)
    effort_tool_use: str = "high"             # web-research refinement

    # Lever 5: cache the strategic batch path per-market (it was entirely
    # uncached — the dominant call path had zero cache hits). Markets with a
    # fresh, price-stable cached result are reused and excluded from the batch.
    strategic_cache_enabled: bool = True

    # Info-content tuning — maximize signal per token sent to the LLM.
    # Evidence is globally re-ranked (recency x authority x relevance) before
    # truncation, so the model sees the best N items, not the first N.
    evidence_top_n: int = 8  # per-market evidence items kept after ranking
    # Relevance backend: "embeddings" (semantic, needs the embeddings extra),
    # "tfidf" (scikit-learn, no model download), or "heuristic" (no deps).
    # Falls back automatically if the chosen backend is unavailable.
    relevance_backend: Literal["embeddings", "tfidf", "heuristic"] = "embeddings"
    embedding_model: str = "all-MiniLM-L6-v2"
    # Calibration feedback as a reliability curve (over/under-confidence per
    # probability band + top misses) instead of dumping raw resolved rows.
    calibration_buckets: bool = True

    def model_post_init(self, __context) -> None:
        """Apply intensity preset as defaults — explicit overrides win."""
        # We only apply the preset when the individual values match
        # the "medium" defaults, meaning the user didn't set them explicitly.
        preset = _INTENSITY_PRESETS.get(self.api_intensity, {})
        medium = _INTENSITY_PRESETS["medium"]
        for key, preset_val in preset.items():
            current = getattr(self, key)
            default = medium[key]
            if current == default and preset_val != default:
                object.__setattr__(self, key, preset_val)


class CalibrationConfig(BaseModel):
    min_samples: int = 30
    refit_interval_hours: int = 6


class MarketMakerConfig(BaseModel):
    enabled: bool = True
    min_spread_bps: int = 40  # minimum spread in bps; below the 1-tick improvement, join BBO
    # Upper spread bound. A nominal spread this wide is not a fat opportunity —
    # it's a dead/empty book (e.g. bid 0.02 / ask 0.98 = 9600 bps), where neither
    # leg ever fills and we churn cancel/replace forever. Real MM edge lives in
    # the ~100-1000 bps range; reject anything wider at both selection and quote time.
    max_spread_bps: int = 1500
    quote_size: float = 10.0  # tokens per side
    max_inventory: float = 50.0  # max directional exposure per market
    max_markets: int = 5  # max simultaneous MM markets
    refresh_seconds: int = 30  # re-quote frequency


class TechnicalConfig(BaseModel):
    enabled: bool = True
    min_move_pct: float = 5.0
    mean_rev_threshold: float = 0.10
    min_history_points: int = 5


class BiasHarvestConfig(BaseModel):
    """Favorite-longshot bias harvest (see strategy/bias_harvest.py).

    PAPER-FORCED by default: flip ``paper`` to false only after the paper
    ledger (``auramaur pnl --paper``) proves the edge live-shaped. The
    backtested edge dies above ~2c slippage, so entries are passive limits
    at the observed price; ``edge_uplift`` must stay below the divergence
    filter's adverse-band floor (0.05) or entries get blocked at MEDIUM
    confidence — by design.
    """

    enabled: bool = False
    paper: bool = True
    band_lo: float = 0.80
    band_hi: float = 0.97
    edge_uplift: float = 0.04
    stake_usd: float = 10.0
    max_open: int = 40
    max_entries_per_cycle: int = 5
    scan_limit: int = 300
    min_liquidity: float = 1000.0
    min_hours_to_resolution: float = 6.0
    max_days_to_resolution: float = 45.0
    interval_seconds: int = 600


class GraduationConfig(BaseModel):
    """Graduation ladder (risk/graduation.py) — capital earned per
    (strategy × category) cell from the pnl_ledger record.

    mode: "observe" logs what enforce WOULD do (rollout default);
    "enforce" paper-forces unproven/demoted cells and applies the
    probation multiplier; "off" disables. Entries only — exits never
    pass through the risk manager.
    """

    mode: str = "observe"
    min_events: int = 20
    window_days: int = 90
    probation_multiplier: float = 0.5
    cache_seconds: int = 300
    exempt_strategies: list[str] = ["arbitrage", "market_maker", "order_monitor"]


class BrokerConfig(BaseModel):
    sync_interval_seconds: int = 60
    use_limit_orders: bool = True
    limit_spread_threshold: float = 0.03  # Use limits when spread >= 3 cents
    limit_edge_threshold: float = 20.0    # Use market orders when edge > 20%
    limit_price_improvement_ticks: int = 1  # Improve on BBO by 1 tick
    max_slippage_bps: int = 100


class KalshiConfig(BaseModel):
    enabled: bool = False
    api_key: str = ""
    private_key_path: str = ""
    environment: str = "demo"  # "demo" | "prod"


class IBKRConfig(BaseModel):
    enabled: bool = False
    # `enabled` is the master switch (connect to IBKR at all). The two books
    # beneath it are gated independently: options_enabled (the option-chain
    # scanner) and directional_equity_enabled (the stocks book). Keep options
    # off to run equities without the OPRA-less scanner spamming Error 200/10091.
    options_enabled: bool = False
    host: str = "127.0.0.1"
    paper_port: int = 7497
    live_port: int = 7496
    client_id: int = 1
    environment: str = "paper"  # "paper" | "live"
    watchlist: list[str] = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "GOOGL"]
    max_contracts_per_symbol: int = 10
    # IB market-data type: 1=live (needs paid subscription), 2=frozen,
    # 3=delayed, 4=delayed-frozen. Default 3 so the scanner works WITHOUT an
    # OPRA/equity subscription (delayed quotes + greeks). Set 1 once subscribed.
    market_data_type: int = 3
    # The options client connects read-only by default (data only). To place
    # equity orders the trading connection must be read-only=False.
    readonly: bool = True
    # Paper-trade mode: route IBKR orders to the paper ledger and size against
    # paper_budget_usd instead of the live account. Lets IBKR exercise the full
    # discovery->analysis->execution pipeline (and feed calibration) while the
    # live account is unfunded / cash-starved. Flip to false once funded to go
    # live. Options are expensive (~$300+/contract) so the live $0 balance would
    # size to zero — the paper budget is what makes prep trading possible.
    paper_trade: bool = True
    paper_budget_usd: float = 5000.0

    # --- Directional equity speculation (gated; no validated edge) ---
    # Mirrors the Kraken directional pillar. Uses its OWN socket connection
    # (equity_client_id) so it doesn't clash with the options client.
    directional_equity_enabled: bool = False
    directional_equity_symbols: list[str] = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
    directional_equity_budget_usd: float = 100.0   # total open speculative exposure
    equity_max_order_usd: float = 50.0             # hard per-order ceiling
    directional_equity_momentum_pct: float = 2.0
    directional_equity_lookback: int = 12          # bars of recent history
    equity_client_id: int = 2                      # distinct from client_id

    # --- Auto FX top-up (CAD->USD funding for the USD-priced stock book) ---
    # Mirrors the Kraken fiat->USDC treasury convert: keep enough *settled* USD
    # buying power for the equity book by converting idle base-currency cash.
    # Off by default. A real conversion still needs the three live gates;
    # otherwise it dry-runs (logs the intended convert). On a cash account
    # converted funds settle T+1, so this maintains a buffer AHEAD of trading
    # rather than funding a same-cycle order. Small orders auto-route as odd
    # lots, so the per-convert cap can sit well under the IDEALPRO 25k minimum.
    auto_fx_enabled: bool = False
    fx_source_currency: str = "CAD"                 # idle cash to draw down
    fx_target_usd: float = 120.0                    # keep >= this much settled USD
    fx_max_convert_usd: float = 150.0              # hard per-conversion ceiling
    fx_min_convert_usd: float = 20.0              # skip dust conversions


class CryptoComConfig(BaseModel):
    enabled: bool = False
    api_key: str = ""
    api_secret: str = ""
    environment: str = "sandbox"  # "sandbox" | "prod"


class KrakenConfig(BaseModel):
    """Kraken SPOT venue — treasury/conversion + (gated) directional.

    Not a binary prediction venue, so it is NOT wired into the binary
    TradingEngine. Spot orders still pass the same three-gate live model;
    until then they run validate-only against Kraken (no execution).
    """

    enabled: bool = False
    quote_currency: str = "USD"
    # Hard ceiling per spot order, independent of the binary risk manager.
    max_order_usd: float = 25.0

    # --- Treasury pillar (always-on when enabled) ---
    treasury_interval_seconds: int = 300
    auto_convert: bool = True             # auto idle-fiat -> USDC
    target_usdc: float = 50.0             # convert fiat until USDC reserve hits this
    fiat_assets: list[str] = ["ZCAD", "ZUSD", "ZEUR"]  # balances treated as idle fiat
    refill_cash_floor: float = 20.0       # alert to refill Polymarket below this cash

    # --- Directional spot (gated; no validated edge — flip on deliberately) ---
    directional_enabled: bool = False
    directional_pairs: list[str] = []     # e.g. ["XBTUSDC", "ETHUSDC"] (USDC-funded)
    # Liquidate "orphaned" directional positions — crypto we still hold on Kraken
    # whose pair was pruned from the valid set or removed from directional_pairs.
    # Without this they sit unsold forever (the exit loop only iterates configured
    # pairs). Safe because treasury holds only USDC/fiat, so non-stable crypto on
    # the account is by definition directional exposure. Set False to only detect
    # + log orphans without auto-selling.
    directional_liquidate_orphans: bool = True
    directional_momentum_pct: float = 3.0  # legacy symmetric threshold (fallback)
    # Asymmetric long bias: enter on a smaller up-move, exit only on a larger
    # down-move so winners ride longer. Fall back to directional_momentum_pct.
    directional_entry_momentum_pct: float = 2.0
    directional_exit_momentum_pct: float = 4.0
    directional_lookback: int = 12        # OHLC candles (hourly) for the momentum read
    # Hard downside stop: exit a held directional pair when it's down this many
    # percent from entry, regardless of momentum. The momentum exit alone (with
    # the asymmetric ride-winners bias) leaves no floor under a loser; this caps
    # it. 0 disables the stop (pure momentum). Default conservative.
    directional_stop_loss_pct: float = 12.0
    # Per-side taker fee estimate (round trip = 2x). Models the fee on
    # paper/validate fills and is folded into the take-profit threshold so a TP
    # only fires once the move clears costs. Live fills use the actual fee.
    directional_fee_pct: float = 0.26
    # Take-profit: exit a winner up this much from entry, NET of round-trip fees.
    # 0 disables it (winners ride, protected only by the trailing stop) — but with
    # a wide trailing_stop a sub-(trailing)% rally can never be banked, so winners
    # decay back into a momentum/stop loss (observed: 0W/7L). Default to a real
    # target so the book can actually realize gains.
    directional_take_profit_pct: float = 4.0
    # Trailing stop: once a position has been in profit, exit if it gives back
    # this many percent from its peak gain. Lets winners run while protecting
    # unrealized gains the from-entry stop can't (peak tracked in position_peaks,
    # so it survives restarts). 0 disables.
    directional_trailing_stop_pct: float = 8.0
    # After any exit, block re-entry on the same pair for this many minutes —
    # damps whipsaw churn (and the fees it bleeds). 0 disables.
    directional_reentry_cooldown_min: float = 30.0
    # Total $ the speculation engine may hold in open directional positions at
    # once — a hard ceiling so it can't consume the treasury reserve / CAD.
    directional_budget_usd: float = 50.0

    # --- LLM/news-driven directional signal (replaces price-only momentum) ---
    # Price-only momentum has no edge (backtested: every variant net-negative
    # after fees). This routes the bot's proven news->LLM crypto pipeline (72%
    # accuracy on resolved crypto markets) into the directional book instead: a
    # per-asset P(up over horizon) gates long entries. Default OFF; when on it
    # is PAPER-forced (validate-only orders) until the paper track record proves
    # edge — flip directional_llm_paper to False to go live.
    directional_llm_enabled: bool = False
    directional_llm_paper: bool = True       # force validate-only orders until proven
    directional_llm_min_prob: float = 0.60   # enter long when P(up) >= this
    directional_llm_exit_prob: float = 0.45  # exit long when P(up) falls below this
    directional_llm_min_confidence: str = "MEDIUM"  # Confidence floor to act
    directional_llm_horizon_days: int = 3    # prediction horizon in the question
    directional_llm_refresh_hours: float = 8.0  # re-run the LLM per pair at most this often
    # Conviction-weighted crypto budget (Tier 1). When enabled, the directional
    # budget ceiling is multiplied by an aggregate-conviction factor in
    # [min_mult, 1.0] derived from the cached LLM P(up) views — so a broadly
    # bullish+confident book leans toward the full ceiling, while a neutral one
    # holds more USDC. The factor is <=1.0 by construction, so this can only
    # REDUCE crypto exposure vs the static ceiling, never increase it.
    directional_conviction_budget_enabled: bool = False
    directional_conviction_min_mult: float = 0.34  # floor on the budget multiplier


class TransfersConfig(BaseModel):
    """Cross-venue fund movement (Kraken <-> Polymarket, Polygon USDC only).

    Gated by AURAMAUR_ENABLE_TRANSFERS *and* per-move approval. Withdrawals can
    only target Kraken withdrawal-address KEYS you pre-whitelisted in the Kraken
    UI — the API cannot send to an arbitrary address. Kalshi is bank-rail and
    not automatable.
    """

    enabled: bool = False
    per_transfer_cap_usd: float = 100.0
    daily_cap_usd: float = 250.0
    min_transfer_usd: float = 10.0
    # Names of withdrawal-address keys configured in the Kraken UI that the bot
    # is allowed to send to (e.g. your Polymarket Polygon USDC deposit address).
    allowed_withdraw_keys: list[str] = []
    # Require an explicit human approval for every transfer (recommended).
    require_approval: bool = True


class EnsembleConfig(BaseModel):
    enabled: bool = False
    source_weights_update_hours: int = 24
    price_move_threshold_pct: float = 5.0


class LLMEnsembleConfig(BaseModel):
    """Config for multi-LLM ensemble (runs multiple models in parallel)."""

    enabled: bool = True  # Enable by default since we have 2 Max+ accounts
    models: list[str] = ["opus", "sonnet"]
    min_samples_for_weights: int = 10  # Min resolved predictions before weighting
    default_weight: float = 0.5  # Starting weight per model (50/50)
    # Burn control: the ensemble doubles the heaviest (full-context) call on
    # every cycle. Gate it so the extra model(s) only run when the primary
    # already found a tradeable edge in the batch — quiet cycles stay 1-call.
    gate_on_edge: bool = True
    edge_threshold_pct: float = 5.0  # |primary_prob - market_price| to trigger the ensemble


class MomentumCouplingConfig(BaseModel):
    """Fast path — the short-turnaround spot->prediction lead-lag pillar.

    Runs on a fast cadence with a momentum signal (NOT the LLM loop): when crypto
    spot moves, the coupled prediction market is expected to reprice ~minutes
    later, so we'd take the matching side early. OFF by default — detection-only
    scaffold until coupling_tradeability.py confirms it's profitable after cost.
    """

    enabled: bool = False
    poll_seconds: int = 60
    lookback_seconds: int = 600        # window over which a spot move is measured
    move_threshold_pct: float = 0.5    # |spot move| over lookback to fire
    assets: list[str] = ["BTC", "ETH"]
    near_money_pct: float = 0.05       # only trade markets whose strike is within this of spot
    max_position_usd: float = 25.0
    execute: bool = False              # detection-only until True (post-validation)


class GeminiConfig(BaseModel):
    """Gemini as an off-hours / budget-relief LLM. Routes analysis to Gemini when
    it's off-hours OR Claude's daily budget is near-exhausted; falls back to
    Claude if Gemini errors."""

    enabled: bool = False
    model: str = "gemini-3.1-pro-preview"
    # UTC hours to prefer Gemini (default = deep-night quiet hours).
    off_hours_utc: list[int] = Field(default_factory=lambda: [4, 5, 6, 7, 8, 9])
    # Switch to Gemini once Claude calls reach this fraction of the daily budget.
    claude_budget_threshold: float = 0.8


class ArbitrageConfig(BaseModel):
    enabled: bool = True
    min_profit_after_fees_pct: float = 1.5
    max_arb_size: float = 25.0
    cross_exchange_auto_execute: bool = True
    negrisk_auto_execute: bool = False
    exchange_fees: dict[str, float] = Field(default_factory=lambda: {
        "polymarket": 0.0,
        "kalshi": 0.07,
    })


class AnalysisConfig(BaseModel):
    """Controls which analysis backend is used."""

    mode: Literal["pipeline", "strategic", "agent"] = "strategic"


class HybridConfig(BaseModel):
    """Multi-strategy mode: arb + news speed + domain LLM + market making."""

    arb_scan_seconds: int = 60
    news_fast_analysis: bool = True
    news_cycle_seconds: int = 30
    llm_domain_filter: bool = True
    llm_whitelist_min_accuracy: float = 0.50
    llm_whitelist_min_trades: int = 5
    market_maker_auto_enable: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json_format: bool = True
    file: str = "auramaur.log"


class Settings(BaseSettings):
    # API Keys
    anthropic_api_key_primary: str = ""
    anthropic_api_key_secondary: str = ""
    polygon_private_key: str = ""
    polymarket_api_key: str = ""
    polymarket_api_secret: str = ""
    polymarket_passphrase: str = ""
    polymarket_proxy_address: str = ""
    newsapi_key: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "auramaur/0.1"
    twitter_bearer_token: str = ""
    fred_api_key: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook_url: str = ""

    # Kalshi
    kalshi_api_key: str = ""
    kalshi_private_key_path: str = ""

    # Crypto.com
    cryptodotcom_api_key: str = ""
    cryptodotcom_api_secret: str = ""

    # Kraken (spot). Used for read-only wallet/balance checks today; no trading
    # adapter is wired yet. Key needs only the "Query Funds" permission —
    # leave "Withdraw Funds" OFF.
    kraken_api_key: str = ""
    kraken_api_secret: str = ""

    # Google Gemini — LLM fallback for off-hours / when Claude budget is low.
    gemini_api_key: str = ""

    # Global risk-tolerance lever: 0=most conservative, 50=neutral, 100=YOLO.
    # Scales the whole prob/stat/risk surface at the RiskManager gateway.
    # From defaults.yaml (risk_tolerance:) and overridable via env RISK_TOLERANCE.
    risk_tolerance: float = Field(default_factory=lambda: float(_DEFAULTS.get("risk_tolerance", 50.0)))

    # Safety
    auramaur_live: bool = False
    # On-chain redemption — real Polygon transactions. Defaulted ON so resolved
    # winners auto-claim back to USDC (recycling capital). Still requires the
    # full live triple-gate (auramaur_live + execution.live + no KILL_SWITCH) in
    # _is_live_submission_allowed(), so this never fires outside live trading;
    # override with env AURAMAUR_ENABLE_REDEMPTION=false to disable.
    auramaur_enable_redemption: bool = True

    # Separate opt-in for cross-venue fund transfers (Kraken -> Polymarket
    # withdrawals). Gated independently of auramaur_live AND of redemption so
    # that enabling live trading never implies the bot can move funds off-venue.
    auramaur_enable_transfers: bool = False

    # Polygon RPC for on-chain redemption. Defaults to a public endpoint;
    # override with a paid provider (Alchemy/Infura/QuickNode) for reliability.
    polygon_rpc_url: str = "https://polygon-bor-rpc.publicnode.com"

    # Sub-configs
    execution: ExecutionConfig = Field(default_factory=lambda: ExecutionConfig(**_DEFAULTS.get("execution", {})))
    risk: RiskConfig = Field(default_factory=lambda: RiskConfig(**_DEFAULTS.get("risk", {})))
    kelly: KellyConfig = Field(default_factory=lambda: KellyConfig(**_DEFAULTS.get("kelly", {})))
    intervals: IntervalsConfig = Field(default_factory=lambda: IntervalsConfig(**_DEFAULTS.get("intervals", {})))
    nlp: NLPConfig = Field(default_factory=lambda: NLPConfig(**_DEFAULTS.get("nlp", {})))
    calibration: CalibrationConfig = Field(default_factory=lambda: CalibrationConfig(**_DEFAULTS.get("calibration", {})))
    broker: BrokerConfig = Field(default_factory=lambda: BrokerConfig(**_DEFAULTS.get("broker", {})))
    kalshi: KalshiConfig = Field(default_factory=lambda: KalshiConfig(**_DEFAULTS.get("kalshi", {})))
    ibkr: IBKRConfig = Field(default_factory=lambda: IBKRConfig(**_DEFAULTS.get("ibkr", {})))
    cryptodotcom: CryptoComConfig = Field(default_factory=lambda: CryptoComConfig(**_DEFAULTS.get("cryptodotcom", {})))
    kraken: KrakenConfig = Field(default_factory=lambda: KrakenConfig(**_DEFAULTS.get("kraken", {})))
    transfers: TransfersConfig = Field(default_factory=lambda: TransfersConfig(**_DEFAULTS.get("transfers", {})))
    ensemble: EnsembleConfig = Field(default_factory=lambda: EnsembleConfig(**_DEFAULTS.get("ensemble", {})))
    llm_ensemble: LLMEnsembleConfig = Field(default_factory=lambda: LLMEnsembleConfig(**_DEFAULTS.get("llm_ensemble", {})))
    gemini: GeminiConfig = Field(default_factory=lambda: GeminiConfig(**_DEFAULTS.get("gemini", {})))
    momentum_coupling: MomentumCouplingConfig = Field(default_factory=lambda: MomentumCouplingConfig(**_DEFAULTS.get("momentum_coupling", {})))
    market_maker: MarketMakerConfig = Field(default_factory=lambda: MarketMakerConfig(**_DEFAULTS.get("market_maker", {})))
    technical: TechnicalConfig = Field(default_factory=lambda: TechnicalConfig(**_DEFAULTS.get("technical", {})))
    bias_harvest: BiasHarvestConfig = Field(default_factory=lambda: BiasHarvestConfig(**_DEFAULTS.get("bias_harvest", {})))
    graduation: GraduationConfig = Field(default_factory=lambda: GraduationConfig(**_DEFAULTS.get("graduation", {})))
    arbitrage: ArbitrageConfig = Field(default_factory=lambda: ArbitrageConfig(**_DEFAULTS.get("arbitrage", {})))
    analysis: AnalysisConfig = Field(default_factory=lambda: AnalysisConfig(**_DEFAULTS.get("analysis", {})))
    hybrid: HybridConfig = Field(default_factory=lambda: HybridConfig(**_DEFAULTS.get("hybrid", {})))
    logging: LoggingConfig = Field(default_factory=lambda: LoggingConfig(**_DEFAULTS.get("logging", {})))

    # Resolve .env to an absolute path anchored at the repo root so Settings
    # loads the same secrets regardless of the caller's CWD. A bare ".env"
    # would be searched relative to CWD, which fails when the bot is
    # launched from the inner `auramaur/` package directory.
    model_config = {
        "env_file": str(Path(__file__).resolve().parent.parent / ".env"),
        "env_file_encoding": "utf-8",
        # Ignore env vars we don't declare. The process shares its environment
        # with libraries that read their own tokens directly (e.g. HF_TOKEN /
        # hf_token for huggingface_hub via sentence-transformers), and an
        # unrelated stray var shouldn't crash Settings on startup.
        "extra": "ignore",
    }

    @property
    def kill_switch_active(self) -> bool:
        # Anchor the kill switch to the repo root (two levels up from this
        # file: config/ -> repo root), not the caller's CWD. A bare
        # Path("KILL_SWITCH") would be CWD-relative and could miss the switch
        # when the bot is launched from the inner package directory.
        repo_root = Path(__file__).resolve().parent.parent
        return (repo_root / "KILL_SWITCH").exists() or Path("KILL_SWITCH").exists()

    @property
    def is_live(self) -> bool:
        """All three gates must be true for live trading."""
        return self.auramaur_live and self.execution.live and not self.kill_switch_active

    @property
    def transfers_armed(self) -> bool:
        """Whether real cross-venue withdrawals may execute.

        Independent of is_live: a transfer needs its OWN env gate
        (AURAMAUR_ENABLE_TRANSFERS) plus config enablement, and is always
        halted by the kill switch. Per-transfer caps, the whitelist, and
        human approval are enforced at the transfer call site on top of this.
        """
        return (
            self.auramaur_enable_transfers
            and self.transfers.enabled
            and not self.kill_switch_active
        )
