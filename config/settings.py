"""Pydantic Settings for Auramaur configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge ``over`` onto ``base`` (override wins; dicts merge)."""
    out = dict(base)
    for key, value in (over or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_defaults() -> dict:
    """Load tracked ``defaults.yaml`` then deep-merge optional local overrides.

    ``defaults.local.yaml`` (gitignored) holds operational / never-commit
    values — the live gate, paper-book headroom, model-budget conservation
    knobs — so the tracked file stays a paper-safe baseline. The local file is
    absent in CI and fresh clones, so behavior there falls back to the tracked
    defaults (and, for keys not present in either, the pydantic field defaults).
    """
    base: dict = {}
    defaults_path = Path(__file__).parent / "defaults.yaml"
    if defaults_path.exists():
        with open(defaults_path) as f:
            base = yaml.safe_load(f) or {}
    local_path = Path(os.environ.get(
        "AURAMAUR_LOCAL_CONFIG",
        Path(__file__).parent / "defaults.local.yaml",
    ))
    if local_path.exists():
        with open(local_path) as f:
            base = _deep_merge(base, yaml.safe_load(f) or {})
    return base


_DEFAULTS = _load_defaults()


class ExecutionConfig(BaseModel):
    live: bool = False
    # Paper book capital. Sized for headroom, not realism: the provisional
    # paper-forced strategies must hold enough concurrent positions across
    # (strategy x category) cells to accrue the graduation sample. Sized so
    # paper cash is never the binding constraint on entries; a too-small book
    # starves them. Recycles on settlement (see #132).
    paper_initial_balance: float = 5000.0
    limit_order_ttl_seconds: int = 120
    # Max cents the router may pay above the signal's reference price to lift
    # the ask (marketable entry) instead of resting a maker quote at bid+1
    # that the TTL reaper usually kills unfilled. The cross also has to leave
    # net edge above risk.min_edge_pct, so this cap only binds on wide edges.
    entry_max_cross_cents: int = 4
    # Exit twin of entry_max_cross_cents: the slippage band for marketable
    # exits. A SELL only fills by crossing down to the real bid; pricing at the
    # snapshot (or anywhere inside the spread) rests above the bid and
    # TTL-cancels forever (a held winner once looped this way for days). So we
    # take the bid outright when it is within this many cents of the snapshot,
    # otherwise skip and let the portfolio monitor back off until the book
    # tightens or the position redeems at resolution.
    exit_max_cross_cents: int = 10
    # Absolute floor on the bid an exit will cross into. Below this, redeeming
    # at resolution beats dumping into a near-zero buyer (the "junk 1c bid"
    # guard), regardless of the slippage band above.
    exit_min_bid_price: float = 0.05
    spread_capture_min_bps: int = 50
    # Depth-aware entry routing: size an entry against the ACTUAL book, not just
    # the top-of-book ask. The router walks the asks up to the slippage budget
    # (the price at which realizable edge would fall to min_edge, also bounded
    # by entry_max_cross_cents), and trims the order to the depth available
    # there. depth_aware_routing toggles the behavior; book_capacity_fraction
    # caps how much of that in-budget depth a single order may take (don't be
    # the whole book); min_fill_fraction skips the entry when less than this
    # share of the requested size fits within budget (a dust fill isn't worth
    # the entry). Set depth_aware_routing False to restore top-of-book pricing.
    depth_aware_routing: bool = True
    book_capacity_fraction: float = 0.5
    min_fill_fraction: float = 0.5
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
    # Hard absolute ceiling on per-market stake, in dollars. max_stake_per_market
    # is interpreted as a fraction of EQUITY when <= 1.0, which grows as the book
    # grows; the regime scaler can lift it further. This is the final clamp so the
    # documented per-market cap actually binds regardless of equity/regime
    # (a $40 entry at ~3.3% of equity slipped through before, 2026-06-15).
    max_stake_abs_ceiling: float = 25.0
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
    # Name-the-gap gate: an LLM signal whose probability diverges from the
    # market by >= min_divergence must carry a NAMED mispricing mechanism
    # (structural/behavioral/informational) from the post-hoc gap audit, or
    # it does not trade. "none"/unauditable blocks. The estimation pipeline
    # stays price-blind; the audit is a separate lazy LLM call made only for
    # otherwise-approved trades.
    mispricing_gate_enabled: bool = False
    mispricing_min_divergence: float = 0.05
    mispricing_audit_ttl_hours: float = 12.0
    # sports: the LLM has no structural edge on game outcomes / spreads / O-U
    # (driven by live injury & lineup info we don't ingest), and resolved-market
    # performance bears that out. Blocked outright rather than left to the
    # feedback loop, which needs a sample sports keeps losing money to build.
    blocked_categories: list[str] = ["sports"]
    # LIVE entries are allowlist-gated (fail-safe): a category must be ON this
    # list to trade real money; unknown/''/'other'/mislabeled categories can
    # paper-trade but never go live. The blocklist above still governs paper
    # (exploration) and the engine's candidate filter. Rationale: a blocklist
    # fails OPEN on every classification gap — the 2026-06 mislabel leak
    # bought tennis and Senate-control markets live. An allowlist makes
    # classifier bugs cost opportunity instead of money.
    allowed_categories_live: list[str] = [
        "crypto", "tech", "politics_intl", "economics", "science", "legal",
        "entertainment", "weather", "esports",
    ]
    # Per-strategy live-category extensions, keyed by strategy_source. Lets a
    # proven graduation cell (e.g. bias_harvest x other) earn its category
    # WITHOUT adding it to the global list above — which the graduation-exempt
    # market maker / arb executor consume directly, so a global 'other' would
    # fail-open the exact classifier-gap hole the allowlist exists to close.
    # Consulted only by the gateway's category allowlist check (#18).
    allowed_categories_live_extra: dict[str, list[str]] = Field(
        default_factory=dict)


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
    # Order-book recorder (data capture for the reversion cost-gate). Read-only;
    # shares the live CLOB client, so it's throttled + capped. Real off-switch /
    # tuning so it can be backed off if it ever slows live trade/exit calls.
    orderbook_recorder_enabled: bool = True
    orderbook_seconds: int = 300
    orderbook_min_liquidity: float = 1000.0
    orderbook_max_markets: int = 150
    orderbook_call_pause_seconds: float = 0.25


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
    model: str = "claude-opus-4-8"
    max_tokens: int = 4096
    api_intensity: Literal["low", "medium", "full_blast"] = "medium"
    skip_second_opinion: bool = False
    max_markets_per_cycle: int = 10
    evidence_per_source: int = 3
    daily_claude_call_budget: int = 100
    # Slice of the daily budget held back for pin_claude callers (the proven
    # edges whose quality depends on the specific model). Unpinned callers stop
    # at budget - reserve; pinned callers can spend up to the full budget, so a
    # bulk consumer can never starve the money-making calls.
    claude_reserve_for_pinned: int = 25
    # Pacing envelope on the NON-RESERVED pool (call_budget.paced_limit). The
    # counter resets at midnight UTC but opportunity flow peaks 12-22 UTC
    # (measured; US prints land 12:30 UTC), and greedy consumption exhausted
    # the pool by early afternoon — dead when the flow arrives. Before
    # peak_start only offpeak_share of the pool may be spent; inside/after
    # the window the remainder unlocks. offpeak_share: 1.0 disables.
    budget_peak_start_hour_utc: int = 12
    budget_peak_end_hour_utc: int = 22
    budget_offpeak_share: float = 0.4

    # Tool-use analyzer — refines strategic-batch results on top-edge markets
    # by letting Claude Code drive its own web_search / web_fetch. "auto"
    # fires tool-use only when the strategic batch already showed a strong
    # edge signal; "tool_use" forces it for every batched market;
    # "strategic_batch" disables the refinement path entirely.
    analysis_mode: Literal["strategic_batch", "tool_use", "auto"] = "auto"
    # Min seconds between strategic batch+adversarial LLM runs. The engine calls
    # it per scan cycle (~every 10 min) but directional signals are paper-forced,
    # so per-cycle batching mostly burned budget; cap the cadence and serve
    # cached results in between. 0 disables the throttle.
    strategic_min_interval_seconds: int = 1800
    # Lever 3: tool-use refinement is the single heaviest call (multi-turn web).
    # Tightened from 5.0/4 — only strong edges earn a web-research pass, and at
    # most 2 per cycle — which is where most of the realized token burn lived.
    tool_use_edge_threshold_pct: float = 8.0  # edge % above which tool-use fires in auto mode
    tool_use_max_budget_usd: float = 0.50  # per-market tool-use budget cap
    tool_use_max_markets_per_cycle: int = 2  # cap concurrent refinements per cycle
    tool_use_model: str = "claude-opus-4-8"  # can differ from strategic batch model

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

    # Lever 6: rejection cooldown. A risk-rejected market re-entered the
    # candidate pool as soon as the 15-minute recently-analyzed window lapsed,
    # so the same dud markets burned a fresh evidence pass + LLM call every
    # ~20 minutes all day (on Kalshi this was the entire signal stream). The
    # verdict can't flip until something moves, so bench the market until the
    # cooldown expires, it reprices by the escape threshold, or a news flag
    # promotes it — the cooldown must never blind the bot to new information.
    rejection_cooldown_minutes: int = 240
    rejection_reprice_threshold: float = 0.03  # abs yes-price move that lifts the bench early

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
    # Live-cash floor MM must leave untouched: no NEW quote pairs while
    # spendable collateral <= this (exits/cancels never gated). Stops MM —
    # the only always-live cell — from auto-claiming every deposited dollar
    # as inventory working capital.
    cash_reserve_usd: float = 50.0
    refresh_seconds: int = 30  # re-quote frequency
    # Per-operation watchdog: a single Polymarket call (orderbook fetch, quote
    # placement) that stalls without a timeout will hang the WHOLE MM loop
    # indefinitely — observed twice 2026-06-30 (the loop went silent for 12-24
    # min on a stuck request). Bound each per-market quote op and the stale-cancel
    # with this timeout so a stuck call is abandoned and the cycle continues.
    op_timeout_seconds: float = 15.0


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
    # Deep band only. The backtest's edge lived in 0.90-0.97 (won 99.3% of 151);
    # the shallow 0.80-0.90 tier has no favorite-longshot edge in practice — paper
    # showed it busting ~24% vs the ~12-20% its price implied (net-negative), while
    # 0.90-0.97 ran 94% win / net-positive. Raised 0.80->0.90 so the cell harvests
    # only where the bias is real. See [[bias-harvest-strategy]].
    band_lo: float = 0.90
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
    # Tail-filter: skip a favorite whose UMA resolution is actively disputed.
    # The deep-band backtest won 99.3%, but the rare flips that produced the
    # paper track's fat-tail losses are disproportionately contested
    # resolutions — a disputed market is price-pinned to the *proposed* outcome
    # and can reverse. Fails open (only an ACTIVE dispute is skipped), so a
    # market with no UMA data still enters as before. See [[uma-dispute-gate]].
    skip_disputed: bool = True
    # Categories where the favorite-longshot harvest has NO edge and bleeds —
    # the "longshot" carries genuine directional signal, not mispricing, so the
    # band sells correctly-priced outcomes and pays the asymmetric tail. The
    # paper track localised the loss to weather (summer heat genuinely hits temp
    # thresholds) and sports/politics_us (already in risk.blocked_categories,
    # but those are bias-harvest-specific no-edge zones too — weather can't be a
    # global block because weather_temp trades it profitably). Checked on top of
    # risk.blocked_categories, against the CLASSIFIED category (not the raw label
    # that an unclassified market would slip through). See [[bias-harvest-strategy]].
    exclude_categories: list[str] = ["weather", "sports", "politics_us"]
    # Maker entry (research: GWU WP 2026-001 / Whelan — the favorite-longshot edge
    # accrues to MAKERS (~-9.6% avg return) not TAKERS (~-31.5%); the prior entry
    # paid the observed price = taker economics, which the paper ledger surfaced as
    # the strategy being "slippage-bled" despite 88% win). When true, post the BUY
    # at the favored-side BID (capture the spread) instead of the last price, and
    # only when there is a real spread to capture (>= maker_min_spread).
    maker_entry: bool = True
    maker_min_spread: float = 0.02
    # Paper realism: maker posts do NOT always get hit. Without modelling this the
    # paper book would assume 100% maker fills and read far too rosy — dangerous
    # because the graduation ladder auto-promotes at 20 positive events. So in
    # paper, deterministically (stable market-id hash) admit only this fraction of
    # otherwise-eligible markets, modelling a realistic maker CAPTURE rate. The
    # real fill rate is THE risk to validate before any live-arm. 1.0 disables.
    paper_maker_fill_rate: float = 0.5


class InformedFlowConfig(BaseModel):
    """Informed-flow follower over Kalshi (strategy/informed_flow_pillar.py).

    Mimics the side of abnormally-large (informed) order flow — abnormal trade
    size (ATS) proxies non-liquidity-motivated trading that predicts resolution
    (Delvecchio CMC thesis #4166; Bartlett & O'Hara). Forecast-free: we don't
    estimate a probability, we follow the informed side with a small uplift.

    MEDIUM confidence (single in-sample thesis) + adverse-selection risk (the edge
    needs us on the informed, not picked-off, side) -> PAPER-FORCED, own cell.
    No-ops cleanly when the Kalshi venue isn't composed.
    """

    enabled: bool = False
    paper: bool = True
    # Detector params (see strategy/informed_flow.detect_informed_flow).
    min_abnormal_sample: int = 20    # min sized trades for a stable baseline
    size_mult: float = 3.0           # abnormal = size >= this x median size
    min_dominance: float = 0.6       # informed side must carry this abnormal share
    trades_limit: int = 200          # tape depth pulled per market
    # Follow with a small uplift; MUST stay < 0.05 (divergence filter floor) so
    # the forecast-free entry isn't blocked at MEDIUM confidence (as bias_harvest).
    uplift: float = 0.04
    # Skip extremes: near 0/1 the ATS signal is noise / already-resolved.
    band_lo: float = 0.10
    band_hi: float = 0.90
    stake_usd: float = 10.0
    # Kalshi-SCALE liquidity floor. Kalshi's liquidity values run ~40x smaller
    # than Polymarket's (active-market MEDIAN ~26, only ~9/589 clear 1000): the
    # original Poly-scale 1000 left informed_flow with ZERO eligible markets — it
    # never even pulled a trade tape (found 2026-06-29). 50 admits a real candidate
    # pool; the detector's min_abnormal_sample (>=20 trades) is the true activity
    # gate. (Matches the kalshi_min_liquidity=50 used elsewhere in config.)
    min_liquidity: float = 50.0
    min_hours_to_resolution: float = 6.0
    max_days_to_resolution: float = 30.0
    max_open: int = 30
    max_entries_per_cycle: int = 5
    scan_limit: int = 200            # near-dated /markets window; tape pulled per eligible
    interval_seconds: int = 1800


class LongHorizonConfig(BaseModel):
    """Long-horizon favorite underpricing (strategy/long_horizon.py).

    Research basis (arXiv 2602.19520, 292M trades): prices are systematically
    UNDERCONFIDENT at long horizons — the calibration slope rises from ~0.99 near
    resolution to ~1.32 beyond a month, so long-dated favorites are underpriced. We
    apply ``slope`` to the market's OWN price (logit space) to get a fair and trade
    only the favored side's underpricing — never a forecast of our own.

    PAPER-FORCED. Politics is EXCLUDED (the paper's effect is strongest there, but
    politics is the bot's documented no-edge zone) so this cell tests whether the
    effect GENERALIZES to tech/crypto/macro net of cost. ``slope`` defaults below
    the paper's 1.32 — it rests on a single non-peer-reviewed preprint, so we size
    the correction conservatively.
    """

    enabled: bool = False
    paper: bool = True
    # Calibration slope applied in logit space. Conservative vs the paper's 1.32
    # (one preprint); raise toward 1.32 only once the paper ledger shows edge.
    slope: float = 1.25
    # Moderate-favorite band. Below ~0.52 the slope correction is negligible;
    # 0.90+ overlaps bias_harvest's near-resolution deep band and locks capital.
    # band_lo widened 0.55->0.52 (2026-06-29) to surface more candidates.
    band_lo: float = 0.52
    band_hi: float = 0.92
    min_edge: float = 0.03
    stake_usd: float = 10.0
    max_open: int = 30
    max_entries_per_cycle: int = 5
    # Raised 300->500 (2026-06-29): the binding constraint on data collection was
    # Pagination depth for the DATED scan (see _scan_long_dated). The 06-29
    # widening to 500 was a NO-OP because Gamma caps a page at 100 AND ordered by
    # volume — the top-100-by-volume contains zero 14-365d moderate favorites. The
    # fix queries the resolution-date window ordered by liquidity, paginated up to
    # this many markets, which surfaces the real candidates.
    scan_limit: int = 500
    min_liquidity: float = 1000.0
    min_days_to_resolution: float = 14.0
    # 365 (was 180): the >180d bucket is the LARGEST pool of long-dated favorites,
    # and the underconfidence slope is strongest at long tenor — capping at 180
    # threw away most candidates. 1yr lock-up is fine at paper / tiny stake.
    max_days_to_resolution: float = 365.0
    interval_seconds: int = 1800
    # Politics excluded: that's where the effect is strongest in the paper but
    # where the bot has no edge — so we test generalization elsewhere. Checked on
    # top of risk.blocked_categories, against the CLASSIFIED category.
    exclude_categories: list[str] = ["politics_us", "politics_intl"]
    # Kalshi instance (long_horizon_kalshi cell, paper-first on its own ledger).
    # Its exclusions ADMIT politics_intl: the live Kalshi evidence for the slope
    # edge is long-tenor geopolitical persistence — a price-slope trade, not a
    # forecast.
    kalshi_enabled: bool = False
    kalshi_exclude_categories: list[str] = ["politics_us"]
    # Kalshi's bulk liquidity field UNDERREPORTS (the lens hit the same wall:
    # floor 300 -> 50); the Polymarket floor rejected 98% of far-dated Kalshi
    # markets. And the persistence lane on Kalshi lives at MULTI-YEAR tenors
    # (the live-book winners resolve 2028-2035) — the 365d lock-up cap that
    # protects the Poly instance would exclude the entire lane; the decay
    # harvest is what makes long tenor affordable.
    kalshi_min_liquidity: float = 50.0
    kalshi_max_days_to_resolution: float = 1460.0
    # Decay harvest: exit once the side has captured this fraction of the
    # entry->$1 distance. 0 disables. Realizes the front-loaded premium on a
    # weeks clock (ladder-compatible) instead of waiting years for resolution.
    take_profit_capture: float = 0.6


class AgentTraderModel(BaseModel):
    """One experiment arm of the intelligence-cap A/B: a model identity plus
    the CLI effort it runs at. ``alias`` names the attribution cell
    (``agent_trader_<alias>``) — keep it stable or the cell's history splits.

    ``provider``: 'claude' arms run through the Max+ CLI (zero marginal
    cost); 'gemini' arms call the REST API (PAID per token — their usage is
    metered into agent_trader_costs so each arm's record can be judged net
    of its own intelligence bill; the operator's cost-inclusive rule)."""

    alias: str
    model: str
    effort: str = "medium"
    provider: str = "claude"  # 'claude' | 'gemini'


class AgentTraderConfig(BaseModel):
    """LLM day-trader pillar (strategy/agent_trader.py) — the Hermes paradigm
    rebuilt on the bot's rails after the external-agent ledger fabrication
    (see agentmcp/book.py S4).

    Runs the SAME mandate/candidates/memory across every model in ``models``;
    each is its own strategy cell (``agent_trader_<alias>``) so the paper
    ledger answers whether model tier changes directional edge. PAPER-FORCED
    (``paper`` + new directional cells under the enforced graduation ladder).

    Budget: one non-reserved Claude CLI call per model per cycle — at the
    default 2h interval and 3 models that is ~36 calls/day, and the pillar
    stops early whenever the shared non-reserved budget is gone, so it can
    never starve the pinned (lens) slice.
    """

    enabled: bool = False
    paper: bool = True
    interval_seconds: int = 7200
    models: list[AgentTraderModel] = [
        AgentTraderModel(alias="haiku", model="claude-haiku-4-5"),
        AgentTraderModel(alias="sonnet", model="claude-sonnet-5"),
        AgentTraderModel(alias="opus", model="claude-opus-4-8"),
        AgentTraderModel(alias="gflash", model="gemini-3.1-flash-lite",
                         provider="gemini"),
        AgentTraderModel(alias="g35flash", model="gemini-3.5-flash",
                         provider="gemini"),
        AgentTraderModel(alias="gpro", model="gemini-3.1-pro-preview",
                         provider="gemini"),
    ]
    # Gemini arms: daily REST-call ceiling across all gemini arms (paid API,
    # independent of the Claude paced pool) and per-model $/1M-token prices
    # [input, output] for the cost meter. Prices are config, not gospel —
    # update from the current rate card.
    gemini_daily_call_limit: int = 30
    gemini_price_per_mtok: dict[str, list[float]] = {
        "gemini-3.1-flash-lite": [0.10, 0.40],
        "gemini-3.5-flash": [0.50, 3.50],
        "gemini-3.1-pro-preview": [2.00, 12.00],
    }
    scan_limit: int = 200
    markets_per_cycle: int = 10
    max_entries_per_cycle: int = 2
    max_open_per_model: int = 10
    stake_usd: float = 10.0
    min_liquidity: float = 1000.0
    # Day-trader horizon: near-dated books turn over fast enough to feed the
    # memory loop; min keeps out markets resolving mid-cycle.
    min_days_to_resolution: float = 0.25
    max_days_to_resolution: float = 30.0
    min_edge_pts: float = 5.0
    memory_events: int = 12
    exclude_categories: list[str] = []
    # Generous: the arms may run WebSearch rounds before answering.
    llm_timeout_seconds: int = 420
    # How long a pass on an offered market keeps it out of that arm's
    # candidate slate (prevents burning calls re-declining the same markets
    # every cycle; after the TTL prices have moved enough to re-ask).
    decline_ttl_hours: float = 24.0


class TermStructureConfig(BaseModel):
    """Deadline-ladder curve reader (strategy/term_structure.py).

    One LLM read per FAMILY (same event, multiple 'by <date>' strikes) into an
    event-time curve, then every strike is priced off that curve — one call
    amortizes across up to a dozen markets, attacking the budget-throughput
    constraint that starves per-market readers. PAPER-FORCED new directional
    cell. Curves are cached ``curve_ttl_hours`` so steady-state spends calls
    only on new/expired families.
    """

    enabled: bool = False
    paper: bool = True
    interval_seconds: int = 7200
    model: str = "claude-opus-4-8"
    effort: str = "medium"
    scan_limit: int = 300
    min_strikes: int = 3
    max_families: int = 16
    families_per_cycle: int = 5   # fresh LLM reads per cycle (cached fams free)
    curve_ttl_hours: float = 24.0
    max_entries_per_family: int = 2
    stake_usd: float = 10.0
    min_liquidity: float = 1000.0
    min_days: float = 0.25
    max_days: float = 90.0
    min_edge_pts: float = 8.0
    llm_timeout_seconds: int = 420
    exclude_categories: list[str] = []


class VolAnchorConfig(BaseModel):
    """Deterministic vol-anchored crypto threshold pricing (strategy/vol_anchor.py).

    Edge: crowd threshold prices back out to a FLAT implied-vol term structure
    anchored on the recent tape; vol mean-reverts, so long-dated touch markets
    are mispriced whenever recent realized sits far from the long-run anchor.
    Zero LLM cost (spot/closes from CoinGecko + closed-form GBM, martingale
    convention, no drift view). PAPER-FORCED, own graduation cell.
    """

    enabled: bool = False
    paper: bool = True
    interval_seconds: int = 3600
    scan_limit: int = 300
    min_liquidity: float = 1000.0
    min_days: float = 1.0
    max_days: float = 240.0
    min_edge_pts: float = 8.0
    stake_usd: float = 10.0
    max_entries_per_cycle: int = 3
    realized_window_days: int = 30
    # Mean-reversion horizon for the sigma blend (years). Calibrated
    # 2026-07-09: weekly-AR(1) half-life measured ~1wk (biased low by RV
    # estimation noise); 0.05 (~18d decay) splits the measurement and the
    # vol-literature slow component. A 4-day market prices off the tape;
    # anything beyond ~6 weeks prices mostly off the anchor.
    tau_years: float = 0.05
    # Sigma source: 'deribit_iv' prices off Deribit's ATM implied-vol term
    # structure (the market-clearing surface; read-only public API) with the
    # calibrated blend as automatic per-asset fallback; 'blend' uses the
    # estimate alone. The priced log carries sigma_src either way.
    sigma_source: str = "deribit_iv"
    deribit_currencies: dict[str, str] = {
        "bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL",
    }
    deribit_ttl_seconds: float = 1800.0
    # Long-run annualized vol anchors, by coingecko id: 1y realized vol,
    # calibrated 2026-07-09 (year of daily closes, cross-checked against
    # ~30d of recorded tick data).
    long_run_vol: dict[str, float] = {
        "bitcoin": 0.45, "ethereum": 0.67, "solana": 0.72,
        "ripple": 0.68, "dogecoin": 0.78,
    }
    exclude_categories: list[str] = []


class EconIndicatorConfig(BaseModel):
    """Data-driven Kalshi economic-indicator bin pricing (strategy/econ_indicator.py).

    PAPER-FORCED by default (and disabled until opted in). Prices "Above X"
    ladders from FRED history; directional, so the graduation ladder keeps it
    paper until the ledger + calibration prove it. The edge must clear the
    Kalshi taker fee on top of min_edge (computed at runtime from the fee model).
    series=[] means "all registered" (econ_pricing.ECON_SERIES).
    """

    enabled: bool = False
    paper: bool = True
    series: list[str] = Field(default_factory=list)
    stake_usd: float = 10.0
    min_edge: float = 0.07
    # Upper sanity bound on |model - market|. A random-walk nowcast disagreeing
    # with the market by more than this isn't edge — it's the model being naive
    # against a forward-looking crowd (e.g. CPI YoY: the model anchors to the
    # last stale print while the market prices expected disinflation). Beyond
    # this gap, trust the market and skip — the econ analog of name-the-gap.
    max_divergence: float = 0.30
    max_open: int = 30
    max_entries_per_cycle: int = 5
    history_n: int = 60
    interval_seconds: int = 1800


class SettlementArbConfig(BaseModel):
    """Settlement-lag / known-outcome arb, FRED-first (strategy/settlement_arb.py).

    The structural generalization of the graduated resolution_lens × weather edge:
    trade a Polymarket econ market ONLY when the referenced FRED indicator's print
    for the reference period is ALREADY PUBLISHED and deterministically
    satisfies/fails the criterion, and the market hasn't repriced (the lag is the
    edge). No forecasting — undetermined prints are skipped. PAPER-FORCED, its own
    graduation cell. Default OFF; flip enabled:true to start the measurement.
    """

    enabled: bool = False
    paper: bool = True
    stake_usd: float = 10.0
    # Required gap between the locked outcome (0/1) and the market price — the
    # un-converged distance the lag leaves on the table.
    min_edge: float = 0.05
    # Liquidity floor is a DUST guard, not a quality filter. NBER w34702 (2026):
    # liquid macro contracts reprice intraday and are well-calibrated — the
    # settlement LAG survives mostly in illiquid, low-volume TAIL/bin contracts
    # with stale prices. This pillar holds to resolution (known-outcome
    # convergence, no exit), so illiquidity does NOT block the exit the way it
    # would for a round-trip strategy. So the floor is set low — just high enough
    # to exclude untradeable dust — to ADMIT the tail bins where the edge lives,
    # not to demand the liquid headline contracts where it doesn't. (Live
    # execution against a single stale quote is the key validation risk.)
    min_liquidity: float = 100.0
    # The LLM only extracts the predicate; both gates default conservative.
    min_extract_confidence: float = 0.8
    verify_min_confidence: float = 0.8
    max_entries_per_cycle: int = 5
    history_n: int = 60
    interval_seconds: int = 1800


class IntradayDriftConfig(BaseModel):
    """Intraday-drift measurement spike (monitoring/intraday_drift.py). NO
    trading — reuses the signals table + snapshots mids to test whether price
    drifts toward the LLM estimate intraday (the under-reaction thesis) before
    any intraday strategy is built. Cheap; disabled by default."""

    enabled: bool = False
    strategies: list[str] = Field(default_factory=lambda: ["news_speed", "llm"])
    interval_seconds: int = 300
    register_lookback_min: int = 15
    time_box_hours: float = 8.0
    max_tracks_per_cycle: int = 60
    fee_threshold: float = 0.02
    report_min_signals: int = 20


class HydroWatchConfig(BaseModel):
    """Hydrology-market watcher (monitoring/hydro_market_watch.py). Alert-only:
    no liquid water markets exist today, so this just flags the first time one
    appears on a venue, so the compHydro data moat can be deployed. Cheap."""

    enabled: bool = False
    scan_limit: int = 500
    min_liquidity: float = 100.0
    interval_seconds: int = 21600  # every 6h — new markets aren't urgent


class WeatherTempConfig(BaseModel):
    """Open-Meteo ensemble pricing of Polymarket city-temperature bins
    (strategy/weather_temp.py). Measurement spike: PAPER-FORCED and disabled
    by default. Every bin is logged (model vs market) regardless of trading;
    the edge must clear the Polymarket taker fee on top of min_edge, and
    max_divergence skips implausibly large gaps (likely a bin-rounding or
    station-match artifact, not edge) until realized highs validate the model.
    """

    enabled: bool = False
    paper: bool = True
    stake_usd: float = 10.0
    min_edge: float = 0.10
    max_divergence: float = 0.40
    max_open: int = 40
    max_entries_per_cycle: int = 8
    scan_limit: int = 500
    interval_seconds: int = 3600


class EntailmentArbConfig(BaseModel):
    """Entailment arbitrage (strategy/entailment_arb.py).

    Trades P(implier) > P(implied) violations between logically linked
    markets. Ladder pairs (numeric threshold / Top-N families) are
    deterministic; fuzzy 'conditional' pairs are LLM-verified
    adversarially and cached. PAPER-FORCED by default.
    """

    enabled: bool = False
    paper: bool = True
    min_gap: float = 0.04
    stake_usd: float = 10.0
    max_pairs_per_cycle: int = 3
    scan_limit: int = 300
    min_liquidity: float = 1000.0
    max_spread_pct: float = 5.0
    min_hours_to_resolution: float = 2.0
    llm_enabled: bool = True
    llm_min_confidence: float = 0.9
    interval_seconds: int = 900
    # Kalshi "Above X" economic-indicator ladders (model-free monotonicity arb).
    # Fetched per-series (not via the generic scan — econ bins are niche and a
    # top-N scan misses them). Unlike Polymarket (0% maker), Kalshi charges a
    # taker fee per leg, so a violation must clear BOTH legs' fees + a buffer to
    # be real — kalshi_min_gap is computed from the fee model at runtime, not
    # this flat min_gap. Paper-forced by the graduation ladder like every cell.
    kalshi_ladders_enabled: bool = True
    kalshi_series: list[str] = Field(default_factory=lambda: [
        "KXCPIYOY", "KXGDP", "KXU3", "KXPAYROLLS", "KXPCEYOY",
    ])
    kalshi_min_liquidity: float = 50.0
    kalshi_gap_buffer: float = 0.01


class CrossVenueArbConfig(BaseModel):
    """Cross-venue semantic-equivalence arbitrage (strategy/cross_venue_arb.py).

    Trades price gaps between Polymarket and Kalshi markets that are logically
    equivalent but worded differently. Candidate pairs are pre-filtered by word
    overlap, then verified ADVERSARIALLY by an LLM (default: not equivalent) at a
    high confidence floor — a false match is a paired loss, not a free arb. The
    gap must clear both legs' taker fees + a buffer. PAPER-FORCED by default and
    NOT graduation-exempt (resolution-mismatch risk = a real directional loss).
    """

    enabled: bool = False
    paper: bool = True
    min_word_overlap: float = 0.5
    gap_buffer: float = 0.02
    stake_usd: float = 10.0
    max_pairs_per_cycle: int = 2
    max_llm_calls_per_cycle: int = 8
    scan_limit: int = 200
    min_liquidity: float = 1000.0
    kalshi_min_liquidity: float = 50.0
    max_spread_pct: float = 5.0
    min_hours_to_resolution: float = 6.0
    llm_min_confidence: float = 0.9
    interval_seconds: int = 1200


class OddLotTenderConfig(BaseModel):
    """Odd-lot tender harvester (strategy/oddlot_tender.py).

    Scans EDGAR for issuer tender offers with odd-lot priority; LLM reads
    the fine print adversarially; alerts the operator and (when IBKR is
    enabled) buys 99 shares PAPER-FORCED. Tendering itself is manual.
    Detection runs even with ibkr.enabled=false.
    """

    enabled: bool = False
    paper: bool = True
    lookback_days: int = 7
    max_filings_per_cycle: int = 5
    llm_min_confidence: float = 0.8
    min_premium_pct: float = 2.0
    max_position_usd: float = 2500.0
    interval_seconds: int = 21600  # 6h — filings are daily-cadence events


class ResolutionLensConfig(BaseModel):
    """Resolution-language lens (strategy/resolution_lens.py).

    Trades headline-vs-fine-print gaps found by an adversarial criteria
    read. Lexical triggers + real-book guards select candidates; the LLM
    lens is the precision stage (verdicts cached forever — criteria are
    static). PAPER-FORCED by default.
    """

    enabled: bool = False
    paper: bool = True
    min_gap_score: float = 0.4
    high_conf_gap_score: float = 0.7
    min_edge: float = 0.08
    # Favorite-discipline floor on BUY entries (0 = off). A 2026-06-24 edge
    # audit of the lens×weather cell found every loss was a BUY of a narrow
    # temperature bin entered in the near-coin-flip band (<~0.65 YES), where
    # favorite-longshot variance dominates and the LLM's named mechanism is
    # post-hoc — the position is identical regardless. Requiring the YES side
    # we buy to already be a market favorite (>= this) cut the cell from 83%
    # to 100% win in-sample. Gates BUYs ONLY: the lens's other documented
    # edge is SELLing overpriced-YES longshots (permanence/announce bars),
    # which by construction sit below this floor and must stay untouched.
    min_entry_price: float = 0.0
    stake_usd: float = 10.0
    max_entries_per_cycle: int = 3
    max_llm_calls_per_cycle: int = 5
    scan_limit: int = 300
    min_liquidity: float = 1000.0
    max_spread_pct: float = 5.0
    min_description_chars: int = 80
    min_hours_to_resolution: float = 12.0
    max_days_to_resolution: float = 90.0
    # Phase 1: read the FULL criteria, not the first 800 chars (the decisive
    # qualifier usually lives at the end). Cap to bound LLM cost; head+tail kept.
    criteria_char_cap: int = 4500
    # Phase 2: adversarially verify the named mechanism (a 2nd skeptical LLM
    # pass that defaults to refuted) before trading — kills hallucinated
    # fine-print. Only trade when confirmed at >= verify_min_confidence.
    verify_enabled: bool = True
    verify_min_confidence: float = 0.7
    # Phase 3: evidence-grounded comprehension. The lens reads CURRENT evidence
    # (the same aggregator pipeline the ensemble uses) AGAINST the strict
    # criteria — "do the literal criteria resolve YES given this evidence and the
    # deadline?" — not a re-forecast. This fuses the two things that individually
    # work (LLM comprehension + live evidence) on the task where the LLM has edge
    # (reading a rule against facts), instead of forecasting (where it loses).
    # Runs ONLY on candidates that already cleared gap_score + adversarial verify,
    # so evidence/LLM spend lands only on real fine-print gaps. The grounded
    # estimate is NOT permanently cached (evidence is fresh): re-grounds when
    # older than phase3_ttl_hours. Falls back to the criteria-strict fair (already
    # Phase 1+2 validated) if evidence/grounding is unavailable — never blocks
    # accrual on a fetch miss.
    phase3_grounding_enabled: bool = True
    phase3_ttl_hours: float = 12.0
    phase3_min_confidence: float = 0.5
    phase3_max_evidence: int = 6
    interval_seconds: int = 1800
    # Paper-phase eligibility: while the cell is hard paper-forced (paper=True)
    # it can never reach the venue, so the live-trading guards (hold horizon,
    # liquidity for fillability) don't apply — and the strict live values starve
    # accrual (only ~92 of 3700 lexical candidates pass; most rejected as
    # <12h-to-resolve or <$1k liquidity). Loosen them in paper to build the
    # graduation record: short-dated markets also SETTLE fast, so paper events
    # accrue quickly. Auto-reverts to the strict values above if paper is
    # flipped to graduate the cell to live.
    paper_min_hours_to_resolution: float = 1.0
    paper_min_liquidity: float = 250.0
    # Kalshi measurement spike (default OFF). When true, a SECOND lens instance
    # scans Kalshi — paper-forced, attributed to 'resolution_lens_kalshi' so it
    # gets its own graduation cells and can't dilute the proven Poly lens. Tests
    # the hypothesis that Kalshi's CFTC-legalistic resolution criteria carry
    # fine-print mispricing. Kalshi's book is thinner, so it gets its own floor.
    kalshi_enabled: bool = False
    # Kalshi's bulk liquidity field underreports badly (top-of-book only, often
    # 0 even on active econ ladders) — 300 starved the spike to zero verdicts
    # (2026-07-03 funnel: 14/218 candidates cleared it, none also in-window).
    # 50 matches the cross_venue floor; safe for a paper spike that holds to
    # resolution, where illiquidity can't block an exit.
    kalshi_min_liquidity: float = 50.0


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
    # Restrict unproven SPRAY (2026-06-29 winner reverse-engineering: winners are
    # FEW + LARGE + SELECTIVE; the loser profile is high-frequency tiny-size spray
    # across hundreds of low-conviction cells). When the open PAPER/exploratory
    # book is already this wide, "unproven" cells stop opening NEW positions
    # (size x0) so exploration concentrates rather than sprays. A RESTRICTION
    # only — it never upsizes, never touches proven/probation/exempt cells, and
    # never affects exits. 0 disables.
    max_unproven_positions: int = 100


class InformationGraduationConfig(BaseModel):
    """Evidence influence earned from paired source-ablation trials."""

    min_resolved: int = 30
    min_paired: int = 50
    min_success_rate: float = 0.98
    probation_multiplier: float = 0.25


class BrokerConfig(BaseModel):
    sync_interval_seconds: int = 60
    use_limit_orders: bool = True
    limit_spread_threshold: float = 0.03  # Use limits when spread >= 3 cents
    limit_edge_threshold: float = 20.0    # Use market orders when edge > 20%
    limit_price_improvement_ticks: int = 1  # Improve on BBO by 1 tick
    max_slippage_bps: int = 100


class KalshiConfig(BaseModel):
    enabled: bool = False
    api_key: str = Field(default="", repr=False, exclude=True)
    private_key_path: str = Field(default="", repr=False, exclude=True)
    environment: str = "demo"  # "demo" | "prod"


class OpenAIETFModel(BaseModel):
    """One isolated intelligence-comparison arm for the ETF paper mandate."""

    alias: str
    model: str
    effort: Literal["low", "medium", "high"] = "medium"
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0

    @model_validator(mode="after")
    def validate_identity(self):
        alias = self.alias.strip().lower()
        if not alias or not alias.replace("_", "").isalnum():
            raise ValueError("ETF model alias must contain only letters, numbers, and underscores")
        if not self.model.strip():
            raise ValueError("ETF model name must not be empty")
        if self.input_cost_per_million < 0 or self.output_cost_per_million < 0:
            raise ValueError("ETF model token prices must be non-negative")
        self.alias = alias
        self.model = self.model.strip()
        return self


class IBKRMultiAssetBookConfig(BaseModel):
    """Risk envelope for one isolated, locally simulated IBKR book."""

    enabled: bool = True
    budget_usd: float = 5_000.0
    max_positions: int = 4
    max_position_pct: float = 15.0
    max_deployment_pct: float = 50.0
    daily_loss_limit_usd: float = 100.0
    stop_loss_pct: float = 5.0
    take_profit_pct: float = 10.0
    max_spread_bps: float = 40.0
    risk_per_position_pct: float = 0.25
    max_asset_class_risk_pct: float = 0.50
    stop_vol_multiple: float = 2.0
    min_stop_pct: float = 0.50
    slippage_bps: float = 2.0

    @model_validator(mode="after")
    def validate_risk(self):
        if self.budget_usd <= 0 or self.max_positions <= 0:
            raise ValueError("IBKR paper book budget and position count must be positive")
        if not 0 < self.max_position_pct <= self.max_deployment_pct <= 100:
            raise ValueError("IBKR paper book deployment percentages are inconsistent")
        if self.daily_loss_limit_usd <= 0 or self.stop_loss_pct <= 0:
            raise ValueError("IBKR paper book loss limits must be positive")
        if self.take_profit_pct <= 0 or self.max_spread_bps <= 0:
            raise ValueError("IBKR paper book exit/spread limits must be positive")
        if not 0 < self.risk_per_position_pct <= self.max_asset_class_risk_pct <= 5:
            raise ValueError("IBKR paper risk percentages are inconsistent")
        if self.stop_vol_multiple <= 0 or self.min_stop_pct <= 0 or self.slippage_bps < 0:
            raise ValueError("IBKR paper volatility/execution inputs are invalid")
        return self


def _canonical_ibkr_etf_symbols() -> list[str]:
    """Keep the legacy ETF experiment on the typed multi-asset manifest."""
    from auramaur.exchange.ibkr_instruments import GLOBAL_ETFS
    return [spec.symbol for spec in GLOBAL_ETFS]


class IBKRConfig(BaseModel):
    enabled: bool = False
    # `enabled` is the master switch (connect to IBKR at all). The two books
    # beneath it are gated independently: options_enabled (the option-chain
    # scanner) and the odd-lot tender pillar (the stocks book). Keep options
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

    # --- Structurally paper-only broad-market ETF experiment ---
    etf_paper_enabled: bool = False
    # Quote-session login is independent of execution: "live" permits the
    # structurally simulated ETF pillar to read TWS port 7496, still readonly.
    etf_quote_port: int = 7497
    multiasset_client_id: int = 3
    multiasset_preflight_client_id: int = 97
    multiasset_paper_enabled: bool = False
    multiasset_cycle_seconds: int = 900
    multiasset_refreshes_per_cycle: int = 12
    multiasset_max_quote_age_seconds: int = 120
    multiasset_contract_cache_seconds: int = 21_600
    multiasset_preflight_concurrency: int = 2
    multiasset_preflight_pacing_retries: int = 2
    multiasset_preflight_retry_seconds: float = 2.0
    # Require a current broker-qualified identity before opening new risk.
    # Kept false in model defaults so isolated test/library users can opt in;
    # tracked deployment defaults enable it.
    multiasset_registry_required: bool = False
    multiasset_disabled_instruments: list[str] = []
    multiasset_min_momentum_pct: float = 1.0
    multiasset_exit_momentum_pct: float = -0.5
    multiasset_min_normalized_momentum: float = 0.25
    multiasset_exit_normalized_momentum: float = -0.10
    multiasset_books: dict[str, IBKRMultiAssetBookConfig] = Field(default_factory=lambda: {
        name: IBKRMultiAssetBookConfig() for name in (
            "global_etf", "fx", "futures", "international_equity", "options", "bonds")
    })
    etf_symbols: list[str] = Field(default_factory=_canonical_ibkr_etf_symbols)
    etf_paper_budget_usd: float = 5_000.0
    etf_max_entry_usd: float = 250.0
    etf_max_deployment_pct: float = 50.0
    etf_max_asset_class_pct: float = 30.0
    etf_max_positions: int = 4
    etf_daily_loss_limit_usd: float = 100.0
    etf_max_signal_refreshes_per_cycle: int = 4
    etf_fee_per_order_usd: float = 1.00
    etf_max_spread_bps: float = 20.0
    etf_min_prob: float = 0.62
    etf_exit_prob: float = 0.47
    etf_min_confidence: str = "MEDIUM"
    etf_signal_horizon_days: int = 5
    etf_signal_refresh_hours: float = 6.0
    etf_cycle_seconds: int = 900
    etf_stop_loss_pct: float = 5.0
    etf_take_profit_pct: float = 8.0
    etf_trailing_stop_pct: float = 3.0
    etf_reentry_cooldown_hours: float = 24.0
    etf_risk_per_position_pct: float = 0.25
    etf_stop_vol_multiple: float = 2.0
    etf_min_stop_pct: float = 1.0
    etf_slippage_bps: float = 2.0
    etf_max_portfolio_risk_pct: float = 1.0
    etf_models: list[OpenAIETFModel] = [
        OpenAIETFModel(alias="luna", model="gpt-5.6-luna", effort="low"),
        OpenAIETFModel(alias="terra", model="gpt-5.6-terra", effort="medium"),
        OpenAIETFModel(alias="sol", model="gpt-5.6-sol", effort="high"),
    ]
    etf_openai_timeout_seconds: int = 120
    etf_openai_daily_call_limit: int = 100

    @model_validator(mode="after")
    def validate_etf_experiment(self):
        if not 1 <= self.etf_quote_port <= 65535:
            raise ValueError("IBKR ETF quote port must be a valid TCP port")
        expected_books = {"global_etf", "fx", "futures", "international_equity",
                          "options", "bonds"}
        if set(self.multiasset_books) != expected_books:
            raise ValueError("IBKR multi-asset config must define exactly six books")
        client_ids = {self.client_id, self.equity_client_id,
                      self.multiasset_client_id, self.multiasset_preflight_client_id}
        if len(client_ids) != 4:
            raise ValueError("IBKR API client ids must be unique")
        if self.multiasset_cycle_seconds <= 0 or self.multiasset_refreshes_per_cycle <= 0:
            raise ValueError("IBKR multi-asset cycle and refresh limits must be positive")
        if (self.multiasset_max_quote_age_seconds <= 0
                or self.multiasset_contract_cache_seconds <= 0):
            raise ValueError("IBKR multi-asset quote/cache limits must be positive")
        if (self.multiasset_preflight_concurrency <= 0
                or self.multiasset_preflight_pacing_retries < 0
                or self.multiasset_preflight_retry_seconds < 0):
            raise ValueError("IBKR multi-asset preflight pacing limits are invalid")
        if len(self.multiasset_disabled_instruments) != len(
                set(self.multiasset_disabled_instruments)):
            raise ValueError("IBKR disabled instrument keys must be unique")
        symbols = [symbol.strip().upper() for symbol in self.etf_symbols]
        if not symbols:
            raise ValueError("IBKR ETF experiment requires at least one symbol")
        if any(not symbol.isalnum() for symbol in symbols):
            raise ValueError("IBKR ETF symbols must be non-empty alphanumeric tickers")
        if len(symbols) != len(set(symbols)):
            raise ValueError("IBKR ETF symbols must be unique")
        self.etf_symbols = symbols

        aliases = [arm.alias for arm in self.etf_models]
        if not aliases:
            raise ValueError("IBKR ETF experiment requires at least one model arm")
        if len(aliases) != len(set(aliases)):
            raise ValueError("IBKR ETF model aliases must be unique")
        if self.etf_paper_budget_usd <= 0:
            raise ValueError("IBKR ETF paper budget must be positive")
        if not 0 < self.etf_max_entry_usd <= self.etf_paper_budget_usd:
            raise ValueError("IBKR ETF entry cap must be positive and no larger than its budget")
        if not 0 < self.etf_max_asset_class_pct <= self.etf_max_deployment_pct <= 100:
            raise ValueError("IBKR ETF asset-class/deployment percentages are inconsistent")
        if self.etf_max_positions <= 0 or self.etf_max_signal_refreshes_per_cycle <= 0:
            raise ValueError("IBKR ETF position and refresh limits must be positive")
        if self.etf_daily_loss_limit_usd <= 0 or self.etf_fee_per_order_usd < 0:
            raise ValueError("IBKR ETF loss limit must be positive and fees non-negative")
        if not 0 < self.etf_risk_per_position_pct <= self.etf_max_portfolio_risk_pct <= 5:
            raise ValueError("IBKR ETF risk percentages are inconsistent")
        if self.etf_stop_vol_multiple <= 0 or self.etf_min_stop_pct <= 0 or self.etf_slippage_bps < 0:
            raise ValueError("IBKR ETF volatility/execution inputs are invalid")
        if not 0 <= self.etf_exit_prob < self.etf_min_prob <= 1:
            raise ValueError("IBKR ETF probability thresholds must satisfy exit < entry")
        if self.etf_openai_daily_call_limit < len(self.etf_models):
            raise ValueError("IBKR ETF daily OpenAI limit must allow every model arm one call")
        if self.etf_paper_enabled and any(
            arm.input_cost_per_million <= 0 or arm.output_cost_per_million <= 0
            for arm in self.etf_models
        ):
            raise ValueError("Enabled IBKR ETF arms require explicit nonzero token prices")
        if self.etf_openai_timeout_seconds <= 0 or self.etf_cycle_seconds <= 0:
            raise ValueError("IBKR ETF timeout and cycle interval must be positive")
        return self

    # --- Directional equity speculation (gated; no validated edge) ---
    # Mirrors the Kraken directional pillar. Uses its OWN socket connection
    # (equity_client_id) so it doesn't clash with the options client.
    # Directional equity momentum book REMOVED 2026-06-09 (pre-failed: same
    # strategy shape went 0W/20L on Kraken, backtested negative in every
    # variant). The equity client + per-order cap remain for the odd-lot
    # tender pillar.
    equity_max_order_usd: float = 2500.0           # hard per-order ceiling (99 sh x ~$25)
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
    api_key: str = Field(default="", repr=False, exclude=True)
    api_secret: str = Field(default="", repr=False, exclude=True)
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


class CoinbaseConfig(BaseModel):
    """Read-only Coinbase shadow book; live execution is intentionally absent."""

    paper_enabled: bool = False
    paper_fee_pct: float = 0.60


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
    # TTL for the LLM batch-pairing cache keyed by the candidate id sets. The
    # scanner re-matches near-identical top-N lists every cycle; matching is a
    # function of the QUESTIONS, not prices, so answers are stable for hours.
    llm_match_cache_seconds: int = 21600
    cross_exchange_auto_execute: bool = True
    negrisk_auto_execute: bool = False
    # Flat per-exchange TAKER fee coefficients (fee = rate * P*(1-P)).
    # polymarket=0.0 here is only the MAKER rate; Polymarket TAKERS pay a
    # per-category rate resolved by signals.taker_fee_rate() (POLYMARKET_TAKER_FEES),
    # NOT this entry. Crossing/taker code paths must go through taker_fee_rate.
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


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json_format: bool = True
    file: str = "auramaur.log"
    # Size-based rotation of the structlog file. Tighter than the old hardcoded
    # 50MB×5 (≈300MB of mostly-noise): 10MB×3 keeps the on-disk log current and
    # bounded at ~40MB so a noisy source can't bury the recent signal in a giant
    # file. Tune up if you need deeper history.
    rotate_max_mb: int = 10
    rotate_backups: int = 3


class Settings(BaseSettings):
    # API Keys
    anthropic_api_key_primary: str = Field(default="", repr=False, exclude=True)
    anthropic_api_key_secondary: str = Field(default="", repr=False, exclude=True)
    openai_api_key: str = Field(default="", repr=False, exclude=True)
    polygon_private_key: str = Field(default="", repr=False, exclude=True)
    polymarket_api_key: str = Field(default="", repr=False, exclude=True)
    polymarket_api_secret: str = Field(default="", repr=False, exclude=True)
    polymarket_passphrase: str = Field(default="", repr=False, exclude=True)
    polymarket_proxy_address: str = ""
    newsapi_key: str = Field(default="", repr=False, exclude=True)
    reddit_client_id: str = Field(default="", repr=False, exclude=True)
    reddit_client_secret: str = Field(default="", repr=False, exclude=True)
    reddit_user_agent: str = "auramaur/0.1"
    twitter_bearer_token: str = Field(default="", repr=False, exclude=True)
    fred_api_key: str = Field(default="", repr=False, exclude=True)
    bls_api_key: str = Field(default="", repr=False, exclude=True)
    bea_api_key: str = Field(default="", repr=False, exclude=True)
    congress_api_key: str = Field(default="", repr=False, exclude=True)
    eia_api_key: str = Field(default="", repr=False, exclude=True)
    telegram_bot_token: str = Field(default="", repr=False, exclude=True)
    telegram_chat_id: str = ""
    discord_webhook_url: str = Field(default="", repr=False, exclude=True)

    # Kalshi
    kalshi_api_key: str = Field(default="", repr=False, exclude=True)
    kalshi_private_key_path: str = Field(default="", repr=False, exclude=True)

    # Crypto.com
    cryptodotcom_api_key: str = Field(default="", repr=False, exclude=True)
    cryptodotcom_api_secret: str = Field(default="", repr=False, exclude=True)

    # Kraken (spot). Used for read-only wallet/balance checks today; no trading
    # adapter is wired yet. Key needs only the "Query Funds" permission —
    # leave "Withdraw Funds" OFF.
    kraken_api_key: str = Field(default="", repr=False, exclude=True)
    kraken_api_secret: str = Field(default="", repr=False, exclude=True)

    # Google Gemini — LLM fallback for off-hours / when Claude budget is low.
    gemini_api_key: str = Field(default="", repr=False, exclude=True)

    # Hugging Face Hub token — used by the sentence-transformers evidence
    # embedder (nlp/relevance.py). Anonymous downloads work but are
    # rate-limited and warn; a free read token lifts both. huggingface_hub
    # reads HF_TOKEN from the process environment, not from our Settings, so
    # model_post_init exports it (pydantic-settings parses .env into fields
    # without touching os.environ).
    hf_token: str = Field(default="", repr=False, exclude=True)

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
    coinbase: CoinbaseConfig = Field(default_factory=lambda: CoinbaseConfig(**_DEFAULTS.get("coinbase", {})))
    transfers: TransfersConfig = Field(default_factory=lambda: TransfersConfig(**_DEFAULTS.get("transfers", {})))
    ensemble: EnsembleConfig = Field(default_factory=lambda: EnsembleConfig(**_DEFAULTS.get("ensemble", {})))
    llm_ensemble: LLMEnsembleConfig = Field(default_factory=lambda: LLMEnsembleConfig(**_DEFAULTS.get("llm_ensemble", {})))
    gemini: GeminiConfig = Field(default_factory=lambda: GeminiConfig(**_DEFAULTS.get("gemini", {})))
    momentum_coupling: MomentumCouplingConfig = Field(default_factory=lambda: MomentumCouplingConfig(**_DEFAULTS.get("momentum_coupling", {})))
    market_maker: MarketMakerConfig = Field(default_factory=lambda: MarketMakerConfig(**_DEFAULTS.get("market_maker", {})))
    technical: TechnicalConfig = Field(default_factory=lambda: TechnicalConfig(**_DEFAULTS.get("technical", {})))
    bias_harvest: BiasHarvestConfig = Field(default_factory=lambda: BiasHarvestConfig(**_DEFAULTS.get("bias_harvest", {})))
    graduation: GraduationConfig = Field(default_factory=lambda: GraduationConfig(**_DEFAULTS.get("graduation", {})))
    information_graduation: InformationGraduationConfig = Field(
        default_factory=lambda: InformationGraduationConfig(
            **_DEFAULTS.get("information_graduation", {})))
    entailment_arb: EntailmentArbConfig = Field(default_factory=lambda: EntailmentArbConfig(**_DEFAULTS.get("entailment_arb", {})))
    cross_venue_arb: CrossVenueArbConfig = Field(default_factory=lambda: CrossVenueArbConfig(**_DEFAULTS.get("cross_venue_arb", {})))
    econ_indicator: EconIndicatorConfig = Field(default_factory=lambda: EconIndicatorConfig(**_DEFAULTS.get("econ_indicator", {})))
    long_horizon: LongHorizonConfig = Field(default_factory=lambda: LongHorizonConfig(**_DEFAULTS.get("long_horizon", {})))
    agent_trader: AgentTraderConfig = Field(default_factory=lambda: AgentTraderConfig(**_DEFAULTS.get("agent_trader", {})))
    term_structure: TermStructureConfig = Field(default_factory=lambda: TermStructureConfig(**_DEFAULTS.get("term_structure", {})))
    vol_anchor: VolAnchorConfig = Field(default_factory=lambda: VolAnchorConfig(**_DEFAULTS.get("vol_anchor", {})))
    informed_flow: InformedFlowConfig = Field(default_factory=lambda: InformedFlowConfig(**_DEFAULTS.get("informed_flow", {})))
    settlement_arb: SettlementArbConfig = Field(default_factory=lambda: SettlementArbConfig(**_DEFAULTS.get("settlement_arb", {})))
    weather_temp: WeatherTempConfig = Field(default_factory=lambda: WeatherTempConfig(**_DEFAULTS.get("weather_temp", {})))
    hydro_watch: HydroWatchConfig = Field(default_factory=lambda: HydroWatchConfig(**_DEFAULTS.get("hydro_watch", {})))
    intraday_drift: IntradayDriftConfig = Field(default_factory=lambda: IntradayDriftConfig(**_DEFAULTS.get("intraday_drift", {})))
    resolution_lens: ResolutionLensConfig = Field(default_factory=lambda: ResolutionLensConfig(**_DEFAULTS.get("resolution_lens", {})))
    oddlot_tender: OddLotTenderConfig = Field(default_factory=lambda: OddLotTenderConfig(**_DEFAULTS.get("oddlot_tender", {})))
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
        # Portable deployments can override nested settings without editing a
        # tracked YAML file, e.g. IBKR__HOST=ibgateway.
        "env_nested_delimiter": "__",
        # Ignore env vars we don't declare. The process shares its environment
        # with libraries that read their own tokens directly (e.g. HF_TOKEN /
        # hf_token for huggingface_hub via sentence-transformers), and an
        # unrelated stray var shouldn't crash Settings on startup.
        "extra": "ignore",
    }

    def model_post_init(self, __context) -> None:
        """Export pass-through tokens to the process environment.

        huggingface_hub (via sentence-transformers) reads HF_TOKEN from
        os.environ — pydantic-settings parses .env into fields without
        touching the environment, so a token set only in .env would never
        reach it. setdefault: a token already exported in the shell wins.
        """
        if self.hf_token and not os.environ.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = self.hf_token

    @property
    def kill_switch_active(self) -> bool:
        # Delegate to the shared root-aware helper so this and every bare call
        # site agree on one definition (repo root OR CWD) and can't drift.
        from auramaur.killswitch import kill_switch_present
        return kill_switch_present()

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
