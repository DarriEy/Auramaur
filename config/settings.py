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
    limit_order_ttl_seconds: int = 300
    spread_capture_min_bps: int = 50
    stop_loss_pct: float = 30.0
    profit_target_pct: float = 50.0
    edge_erosion_min_pct: float = 2.0
    time_decay_hours: float = 12.0


class RiskConfig(BaseModel):
    max_drawdown_pct: float = 15.0
    max_stake_per_market: float = 25.0
    daily_loss_limit: float = 200.0
    max_open_positions: int = 30
    min_edge_pct: float = 5.0
    min_liquidity: float = 1000.0
    max_spread_pct: float = 5.0
    confidence_floor: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"
    implied_prob_min: float = 0.05
    implied_prob_max: float = 0.95
    category_exposure_cap_pct: float = 30.0
    time_to_resolution_min_hours: int = 24
    second_opinion_divergence_max: float = 0.15


class KellyConfig(BaseModel):
    fraction: float = 0.25


class IntervalsConfig(BaseModel):
    market_scan_seconds: int = 300
    news_poll_seconds: int = 120
    analysis_seconds: int = 180
    portfolio_check_seconds: int = 60
    dashboard_refresh_seconds: int = 5


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
    min_spread_bps: int = 200  # minimum spread to participate (200 bps = 2%)
    quote_size: float = 10.0  # tokens per side
    max_inventory: float = 50.0  # max directional exposure per market
    max_markets: int = 5  # max simultaneous MM markets
    refresh_seconds: int = 30  # re-quote frequency


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
    host: str = "127.0.0.1"
    paper_port: int = 7497
    live_port: int = 7496
    client_id: int = 1
    environment: str = "paper"  # "paper" | "live"
    watchlist: list[str] = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "GOOGL"]
    max_contracts_per_symbol: int = 10


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


class AnalysisConfig(BaseModel):
    """Controls which analysis backend is used."""

    mode: Literal["pipeline", "strategic", "agent"] = "strategic"


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

    # Safety
    auramaur_live: bool = False

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
    ensemble: EnsembleConfig = Field(default_factory=lambda: EnsembleConfig(**_DEFAULTS.get("ensemble", {})))
    llm_ensemble: LLMEnsembleConfig = Field(default_factory=lambda: LLMEnsembleConfig(**_DEFAULTS.get("llm_ensemble", {})))
    market_maker: MarketMakerConfig = Field(default_factory=lambda: MarketMakerConfig(**_DEFAULTS.get("market_maker", {})))
    analysis: AnalysisConfig = Field(default_factory=lambda: AnalysisConfig(**_DEFAULTS.get("analysis", {})))
    logging: LoggingConfig = Field(default_factory=lambda: LoggingConfig(**_DEFAULTS.get("logging", {})))

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def kill_switch_active(self) -> bool:
        return Path("KILL_SWITCH").exists()

    @property
    def is_live(self) -> bool:
        """All three gates must be true for live trading."""
        return self.auramaur_live and self.execution.live and not self.kill_switch_active
