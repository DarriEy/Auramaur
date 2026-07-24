"""Consumer-level data contracts and delivery telemetry.

Provider uptime is not the same thing as a strategy receiving usable data.
This module names the datasets each strategy consumes and records deliveries
with event time, observation time, coverage, fallbacks, and snapshot identity.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Literal

import structlog
from pydantic import BaseModel, Field

log = structlog.get_logger()

_HEALTHY_THROTTLE_SECONDS = {"order_book": 15.0}
_last_healthy_recorded: dict[tuple[int, str, str], float] = {}

DeliveryStatus = Literal[
    "ok", "empty", "stale", "partial", "timeout", "error", "unavailable"
]


class DataRequirement(BaseModel):
    component: str
    max_age_seconds: float
    min_items: int = 1
    required_fields: tuple[str, ...] = ()
    fail_closed: bool = True


class DataDelivery(BaseModel):
    strategy: str
    component: str
    status: DeliveryStatus
    provider: str = ""
    market_id: str = ""
    snapshot_id: str = ""
    observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_at: datetime | None = None
    latency_ms: int = 0
    item_count: int = 0
    required_fields: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    fallback_used: str = ""
    detail: dict = Field(default_factory=dict)

    @property
    def age_seconds(self) -> float | None:
        if self.source_at is None:
            return None
        source = self.source_at
        if source.tzinfo is None:
            source = source.replace(tzinfo=timezone.utc)
        observed = self.observed_at
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        return max(0.0, (observed - source).total_seconds())


# Explicit defaults. Strategies may override these with a ``data_requirements``
# attribute; keeping the registry here makes readiness useful before every
# pillar has bespoke telemetry.
_EVIDENCE = DataRequirement(component="evidence", max_age_seconds=48 * 3600)
_MARKETS = DataRequirement(
    component="market_snapshot", max_age_seconds=300,
    required_fields=("outcome_yes_price", "outcome_no_price"),
)
_BOOK = DataRequirement(
    component="order_book", max_age_seconds=15,
    required_fields=("best_bid", "best_ask"),
)
_CYCLE = DataRequirement(component="strategy_cycle", max_age_seconds=900, min_items=0)

REQUIREMENTS: dict[str, tuple[DataRequirement, ...]] = {
    "strategic": (_MARKETS, _EVIDENCE),
    "llm": (_MARKETS, _EVIDENCE),
    "technical": (_MARKETS, DataRequirement(component="price_history", max_age_seconds=300)),
    "market_maker": (_BOOK,),
    "arbitrage": (_MARKETS,),
    "bias_harvest": (_MARKETS, _BOOK),
    "platform_consensus": (_MARKETS, DataRequirement(component="platform_forecasts", max_age_seconds=900)),
    "cross_venue_arb": (DataRequirement(component="cross_venue_snapshot", max_age_seconds=30),),
    "informed_flow": (DataRequirement(component="venue_trades", max_age_seconds=120),),
    "econ_indicator": (DataRequirement(component="fred_observations", max_age_seconds=3600), _MARKETS),
    "settlement_arb": (DataRequirement(component="fred_observations", max_age_seconds=3600), _MARKETS),
    "weather_temp": (DataRequirement(component="weather_ensemble", max_age_seconds=3600), _MARKETS),
    "vol_anchor": (DataRequirement(component="spot_volatility", max_age_seconds=3600), _MARKETS),
    "oddlot_tender": (DataRequirement(component="edgar_filings", max_age_seconds=21600),),
    "resolution_lens": (_MARKETS, _EVIDENCE),
    "resolution_lens_kalshi": (_MARKETS, _EVIDENCE),
    "term_structure": (_MARKETS, _EVIDENCE),
    "agent_trader": (_MARKETS, _EVIDENCE),
    "long_horizon": (_MARKETS,),
    "entailment_arb": (_MARKETS,),
    "ibkr_etf": (DataRequirement(component="equity_quote", max_age_seconds=120), _EVIDENCE),
    "ibkr_multiasset": (DataRequirement(component="multiasset_quote", max_age_seconds=120),),
    "kraken_treasury": (DataRequirement(component="crypto_quote", max_age_seconds=120), _EVIDENCE),
}


def requirements_for(strategy: str) -> tuple[DataRequirement, ...]:
    return REQUIREMENTS.get(strategy, (_CYCLE,))


def snapshot_id(*parts: object) -> str:
    """Stable identity for observations intended to be contemporaneous."""
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()[:24]


async def record_delivery(db, delivery: DataDelivery) -> None:
    """Persist one delivery without allowing monitoring to kill a strategy."""
    try:
        throttle = _HEALTHY_THROTTLE_SECONDS.get(delivery.component, 0.0)
        key = (id(db), delivery.strategy, delivery.component)
        now_mono = time.monotonic()
        if (delivery.status == "ok" and throttle > 0
                and now_mono - _last_healthy_recorded.get(key, -throttle) < throttle):
            return
        observed = delivery.observed_at.astimezone(timezone.utc)
        source = delivery.source_at
        if source is not None:
            source = (source.replace(tzinfo=timezone.utc) if source.tzinfo is None
                      else source.astimezone(timezone.utc))
        async with db.transaction(owner="data_edge"):
            await db.execute(
                """INSERT INTO strategy_data_deliveries
               (delivery_id,strategy,component,status,provider,market_id,snapshot_id,
                observed_at,source_at,age_seconds,latency_ms,item_count,
                required_fields,missing_fields,fallback_used,detail)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (uuid.uuid4().hex, delivery.strategy, delivery.component,
                 delivery.status, delivery.provider, delivery.market_id,
                 delivery.snapshot_id, observed.isoformat(),
                 source.isoformat() if source else None, delivery.age_seconds,
                 max(0, delivery.latency_ms), max(0, delivery.item_count),
                 json.dumps(delivery.required_fields), json.dumps(delivery.missing_fields),
                delivery.fallback_used, json.dumps(delivery.detail, default=str)[:4000]),
            )
        if delivery.status == "ok" and throttle > 0:
            _last_healthy_recorded[key] = now_mono
    except Exception as exc:  # noqa: BLE001 - telemetry must be best effort
        log.debug("data_edge.record_failed", strategy=delivery.strategy,
                  component=delivery.component, error=str(exc))
