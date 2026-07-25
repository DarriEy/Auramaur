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
    source_time_required: bool = True


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
#
# ``fail_closed=True`` is reserved for components a strategy records
# UNCONDITIONALLY every cycle (market snapshots, cycle telemetry): for those,
# a missing/expired record means the pipeline broke. Components recorded only
# when a candidate/event materializes (per-market books, FRED cache misses,
# weather ensembles, model calls, venue quotes outside market hours) are
# ``fail_closed=False`` — absence means idle, not broken, and routing them to
# FAIL would leave the criterion permanently red during normal quiet periods.
_EVIDENCE = DataRequirement(component="evidence", max_age_seconds=48 * 3600)
_MARKETS = DataRequirement(
    component="market_snapshot", max_age_seconds=300,
    required_fields=("outcome_yes_price", "outcome_no_price"),
)
_BOOK = DataRequirement(
    component="order_book", max_age_seconds=15,
    required_fields=("best_bid", "best_ask"), fail_closed=False,
)
_CYCLE = DataRequirement(
    component="strategy_cycle", max_age_seconds=900, min_items=0,
    source_time_required=False,
)
_FRED = DataRequirement(
    component="fred_observations", max_age_seconds=3600, fail_closed=False,
)
# IBKR/venue quotes: recorded every cycle, but outside market hours the
# honest record is "empty, market closed" — that must not read as failure.
_VENUE_QUOTE = DataRequirement(
    component="multiasset_quote", max_age_seconds=120, fail_closed=False,
)

REQUIREMENTS: dict[str, tuple[DataRequirement, ...]] = {
    "strategic": (_MARKETS, _EVIDENCE),
    "llm": (_MARKETS, _EVIDENCE),
    "technical": (_MARKETS, DataRequirement(component="price_history", max_age_seconds=300)),
    "market_maker": (_MARKETS, _BOOK),
    "arbitrage": (_MARKETS,),
    "bias_harvest": (_MARKETS, _BOOK),
    "platform_consensus": (_MARKETS, DataRequirement(component="platform_forecasts", max_age_seconds=900, fail_closed=False)),
    "cross_venue_arb": (DataRequirement(component="cross_venue_snapshot", max_age_seconds=30),),
    "informed_flow": (DataRequirement(component="venue_trades", max_age_seconds=120),),
    "econ_indicator": (_FRED, _MARKETS),
    "settlement_arb": (_FRED, _MARKETS),
    "weather_temp": (DataRequirement(component="weather_ensemble", max_age_seconds=3600, fail_closed=False), _MARKETS),
    "vol_anchor": (DataRequirement(component="spot_volatility", max_age_seconds=3600, fail_closed=False), _MARKETS),
    "oddlot_tender": (DataRequirement(component="edgar_filings", max_age_seconds=21600),),
    "resolution_lens": (_MARKETS, _EVIDENCE.model_copy(update={"fail_closed": False})),
    "resolution_lens_kalshi": (_MARKETS, _EVIDENCE.model_copy(update={"fail_closed": False})),
    "term_structure": (_MARKETS,),
    "agent_trader": (_MARKETS, DataRequirement(component="model_response", max_age_seconds=10800, fail_closed=False)),
    "long_horizon": (_MARKETS,),
    "entailment_arb": (_MARKETS,),
    "ibkr_etf_paper": (DataRequirement(component="equity_quote", max_age_seconds=120, fail_closed=False), _EVIDENCE),
    "ibkr_fx_paper": (_VENUE_QUOTE,),
    "ibkr_futures_paper": (_VENUE_QUOTE,),
    "ibkr_global_etf_paper": (_VENUE_QUOTE,),
    "ibkr_international_equity_paper": (_VENUE_QUOTE,),
    "ibkr_bonds_paper": (_VENUE_QUOTE,),
    "ibkr_options_paper": (_VENUE_QUOTE,),
    "kraken_treasury": (DataRequirement(component="crypto_quote", max_age_seconds=120, fail_closed=False), _EVIDENCE),
    # Operational/non-trading workers. These have no market-data input
    # contract, but naming them explicitly prevents the dangerous unknown-name
    # fallback that used to make new strategies look healthy automatically.
    "evidence_distiller": (_CYCLE,),
    "interim_manager": (_CYCLE,),
    # No delivery recorder yet — hold it to the cycle contract until one lands
    # (a crypto_quote requirement with no emitter would fail unconditionally).
    "momentum_coupling": (_CYCLE,),
}


STRATEGY_ALIASES: dict[str, str] = {
    "KrakenPillar": "kraken_treasury",
    "EvidenceDistiller": "evidence_distiller",
}


def canonical_strategy(strategy: str) -> str:
    return STRATEGY_ALIASES.get(strategy, strategy)


def requirements_for(strategy: str) -> tuple[DataRequirement, ...]:
    return REQUIREMENTS.get(canonical_strategy(strategy), ())


def snapshot_id(*parts: object) -> str:
    """Stable identity for observations intended to be contemporaneous."""
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()[:24]


async def record_delivery(db, delivery: DataDelivery) -> None:
    """Persist one delivery without allowing monitoring to kill a strategy."""
    try:
        delivery.strategy = canonical_strategy(delivery.strategy)
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


async def record_market_snapshot(db, strategy: str, markets, *,
                                 provider: str = "discovery") -> str:
    """Record the exact discovery result delivered to one consumer.

    A snapshot is usable when at least one market carries prices — venues
    routinely include a few unpriced markets, and strategies skip those, so
    "partial" (a hard readiness reason on fail-closed contracts) is reserved
    for the degenerate case where rows arrived but NONE are priced. The
    unpriced count is always surfaced in ``detail``.
    """
    observed = datetime.now(timezone.utc)
    rows = list(markets or [])
    missing = sum(
        1 for market in rows
        if getattr(market, "outcome_yes_price", None) is None
        or getattr(market, "outcome_no_price", None) is None
    )
    degraded = bool(rows) and missing == len(rows)
    sid = snapshot_id(strategy, provider, observed.isoformat(), len(rows))
    await record_delivery(db, DataDelivery(
        strategy=strategy, component="market_snapshot",
        status="empty" if not rows else "partial" if degraded else "ok",
        provider=provider, snapshot_id=sid, source_at=observed,
        item_count=len(rows),
        required_fields=("outcome_yes_price", "outcome_no_price"),
        missing_fields=(("outcome_prices",) if degraded else ()),
        detail={"missing_market_count": missing},
    ))
    return sid


async def _record_input_manifest(db, strategy: str,
                                 cycle_started_at: datetime) -> None:
    """Persist the datasets that actually participated in one strategy cycle."""
    canonical = canonical_strategy(strategy)
    requirements = requirements_for(canonical)
    components: list[dict] = []
    healthy = bool(requirements)
    for requirement in requirements:
        row = await db.fetchone(
            """SELECT component,status,provider,market_id,snapshot_id,source_at,
                      age_seconds,item_count,missing_fields,fallback_used
                 FROM strategy_data_deliveries
                WHERE strategy=? AND component=? AND observed_at>=?
                ORDER BY observed_at DESC,id DESC LIMIT 1""",
            (canonical, requirement.component, cycle_started_at.isoformat()),
        )
        if row is None:
            components.append({"component": requirement.component,
                               "status": "unobserved"})
            healthy = healthy and not requirement.fail_closed
            continue
        item = dict(row)
        components.append(item)
        if requirement.fail_closed and (
                item["status"] != "ok"
                or int(item["item_count"] or 0) < requirement.min_items
                or (requirement.source_time_required and not item["source_at"])
                or json.loads(item["missing_fields"] or "[]")):
            healthy = False
    await record_delivery(db, DataDelivery(
        strategy=canonical, component="input_manifest",
        status="ok" if healthy else "error",
        source_at=datetime.now(timezone.utc), item_count=len(components),
        detail={"cycle_started_at": cycle_started_at.isoformat(),
                "components": components},
    ))


async def record_input_manifest(db, strategy: str,
                                cycle_started_at: datetime) -> None:
    """Best-effort wrapper: telemetry must never break a strategy cycle."""
    try:
        await _record_input_manifest(db, strategy, cycle_started_at)
    except Exception as exc:  # noqa: BLE001
        log.debug("data_edge.manifest_failed", strategy=strategy, error=str(exc))
