"""Auditable migration inventory for every production strategy family."""

from __future__ import annotations

from enum import Enum


class MigrationTrack(str, Enum):
    PORTABLE_DIRECTIONAL = "portable_directional"
    ROUTING = "routing"
    PAIRED_PACKAGE = "paired_package"
    QUOTING = "quoting"
    EXTERNAL_ASSET = "external_asset"


class MigrationStatus(str, Enum):
    PROOF_OF_CONCEPT = "proof_of_concept"
    MIGRATED = "migrated"
    PLANNED = "planned"
    SPECIALIZED_PLANNED = "specialized_planned"


# ``MIGRATED`` means production delegates its deterministic decision seam to a
# pure contract and the existing execution-contract tests still pass. It does
# not claim that every contract fits the generic TargetPosition replay runtime.
# Keys mirror strategy.registry. Specialized strategies are not forced through
# TargetPosition: routing, paired legs, quoting inventory, and off-PM assets use
# contracts that retain their production semantics.
MIGRATION_INVENTORY: dict[str, tuple[MigrationTrack, MigrationStatus]] = {
    "core_trading": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.PLANNED),
    "news_reactor": (MigrationTrack.ROUTING, MigrationStatus.PLANNED),
    "kraken": (MigrationTrack.EXTERNAL_ASSET, MigrationStatus.SPECIALIZED_PLANNED),
    "arbitrage": (MigrationTrack.PAIRED_PACKAGE, MigrationStatus.SPECIALIZED_PLANNED),
    "bias_harvest": (
        MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.MIGRATED
    ),
    "platform_consensus": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.PLANNED),
    "long_horizon": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.PLANNED),
    "agent_trader": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.PLANNED),
    "agent_trader_kalshi": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.PLANNED),
    "term_structure": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.PLANNED),
    "vol_anchor": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.PLANNED),
    "informed_flow": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.PLANNED),
    "entailment_arb": (MigrationTrack.PAIRED_PACKAGE, MigrationStatus.SPECIALIZED_PLANNED),
    "cross_venue_arb": (MigrationTrack.PAIRED_PACKAGE, MigrationStatus.SPECIALIZED_PLANNED),
    "econ_indicator": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.PLANNED),
    "interim_manager": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.PLANNED),
    "settlement_arb": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.PLANNED),
    "weather_temp": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.PLANNED),
    "resolution_lens": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.PLANNED),
    "resolution_lens_kalshi": (
        MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.PLANNED
    ),
    "oddlot_tender": (MigrationTrack.EXTERNAL_ASSET, MigrationStatus.SPECIALIZED_PLANNED),
    "momentum_coupling": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.PLANNED),
    "market_maker": (MigrationTrack.QUOTING, MigrationStatus.SPECIALIZED_PLANNED),
    "ibkr_etf_paper": (MigrationTrack.EXTERNAL_ASSET, MigrationStatus.SPECIALIZED_PLANNED),
    "ibkr_multiasset": (MigrationTrack.EXTERNAL_ASSET, MigrationStatus.SPECIALIZED_PLANNED),
}
