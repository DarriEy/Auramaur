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
    "core_trading": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.MIGRATED),
    "news_reactor": (MigrationTrack.ROUTING, MigrationStatus.MIGRATED),
    "kraken": (MigrationTrack.EXTERNAL_ASSET, MigrationStatus.SPECIALIZED_PLANNED),
    "arbitrage": (MigrationTrack.PAIRED_PACKAGE, MigrationStatus.MIGRATED),
    "bias_harvest": (
        MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.MIGRATED
    ),
    "platform_consensus": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.MIGRATED),
    "long_horizon": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.MIGRATED),
    "agent_trader": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.MIGRATED),
    "agent_trader_kalshi": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.MIGRATED),
    "term_structure": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.MIGRATED),
    "vol_anchor": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.MIGRATED),
    "informed_flow": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.MIGRATED),
    "entailment_arb": (MigrationTrack.PAIRED_PACKAGE, MigrationStatus.MIGRATED),
    "cross_venue_arb": (MigrationTrack.PAIRED_PACKAGE, MigrationStatus.MIGRATED),
    "econ_indicator": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.MIGRATED),
    "interim_manager": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.MIGRATED),
    "settlement_arb": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.MIGRATED),
    "weather_temp": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.MIGRATED),
    "resolution_lens": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.MIGRATED),
    "resolution_lens_kalshi": (
        MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.MIGRATED
    ),
    "oddlot_tender": (MigrationTrack.EXTERNAL_ASSET, MigrationStatus.SPECIALIZED_PLANNED),
    "momentum_coupling": (MigrationTrack.PORTABLE_DIRECTIONAL, MigrationStatus.MIGRATED),
    "market_maker": (MigrationTrack.QUOTING, MigrationStatus.MIGRATED),
    "ibkr_etf_paper": (MigrationTrack.EXTERNAL_ASSET, MigrationStatus.SPECIALIZED_PLANNED),
    "ibkr_multiasset": (MigrationTrack.EXTERNAL_ASSET, MigrationStatus.SPECIALIZED_PLANNED),
}
