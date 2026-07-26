from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest
from click.testing import CliRunner

from auramaur.db.database import Database
from auramaur.cli._base import main
from auramaur.experiments.operations import ExperimentManifest, bind_manifest
from auramaur.experiments.registry import RegisteredExperiment
from auramaur.experiments.specialized import (
    PackageSemantics,
    RegisteredSpecializedExperiment,
    SpecializedReplayRuntime,
)
from auramaur.research.experiment_repository import ExperimentRepository
from auramaur.research.experiment_scoreboard import ExperimentScoreboard


NOW = datetime(2026, 7, 25, tzinfo=timezone.utc)


def _definition(source: str = "bias_harvest") -> dict:
    return {
        "key": f"ops-{source}", "strategy_source": source,
        "hypothesis": "A deterministic proposal is testable.",
        "mechanism": "fixture", "implementation_version": "v1",
        "parameters": {}, "venues": ["polymarket"],
        "primary_metric": "net_pnl", "baseline": "no_position",
        "min_observations": 1, "holdout_days": 1, "max_drawdown": 0.2,
        "cost_model": "linear-v1",
        "rejection_criteria": ["max_drawdown_exceeds_limit"],
    }


def _portfolio() -> dict:
    return {"observed_at": NOW.isoformat(), "cash": 100, "equity": 100}


def _feature(name: str, payload: dict) -> dict:
    return {
        "name": name, "payload": payload, "source": "fixture",
        "observed_at": NOW.isoformat(), "available_at": NOW.isoformat(),
    }


def test_portable_manifest_binds_dataclass_rules():
    manifest = ExperimentManifest.model_validate({
        "definition": _definition(),
        "binding": {
            "kind": "portable",
            "implementation": (
                "auramaur.experiments.strategies.bias_harvest:BiasHarvestExperiment"
            ),
            "constructor_arguments": {"rules": {
                "band_lo": 0.05, "band_hi": 0.95, "edge_uplift": 0.05,
                "min_liquidity": 10,
                "min_hours_to_resolution": 1, "max_days_to_resolution": 30,
                "stake_usd": 10, "skip_disputed": True, "maker_entry": False,
                "maker_min_spread": 0.01, "paper": True,
            }},
        },
        "snapshots": [{
            "observed_at": NOW.isoformat(), "sequence": 1, "market_id": "m1",
            "venue": "polymarket",
            "prices": [{"instrument_id": "m1:YES", "price": 0.5}],
            "features": [_feature("bias_harvest_candidate", {
                "market_id": "m1", "venue": "polymarket", "question": "Will X?",
                "description": "", "category": "other", "active": True,
                "accepting_orders": True, "liquidity": 100, "volume": 100,
                "yes_price": 0.5, "no_price": 0.5,
                "end_at": (NOW + timedelta(days=2)).isoformat(),
                "already_entered_or_held": False, "maker_book_available": True,
            })], "data_version": "fixture-v1",
        }],
        "initial_portfolio": _portfolio(),
        "execution_model": {"version": "linear-v1"},
    })
    assert isinstance(bind_manifest(manifest), RegisteredExperiment)


def _specialized_manifest() -> ExperimentManifest:
    return ExperimentManifest.model_validate({
        "definition": _definition("cross_venue_arb"),
        "binding": {
            "kind": "specialized",
            "evaluator": (
                "auramaur.experiments.strategies.cross_venue_arb:paired_arb_proposal"
            ),
            "feature_name": "cross_venue_pair", "proposal_kind": "cross_venue_pair",
            "semantics": "all_or_none",
            "fixed_arguments": {
                "min_confidence": 0.8, "required_gap": 0.05, "stake_usd": 10,
            },
        },
        "snapshots": [{
            "observed_at": NOW.isoformat(), "sequence": 1, "market_id": "pair-1",
            "venue": "multi", "prices": [{"instrument_id": "marker", "price": 1}],
            "features": [_feature("cross_venue_pair", {
                "market_a_id": "a", "venue_a": "polymarket", "yes_price_a": 0.3,
                "market_b_id": "b", "venue_b": "kalshi", "yes_price_b": 0.5,
                "orientation": "same", "confidence": 0.9,
            })], "data_version": "fixture-v1",
        }],
        "initial_portfolio": _portfolio(),
    })


def test_specialized_replay_preserves_atomic_package():
    manifest = _specialized_manifest()
    registered = bind_manifest(manifest)
    assert isinstance(registered, RegisteredSpecializedExperiment)
    report = asyncio.run(SpecializedReplayRuntime().run(
        registered, manifest.snapshots, manifest.initial_portfolio
    ))
    proposal = report.results[0].proposal
    assert proposal is not None
    assert proposal.semantics is PackageSemantics.ALL_OR_NONE
    assert len(proposal.payload.decoded()["legs"]) == 2


def test_specialized_adapter_coerces_nested_external_asset_contracts():
    raw = {
        "definition": _definition("kraken"),
        "binding": {
            "kind": "specialized",
            "evaluator": "auramaur.experiments.strategies.kraken:assess_kraken_spot",
            "feature_name": "kraken_order", "proposal_kind": "kraken_spot_order",
            "semantics": "asset_order", "package_id_field": "pair",
        },
        "snapshots": [{
            "observed_at": NOW.isoformat(), "sequence": 1, "market_id": "BTCUSD",
            "venue": "kraken", "prices": [{"instrument_id": "BTCUSD", "price": 50000}],
            "features": [_feature("kraken_order", {
                "inputs": {
                    "pair": "BTCUSD", "price": 50000, "holding": False,
                    "orphaned": False, "signal_accepted": True,
                    "sized_entry_volume": 0.001, "allocated_usd": 0,
                    "budget_usd": 100, "paper": True,
                },
                "rules": {"max_order_usd": 50, "lot_decimals": 6},
            })], "data_version": "fixture-v1",
        }],
        "initial_portfolio": _portfolio(),
    }
    manifest = ExperimentManifest.model_validate(raw)
    registered = bind_manifest(manifest)
    assert isinstance(registered, RegisteredSpecializedExperiment)
    report = asyncio.run(SpecializedReplayRuntime().run(
        registered, manifest.snapshots, manifest.initial_portfolio
    ))
    proposal = report.results[0].proposal
    assert proposal is not None
    assert proposal.package_id == "BTCUSD"
    assert proposal.semantics is PackageSemantics.ASSET_ORDER
    assert proposal.payload.decoded()["action"] == "buy"


@pytest.mark.asyncio
async def test_specialized_replay_persists_without_pnl_and_scores_separately(tmp_path):
    manifest = _specialized_manifest()
    registered = bind_manifest(manifest)
    assert isinstance(registered, RegisteredSpecializedExperiment)
    report = await SpecializedReplayRuntime().run(
        registered, manifest.snapshots, manifest.initial_portfolio
    )
    db = Database(str(tmp_path / "ops.db"))
    await db.connect()
    try:
        inserted = await ExperimentRepository(db).record_specialized_replay(
            registered, report
        )
        assert inserted == 1
        assert await ExperimentRepository(db).record_specialized_replay(
            registered, report
        ) == 0
        row = await db.fetchone("SELECT score,payload_json FROM strategy_evaluations")
        payload = json.loads(row["payload_json"])
        assert row["score"] == 0
        assert "net_pnl" not in payload
        board = await ExperimentScoreboard(db).build()
        assert board[0].status == "proposal_replay"
        assert board[0].replay_observations == 1
        assert board[0].net_pnl is None
    finally:
        await db.close()


def test_experiment_commands_are_registered():
    from auramaur.cli import experiments  # noqa: F401

    result = CliRunner().invoke(main, ["experiment", "--help"])
    assert result.exit_code == 0
    for command in ("define", "register", "replay", "shadow", "report", "scoreboard"):
        assert command in result.output
