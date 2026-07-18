"""Security invariants for irreversible cross-venue withdrawals."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from auramaur.treasury.transfers import TransferManager


def _settings(*, cap: float = 100.0, armed: bool = True):
    return SimpleNamespace(
        transfers_armed=armed,
        transfers=SimpleNamespace(
            allowed_withdraw_keys=["polymarket-usdc"],
            min_transfer_usd=1.0,
            per_transfer_cap_usd=cap,
            daily_cap_usd=cap,
            require_approval=True,
        ),
    )


@pytest.mark.asyncio
async def test_concurrent_transfers_cannot_race_daily_cap(tmp_path):
    kraken = SimpleNamespace(
        _private=AsyncMock(return_value={"error": [], "result": {"refid": "ok"}})
    )
    ledger = tmp_path / "transfers.sqlite3"
    managers = [TransferManager(_settings(), kraken, ledger) for _ in range(2)]

    with patch("auramaur.treasury.transfers.kill_switch_present", return_value=False):
        first, second = await asyncio.gather(*[
            manager.transfer("polymarket-usdc", 60.0, approver=lambda _: True)
            for manager in managers
        ])

    assert sorted([first.status, second.status]) == ["blocked", "executed"]
    assert kraken._private.await_count == 1


@pytest.mark.asyncio
async def test_api_rejection_releases_reservation(tmp_path):
    kraken = SimpleNamespace(
        _private=AsyncMock(side_effect=[
            {"error": ["rejected"], "result": {}},
            {"error": [], "result": {"refid": "second"}},
        ])
    )
    manager = TransferManager(_settings(), kraken, tmp_path / "transfers.sqlite3")

    with patch("auramaur.treasury.transfers.kill_switch_present", return_value=False):
        rejected = await manager.transfer(
            "polymarket-usdc", 100.0, approver=lambda _: True,
        )
        retried = await manager.transfer(
            "polymarket-usdc", 100.0, approver=lambda _: True,
        )

    assert rejected.status == "rejected"
    assert retried.status == "executed"


@pytest.mark.asyncio
async def test_uncertain_outcome_retains_reservation(tmp_path):
    kraken = SimpleNamespace(_private=AsyncMock(side_effect=TimeoutError("timeout")))
    manager = TransferManager(_settings(), kraken, tmp_path / "transfers.sqlite3")

    with patch("auramaur.treasury.transfers.kill_switch_present", return_value=False):
        uncertain = await manager.transfer(
            "polymarket-usdc", 60.0, approver=lambda _: True,
        )
        blocked = await manager.transfer(
            "polymarket-usdc", 60.0, approver=lambda _: True,
        )

    assert "uncertain" in uncertain.reason
    assert blocked.status == "blocked"


def test_legacy_json_total_is_migrated_and_enforced(tmp_path):
    import json
    from datetime import datetime, timezone

    ledger = tmp_path / "transfer_ledger.sqlite3"
    ledger.with_suffix(".json").write_text(json.dumps({
        datetime.now(timezone.utc).date().isoformat(): 80.0,
    }))
    manager = TransferManager(_settings(), SimpleNamespace(), ledger)

    assert manager._spent_today() == 80.0


def test_default_ledger_path_honours_state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AURAMAUR_STATE_DIR", str(tmp_path / "state"))
    manager = TransferManager(_settings(), SimpleNamespace())

    assert manager._ledger_path == tmp_path / "state" / "data" / "transfer_ledger.sqlite3"


@pytest.mark.asyncio
async def test_unarmed_preview_with_corrupt_ledger_reports_blocked(tmp_path):
    ledger = tmp_path / "transfer_ledger.sqlite3"
    ledger.with_suffix(".json").write_text("{not valid json")
    manager = TransferManager(_settings(armed=False), SimpleNamespace(), ledger)

    with patch("auramaur.treasury.transfers.kill_switch_present", return_value=False):
        result = await manager.transfer(
            "polymarket-usdc", 10.0, approver=lambda _: True,
        )

    assert result.status == "blocked"
    assert "ledger unavailable" in result.reason
