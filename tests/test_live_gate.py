"""Tests for the operational live-readiness preflight + the config CI guard."""

from __future__ import annotations

import subprocess

import pytest
import yaml

from auramaur.db.database import Database
from auramaur.monitoring.live_gate import GateResult, PreflightReport, preflight
from config.settings import Settings


def test_report_live_allowed_logic():
    rep = PreflightReport(results=[GateResult("a", "OK", "x"), GateResult("b", "WARN", "y")])
    assert rep.live_allowed
    assert rep.warnings and not rep.blocks
    rep.results.append(GateResult("c", "BLOCK", "z"))
    assert not rep.live_allowed
    assert [b.name for b in rep.blocks] == ["c"]


async def _db() -> Database:
    db = Database(":memory:")
    await db.connect()
    return db


@pytest.mark.asyncio
async def test_clean_state_allows_live(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no KILL_SWITCH in this cwd
    db = await _db()
    try:
        rep = await preflight(Settings(), db)
        assert rep.live_allowed, [b.detail for b in rep.blocks]
        sev = {r.name: r.severity for r in rep.results}
        assert sev["kill_switch"] == "OK"
        assert sev["fee_model"] == "OK"
        assert sev["position_marks"] == "OK"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_kill_switch_blocks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "KILL_SWITCH").write_text("")  # in tmp cwd — never the real file
    db = await _db()
    try:
        rep = await preflight(Settings(), db)
        assert not rep.live_allowed
        assert any(b.name == "kill_switch" for b in rep.blocks)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_disabled_fee_model_blocks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = Settings()
    s.arbitrage.exchange_fees = {"kalshi": 0.0, "polymarket": 0.0}  # fees disabled
    db = await _db()
    try:
        rep = await preflight(s, db)
        assert not rep.live_allowed
        assert any(b.name == "fee_model" for b in rep.blocks)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_zero_marked_live_positions_block(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = await _db()
    try:
        for i in range(6):  # > 5 zero-marked live positions => BLOCK
            await db.execute(
                """INSERT INTO portfolio
                   (market_id, exchange, side, size, avg_price, current_price,
                    token, is_paper, updated_at)
                   VALUES (?, 'polymarket', 'BUY', 10, 0.5, 0, 'YES', 0, datetime('now'))""",
                (f"m{i}",),
            )
        await db.commit()
        rep = await preflight(Settings(), db)
        assert not rep.live_allowed
        assert any(b.name == "position_marks" for b in rep.blocks)
    finally:
        await db.close()


def test_committed_defaults_yaml_is_not_live():
    """The TRACKED config/defaults.yaml must ship with execution.live: false.

    Reads the committed blob (not the working tree) so a local live override
    can't satisfy or break the guard — this catches accidentally committing it.
    """
    out = subprocess.run(
        ["git", "show", "HEAD:config/defaults.yaml"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    cfg = yaml.safe_load(out.stdout)
    assert cfg.get("execution", {}).get("live") is False, (
        "committed config/defaults.yaml must have execution.live: false — "
        "the local live override must never be committed"
    )
