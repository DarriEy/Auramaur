"""Operator-directed order gates.

This is a live-order path on a real account, so every refusal condition is
pinned. Today produced three separate cases of a flag not doing what its name
said; these tests exist so this gate cannot join them.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from auramaur.broker.directed_orders import DirectedOrder, DirectedOrderExecutor
from auramaur.db.database import Database


def _settings(**over):
    ibkr = SimpleNamespace(
        directed_orders_enabled=True, directed_orders_confirm_live=True,
        directed_orders_account="U24897594",
        directed_orders_allowlist=["SPY", "SLV"],
        directed_orders_max_notional_usd=250.0,
        directed_orders_daily_notional_usd=1000.0,
        directed_orders_client_id=5,
        multiasset_execution_fill_timeout_seconds=5,
    )
    for k, v in over.items():
        setattr(ibkr, k, v)
    return SimpleNamespace(ibkr=ibkr, is_live=over.pop("is_live", True))


def _order(**over):
    base = dict(symbol="SPY", sec_type="STK", currency="USD", exchange="SMART",
                side="BUY", quantity=1.0, order_type="LMT", limit_price=100.0,
                label="probe-us-etf", dry_run=True)
    base.update(over)
    return DirectedOrder(**base)


async def _exec(tmp_path, settings=None):
    db = Database(str(tmp_path / "d.db"))
    await db.connect()
    return DirectedOrderExecutor(settings or _settings(), db), db


@pytest.mark.asyncio
async def test_clean_order_passes_every_gate(tmp_path):
    ex, db = await _exec(tmp_path)
    try:
        assert await ex.gate_reason(_order(), 100.0, "U24897594") == ""
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_kill_switch_outranks_everything(tmp_path):
    """Checked first so a config detail can never mask it."""
    ex, db = await _exec(tmp_path)
    try:
        with patch("auramaur.broker.directed_orders.kill_switch_present",
                   lambda: True):
            assert await ex.gate_reason(_order(), 100.0, "U24897594") == "kill switch"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_unset_account_refuses_everything(tmp_path):
    """Fail closed. The environment label cannot be trusted to say which
    account is at risk — `environment: paper` served a live account on
    2026-07-29 — so the operator must declare it."""
    ex, db = await _exec(tmp_path, _settings(directed_orders_account=""))
    try:
        r = await ex.gate_reason(_order(), 100.0, "U24897594")
        assert "fail-closed" in r
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_wrong_account_is_refused(tmp_path):
    ex, db = await _exec(tmp_path)
    try:
        r = await ex.gate_reason(_order(), 100.0, "DU9999999")
        assert "DU9999999" in r and "U24897594" in r
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("over,expect", [
    ({"directed_orders_enabled": False}, "directed_orders_enabled=false"),
    ({"directed_orders_confirm_live": False}, "confirm_live=false"),
    ({"is_live": False}, "global live gates are closed"),
])
async def test_each_flag_refuses_on_its_own(tmp_path, over, expect):
    ex, db = await _exec(tmp_path, _settings(**over))
    try:
        assert expect in await ex.gate_reason(_order(), 100.0, "U24897594")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_symbol_must_be_explicitly_allowlisted(tmp_path):
    ex, db = await _exec(tmp_path)
    try:
        r = await ex.gate_reason(_order(symbol="TSLA"), 100.0, "U24897594")
        assert "not in ibkr.directed_orders_allowlist" in r
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_per_order_notional_cap(tmp_path):
    ex, db = await _exec(tmp_path)
    try:
        assert await ex.gate_reason(_order(), 250.0, "U24897594") == ""
        assert "per-order cap" in await ex.gate_reason(_order(), 250.01, "U24897594")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_daily_cap_counts_submitted_not_just_filled(tmp_path):
    """An unfilled order is still committed exposure until it is cancelled."""
    ex, db = await _exec(tmp_path)
    try:
        for _ in range(4):
            await db.execute(
                """INSERT INTO directed_orders
                     (symbol, sec_type, side, quantity, order_type,
                      notional_usd, status)
                   VALUES ('SPY','STK','BUY',1,'LMT',250.0,'submitted')""")
        await db.commit()
        r = await ex.gate_reason(_order(), 100.0, "U24897594")
        assert "daily cap" in r
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_limit_order_needs_a_price(tmp_path):
    ex, db = await _exec(tmp_path)
    try:
        r = await ex.gate_reason(_order(limit_price=None), 100.0, "U24897594")
        assert "limit order without a limit price" in r
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_refused_order_is_audited_and_never_sent(tmp_path):
    ex, db = await _exec(tmp_path, _settings(directed_orders_enabled=False))
    try:
        ib = SimpleNamespace(managedAccounts=lambda: ["U24897594"])
        ex._send = AsyncMock()
        res = await ex.place(_order(dry_run=False), ib=ib, reference_price=100.0)
        assert res.accepted is False and res.status == "refused"
        ex._send.assert_not_awaited()
        row = await db.fetchone("SELECT status, refuse_reason FROM directed_orders")
        assert row["status"] == "refused"
        assert "directed_orders_enabled=false" in row["refuse_reason"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_dry_run_is_audited_and_never_sent(tmp_path):
    """dry_run is gate 3 and must be cleared explicitly, per CLAUDE.md."""
    ex, db = await _exec(tmp_path)
    try:
        ib = SimpleNamespace(managedAccounts=lambda: ["U24897594"])
        ex._send = AsyncMock()
        res = await ex.place(_order(dry_run=True), ib=ib, reference_price=100.0)
        assert res.accepted is True and res.status == "dry_run"
        ex._send.assert_not_awaited()
        row = await db.fetchone("SELECT status, dry_run FROM directed_orders")
        assert row["status"] == "dry_run" and row["dry_run"] == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_audit_row_is_written_before_sending(tmp_path):
    """An order that errors or never returns must still be answerable."""
    ex, db = await _exec(tmp_path)
    try:
        ib = SimpleNamespace(managedAccounts=lambda: ["U24897594"])
        ex._send = AsyncMock(side_effect=RuntimeError("gateway exploded"))
        res = await ex.place(_order(dry_run=False), ib=ib, reference_price=100.0)
        assert res.accepted is False and res.status == "error"
        row = await db.fetchone("SELECT status, refuse_reason FROM directed_orders")
        assert row["status"] == "error"
        assert "gateway exploded" in row["refuse_reason"]
    finally:
        await db.close()
