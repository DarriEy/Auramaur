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
        directed_orders_allowlist=["SPY", "SLV", "USDCAD"],
        directed_orders_max_notional_usd=250.0,
        directed_orders_daily_notional_usd=1000.0,
        directed_orders_treasury_max_notional_usd=1000.0,
        directed_orders_treasury_daily_notional_usd=2000.0,
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


def _fx(**over):
    base = dict(symbol="USDCAD", sec_type="CASH", currency="CAD", exchange="IDEALPRO",
                side="BUY", quantity=800.0, order_type="MKT", limit_price=None,
                label="probe-fx", dry_run=True)
    base.update(over)
    return DirectedOrder(**base)


def test_usd_base_fx_notional_is_the_quantity_not_quantity_times_rate():
    """USDCAD quantity is already USD. Multiplying by the 1.37 rate would
    report 800 USD as 1,096 and measure it against a USD cap it never
    breached."""
    assert _fx(quantity=800.0).notional_usd(1.37) == pytest.approx(800.0)
    # A non-USD-base pair still converts through the rate.
    assert _fx(symbol="EURUSD", quantity=100.0).notional_usd(1.08) == pytest.approx(108.0)
    # Equities are unaffected.
    assert _order(quantity=2.0).notional_usd(50.0) == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_conversion_uses_the_treasury_cap_not_the_trading_cap(tmp_path):
    """$800 exceeds the $250 trading cap but sits inside the treasury cap —
    a currency conversion is not a market position."""
    ex, db = await _exec(tmp_path)
    try:
        assert await ex.gate_reason(_fx(), 800.0, "U24897594") == ""
        # The same notional as an EQUITY order is still refused.
        r = await ex.gate_reason(_order(), 800.0, "U24897594")
        assert "per-order cap" in r
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_treasury_cap_still_binds(tmp_path):
    ex, db = await _exec(tmp_path)
    try:
        r = await ex.gate_reason(_fx(), 1000.01, "U24897594")
        assert "treasury cap" in r
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_trading_and_treasury_daily_budgets_are_separate(tmp_path):
    """A conversion must not consume the trading allowance, nor the reverse —
    otherwise one $800 conversion eats most of the day's trading budget."""
    ex, db = await _exec(tmp_path)
    try:
        await db.execute(
            """INSERT INTO directed_orders
                 (symbol, sec_type, side, quantity, order_type, notional_usd, status)
               VALUES ('USDCAD','CASH','BUY',800,'MKT',800.0,'filled')""")
        await db.commit()
        # Trading budget untouched by the conversion.
        assert await ex.gate_reason(_order(), 250.0, "U24897594") == ""
        # Treasury budget did move.
        r = await ex.gate_reason(_fx(quantity=1300.0), 1300.0, "U24897594")
        assert "treasury" in r
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("broker_status", [
    "ValidationError", "Cancelled", "ApiCancelled", "Inactive"])
async def test_broker_refusal_is_recorded_as_error_not_submitted(
        tmp_path, broker_status):
    """A refusal is not a submission.

    Observed 2026-07-30: IB Gateway in Read-Only API mode returned
    ValidationError, which was missing from _TERMINAL, so the executor burned
    the full fill timeout and then recorded a REJECTED order as 'submitted' —
    claiming a live working order, and consuming the daily cap for exposure
    that never existed.
    """
    from auramaur.broker.directed_orders import DirectedOrderResult

    ex, db = await _exec(tmp_path)
    try:
        ib = SimpleNamespace(managedAccounts=lambda: ["U24897594"])
        ex._send = AsyncMock(return_value=DirectedOrderResult(
            True, broker_status, ib_order_id="7", filled_qty=0.0))
        res = await ex.place(_order(dry_run=False), ib=ib, reference_price=100.0)

        assert res.accepted is False
        assert res.status == "error"
        assert broker_status in res.reason
        row = await db.fetchone(
            "SELECT status, refuse_reason FROM directed_orders")
        assert row["status"] == "error"
        assert broker_status in row["refuse_reason"]

        # And it must not have consumed the daily trading budget.
        assert await ex.gate_reason(_order(), 250.0, "U24897594") == ""
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_real_fill_is_still_recorded_as_filled(tmp_path):
    from auramaur.broker.directed_orders import DirectedOrderResult

    ex, db = await _exec(tmp_path)
    try:
        ib = SimpleNamespace(managedAccounts=lambda: ["U24897594"])
        ex._send = AsyncMock(return_value=DirectedOrderResult(
            True, "Filled", ib_order_id="8", filled_qty=1.0, filled_price=99.5))
        res = await ex.place(_order(dry_run=False), ib=ib, reference_price=100.0)
        assert res.accepted is True and res.status == "filled"
        row = await db.fetchone("SELECT status, filled_price FROM directed_orders")
        assert row["status"] == "filled" and row["filled_price"] == 99.5
    finally:
        await db.close()
