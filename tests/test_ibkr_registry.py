from types import SimpleNamespace

import pytest

from auramaur.db.database import Database
from auramaur.exchange.ibkr_instruments import BY_KEY
from auramaur.exchange.ibkr_registry import approve, eligible_keys, record_validation


@pytest.mark.asyncio
async def test_registry_auto_approves_declared_static_instrument():
    db = Database(":memory:")
    await db.connect()
    spec = BY_KEY["SPY"]
    status = await record_validation(
        db, spec, SimpleNamespace(conId=756733, localSymbol="SPY",
                                  tradingClass="SPY", exchange="SMART",
                                  currency="USD", multiplier="1"),
        quote_source="ibkr_live", has_history=True)
    assert status == "eligible"
    assert await eligible_keys(db) == {"SPY"}
    await db.close()


@pytest.mark.asyncio
async def test_corporate_bond_requires_operator_approval():
    db = Database(":memory:")
    await db.connect()
    spec = BY_KEY["CORP_5Y"]
    contract = SimpleNamespace(conId=1234, exchange="SMART", currency="USD",
                               multiplier="10")
    assert await record_validation(db, spec, contract, quote_source="ibkr_live",
                                   has_history=True) == "pending_approval"
    assert "CORP_5Y" not in await eligible_keys(db)
    assert await approve(db, "CORP_5Y", "reviewed issue and maturity")
    assert "CORP_5Y" in await eligible_keys(db)
    await db.close()


@pytest.mark.asyncio
async def test_manifest_drift_revokes_existing_approval(monkeypatch):
    db = Database(":memory:")
    await db.connect()
    spec = BY_KEY["SPY"]
    contract = SimpleNamespace(conId=1, exchange="SMART", currency="USD",
                               multiplier="1")
    await record_validation(db, spec, contract, quote_source="ibkr_live",
                            has_history=True)
    monkeypatch.setattr(type(spec), "manifest_hash", lambda _self: "changed")
    assert await record_validation(db, spec, contract, quote_source="ibkr_live",
                                   has_history=True) == "drifted"
    assert "SPY" not in await eligible_keys(db)
    await db.close()


@pytest.mark.asyncio
async def test_contract_identity_drift_requires_reapproval_but_failure_preserves_identity():
    db = Database(":memory:")
    await db.connect()
    spec = BY_KEY["SPY"]
    original = SimpleNamespace(conId=11, localSymbol="SPY", exchange="SMART",
                               currency="USD", multiplier="1")
    await record_validation(db, spec, original, quote_source="ibkr_live",
                            has_history=True)
    failed = SimpleNamespace(conId=0, exchange="SMART", currency="USD", multiplier="1")
    assert await record_validation(db, spec, failed, quote_source="none",
                                   has_history=False, error="socket lost") == "quarantined"
    row = await db.fetchone(
        "SELECT con_id, local_symbol FROM ibkr_contract_registry WHERE instrument_key='SPY'")
    assert (row["con_id"], row["local_symbol"]) == (11, "SPY")
    changed = SimpleNamespace(conId=12, localSymbol="SPY", exchange="SMART",
                              currency="USD", multiplier="1")
    assert await record_validation(db, spec, changed, quote_source="ibkr_live",
                                   has_history=True) == "drifted"
    assert "SPY" not in await eligible_keys(db)
    assert await approve(db, "SPY", "verified broker identity replacement")
    assert "SPY" in await eligible_keys(db)
    await db.close()
