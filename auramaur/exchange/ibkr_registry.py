"""Persistence boundary between the reviewed manifest and IBKR discovery."""

from __future__ import annotations

from auramaur.exchange.ibkr_instruments import InstrumentSpec


async def record_validation(db, spec: InstrumentSpec, contract, *, quote_source: str,
                            has_history: bool, error: str = "") -> str:
    """Upsert one declared instrument without overwriting operator approval."""
    existing = await db.fetchone(
        "SELECT manifest_hash, con_id, approved FROM ibkr_contract_registry "
        "WHERE instrument_key=?",
        (spec.key,))
    same_manifest = bool(existing and existing["manifest_hash"] == spec.manifest_hash())
    discovered_con_id = int(getattr(contract, "conId", 0) or 0)
    identity_changed = bool(
        existing and discovered_con_id and int(existing["con_id"])
        and discovered_con_id != int(existing["con_id"]))
    approved = int(existing["approved"]) if same_manifest else int(not spec.approval_required)
    if error:
        status = "quarantined"
    elif existing and (not same_manifest or identity_changed):
        status = "drifted"
        approved = 0
    elif spec.approval_required and not approved:
        status = "pending_approval"
    elif quote_source != "ibkr_live" or not has_history:
        status = "qualified_no_live_data"
    else:
        status = "eligible"
    con_id = discovered_con_id
    await db.execute(
        """INSERT INTO ibkr_contract_registry
           (instrument_key, book, kind, manifest_hash, con_id, local_symbol,
            trading_class, exchange, currency, multiplier, status, approved,
            quote_source, has_history, last_error)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(instrument_key) DO UPDATE SET
             book=excluded.book, kind=excluded.kind,
             manifest_hash=excluded.manifest_hash,
             con_id=CASE WHEN excluded.con_id != 0 THEN excluded.con_id ELSE con_id END,
             local_symbol=CASE WHEN excluded.local_symbol != ''
                               THEN excluded.local_symbol ELSE local_symbol END,
             trading_class=CASE WHEN excluded.trading_class != ''
                                THEN excluded.trading_class ELSE trading_class END,
             exchange=CASE WHEN excluded.exchange != ''
                           THEN excluded.exchange ELSE exchange END,
             currency=CASE WHEN excluded.currency != ''
                           THEN excluded.currency ELSE currency END,
             multiplier=CASE WHEN excluded.con_id != 0
                             THEN excluded.multiplier ELSE multiplier END,
             status=excluded.status, approved=excluded.approved,
             quote_source=excluded.quote_source, has_history=excluded.has_history,
             last_error=excluded.last_error, validated_at=datetime('now'),
             qualified_at=CASE WHEN excluded.con_id != 0
                                    AND ibkr_contract_registry.con_id != excluded.con_id
                               THEN datetime('now') ELSE qualified_at END""",
        (spec.key, spec.book.value, spec.kind.value, spec.manifest_hash(), con_id,
         str(getattr(contract, "localSymbol", "") or ""),
         str(getattr(contract, "tradingClass", "") or ""),
         str(getattr(contract, "exchange", spec.exchange) or spec.exchange),
         str(getattr(contract, "currency", spec.currency) or spec.currency),
         float(getattr(contract, "multiplier", spec.multiplier) or spec.multiplier),
         status, approved, quote_source, int(has_history), error[:300]))
    await db.commit()
    return status


async def approve(db, instrument_key: str, reason: str) -> bool:
    """Approve the current manifest identity; drift invalidates this approval."""
    cursor = await db.execute(
        """UPDATE ibkr_contract_registry SET approved=1,
           status=CASE WHEN quote_source='ibkr_live' AND has_history=1
                       THEN 'eligible' ELSE 'qualified_no_live_data' END,
           approval_reason=?, approved_at=datetime('now'), validated_at=datetime('now')
           WHERE instrument_key=? AND status IN ('pending_approval', 'drifted')""",
        (reason.strip(), instrument_key))
    await db.commit()
    return cursor.rowcount == 1


async def eligible_keys(db) -> set[str]:
    rows = await db.fetchall(
        "SELECT instrument_key, manifest_hash FROM ibkr_contract_registry "
        "WHERE status='eligible' AND approved=1 AND has_history=1")
    from auramaur.exchange.ibkr_instruments import BY_KEY
    return {row["instrument_key"] for row in rows
            if row["instrument_key"] in BY_KEY
            and row["manifest_hash"] == BY_KEY[row["instrument_key"]].manifest_hash()}
