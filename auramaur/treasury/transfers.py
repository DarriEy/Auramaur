"""Guarded cross-venue fund transfers — Kraken -> Polymarket (Polygon USDC).

This is the single most dangerous capability in the codebase (it moves real
money off-venue, irreversibly), so it is wrapped in layered guardrails. A real
withdrawal executes ONLY when ALL of the following hold:

  1. Kill switch absent.
  2. settings.transfers_armed  (AURAMAUR_ENABLE_TRANSFERS env + transfers.enabled
     config + no kill switch) — a gate separate from live trading.
  3. Destination is a Kraken withdrawal-address KEY that is BOTH pre-whitelisted
     in the Kraken UI AND listed in settings.transfers.allowed_withdraw_keys.
     (Kraken's API can only send to addresses you pre-approved in the UI, so even
     a leaked key cannot invent a destination.)
  4. amount within [min_transfer_usd, per_transfer_cap_usd].
  5. today's running total + amount <= daily_cap_usd.
  6. an approver callback explicitly returns True (when require_approval).

Anything short of all six → a dry-run preview that moves nothing. Kalshi is a
bank-rail destination and is intentionally not automatable here.

Policy lives here; the signed transport is borrowed from KrakenSpotClient so we
don't duplicate the HMAC.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from auramaur.killswitch import kill_switch_present
from typing import Callable

import structlog
from pydantic import BaseModel

log = structlog.get_logger()

_LEDGER = Path("data/transfer_ledger.sqlite3")

# An approver receives a summary dict and returns True to authorize. The default
# denies — approval must be an explicit, deliberate act.
Approver = Callable[[dict], bool]


def _deny(_summary: dict) -> bool:
    return False


class TransferResult(BaseModel):
    status: str  # "executed" | "dry_run" | "blocked" | "rejected"
    reason: str = ""
    asset: str = ""
    amount: float = 0.0
    dest_key: str = ""
    refid: str = ""


class TransferManager:
    def __init__(self, settings, kraken_client, ledger_path: Path = _LEDGER):
        self._settings = settings
        self._kraken = kraken_client
        self._ledger_path = ledger_path

    # ------------------------------------------------------------------
    # Daily ledger (persisted so the daily cap survives restarts)
    # ------------------------------------------------------------------

    def _today(self) -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def _connect_ledger(self) -> sqlite3.Connection:
        """Open the enforcement ledger; errors intentionally propagate closed."""
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self._ledger_path, timeout=30, isolation_level=None)
        db.execute("PRAGMA busy_timeout=30000")
        db.execute(
            """CREATE TABLE IF NOT EXISTS transfer_reservations (
                   id TEXT PRIMARY KEY,
                   day TEXT NOT NULL,
                   asset TEXT NOT NULL,
                   dest_key TEXT NOT NULL,
                   amount_usd REAL NOT NULL CHECK(amount_usd > 0),
                   status TEXT NOT NULL CHECK(status IN ('reserved', 'executed')),
                   refid TEXT NOT NULL DEFAULT '',
                   created_at TEXT NOT NULL DEFAULT (datetime('now'))
               )"""
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS transfer_ledger_meta "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._migrate_legacy_ledger(db)
        return db

    def _migrate_legacy_ledger(self, db: sqlite3.Connection) -> None:
        """Import the old JSON totals once; malformed state fails closed."""
        marker = db.execute(
            "SELECT 1 FROM transfer_ledger_meta WHERE key = 'legacy_json_migrated'"
        ).fetchone()
        if marker:
            return
        legacy_path = self._ledger_path.with_suffix(".json")
        if legacy_path.exists():
            try:
                legacy = json.loads(legacy_path.read_text())
                if not isinstance(legacy, dict):
                    raise ValueError("root must be an object")
                rows = []
                for day, amount in legacy.items():
                    amount_f = float(amount)
                    if amount_f < 0:
                        raise ValueError(f"negative amount for {day}")
                    if amount_f:
                        rows.append((f"legacy-{day}", str(day), amount_f))
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as e:
                raise RuntimeError(f"legacy transfer ledger is invalid: {e}") from e
            db.executemany(
                "INSERT OR IGNORE INTO transfer_reservations "
                "(id, day, asset, dest_key, amount_usd, status) "
                "VALUES (?, ?, 'USDC', 'legacy-json', ?, 'executed')",
                rows,
            )
        db.execute(
            "INSERT INTO transfer_ledger_meta (key, value) "
            "VALUES ('legacy_json_migrated', datetime('now'))"
        )

    def _spent_today(self) -> float:
        db = self._connect_ledger()
        try:
            row = db.execute(
                "SELECT COALESCE(SUM(amount_usd), 0) FROM transfer_reservations "
                "WHERE day = ? AND status IN ('reserved', 'executed')",
                (self._today(),),
            ).fetchone()
            return float(row[0])
        finally:
            db.close()

    def _reserve(self, reservation_id: str, asset: str, dest_key: str,
                 amount_usd: float, daily_cap_usd: float) -> float:
        """Atomically enforce the cap and reserve allowance before withdrawal."""
        db = self._connect_ledger()
        try:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT COALESCE(SUM(amount_usd), 0) FROM transfer_reservations "
                "WHERE day = ? AND status IN ('reserved', 'executed')",
                (self._today(),),
            ).fetchone()
            spent = float(row[0])
            if spent + amount_usd > daily_cap_usd:
                db.rollback()
                raise ValueError(
                    f"daily cap ${daily_cap_usd:.2f} would be exceeded "
                    f"(${spent:.2f} already moved or reserved today)"
                )
            db.execute(
                "INSERT INTO transfer_reservations "
                "(id, day, asset, dest_key, amount_usd, status) "
                "VALUES (?, ?, ?, ?, ?, 'reserved')",
                (reservation_id, self._today(), asset, dest_key, amount_usd),
            )
            db.commit()
            return spent
        finally:
            db.close()

    def _release(self, reservation_id: str) -> None:
        db = self._connect_ledger()
        try:
            db.execute(
                "DELETE FROM transfer_reservations WHERE id = ? AND status = 'reserved'",
                (reservation_id,),
            )
            db.commit()
        finally:
            db.close()

    def _mark_executed(self, reservation_id: str, refid: str) -> None:
        db = self._connect_ledger()
        try:
            cursor = db.execute(
                "UPDATE transfer_reservations SET status = 'executed', refid = ? "
                "WHERE id = ? AND status = 'reserved'",
                (refid, reservation_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("transfer reservation missing during finalization")
            db.commit()
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Transfer
    # ------------------------------------------------------------------

    async def transfer(
        self,
        dest_key: str,
        amount_usd: float,
        asset: str = "USDC",
        approver: Approver = _deny,
    ) -> TransferResult:
        """Move `amount_usd` of `asset` to the whitelisted Kraken address `dest_key`.

        Runs every guardrail; only executes a real Kraken Withdraw when all pass
        and the approver authorizes. Otherwise returns a dry-run preview.
        """
        cfg = self._settings.transfers

        def blocked(reason: str) -> TransferResult:
            log.warning("transfer.blocked", dest_key=dest_key, amount=amount_usd, reason=reason)
            return TransferResult(status="blocked", reason=reason, asset=asset,
                                  amount=amount_usd, dest_key=dest_key)

        # 1. Kill switch.
        if kill_switch_present():
            return blocked("kill switch active")

        # 2. Whitelist (config side). Kraken UI is the second, independent gate.
        if dest_key not in cfg.allowed_withdraw_keys:
            return blocked(f"'{dest_key}' not in transfers.allowed_withdraw_keys "
                           f"{cfg.allowed_withdraw_keys}")

        # 3. Amount bounds.
        if amount_usd < cfg.min_transfer_usd:
            return blocked(f"${amount_usd:.2f} below min ${cfg.min_transfer_usd:.2f}")
        if amount_usd > cfg.per_transfer_cap_usd:
            return blocked(f"${amount_usd:.2f} over per-transfer cap ${cfg.per_transfer_cap_usd:.2f}")

        # 4. Armed? If not, this is a preview — nothing moves.
        if not self._settings.transfers_armed:
            spent = self._spent_today()
            summary = {"asset": asset, "amount_usd": amount_usd, "dest_key": dest_key,
                       "spent_today": spent, "daily_cap": cfg.daily_cap_usd}
            log.info("transfer.dry_run", reason="transfers not armed", **summary)
            return TransferResult(status="dry_run", asset=asset, amount=amount_usd,
                                  dest_key=dest_key,
                                  reason="transfers not armed (set AURAMAUR_ENABLE_TRANSFERS "
                                         "+ transfers.enabled) — preview only, nothing moved")

        # 5. Human approval. Approval precedes reservation so a rejected
        # preview cannot consume daily allowance.
        try:
            spent_preview = self._spent_today()
        except (OSError, sqlite3.Error, RuntimeError) as e:
            return blocked(f"transfer ledger unavailable; refusing withdrawal: {e}")
        preview = {"asset": asset, "amount_usd": amount_usd, "dest_key": dest_key,
                   "spent_today": spent_preview, "daily_cap": cfg.daily_cap_usd}
        if cfg.require_approval and not approver(preview):
            return TransferResult(status="rejected", asset=asset, amount=amount_usd,
                                  dest_key=dest_key, reason="approval not granted")

        # 6. Reserve daily allowance in one cross-process transaction. A process
        # crash after this point leaves a reservation that counts against the
        # cap (fail closed) until an operator reconciles it.
        reservation_id = uuid.uuid4().hex
        try:
            spent = self._reserve(
                reservation_id, asset, dest_key, amount_usd, cfg.daily_cap_usd,
            )
        except (OSError, sqlite3.Error, RuntimeError) as e:
            return blocked(f"transfer ledger unavailable; refusing withdrawal: {e}")
        except ValueError as e:
            return blocked(str(e))

        summary = {"asset": asset, "amount_usd": amount_usd, "dest_key": dest_key,
                   "spent_today": spent, "daily_cap": cfg.daily_cap_usd,
                   "reservation_id": reservation_id}

        # === EXECUTE — real, irreversible withdrawal ===
        log.warning("transfer.executing", **summary)
        try:
            resp = await self._kraken._private(
                "Withdraw", {"asset": asset, "key": dest_key, "amount": str(amount_usd)},
            )
        except Exception:
            # An exception can be an uncertain network outcome: retain the
            # reservation so a retry cannot exceed the cap or double-withdraw.
            log.exception("transfer.outcome_uncertain", **summary)
            return TransferResult(status="rejected", asset=asset, amount=amount_usd,
                                  dest_key=dest_key,
                                  reason="withdrawal outcome uncertain; reservation retained")
        if resp.get("error"):
            self._release(reservation_id)
            return TransferResult(status="rejected", asset=asset, amount=amount_usd,
                                  dest_key=dest_key, reason=str(resp["error"])[:200])

        refid = str(resp.get("result", {}).get("refid", ""))
        if not refid:
            log.error("transfer.missing_refid", **summary)
            return TransferResult(status="rejected", asset=asset, amount=amount_usd,
                                  dest_key=dest_key,
                                  reason="withdrawal response missing refid; reservation retained")
        self._mark_executed(reservation_id, refid)
        log.warning("transfer.executed", refid=refid, **summary)
        return TransferResult(status="executed", asset=asset, amount=amount_usd,
                              dest_key=dest_key, refid=refid)
