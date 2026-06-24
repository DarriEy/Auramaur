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
from datetime import datetime, timezone
from pathlib import Path
from auramaur.killswitch import kill_switch_present
from typing import Callable

import structlog
from pydantic import BaseModel

log = structlog.get_logger()

_LEDGER = Path("data/transfer_ledger.json")

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

    def _spent_today(self) -> float:
        try:
            data = json.loads(self._ledger_path.read_text())
        except (OSError, json.JSONDecodeError):
            return 0.0
        return float(data.get(self._today(), 0.0))

    def _record(self, amount_usd: float) -> None:
        try:
            data = json.loads(self._ledger_path.read_text())
        except (OSError, json.JSONDecodeError):
            data = {}
        data[self._today()] = float(data.get(self._today(), 0.0)) + amount_usd
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._ledger_path.write_text(json.dumps(data, indent=2))

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

        # 4. Daily cap.
        spent = self._spent_today()
        if spent + amount_usd > cfg.daily_cap_usd:
            return blocked(f"daily cap ${cfg.daily_cap_usd:.2f} would be exceeded "
                           f"(${spent:.2f} already moved today)")

        summary = {"asset": asset, "amount_usd": amount_usd, "dest_key": dest_key,
                   "spent_today": spent, "daily_cap": cfg.daily_cap_usd}

        # 5. Armed? If not, this is a preview — nothing moves.
        if not self._settings.transfers_armed:
            log.info("transfer.dry_run", reason="transfers not armed", **summary)
            return TransferResult(status="dry_run", asset=asset, amount=amount_usd,
                                  dest_key=dest_key,
                                  reason="transfers not armed (set AURAMAUR_ENABLE_TRANSFERS "
                                         "+ transfers.enabled) — preview only, nothing moved")

        # 6. Human approval.
        if cfg.require_approval and not approver(summary):
            return TransferResult(status="rejected", asset=asset, amount=amount_usd,
                                  dest_key=dest_key, reason="approval not granted")

        # === EXECUTE — real, irreversible withdrawal ===
        log.warning("transfer.executing", **summary)
        resp = await self._kraken._private(
            "Withdraw", {"asset": asset, "key": dest_key, "amount": str(amount_usd)},
        )
        if resp.get("error"):
            return TransferResult(status="rejected", asset=asset, amount=amount_usd,
                                  dest_key=dest_key, reason=str(resp["error"])[:200])

        refid = str(resp.get("result", {}).get("refid", ""))
        self._record(amount_usd)
        log.warning("transfer.executed", refid=refid, **summary)
        return TransferResult(status="executed", asset=asset, amount=amount_usd,
                              dest_key=dest_key, refid=refid)
