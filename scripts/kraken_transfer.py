"""Move funds Kraken -> Polymarket (Polygon USDC) through the guarded path.

DRY-RUN by default — it previews and moves nothing. A real withdrawal needs ALL
of: AURAMAUR_ENABLE_TRANSFERS=true + transfers.enabled + the dest in
transfers.allowed_withdraw_keys + the address whitelisted in the Kraken UI +
within caps + --execute + an explicit typed confirmation.

  Preview:  python scripts/kraken_transfer.py --to polymarket-usdc --amount 50
  Execute:  python scripts/kraken_transfer.py --to polymarket-usdc --amount 50 --execute
"""

from __future__ import annotations

import argparse
import asyncio

from config.settings import Settings
from auramaur.exchange.kraken import KrakenSpotClient
from auramaur.treasury.transfers import TransferManager


def _build_approver(amount_usd: float, assume_yes: bool):
    def approver(summary: dict) -> bool:
        if assume_yes:
            return True
        try:
            ans = input(
                f"\nType the amount ({amount_usd:.2f}) to CONFIRM this real, "
                f"irreversible withdrawal: "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            return False
        return ans == f"{amount_usd:.2f}" or ans == str(amount_usd)
    return approver


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", required=True, help="Kraken withdrawal-address key (whitelisted)")
    ap.add_argument("--amount", required=True, type=float, help="amount in USD/USDC")
    ap.add_argument("--asset", default="USDC")
    ap.add_argument("--execute", action="store_true", help="attempt a real withdrawal")
    ap.add_argument("--yes", action="store_true", help="skip the typed confirmation prompt")
    args = ap.parse_args()

    s = Settings()
    cfg = s.transfers
    mgr = TransferManager(s, KrakenSpotClient(s))

    print("Transfer gate state:")
    print(f"  transfers_armed     : {s.transfers_armed}  "
          f"(env={s.auramaur_enable_transfers}, config.enabled={cfg.enabled}, "
          f"kill_switch={s.kill_switch_active})")
    print(f"  whitelist           : {cfg.allowed_withdraw_keys}")
    print(f"  caps                : per ${cfg.per_transfer_cap_usd:.0f}, daily ${cfg.daily_cap_usd:.0f}")
    print(f"  spent today         : ${mgr._spent_today():.2f}")
    print(f"\nRequest: {args.amount} {args.asset} -> '{args.to}'  "
          f"({'EXECUTE' if args.execute else 'PREVIEW'})\n")

    approver = _build_approver(args.amount, args.yes) if args.execute else (lambda _s: False)
    res = await mgr.transfer(args.to, args.amount, asset=args.asset, approver=approver)
    await mgr._kraken.close()

    print(f"Result: {res.status.upper()}")
    if res.reason:
        print(f"  reason: {res.reason}")
    if res.refid:
        print(f"  refid : {res.refid}")


if __name__ == "__main__":
    asyncio.run(main())
