"""Polymarket redemption detection.

Queries Polymarket's data-api for positions held by a proxy wallet and
identifies those ready to redeem — winning conditional tokens in resolved
markets that can be converted back to USDC.

On-chain redemption (calling CTF.redeemPositions via the Gnosis Safe proxy)
is not yet implemented here.  See the TODO at the bottom of this file.
For now this module surfaces what's redeemable so the user can act on it
via the Polymarket UI or an on-chain script.
"""

from __future__ import annotations

from dataclasses import dataclass

import aiohttp
import structlog

log = structlog.get_logger()


POLYMARKET_DATA_API = "https://data-api.polymarket.com/positions"


@dataclass
class RedeemablePosition:
    """A Polymarket position's redemption / resolution state."""

    condition_id: str
    asset_id: str
    title: str
    outcome: str          # "Yes" or "No" — which side we hold
    size: float           # Number of conditional tokens held
    avg_price: float      # Average entry price
    cur_price: float      # Current market price of our token
    payout: float         # Expected USDC at redemption (size if winning, 0 if losing)
    is_winner: bool       # True if this side resolved in our favor
    redeemable_now: bool  # True if on-chain redemption is unlocked
    status: str           # "redeemable", "pending_oracle", "open"
    neg_risk: bool        # NegRisk markets redeem via NegRiskAdapter, not CTF
    mergeable: bool       # Holds both YES and NO — can merge to recover USDC
    end_date: str         # Market end date (ISO string)
    slug: str             # URL slug for Polymarket UI link

    @property
    def polymarket_url(self) -> str:
        return f"https://polymarket.com/event/{self.slug}" if self.slug else ""

    @property
    def cost_basis(self) -> float:
        return self.size * self.avg_price

    @property
    def realized_pnl(self) -> float:
        return self.payout - self.cost_basis


async def fetch_redeemable_positions(
    proxy_address: str,
    session: aiohttp.ClientSession | None = None,
    include_pending: bool = True,
) -> list[RedeemablePosition]:
    """Fetch redeemable and resolved-but-pending positions for a proxy wallet.

    Polymarket's data-api exposes ``redeemable`` per position, which flips
    to true once the UMA oracle confirms resolution and tokens can be
    converted to USDC on-chain.

    If *include_pending* is True, also returns positions whose market has
    effectively resolved (price at 0 or 1) but are still inside the oracle
    confirmation window — so the user can see what's coming soon.
    """
    if not proxy_address:
        raise ValueError("proxy_address is required")

    close_session = session is None
    if session is None:
        session = aiohttp.ClientSession()

    try:
        params = {"user": proxy_address, "sizeThreshold": "0.01"}
        async with session.get(POLYMARKET_DATA_API, params=params, timeout=15) as resp:
            resp.raise_for_status()
            data = await resp.json()
    finally:
        if close_session:
            await session.close()

    results: list[RedeemablePosition] = []
    for item in data:
        size = float(item.get("size", 0) or 0)
        if size <= 0:
            continue

        redeemable_now = bool(item.get("redeemable"))
        cur_price = float(item.get("curPrice", 0.5) or 0.5)

        # "Effectively resolved" = curPrice converged to 0 or 1.  Polymarket's
        # data-api returns curPrice in the range 0..1 relative to the *user's
        # outcome* (so 1.0 means we're winning, 0.0 means we're losing).
        is_effectively_resolved = cur_price >= 0.99 or cur_price <= 0.01
        is_winner = cur_price >= 0.99

        if redeemable_now:
            status = "redeemable"
        elif is_effectively_resolved and include_pending:
            status = "pending_oracle"
        else:
            continue  # still an open position, not relevant here

        payout = size if is_winner else 0.0

        results.append(RedeemablePosition(
            condition_id=str(item.get("conditionId", "")),
            asset_id=str(item.get("asset", "")),
            title=str(item.get("title", "")),
            outcome=str(item.get("outcome", "")),
            size=size,
            avg_price=float(item.get("avgPrice", 0) or 0),
            cur_price=cur_price,
            payout=payout,
            is_winner=is_winner,
            redeemable_now=redeemable_now,
            status=status,
            neg_risk=bool(item.get("negativeRisk") or item.get("negRisk") or False),
            mergeable=bool(item.get("mergeable") or False),
            end_date=str(item.get("endDate", "")),
            slug=str(item.get("slug", "")),
        ))

    log.info(
        "redeemer.fetched",
        proxy=proxy_address[:10],
        redeemable_now=sum(1 for p in results if p.redeemable_now),
        pending_oracle=sum(1 for p in results if p.status == "pending_oracle"),
        total_payout_redeemable=round(
            sum(p.payout for p in results if p.redeemable_now), 2,
        ),
    )
    return results


def summarize_redemptions(positions: list[RedeemablePosition]) -> dict:
    """Produce a summary of redeemable positions."""
    now_redeemable = [p for p in positions if p.redeemable_now]
    pending = [p for p in positions if p.status == "pending_oracle"]

    winners_now = [p for p in now_redeemable if p.is_winner]
    winners_pending = [p for p in pending if p.is_winner]

    return {
        "redeemable_now": len(now_redeemable),
        "pending_oracle": len(pending),
        "winning_now": len(winners_now),
        "winning_pending": len(winners_pending),
        "payout_now_usdc": round(sum(p.payout for p in winners_now), 2),
        "payout_pending_usdc": round(sum(p.payout for p in winners_pending), 2),
        "cost_basis_now": round(sum(p.cost_basis for p in now_redeemable), 2),
        "net_pnl_now": round(sum(p.realized_pnl for p in now_redeemable), 2),
        "neg_risk_count": sum(1 for p in positions if p.neg_risk),
    }


# --------------------------------------------------------------------------
# TODO: on-chain redemption via Gnosis Safe proxy
# --------------------------------------------------------------------------
# The remaining work to make this fully automated:
#
# 1. Import web3.py and connect to a Polygon RPC.
# 2. For regular markets: build a CTF.redeemPositions() call.
#    Contract: 0x4D97DCd97eC945f40cF65F87097ACe5EA0476045
#    Args: (collateralToken=USDC, parentCollectionId=bytes32(0),
#           conditionId, indexSets=[1,2])
# 3. For NegRisk markets: build a NegRiskAdapter.redeemPositions() call.
#    Contract: 0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296
# 4. Because Polymarket uses a Gnosis Safe proxy wallet (signature_type=2),
#    the call must be wrapped in a Safe transaction:
#    - build exec_transaction payload
#    - sign with the EOA private key (settings.polygon_private_key)
#    - submit to the Safe
# 5. Group multiple redemptions into a single Safe batch transaction to
#    save gas (one MultiSend call instead of N individual calls).
# 6. Verify USDC balance increased after receipt, then mark in DB.
# --------------------------------------------------------------------------
