"""Polymarket redemption detection.

Queries Polymarket's data-api for positions held by a proxy wallet and
identifies those ready to redeem — winning conditional tokens in resolved
markets that can be converted back to USDC.

On-chain redemption (CTF.redeemPositions via the Gnosis Safe proxy) lives in
``auramaur.broker.onchain.OnChainRedeemer``. This module is responsible for
*what* is redeemable; that module for *how* to actually redeem.
"""

from __future__ import annotations

from dataclasses import dataclass

import aiohttp
import structlog

log = structlog.get_logger()


POLYMARKET_DATA_API = "https://data-api.polymarket.com/positions"


@dataclass
class VenuePosition:
    """One current holding returned by Polymarket's position API."""

    condition_id: str
    asset_id: str
    title: str
    outcome: str
    size: float
    avg_price: float
    cur_price: float
    initial_value: float
    current_value: float
    cash_pnl: float
    redeemable: bool
    end_date: str
    slug: str
    neg_risk: bool = False
    mergeable: bool = False


async def fetch_current_positions(
    proxy_address: str,
    session: aiohttp.ClientSession | None = None,
    *,
    size_threshold: float = 0.01,
) -> list[VenuePosition]:
    """Return the venue-native current holdings for an account.

    Unlike the CLOB trade-history reconstruction, this endpoint is already
    netted and includes manual/external trades. A successful empty response is
    authoritative; request failures raise and therefore never erase the last
    good database snapshot.
    """
    if not proxy_address:
        raise ValueError("proxy_address is required")
    page_size = 500
    max_offset = 10_000
    close_session = session is None
    if session is None:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20))
    items: list[dict] = []
    offset = 0
    try:
        while True:
            params = {
                "user": proxy_address, "sizeThreshold": str(size_threshold),
                "limit": str(page_size), "offset": str(offset),
                "sortBy": "TOKENS", "sortDirection": "DESC",
            }
            async with session.get(
                POLYMARKET_DATA_API, params=params, timeout=15,
            ) as resp:
                resp.raise_for_status()
                page = await resp.json()
            if not isinstance(page, list) or any(not isinstance(row, dict) for row in page):
                raise ValueError("Polymarket positions response must be a list of objects")
            items.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
            if offset > max_offset:
                raise RuntimeError("Polymarket positions exceeded supported pagination range")
    finally:
        if close_session:
            await session.close()

    positions: list[VenuePosition] = []
    seen_assets: set[str] = set()
    for item in items:
        asset_id = str(item.get("asset", ""))
        condition_id = str(item.get("conditionId", ""))
        if not asset_id or not condition_id:
            raise ValueError("Polymarket position is missing asset or conditionId")
        if asset_id in seen_assets:
            raise ValueError(f"Polymarket returned duplicate asset {asset_id}")
        seen_assets.add(asset_id)
        size = float(item.get("size", 0) or 0)
        if size <= size_threshold:
            continue
        positions.append(VenuePosition(
            condition_id=condition_id, asset_id=asset_id,
            title=str(item.get("title", "")), outcome=str(item.get("outcome", "")),
            size=size, avg_price=float(item.get("avgPrice", 0) or 0),
            cur_price=float(item.get("curPrice", 0) or 0),
            initial_value=float(item.get("initialValue", 0) or 0),
            current_value=float(item.get("currentValue", 0) or 0),
            cash_pnl=float(item.get("cashPnl", 0) or 0),
            redeemable=bool(item.get("redeemable")),
            neg_risk=bool(item.get("negativeRisk") or item.get("negRisk")),
            mergeable=bool(item.get("mergeable")),
            end_date=str(item.get("endDate", "")), slug=str(item.get("slug", "")),
        ))
    return positions


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
    data = await fetch_current_positions(proxy_address, session, size_threshold=0.01)
    results: list[RedeemablePosition] = []
    for item in data:
        size = item.size
        if size <= 0:
            continue

        redeemable_now = item.redeemable
        cur_price = item.cur_price

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
            condition_id=item.condition_id,
            asset_id=item.asset_id,
            title=item.title,
            outcome=item.outcome,
            size=size,
            avg_price=item.avg_price,
            cur_price=cur_price,
            payout=payout,
            is_winner=is_winner,
            redeemable_now=redeemable_now,
            status=status,
            neg_risk=item.neg_risk,
            mergeable=item.mergeable,
            end_date=item.end_date,
            slug=item.slug,
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
# On-chain redemption lives in auramaur.broker.onchain.OnChainRedeemer.
# Implemented for Gnosis Safe v1.3.0 (Polymarket's proxy type) + CTF
# binary markets. NegRisk adapter path is still TODO — positions with
# neg_risk=True are skipped by the redeemer with a clear error.
# --------------------------------------------------------------------------
