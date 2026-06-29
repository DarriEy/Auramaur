"""Polymarket leaderboard — read-only "whale watch" INTELLIGENCE feed.

Reverse-engineering (2026-06-29) showed the profitable Polymarket wallets split
into two machines: DIRECTIONAL conviction (~75% of profit on ~15% of volume;
PnL/vol 20-68%) and MARKET-MAKER/ARB (~85% of volume for ~25% of profit; PnL/vol
<3%). The directional archetype is the one a small model-driven bot could
plausibly learn from. This module pulls the public leaderboard + per-wallet
activity and classifies winners by that PnL/volume ratio.

INTELLIGENCE ONLY — this is NOT a trading signal and does NOT place orders.
Copy-trading is documented to be a money-loser (lagged fills, baitable), so this
exists to WATCH what proven directional winners accumulate, not to mirror them.
A confirmation-gate use (a human/strategy cross-check) could come later.

Public data-api, no auth: GET /v1/leaderboard, /activity, /positions, /value.
Use the proxyWallet (Gnosis Safe) address.
"""

from __future__ import annotations

from dataclasses import dataclass

import aiohttp
import structlog

log = structlog.get_logger()

DATA_API = "https://data-api.polymarket.com"
_TIMEOUT = aiohttp.ClientTimeout(total=20, sock_connect=10, sock_read=15)

# PnL/volume thresholds separating the empirically-observed archetypes.
_DIRECTIONAL_MIN = 0.10   # >=10% edge per dollar traded = conviction/info edge
_MM_ARB_MAX = 0.03        # <3% = thin-margin high-turnover MM/arb


def classify_archetype(pnl: float, vol: float) -> str:
    """Classify a wallet by profit-per-dollar-of-volume (the empirical
    discriminator from the winner reverse-engineering). Pure."""
    if vol <= 0:
        return "unknown"
    ratio = pnl / vol
    if ratio >= _DIRECTIONAL_MIN:
        return "directional"
    if ratio < _MM_ARB_MAX:
        return "mm_arb"
    return "mixed"


@dataclass(frozen=True)
class Leader:
    rank: int
    wallet: str
    name: str
    pnl: float
    vol: float
    archetype: str

    @property
    def pnl_vol_ratio(self) -> float:
        return self.pnl / self.vol if self.vol else 0.0


def parse_leader(entry: dict) -> Leader | None:
    """Turn a raw leaderboard row into a classified Leader, or None if malformed.
    Pure."""
    try:
        pnl = float(entry.get("pnl", 0) or 0)
        vol = float(entry.get("vol", 0) or 0)
        wallet = str(entry.get("proxyWallet", "") or "")
        if not wallet:
            return None
        return Leader(
            rank=int(entry.get("rank", 0) or 0), wallet=wallet,
            name=str(entry.get("userName", "") or ""), pnl=pnl, vol=vol,
            archetype=classify_archetype(pnl, vol),
        )
    except (TypeError, ValueError):
        return None


class PolymarketLeaderboard:
    """Async read-only client for the public Polymarket leaderboard + activity."""

    def __init__(self, session: aiohttp.ClientSession | None = None) -> None:
        self._session = session
        self._owned = session is None

    async def _get(self, path: str, **params):
        sess = self._session or aiohttp.ClientSession(timeout=_TIMEOUT)
        try:
            async with sess.get(f"{DATA_API}{path}", params=params) as r:
                if r.status != 200:
                    log.debug("leaderboard.bad_status", path=path, status=r.status)
                    return None
                return await r.json()
        except Exception as e:
            log.debug("leaderboard.fetch_failed", path=path, error=str(e))
            return None
        finally:
            if self._owned:
                await sess.close()

    async def top(self, *, order_by: str = "PNL", period: str = "ALL",
                  limit: int = 50, offset: int = 0) -> list[Leader]:
        rows = await self._get("/v1/leaderboard", orderBy=order_by,
                               timePeriod=period, limit=min(limit, 50), offset=offset)
        if not isinstance(rows, list):
            return []
        return [ld for r in rows if (ld := parse_leader(r)) is not None]

    async def top_directional(self, **kwargs) -> list[Leader]:
        """The replicable archetype: winners whose edge is conviction, not
        turnover (the lane a small model-driven bot could learn from)."""
        return [ld for ld in await self.top(**kwargs) if ld.archetype == "directional"]

    async def recent_activity(self, wallet: str, *, limit: int = 20) -> list[dict]:
        """A wallet's recent TRADE activity (read-only intelligence). Returns []
        on error."""
        rows = await self._get("/activity", user=wallet, limit=limit)
        if not isinstance(rows, list):
            return []
        return [r for r in rows if str(r.get("type", "TRADE")).upper() == "TRADE"]
