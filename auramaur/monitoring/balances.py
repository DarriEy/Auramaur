"""Venue balance fetchers + the bot-side recorder.

The web dashboard is read-only by construction and must never hold venue
credentials, so it cannot ask the venues for balances itself. Instead the bot
(which already holds the credentials) records each venue's cash into the
``venue_balances`` table, and any ``mode=ro`` consumer displays the row along
with its age. A fetch failure deliberately leaves the last good row in place:
a balance that is 20 minutes old and says so is more useful than "—".

The per-venue fetchers raise on failure (callers choose between "keep last
good" and "show a dash") and are shared with the TUI cockpit so the formatted
strings can never diverge between the two views.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import structlog

log = structlog.get_logger()

# Venue REST calls carry this timeout so a wedged venue can't stall a caller.
FETCH_TIMEOUT_S = 8


async def kalshi_balance(settings) -> str:
    from auramaur.exchange.kalshi import KalshiClient
    kc = KalshiClient(settings=settings, paper_trader=None)
    try:
        return f"${await asyncio.wait_for(kc.get_balance(), FETCH_TIMEOUT_S):.2f}"
    finally:
        await kc.close()


async def kraken_balance(settings) -> str:
    from auramaur.exchange.kraken import KrakenSpotClient
    kk = KrakenSpotClient(settings)
    try:
        bal = await asyncio.wait_for(kk.get_balance(), FETCH_TIMEOUT_S)
        if not bal:
            # get_balance collapses API errors to {} — that's a failed fetch,
            # not an account holding zero of every asset.
            raise RuntimeError("kraken returned no balances")
        usdc = bal.get("USDC", 0.0)
        cad = bal.get("ZCAD", 0.0)
        # Dust floor: venue rounding residues (AVAX 4e-7 etc.) read as open
        # spec positions on the console/dashboard (2026-07-20 audit). No
        # price feed here, so a quantity floor: real entries are whole-ish
        # units, dust is <=1e-4.
        crypto = [a for a, v in bal.items()
                  if v >= 0.01 and a != "USDC" and not a.startswith("Z")]
        return (f"${usdc:.0f} USDC + {cad:.0f} CAD"
                + (f" | spec: {','.join(crypto)}" if crypto else ""))
    finally:
        await kk.close()


async def ibkr_balance(settings) -> str:
    """Account cash/net-liq via a short-lived read-only gateway session on its
    own clientId (it must never bump a trading or quote session)."""
    from ib_async import IB
    cfg = settings.ibkr
    port = cfg.paper_port if cfg.environment == "paper" else cfg.live_port
    ib = IB()
    await asyncio.wait_for(
        ib.connectAsync(host=cfg.host, port=port,
                        clientId=cfg.balance_client_id, readonly=True),
        FETCH_TIMEOUT_S + 2,
    )
    try:
        rows = await asyncio.wait_for(ib.accountSummaryAsync(), FETCH_TIMEOUT_S)
        tags = {r.tag: (r.value, r.currency) for r in rows}
        if "NetLiquidation" not in tags:
            raise RuntimeError("ibkr account summary missing NetLiquidation")
        avail = float(tags.get("AvailableFunds", ("0", ""))[0] or 0.0)
        netliq, ccy = tags["NetLiquidation"]
        detail = f"${avail:,.2f} avail | ${float(netliq or 0.0):,.2f} net"
        return detail if ccy in ("", "USD") else f"{detail} {ccy}"
    finally:
        ib.disconnect()


async def record_venue_balances(db, settings, *, include_ibkr: bool = True) -> None:
    """Upsert one ``venue_balances`` row per enabled venue.

    ``include_ibkr`` lets the caller run the gateway connect on a slower
    cadence than the cheap REST venues.
    """
    fetchers = {}
    if settings.kalshi.enabled:
        fetchers["kalshi"] = kalshi_balance
    if settings.kraken.enabled:
        fetchers["kraken"] = kraken_balance
    if include_ibkr and settings.ibkr.enabled:
        fetchers["ibkr"] = ibkr_balance

    # All venue fetches complete BEFORE the first write: a slow/hung venue
    # API (the 2026-07-21 gateway outage ran 8h) must never be awaited while
    # the shared connection holds an open write transaction.
    fetched: list[tuple[str, str]] = []
    for venue, fetch in fetchers.items():
        try:
            detail = await fetch(settings)
        except Exception as exc:  # noqa: BLE001 — keep last good row
            # warning, not debug: the stale row silently stood in for a live
            # balance for 8h during the 2026-07-21 gateway 2FA outage — the
            # age column "told the story" but nothing surfaced it.
            log.warning("balance_recorder.fetch_error", venue=venue,
                        error=str(exc)[:120])
            continue
        fetched.append((venue, detail))
    if fetched:
        for venue, detail in fetched:
            await db.execute(
                """INSERT INTO venue_balances (venue, detail, fetched_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(venue) DO UPDATE SET
                       detail = excluded.detail, fetched_at = excluded.fetched_at""",
                (venue, detail, datetime.now(timezone.utc).isoformat()),
            )
        await db.commit()
