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
# A balance row older than this, while fetches are failing, is reported as
# an error rather than a warning: the displayed number is no longer real.
STALE_BALANCE_ALERT_HOURS = 6.0


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


async def _row_age_hours(db, venue: str) -> float | None:
    """Age of the persisted balance row, so a repeated failure can escalate.

    A fetch error alone is routine (a venue blips); a fetch error standing on
    top of a day-old row means the number an operator is reading is fiction.
    """
    try:
        row = await db.fetchone(
            "SELECT fetched_at FROM venue_balances WHERE venue = ?", (venue,))
        if not row or not row["fetched_at"]:
            return None
        fetched = datetime.fromisoformat(str(row["fetched_at"]))
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - fetched).total_seconds() / 3600.0
    except Exception:  # noqa: BLE001 — never let observability break the loop
        return None


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
            #
            # Always carry the exception TYPE. asyncio.TimeoutError — the most
            # common failure here, since every call is wrapped in wait_for —
            # stringifies to "", so the upgrade to warning above still emitted
            # `error: ""` 153 times on 2026-07-27 while the IBKR row sat two
            # days stale. A warning that cannot say what went wrong is the same
            # invisibility in a louder font.
            log.warning("balance_recorder.fetch_error", venue=venue,
                        error_type=type(exc).__name__,
                        error=str(exc)[:120] or type(exc).__name__)
            stale_age = await _row_age_hours(db, venue)
            if stale_age is not None and stale_age >= STALE_BALANCE_ALERT_HOURS:
                log.error("balance_recorder.stale_balance", venue=venue,
                          age_hours=round(stale_age, 1),
                          error_type=type(exc).__name__)
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
