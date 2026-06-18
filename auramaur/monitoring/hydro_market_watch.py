"""Hydrology market watcher — alert when a tradeable water market appears.

Kalshi/Polymarket don't currently list liquid hydrology markets (streamflow,
drought, reservoir level, flood stage, snowpack), so the compHydro data moat
(CSFS streamflow, COS water-obs, CAS attributes) has nothing to price. This
watcher flips that into "armed and waiting": each cycle it scans both venues
and ALERTS (no trading) the first time a genuine hydrology market shows up with
liquidity, so the moat can be deployed.

The matcher is deliberately precise — the naive substring scan false-positived
on "Aust*rain*" / "*Rain*bow Six" — so it requires word-boundaried,
hydrology-specific terms.
"""

from __future__ import annotations

import re

import structlog

log = structlog.get_logger()

# Word-boundaried, hydrology-specific terms. Excluded as ambiguous: bare
# "rain"/"snow"/"water" (Austrian/Rainbow/snowboard), "runoff" (ELECTION
# runoff — the first false positive in the wild), and "swe" (abbreviation).
# Only terms that unambiguously denote a water/hydrology event qualify.
_HYDRO_RE = re.compile(
    r"\b("
    r"streamflow|snowmelt|reservoir|snowpack|snow water equivalent|"
    r"drought|flood|flooding|flood stage|"
    r"water level|river level|lake level|river flow|"
    r"lake mead|lake powell|colorado river|mississippi river|"
    r"rainfall|precipitation|snowfall"
    r")\b",
    re.I,
)


def is_hydro_market(text: str) -> bool:
    """True if the text denotes a hydrology/water market (precise, not substring).

    Matched against the market QUESTION only — descriptions carry verbose
    boilerplate (e.g. "a runoff election will be held") that false-positives.
    """
    return bool(_HYDRO_RE.search(text or ""))


class HydroMarketWatcher:
    def __init__(self, db, settings, discoveries: dict, alerts) -> None:
        self._db = db
        self._settings = settings
        self._discoveries = discoveries or {}
        self._alerts = alerts

    async def _ensure_table(self) -> None:
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS hydro_watch_seen "
            "(market_id TEXT PRIMARY KEY, venue TEXT, question TEXT, first_seen TEXT)")
        await self._db.commit()

    async def run_once(self) -> int:
        cfg = self._settings.hydro_watch
        if not cfg.enabled or self._alerts is None:
            return 0
        await self._ensure_table()
        new = 0
        for venue, disc in self._discoveries.items():
            try:
                markets = await disc.get_markets(limit=cfg.scan_limit)
            except Exception as e:
                log.debug("hydro_watch.scan_error", venue=venue, error=str(e))
                continue
            for m in markets:
                # Question/title only — descriptions false-positive on boilerplate
                # ("a runoff election will be held").
                if not is_hydro_market(m.question or ""):
                    continue
                liq = max(float(m.liquidity or 0), float(m.volume or 0))
                if liq < cfg.min_liquidity:
                    continue
                row = await self._db.fetchone(
                    "SELECT 1 FROM hydro_watch_seen WHERE market_id = ?", (m.id,))
                if row is not None:
                    continue  # already alerted
                await self._db.execute(
                    "INSERT OR IGNORE INTO hydro_watch_seen (market_id, venue, question, first_seen) "
                    "VALUES (?, ?, ?, datetime('now'))",
                    (m.id, venue, (m.question or "")[:200]))
                await self._db.commit()
                await self._alerts.send(
                    f"🌊 Hydrology market listed on {venue}: "
                    f"{(m.question or '')[:120]} (liq ~${liq:.0f}) — CSFS/COS data is ready to price it.",
                    level="info")
                log.info("hydro_watch.new_market", venue=venue, market_id=m.id,
                         question=(m.question or "")[:80])
                new += 1
        if new:
            log.info("hydro_watch.cycle_done", new=new)
        return new
