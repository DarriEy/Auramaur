"""Deribit implied-volatility term structure — vol_anchor's market sigma.

vol_anchor's original sigma was an ESTIMATE: recent realized vol blended
toward a calibrated long-run anchor. Deribit publishes the market-clearing
implied-vol surface for the same underlyings (the deepest crypto vol venue),
free and unauthenticated. Reading ATM IV per expiry off the option book and
interpolating to a market's horizon replaces the estimate with the price
professionals actually clear at — and turns the pillar's edge claim into a
directly measurable spread: prediction-market implied vol vs Deribit IV.

Read-only by construction: one public GET per currency, no credentials, no
order API in this module. Failure of any kind returns None and the pillar
falls back to the calibrated blend (the pre-Deribit behavior), logging which
source priced each market so records stay interpretable.

Term-structure math: per expiry, take the strike nearest the index (ATM) and
average the closest call/put mark IVs; interpolate between tenors LINEARLY IN
TOTAL VARIANCE (w = sigma^2 * T, the standard no-arbitrage-friendly scheme);
clamp flat outside the quoted range.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import aiohttp
import structlog

log = structlog.get_logger()

_API = "https://www.deribit.com/api/v2"

# "ETH-26SEP26-1600-C" -> (expiry, strike, kind)
_INSTRUMENT_RE = re.compile(
    r"^[A-Z]+-(\d{1,2})([A-Z]{3})(\d{2})-(\d+(?:d\d+)?)-([CP])$")

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def parse_instrument(name: str) -> tuple[datetime, float, str] | None:
    """(expiry_utc, strike, 'C'|'P') from a Deribit option name, else None."""
    m = _INSTRUMENT_RE.match(name or "")
    if not m:
        return None
    day, mon, yy, strike_s, kind = m.groups()
    month = _MONTHS.get(mon)
    if month is None:
        return None
    try:
        expiry = datetime(2000 + int(yy), month, int(day), 8, 0,
                          tzinfo=timezone.utc)  # Deribit expires 08:00 UTC
        strike = float(strike_s.replace("d", "."))
    except ValueError:
        return None
    return expiry, strike, kind


def atm_term_structure(
    summaries: list[dict], index_price: float, now: datetime,
) -> list[tuple[float, float]]:
    """[(t_years, sigma)] sorted by tenor, from a book summary list.

    Per expiry: the strike nearest the index; sigma = mean of that strike's
    call/put mark_iv (single side accepted). Entries without a usable
    mark_iv are skipped. Tenors under ~12h are dropped (expiry-noise IVs).
    """
    by_expiry: dict[datetime, dict[float, list[float]]] = {}
    for s in summaries:
        parsed = parse_instrument(s.get("instrument_name", ""))
        iv = s.get("mark_iv")
        if parsed is None or iv is None or iv <= 0:
            continue
        expiry, strike, _kind = parsed
        by_expiry.setdefault(expiry, {}).setdefault(strike, []).append(
            float(iv) / 100.0)

    out: list[tuple[float, float]] = []
    for expiry, strikes in by_expiry.items():
        t_years = (expiry - now).total_seconds() / (365.0 * 86400.0)
        if t_years < 0.5 / 365.0:
            continue
        atm_strike = min(strikes, key=lambda k: abs(k - index_price))
        ivs = strikes[atm_strike]
        out.append((t_years, sum(ivs) / len(ivs)))
    out.sort()
    return out


def interp_sigma(term: list[tuple[float, float]], t_years: float) -> float | None:
    """Sigma at t_years — linear in total variance between quoted tenors,
    flat-clamped outside the range. None on an empty structure."""
    if not term or t_years <= 0:
        return None
    if t_years <= term[0][0]:
        return term[0][1]
    if t_years >= term[-1][0]:
        return term[-1][1]
    for (t0, s0), (t1, s1) in zip(term, term[1:]):
        if t0 <= t_years <= t1:
            w0, w1 = s0 * s0 * t0, s1 * s1 * t1
            w = w0 + (w1 - w0) * (t_years - t0) / (t1 - t0)
            return (w / t_years) ** 0.5
    return term[-1][1]


@dataclass
class _Cached:
    at: float
    term: list[tuple[float, float]]


class DeribitIVSource:
    """ATM IV term structures per currency, cached with a TTL."""

    def __init__(self, currencies: dict[str, str], ttl_seconds: float = 1800.0,
                 timeout_s: float = 20.0) -> None:
        self._currencies = currencies      # coingecko id -> deribit currency
        self._ttl = ttl_seconds
        self._timeout = timeout_s
        self._cache: dict[str, _Cached] = {}

    async def _fetch_json(self, url: str) -> dict:
        async with aiohttp.ClientSession() as s:
            async with s.get(url, timeout=aiohttp.ClientTimeout(
                    total=self._timeout)) as r:
                r.raise_for_status()
                return await r.json()

    async def _term_for_currency(self, ccy: str) -> list[tuple[float, float]]:
        cached = self._cache.get(ccy)
        if cached and (time.monotonic() - cached.at) < self._ttl:
            return cached.term
        idx = await self._fetch_json(
            f"{_API}/public/get_index_price?index_name={ccy.lower()}_usd")
        index_price = float(idx["result"]["index_price"])
        book = await self._fetch_json(
            f"{_API}/public/get_book_summary_by_currency"
            f"?currency={ccy}&kind=option")
        term = atm_term_structure(
            book["result"], index_price, datetime.now(timezone.utc))
        self._cache[ccy] = _Cached(time.monotonic(), term)
        log.info("deribit_iv.term_refreshed", currency=ccy,
                 tenors=len(term),
                 near=round(term[0][1], 3) if term else None,
                 far=round(term[-1][1], 3) if term else None)
        return term

    async def term_sigma(self, cg_id: str, t_years: float) -> float | None:
        """Market IV at the horizon, or None (asset uncovered / API down /
        empty structure) — the caller falls back to its estimate."""
        ccy = self._currencies.get(cg_id)
        if ccy is None:
            return None
        try:
            term = await self._term_for_currency(ccy)
        except Exception as e:
            log.warning("deribit_iv.fetch_failed", currency=ccy,
                        error=f"{type(e).__name__}: {e}"[:160])
            return None
        return interp_sigma(term, t_years)
