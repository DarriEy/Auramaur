"""G10 short-rate provider for the FX carry research recorder.

OECD "immediate rate" series mirrored on FRED, one per currency the FX book
trades. Monthly and lagged one to two months — acceptable for the SIGN and
rough magnitude of carry in a research recording (the recorder stores the
detail string, so the staleness is auditable). Rates are cached for a day;
a missing series or absent FRED key degrades to None and the recorder simply
skips the carry signal for that pair (trend-only still records).
"""

from __future__ import annotations

import time

import structlog

log = structlog.get_logger()

FRED_RATE_SERIES: dict[str, str] = {
    "USD": "DFF",                 # effective federal funds, daily
    "EUR": "ECBDFR",              # ECB deposit facility, daily
    "GBP": "IUDSOIA",             # SONIA, daily
    "JPY": "IRSTCI01JPM156N",     # OECD immediate rate, monthly
    "CAD": "IRSTCI01CAM156N",
    "CHF": "IR3TIB01CHM156N",     # 3m interbank (SNB policy proxy)
    "AUD": "IRSTCI01AUM156N",
    "NZD": "IRSTCI01NZM156N",
}

_CACHE_SECONDS = 86_400.0


class FredRatesProvider:
    """Daily-cached ``currency -> annual decimal rate`` lookups."""

    def __init__(self, fred_source) -> None:
        self._fred = fred_source
        self._cache: dict[str, tuple[float, float | None]] = {}

    async def rate(self, currency: str) -> float | None:
        series = FRED_RATE_SERIES.get(currency.upper())
        if series is None or self._fred is None:
            return None
        hit = self._cache.get(currency)
        now = time.monotonic()
        if hit is not None and now - hit[0] < _CACHE_SECONDS:
            return hit[1]
        value: float | None = None
        try:
            obs = await self._fred.get_observations(series, n=1)
            if obs:
                value = float(obs[-1][1]) / 100.0  # FRED quotes percent
        except Exception as e:  # noqa: BLE001 — research-only, never fatal
            log.debug("fx_rates.fetch_failed", currency=currency,
                      series=series, error=str(e)[:100])
        self._cache[currency] = (now, value)
        return value
