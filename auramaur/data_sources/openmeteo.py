"""Open-Meteo ensemble weather source — free, keyless NWP forecasts.

Used to price Polymarket daily city-temperature bins (#weather strategy). The
ENSEMBLE endpoint returns ~31 GFS members of a daily max/min, i.e. a real
forecast distribution — exactly what a bin market needs. No API key.

Pure pricing math lives in strategy/weather_pricing.py; this is just the fetch.
"""

from __future__ import annotations

import asyncio
from datetime import date

import aiohttp
import structlog

log = structlog.get_logger()

_ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
_TIMEOUT = aiohttp.ClientTimeout(total=20)


class OpenMeteoSource:
    source_name = "openmeteo"

    async def daily_ensemble(self, lat: float, lon: float, target: date,
                             kind: str = "high", unit: str = "C",
                             model: str = "gfs_seamless") -> list[float]:
        """Return the ensemble members' daily max (kind='high') or min ('low')
        for ``target`` at (lat, lon), in the requested unit. [] on any failure
        or if the date isn't in the forecast horizon — callers must handle empty
        (never price a bin off no members).
        """
        var = "temperature_2m_max" if kind == "high" else "temperature_2m_min"
        params = {
            "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
            "daily": var, "models": model, "timezone": "auto",
            "forecast_days": "16",
            "temperature_unit": "fahrenheit" if unit.upper() == "F" else "celsius",
        }
        try:
            async with aiohttp.ClientSession(timeout=_TIMEOUT) as s:
                async with s.get(_ENSEMBLE_URL, params=params) as r:
                    if r.status != 200:
                        log.warning("openmeteo.http", status=r.status)
                        return []
                    data = await r.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.warning("openmeteo.fetch_error", error=str(e)[:80])
            return []
        daily = data.get("daily") or {}
        times = daily.get("time") or []
        iso = target.isoformat()
        if iso not in times:
            return []
        idx = times.index(iso)
        members: list[float] = []
        for key, series in daily.items():
            if not key.startswith(var):
                continue
            if idx < len(series) and series[idx] is not None:
                members.append(float(series[idx]))
        return members
