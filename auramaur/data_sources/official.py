"""Primary U.S. government sources for event-time evidence."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import aiohttp
import structlog

from auramaur.data_sources.base import NewsItem

log = structlog.get_logger()
_TIMEOUT = aiohttp.ClientTimeout(total=15)

# Candidate-source trust is isolated from production ranking.
OFFICIAL_SHADOW_AUTHORITY = {
    "nws": 3.0, "bls": 3.0, "bea": 3.0, "congress": 3.0, "eia": 3.0,
}


def _dt(value: str | None) -> tuple[datetime, str]:
    if value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")), "exact"
        except ValueError:
            pass
    return datetime.now(timezone.utc), "provider_seen"


def _id(source: str, value: str) -> str:
    return hashlib.sha256(f"{source}:{value}".encode()).hexdigest()[:24]


class _HTTPSource:
    source_name = "official"
    categories: set[str] | None = None
    # New information lanes collect lineage but cannot influence forecasts
    # until their source/category/horizon cell graduates.
    information_mode = "shadow"

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None

    async def _get(self, url: str, **kwargs) -> dict:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=_TIMEOUT)
        async with self._session.get(url, **kwargs) as response:
            response.raise_for_status()
            return await response.json(content_type=None)

    async def _post(self, url: str, **kwargs) -> dict:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=_TIMEOUT)
        async with self._session.post(url, **kwargs) as response:
            response.raise_for_status()
            return await response.json(content_type=None)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


class NWSSource(_HTTPSource):
    source_name = "nws"
    categories = {"weather", "science"}
    trial_horizon = "0-6h"
    trial_event_type = "severe_weather_transition"

    async def fetch(self, query: str, limit: int = 20) -> list[NewsItem]:
        try:
            data = await self._get(
                "https://api.weather.gov/alerts/active",
                headers={"User-Agent": "Auramaur/0.1 (research bot)"},
            )
        except Exception as exc:
            log.warning("nws.fetch_failed", error=str(exc)[:120])
            return []
        words = {w.lower() for w in query.split() if len(w) > 3}
        items = []
        for feature in data.get("features", []):
            p = feature.get("properties", {})
            text = f"{p.get('headline','')} {p.get('description','')} {p.get('areaDesc','')}"
            if words and not any(w in text.lower() for w in words):
                continue
            published, quality = _dt(p.get("sent") or p.get("effective"))
            ident = feature.get("id") or p.get("id") or text[:100]
            items.append(NewsItem(
                id=_id("nws", ident), source="nws", title=p.get("headline") or p.get("event", "NWS alert"),
                content=text[:2000], url=feature.get("id", ""), published_at=published,
                timestamp_quality=quality, relevance_score=2.5,
            ))
        return items[:limit]


_BLS_SERIES = {
    "cpi": "CUUR0000SA0", "inflation": "CUUR0000SA0", "unemployment": "LNS14000000",
    "payroll": "CES0000000001", "jobs": "CES0000000001", "ppi": "WPUFD4",
}


class BLSSource(_HTTPSource):
    source_name = "bls"
    categories = {"economics"}
    trial_horizon = "0-30m"
    trial_event_type = "scheduled_release"

    def __init__(self, api_key: str = "") -> None:
        super().__init__()
        self._api_key = api_key

    async def fetch(self, query: str, limit: int = 20) -> list[NewsItem]:
        ids = list(dict.fromkeys(v for k, v in _BLS_SERIES.items() if k in query.lower()))
        if not ids:
            return []
        year = datetime.now(timezone.utc).year
        payload = {"seriesid": ids, "startyear": str(year - 1), "endyear": str(year)}
        if self._api_key:
            payload["registrationkey"] = self._api_key
        try:
            data = await self._post("https://api.bls.gov/publicAPI/v2/timeseries/data/", json=payload)
        except Exception as exc:
            log.warning("bls.fetch_failed", error=str(exc)[:120])
            return []
        items = []
        for series in data.get("Results", {}).get("series", []):
            sid = series.get("seriesID", "")
            for row in series.get("data", [])[:3]:
                period = f"{row.get('year','')}-{row.get('periodName',row.get('period',''))}"
                items.append(NewsItem(
                    id=_id("bls", f"{sid}:{period}:{row.get('value')}"), source="bls",
                    title=f"BLS {sid}: {row.get('value')} ({period})",
                    content=f"Official BLS series {sid}; value {row.get('value')}; period {period}.",
                    url="https://www.bls.gov/developers/", timestamp_quality="provider_seen",
                    relevance_score=2.5, keywords=[sid],
                ))
        return items[:limit]


class BEASource(_HTTPSource):
    source_name = "bea"
    categories = {"economics"}
    trial_horizon = "0-30m"
    trial_event_type = "scheduled_release"

    def __init__(self, api_key: str) -> None:
        super().__init__()
        self._api_key = api_key

    async def fetch(self, query: str, limit: int = 20) -> list[NewsItem]:
        table = "T10101" if "gdp" in query.lower() else "T20804" if "pce" in query.lower() else ""
        if not table:
            return []
        params = {"UserID": self._api_key, "method": "GetData", "datasetname": "NIPA",
                  "TableName": table, "Frequency": "Q", "Year": "X", "ResultFormat": "JSON"}
        try:
            data = await self._get("https://apps.bea.gov/api/data", params=params)
            rows = data["BEAAPI"]["Results"]["Data"]
        except Exception as exc:
            log.warning("bea.fetch_failed", error=str(exc)[:120])
            return []
        return [NewsItem(
            id=_id("bea", f"{table}:{r.get('TimePeriod')}:{r.get('LineNumber')}:{r.get('DataValue')}"),
            source="bea", title=f"BEA {r.get('LineDescription','economic release')}: {r.get('DataValue')}",
            content=f"Official BEA NIPA table {table}, {r.get('TimePeriod')}: {r.get('DataValue')}",
            url="https://apps.bea.gov/iTable/", timestamp_quality="provider_seen", relevance_score=2.5,
        ) for r in rows[:limit]]


class CongressSource(_HTTPSource):
    source_name = "congress"
    categories = {"politics_us", "legal"}
    trial_horizon = "1-14d"
    trial_event_type = "procedural_milestone"

    def __init__(self, api_key: str) -> None:
        super().__init__()
        self._api_key = api_key

    async def fetch(self, query: str, limit: int = 20) -> list[NewsItem]:
        try:
            data = await self._get("https://api.congress.gov/v3/bill",
                                   params={"api_key": self._api_key, "format": "json", "limit": min(limit, 20)})
        except Exception as exc:
            log.warning("congress.fetch_failed", error=str(exc)[:120])
            return []
        words = {w.lower() for w in query.split() if len(w) > 3}
        items = []
        for bill in data.get("bills", []):
            title = bill.get("title", "")
            if words and not any(w in title.lower() for w in words):
                continue
            published, quality = _dt(bill.get("updateDate"))
            items.append(NewsItem(
                id=_id("congress", bill.get("url", title)), source="congress", title=title,
                content=f"Congress.gov latest action/update: {bill.get('updateDate','')}",
                url=bill.get("url", ""), published_at=published, timestamp_quality=quality,
                relevance_score=2.5,
            ))
        return items[:limit]


class EIASource(_HTTPSource):
    source_name = "eia"
    categories = {"economics", "weather"}
    trial_horizon = "0-24h"
    trial_event_type = "scheduled_release"

    def __init__(self, api_key: str) -> None:
        super().__init__()
        self._api_key = api_key

    async def fetch(self, query: str, limit: int = 20) -> list[NewsItem]:
        q = query.lower()
        route = "natural-gas/stor/wkly" if "natural gas" in q or "storage" in q else "petroleum/stoc/wstk" if "oil" in q or "crude" in q else ""
        if not route:
            return []
        try:
            data = await self._get(f"https://api.eia.gov/v2/{route}/data/", params={
                "api_key": self._api_key, "frequency": "weekly", "length": min(limit, 20),
            })
            rows = data.get("response", {}).get("data", [])
        except Exception as exc:
            log.warning("eia.fetch_failed", error=str(exc)[:120])
            return []
        items = []
        for row in rows:
            period = row.get("period", "")
            value = row.get("value")
            items.append(NewsItem(
                id=_id("eia", f"{route}:{period}:{value}"), source="eia",
                title=f"EIA {route}: {value} ({period})", content=str(row)[:1500],
                url="https://www.eia.gov/opendata/", timestamp_quality="provider_seen",
                relevance_score=2.5,
            ))
        return items[:limit]
