import pytest

from auramaur.data_sources.official import BLSSource, CongressSource, EIASource, NWSSource


@pytest.mark.asyncio
async def test_nws_parses_and_filters_alert(monkeypatch):
    source = NWSSource()
    async def fake_get(*args, **kwargs):
        return {"features": [{"id": "alert-1", "properties": {
            "headline": "Tornado warning for Travis County", "description": "Take shelter",
            "areaDesc": "Travis County", "sent": "2026-07-17T12:00:00Z"}}]}
    monkeypatch.setattr(source, "_get", fake_get)
    items = await source.fetch("Travis tornado")
    assert items[0].source == "nws" and items[0].timestamp_quality == "exact"


@pytest.mark.asyncio
async def test_bls_parses_latest_series(monkeypatch):
    source = BLSSource()
    async def fake_post(*args, **kwargs):
        return {"Results": {"series": [{"seriesID": "CUUR0000SA0", "data": [
            {"year": "2026", "periodName": "June", "value": "321.5"}]}]}}
    monkeypatch.setattr(source, "_post", fake_post)
    items = await source.fetch("CPI inflation")
    assert "321.5" in items[0].title and items[0].information_mode == "production"


@pytest.mark.asyncio
async def test_congress_and_eia_parse(monkeypatch):
    congress = CongressSource("key")
    async def congress_get(*args, **kwargs):
        return {"bills": [{"title": "Budget Act", "url": "bill-url",
                            "updateDate": "2026-07-17T12:00:00Z"}]}
    monkeypatch.setattr(congress, "_get", congress_get)
    assert (await congress.fetch("budget"))[0].source == "congress"

    eia = EIASource("key")
    async def eia_get(*args, **kwargs):
        return {"response": {"data": [{"period": "2026-07-10", "value": 100}]}}
    monkeypatch.setattr(eia, "_get", eia_get)
    assert (await eia.fetch("natural gas storage"))[0].source == "eia"
