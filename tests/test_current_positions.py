from __future__ import annotations

import pytest

from auramaur.broker.redeemer import fetch_current_positions


def _row(i: int, *, size: float = 2.0) -> dict:
    return {
        "asset": f"asset-{i}", "conditionId": f"condition-{i}",
        "title": f"Market {i}", "outcome": "Yes", "size": size,
        "avgPrice": .4, "curPrice": .5, "initialValue": .8,
        "currentValue": 1.0, "cashPnl": .2, "redeemable": False,
        "endDate": "", "slug": f"market-{i}",
    }


class _Response:
    def __init__(self, payload, status: int = 200):
        self.payload = payload
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    async def json(self):
        return self.payload


class _Session:
    def __init__(self, pages):
        self.pages = pages
        self.params: list[dict] = []
        self.closed = False

    def get(self, _url, *, params, timeout):
        assert timeout == 15
        self.params.append(params)
        return _Response(self.pages[int(params["offset"])])

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_current_positions_exhausts_offset_pages():
    session = _Session({0: [_row(i) for i in range(500)],
                        500: [_row(500), _row(501)]})
    rows = await fetch_current_positions("0x" + "1" * 40, session=session)
    assert len(rows) == 502
    assert [p["offset"] for p in session.params] == ["0", "500"]
    assert all(p["limit"] == "500" for p in session.params)
    assert session.closed is False


@pytest.mark.asyncio
async def test_current_positions_accepts_authoritative_empty_page():
    session = _Session({0: []})
    assert await fetch_current_positions("0x" + "1" * 40, session=session) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{"error": "bad"}, ["not-an-object"]])
async def test_current_positions_rejects_malformed_payload(payload):
    session = _Session({0: payload})
    with pytest.raises(ValueError, match="list of objects"):
        await fetch_current_positions("0x" + "1" * 40, session=session)


@pytest.mark.asyncio
async def test_current_positions_rejects_duplicate_assets_across_pages():
    page = [_row(i) for i in range(500)]
    session = _Session({0: page, 500: [_row(499)]})
    with pytest.raises(ValueError, match="duplicate asset"):
        await fetch_current_positions("0x" + "1" * 40, session=session)


@pytest.mark.asyncio
async def test_current_positions_rejects_missing_identity():
    row = _row(1)
    row["asset"] = ""
    with pytest.raises(ValueError, match="missing asset or conditionId"):
        await fetch_current_positions(
            "0x" + "1" * 40, session=_Session({0: [row]}))
