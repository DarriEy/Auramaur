"""The syncer must not resurrect settled-but-unredeemed positions.

Settled legs stay in the on-chain wallet until redemption, so every sync
re-created their portfolio rows at a $0 mark — phantom -100% unrealized on
P&L that is already realized in the ledger.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from auramaur.broker.sync import PositionSyncer
from auramaur.exchange.models import LivePosition, TokenType


def _bare_syncer(ledger_refs: list[str]) -> PositionSyncer:
    s = PositionSyncer.__new__(PositionSyncer)
    db = AsyncMock()
    db.fetchall = AsyncMock(
        return_value=[{"source_ref": r} for r in ledger_refs])
    s._db = db
    return s


def _pos(market_id: str, token: TokenType = TokenType.NO) -> LivePosition:
    return LivePosition(market_id=market_id, token_id="t", token=token,
                        size=10.0, avg_cost=0.5, current_price=0.0)


@pytest.mark.asyncio
async def test_settled_keys_parses_ledger_refs():
    s = _bare_syncer([
        "settle:m1:NO:0",         # live settlement -> filtered
        "settle:m2:YES:1",        # paper settlement -> ignored for live
        "exit:m3:YES:0",          # not a settlement ref format
    ])
    assert await s._settled_keys(0) == {("m1", "NO")}
    assert await s._settled_keys(1) == {("m2", "YES")}


@pytest.mark.asyncio
async def test_drop_settled_filters_only_settled_leg():
    s = _bare_syncer(["settle:m1:NO:0"])
    settled = await s._settled_keys(0)
    kept = s._drop_settled(
        [_pos("m1", TokenType.NO),   # settled -> dropped
         _pos("m1", TokenType.YES),  # other leg of same market -> kept
         _pos("m2", TokenType.NO)],  # different market -> kept
        settled)
    assert [(p.market_id, p.token) for p in kept] == [
        ("m1", TokenType.YES), ("m2", TokenType.NO)]


@pytest.mark.asyncio
async def test_settled_keys_survives_db_error():
    s = PositionSyncer.__new__(PositionSyncer)
    db = AsyncMock()
    db.fetchall = AsyncMock(side_effect=RuntimeError("no table"))
    s._db = db
    assert await s._settled_keys(0) == set()
