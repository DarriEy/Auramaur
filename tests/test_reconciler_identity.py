from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from auramaur.broker import redeemer as redeemer_mod
from auramaur.broker.reconciler import PositionReconciler
from auramaur.broker.redeemer import VenuePosition
from auramaur.db.database import Database


COND = "0x" + "a" * 62
STUB_ID = COND[:16]


def _exchange():
    return SimpleNamespace(
        _settings=SimpleNamespace(polymarket_proxy_address="0xPROXY"),
    )


def _position() -> VenuePosition:
    return VenuePosition(
        condition_id=COND,
        asset_id="held-token",
        title="Will identity recovery run?",
        outcome="Yes",
        size=10.0,
        avg_price=0.4,
        cur_price=0.55,
        initial_value=4.0,
        current_value=5.5,
        cash_pnl=1.5,
        redeemable=False,
        slug="identity-recovery",
        end_date="",
        outcome_index=0,
    )


@pytest.mark.asyncio
async def test_existing_condition_prefix_stub_retries_token_identity_recovery(
        tmp_path, monkeypatch):
    db = Database(str(tmp_path / "identity.db"))
    await db.connect()
    try:
        async def fake_positions(_proxy):
            return [_position()]

        monkeypatch.setattr(
            redeemer_mod, "fetch_current_positions", fake_positions)
        reconciler = PositionReconciler(_exchange(), db)
        reconciler._find_market_id = AsyncMock(return_value=STUB_ID)
        reconciler._ingest_market_from_gamma = AsyncMock(return_value="900001")

        positions = await reconciler.reconcile()

        assert [position.market_id for position in positions] == ["900001"]
        reconciler._ingest_market_from_gamma.assert_awaited_once()
    finally:
        await db.close()
