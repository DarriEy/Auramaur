"""Reconciler must not mark an ACTIVE held position at $0.

The CLOB ``get_market`` ``price`` field comes back 0/missing for thinly-traded
tokens. The reconciler discovers these positions from on-chain trades and
(for ones with no DB market row) persists them via ``_merge_new_positions``,
carrying the reconciler's ``current_price``. With no fallback, 47 live
Polymarket positions (~$244 cost) were marked $0 — understating the portfolio
and fabricating -100% unrealized losses that feed the risk gates and Kelly.

The fix: when the CLOB price is 0, fall back to avg_cost (the real fill price).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from auramaur.broker.reconciler import PositionReconciler


def _exchange(trades, market_info):
    async def clob_call(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    client = SimpleNamespace(
        get_trades=lambda: trades,
        get_market=lambda cid: market_info,
    )
    return SimpleNamespace(
        _clob_client=client,
        _init_clob_client=lambda: None,
        _settings=SimpleNamespace(polymarket_proxy_address="0xPROXY"),
        clob_call=clob_call,
        register_market_tokens=lambda *a, **k: None,
    )


def _trade(asset_id="TOKNO", price="0.06", size="100", outcome="NO"):
    return {
        "status": "CONFIRMED",
        "market": "0xCOND",
        "outcome": outcome,
        "trader_side": "TAKER",
        "asset_id": asset_id,
        "side": "BUY",
        "size": size,
        "price": price,
        "maker_orders": [],
    }


async def _run(token_price):
    market_info = {
        "question": "Q?",
        "market_slug": "q",
        "tokens": [{"token_id": "TOKNO", "price": token_price, "outcome": "No"}],
    }
    recon = PositionReconciler(_exchange([_trade()], market_info), db=None)
    recon._find_market_id = AsyncMock(return_value="mkt1")
    return await recon.reconcile_from_trades()


@pytest.mark.asyncio
async def test_zero_clob_price_falls_back_to_avg_cost():
    positions = await _run(token_price=0)
    assert len(positions) == 1
    # avg_cost = total_cost/net = 6.0/100 = 0.06; must NOT be marked $0.
    assert positions[0].current_price == pytest.approx(0.06)
    assert positions[0].avg_cost == pytest.approx(0.06)


@pytest.mark.asyncio
async def test_nonzero_clob_price_is_kept():
    positions = await _run(token_price=0.045)
    assert len(positions) == 1
    # A real CLOB price must be used as-is, not overridden by the fallback.
    assert positions[0].current_price == pytest.approx(0.045)
