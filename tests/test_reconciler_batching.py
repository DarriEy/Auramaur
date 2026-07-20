"""Reconciler stub-market writes must not interleave with network calls.

db-contention plan Phase 1: the per-position loop in ``reconcile()`` awaits
CLOB ``get_market`` over the network. Stub-market INSERTs used to happen (and
commit) inside that loop, holding/queueing write work between network round
trips on the shared connection. They are now collected during the loop and
batch-inserted with ONE commit at the end — after every network call has
completed.
"""

from __future__ import annotations

import pytest
from types import SimpleNamespace

from auramaur.broker.reconciler import PositionReconciler
from auramaur.db.database import Database


def _trade(asset_id: str, condition_id: str, outcome: str = "No") -> dict:
    return {
        "status": "CONFIRMED",
        "market": condition_id,
        "outcome": outcome,
        "trader_side": "TAKER",
        "asset_id": asset_id,
        "side": "BUY",
        "size": "100",
        "price": "0.06",
        "maker_orders": [],
    }


def _exchange(trades, market_infos: dict, events: list[str]):
    """Exchange double whose get_market records a network event per call."""

    def get_market(cid):
        events.append(f"net:{cid}")
        return market_infos[cid]

    async def clob_call(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    client = SimpleNamespace(
        get_trades=lambda: trades,
        get_market=get_market,
    )
    return SimpleNamespace(
        _clob_client=client,
        _init_clob_client=lambda: None,
        _settings=SimpleNamespace(polymarket_proxy_address="0xPROXY"),
        clob_call=clob_call,
        register_market_tokens=lambda *a, **k: None,
    )


def _market_info(question: str, token_id: str) -> dict:
    return {
        "question": question,
        "market_slug": question.lower().rstrip("?"),
        "tokens": [{"token_id": token_id, "price": 0.06, "outcome": "No"}],
    }


@pytest.mark.asyncio
async def test_stub_inserts_happen_after_all_network_calls():
    db = Database(":memory:")
    await db.connect()
    try:
        events: list[str] = []
        cond_a = "0x" + "a" * 62
        cond_b = "0x" + "b" * 62
        exchange = _exchange(
            trades=[_trade("TOK_A", cond_a), _trade("TOK_B", cond_b)],
            market_infos={
                cond_a: _market_info("Alpha?", "TOK_A"),
                cond_b: _market_info("Beta?", "TOK_B"),
            },
            events=events,
        )
        recon = PositionReconciler(exchange, db)

        # Instrument the reconciler-visible write path. Database.fetchone
        # goes straight to the raw connection, so only reconciler writes
        # (db.execute / db.commit) land in `events`.
        real_execute = db.execute
        real_commit = db.commit

        async def _execute(sql, params=()):
            events.append("db")
            return await real_execute(sql, params)

        async def _commit():
            events.append("commit")
            await real_commit()

        db.execute = _execute
        db.commit = _commit

        positions = await recon.reconcile()

        assert len(positions) == 2
        net = [i for i, e in enumerate(events) if e.startswith("net:")]
        writes = [i for i, e in enumerate(events) if e == "db"]
        commits = [i for i, e in enumerate(events) if e == "commit"]
        assert len(net) == 2, "expected one get_market per condition"
        assert len(writes) == 2, "expected one stub INSERT per condition"
        assert len(commits) == 1, "stub batch must commit exactly once"
        assert max(net) < min(writes), (
            f"a stub write ran before all network calls finished: {events}"
        )
        assert commits[0] > max(writes)

        # The stubs themselves must still land.
        for cond in (cond_a, cond_b):
            row = await db.fetchone(
                "SELECT id, condition_id FROM markets WHERE id = ?",
                (cond[:16],),
            )
            assert row is not None
            assert row["condition_id"] == cond
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_duplicate_condition_queues_single_stub():
    """YES+NO tokens of one market share a condition_id — one stub, not two."""
    db = Database(":memory:")
    await db.connect()
    try:
        events: list[str] = []
        cond = "0x" + "c" * 62
        info = {
            "question": "Gamma?",
            "market_slug": "gamma",
            "tokens": [
                {"token_id": "TOK_YES", "price": 0.94, "outcome": "Yes"},
                {"token_id": "TOK_NO", "price": 0.06, "outcome": "No"},
            ],
        }
        exchange = _exchange(
            trades=[
                _trade("TOK_YES", cond, outcome="Yes"),
                _trade("TOK_NO", cond, outcome="No"),
            ],
            market_infos={cond: info},
            events=events,
        )
        recon = PositionReconciler(exchange, db)

        real_execute = db.execute

        async def _execute(sql, params=()):
            events.append("db")
            return await real_execute(sql, params)

        db.execute = _execute

        positions = await recon.reconcile()

        assert len(positions) == 2
        assert events.count("db") == 1, "shared condition_id must queue one stub"
        row = await db.fetchone(
            "SELECT COUNT(*) AS n FROM markets WHERE condition_id = ?", (cond,)
        )
        assert row["n"] == 1
    finally:
        await db.close()
