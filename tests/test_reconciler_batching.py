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

        positions = await recon.reconcile_from_trades()

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

        positions = await recon.reconcile_from_trades()

        assert len(positions) == 2
        assert events.count("db") == 1, "shared condition_id must queue one stub"
        row = await db.fetchone(
            "SELECT COUNT(*) AS n FROM markets WHERE condition_id = ?", (cond,)
        )
        assert row["n"] == 1
    finally:
        await db.close()


def test_transaction_same_task_reentrancy_joins_outer():
    """bot.py's sync task wraps transaction() around calls that open their
    own transaction() — same-task nesting must JOIN, not BEGIN twice
    (the 2026-07-20 position_sync "transaction within a transaction")."""
    import asyncio

    from auramaur.db.database import Database

    async def run():
        db = Database(":memory:")
        await db.connect()
        async with db.transaction():
            await db.execute(
                "CREATE TABLE t_reent (id INTEGER PRIMARY KEY, v TEXT)")
            async with db.transaction():  # nested on the same task
                await db.execute("INSERT INTO t_reent (v) VALUES ('inner')")
            await db.execute("INSERT INTO t_reent (v) VALUES ('outer')")
        rows = await db.fetchall("SELECT v FROM t_reent ORDER BY id")
        assert [r["v"] for r in rows] == ["inner", "outer"]
        await db.close()

    asyncio.run(run())


@pytest.mark.asyncio
async def test_current_positions_are_authoritative_and_snapshotted(monkeypatch):
    from auramaur.broker.redeemer import VenuePosition

    db = Database(":memory:")
    await db.connect()
    try:
        await db.execute(
            """INSERT INTO markets (id,condition_id,question,exchange,last_updated)
               VALUES ('m1','cond-1','Manual holding?','polymarket',datetime('now'))""")
        await db.commit()
        exchange = _exchange([], {}, [])
        held = VenuePosition(
            condition_id="cond-1", asset_id="asset-1", title="Manual holding?",
            outcome="No", size=12.5, avg_price=.4, cur_price=.6,
            initial_value=5, current_value=7.5, cash_pnl=2.5,
            redeemable=False, end_date="", slug="manual-holding",
        )

        async def fake_fetch(_proxy):
            return [held]

        monkeypatch.setattr(
            "auramaur.broker.redeemer.fetch_current_positions", fake_fetch)
        recon = PositionReconciler(exchange, db)
        positions = await recon.reconcile()

        assert recon.last_fetch_ok is True
        assert [(p.market_id, p.token_id, p.size) for p in positions] == [
            ("m1", "asset-1", 12.5)]
        row = await db.fetchone(
            "SELECT * FROM venue_positions WHERE asset_id='asset-1'")
        assert row["current_value"] == pytest.approx(7.5)
        assert row["cash_pnl"] == pytest.approx(2.5)
    finally:
        await db.close()


async def _seed_snapshot(db):
    await db.execute(
        """INSERT INTO venue_positions
           (venue,asset_id,condition_id,size,fetched_at)
           VALUES ('polymarket','old-asset','old-condition',1,datetime('now'))""")
    await db.commit()


@pytest.mark.asyncio
async def test_failed_current_position_fetch_preserves_last_snapshot(monkeypatch):
    db = Database(":memory:")
    await db.connect()
    try:
        await _seed_snapshot(db)

        async def failed_fetch(_proxy):
            raise RuntimeError("temporary outage")

        monkeypatch.setattr(
            "auramaur.broker.redeemer.fetch_current_positions", failed_fetch)
        recon = PositionReconciler(_exchange([], {}, []), db)
        assert await recon.reconcile() == []
        assert recon.last_fetch_ok is False
        row = await db.fetchone(
            "SELECT asset_id FROM venue_positions WHERE venue='polymarket'")
        assert row["asset_id"] == "old-asset"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_successful_empty_fetch_atomically_clears_snapshot(monkeypatch):
    db = Database(":memory:")
    await db.connect()
    try:
        await _seed_snapshot(db)

        async def empty_fetch(_proxy):
            return []

        monkeypatch.setattr(
            "auramaur.broker.redeemer.fetch_current_positions", empty_fetch)
        recon = PositionReconciler(_exchange([], {}, []), db)
        assert await recon.reconcile() == []
        assert recon.last_fetch_ok is True
        count = await db.fetchone(
            "SELECT COUNT(*) AS n FROM venue_positions WHERE venue='polymarket'")
        assert count["n"] == 0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_snapshot_insert_failure_rolls_back_previous_snapshot(monkeypatch):
    from auramaur.broker.redeemer import VenuePosition

    db = Database(":memory:")
    await db.connect()
    try:
        await _seed_snapshot(db)
        await db.execute(
            """INSERT INTO markets (id,condition_id,question,exchange,last_updated)
               VALUES ('m1','new-condition','New?','polymarket',datetime('now'))""")
        await db.commit()
        held = VenuePosition(
            condition_id="new-condition", asset_id="new-asset", title="New?",
            outcome="Yes", size=2, avg_price=.4, cur_price=.5,
            initial_value=.8, current_value=1, cash_pnl=.2,
            redeemable=False, end_date="", slug="new",
        )

        async def fake_fetch(_proxy):
            return [held]

        monkeypatch.setattr(
            "auramaur.broker.redeemer.fetch_current_positions", fake_fetch)
        real_execute = db.execute

        async def fail_snapshot_insert(sql, params=()):
            if "INSERT INTO venue_positions" in sql:
                raise RuntimeError("simulated insert failure")
            return await real_execute(sql, params)

        db.execute = fail_snapshot_insert
        recon = PositionReconciler(_exchange([], {}, []), db)
        with pytest.raises(RuntimeError, match="simulated insert failure"):
            await recon.reconcile()
        assert recon.last_fetch_ok is False
        rows = await db.fetchall(
            "SELECT asset_id FROM venue_positions WHERE venue='polymarket'")
        assert [row["asset_id"] for row in rows] == ["old-asset"]
    finally:
        await db.close()
