"""The 2026-08-05 live-money settlement double-booking (The Hundred incident).

Two outcome-label normalizers disagreed on non-binary (team-name) Polymarket
outcomes: the reconciler's portfolio projection (`to_live_positions`) used an
ad-hoc ternary that labeled every team name NO, while bot.py's cost_basis
mirror used `TokenType.from_str`, which labels every non-"NO" string YES. One
venue asset therefore carried a NO portfolio row and a YES cost_basis row —
two distinct canonical settlement source_refs — and the venue sweep booked
the SAME tokens twice across two cycles (market 0x7557f7ac41736a booked
+24.104 twice; the per-leg truth was +24.104 and -23.696). A both-sides
(arb) holding also collapsed onto one (market_id, is_paper, token) PK per
table, destroying the losing leg's basis. The mappers are now unified in
`reconciler.reconciled_token` (data-api outcomeIndex first, shared from_str
fallback).

Class B: `repair_orphaned_ids` renamed cost_basis/portfolio/fills rows from
the stub id (condition_id[:16]) to the real gamma id but never `pnl_ledger`,
so a settlement booked under the stub id became invisible to the
market_id-keyed dedup checks and the market settled AGAIN under the real id.
The repair now migrates the ledger too, and the prior-settlement lookup is
condition-aware for legs the repair never touched.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from auramaur.broker import redeemer as redeemer_mod
from auramaur.broker.ledger import VENUE_STRATEGY, record_ledger_event
from auramaur.broker.reconciler import (
    PositionReconciler,
    ReconciledPosition,
    reconciled_token,
)
from auramaur.broker.redeemer import RedeemablePosition, VenuePosition
from auramaur.broker.sync import PositionSyncer
from auramaur.db.database import Database
from auramaur.exchange.models import TokenType
from auramaur.strategy.resolution_tracker import ResolutionTracker

COND = "0x7557f7ac41736a" + "0" * 48
MID = "555001"
TITLE = "The Hundred: Welsh Fire v Manchester Super Giants?"

# Per-leg truth from the incident: +24.104 (winner) and -23.696 (loser);
# the bug booked +24.104 twice instead.
WINNER = {"asset": "tok-wf", "outcome": "Welsh Fire", "size": 30.13, "avg": 0.2}
LOSER = {"asset": "tok-msg", "outcome": "Manchester Super Giants",
         "size": 29.62, "avg": 0.8}


def _exchange():
    async def clob_call(fn, *args, **kwargs):
        return fn(*args, **kwargs)
    return SimpleNamespace(
        _settings=SimpleNamespace(polymarket_proxy_address="0xPROXY"),
        _init_clob_client=lambda: None,
        _clob_client=None,
        clob_call=clob_call,
        register_market_tokens=lambda *a, **k: None,
    )


def _vp(asset_id, outcome, index, size, avg, cur):
    return VenuePosition(
        condition_id=COND, asset_id=asset_id, title=TITLE,
        outcome=outcome, size=size, avg_price=avg, cur_price=cur,
        initial_value=size * avg, current_value=size * cur, cash_pnl=0.0,
        redeemable=False, end_date="", slug="the-hundred",
        outcome_index=index,
    )


def _redeemable(asset_id, *, is_winner, outcome, size, avg, cond=COND):
    return RedeemablePosition(
        condition_id=cond, asset_id=asset_id, title=TITLE,
        outcome=outcome, size=size, avg_price=avg,
        cur_price=1.0 if is_winner else 0.0,
        payout=size if is_winner else 0.0, is_winner=is_winner,
        redeemable_now=False, status="pending_oracle", neg_risk=False,
        mergeable=False, end_date="", slug="the-hundred",
    )


# bot.py's live cost_basis mirror upsert (see _task_position_sync); the
# source-pinning test below asserts bot.py feeds it (and the settled-key
# filter) through reconciled_token, so this replica exercises the same
# mapping the bot writes with.
MIRROR_SQL = """INSERT INTO cost_basis (market_id, token, token_id, size, avg_cost, total_cost, is_paper, updated_at)
   VALUES (?, ?, ?, ?, ?, ?, 0, datetime('now'))
   ON CONFLICT(market_id, is_paper, token) DO UPDATE SET
       token = excluded.token, token_id = excluded.token_id,
       size = excluded.size, avg_cost = excluded.avg_cost,
       total_cost = excluded.total_cost, updated_at = excluded.updated_at"""


async def _run_mirror_cycle(db, syncer, reconciler, reconciled):
    """One live sync pass as bot._task_position_sync performs it: settled-key
    skip, cost_basis mirror upsert, portfolio merge — with the unified mapper."""
    settled = await syncer._settled_keys(0)
    unsettled = [
        rp for rp in reconciled
        if (rp.market_id, reconciled_token(rp).value) not in settled
    ]
    for rp in unsettled:
        await db.execute(MIRROR_SQL, (
            rp.market_id, reconciled_token(rp).value, rp.token_id,
            rp.size, rp.avg_cost, rp.size * rp.avg_cost))
    await db.commit()
    await syncer._merge_new_positions(reconciler.to_live_positions(unsettled))
    return unsettled


async def _setup(tmp_path, monkeypatch, venue_positions):
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    await db.execute(
        """INSERT INTO markets (id, exchange, condition_id, question, active,
             clob_token_yes, clob_token_no, last_updated)
           VALUES (?, 'polymarket', ?, ?, 1, 'tok-wf', 'tok-msg',
                   datetime('now'))""",
        (MID, COND, TITLE))
    await db.commit()

    async def fake_positions(_proxy):
        return venue_positions
    monkeypatch.setattr(redeemer_mod, "fetch_current_positions", fake_positions)

    reconciler = PositionReconciler(_exchange(), db)
    syncer = PositionSyncer(
        SimpleNamespace(is_live=True), db, None, None, None)
    tracker = ResolutionTracker(db, None, {}, proxy_address="0xPROXY")
    return db, reconciler, syncer, tracker


# ---------------------------------------------------------------------------
# (a) The Hundred, verbatim: both sides held, two sweep cycles, one booking
#     per ASSET, net equals per-leg truth.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hundred_incident_books_each_leg_exactly_once(tmp_path, monkeypatch):
    vps = [
        _vp(WINNER["asset"], WINNER["outcome"], 0,
            WINNER["size"], WINNER["avg"], 1.0),
        _vp(LOSER["asset"], LOSER["outcome"], 1,
            LOSER["size"], LOSER["avg"], 0.0),
    ]
    db, reconciler, syncer, tracker = await _setup(tmp_path, monkeypatch, vps)
    try:
        async def fake_redeemable(_proxy):
            return [
                _redeemable(WINNER["asset"], is_winner=True,
                            outcome=WINNER["outcome"],
                            size=WINNER["size"], avg=WINNER["avg"]),
                _redeemable(LOSER["asset"], is_winner=False,
                            outcome=LOSER["outcome"],
                            size=LOSER["size"], avg=LOSER["avg"]),
            ]
        monkeypatch.setattr(
            redeemer_mod, "fetch_redeemable_positions", fake_redeemable)

        # -- Cycle 1: reconcile -> mirror -> venue sweep -------------------
        reconciled = await reconciler.reconcile()
        assert len(reconciled) == 2
        await _run_mirror_cycle(db, syncer, reconciler, reconciled)

        # The two tables agree on the per-asset side and hold BOTH legs.
        cb = {r["token"]: dict(r) for r in await db.fetchall(
            "SELECT * FROM cost_basis WHERE market_id = ?", (MID,))}
        pf = {r["token"]: dict(r) for r in await db.fetchall(
            "SELECT * FROM portfolio WHERE market_id = ?", (MID,))}
        assert set(cb) == set(pf) == {"YES", "NO"}
        assert cb["YES"]["token_id"] == pf["YES"]["token_id"] == WINNER["asset"]
        assert cb["NO"]["token_id"] == pf["NO"]["token_id"] == LOSER["asset"]

        settled = await tracker.settle_via_venue("0xPROXY")
        assert len(settled) == 2

        rows = await db.fetchall(
            "SELECT source_ref, pnl, strategy_source FROM pnl_ledger "
            "WHERE kind = 'settlement' ORDER BY source_ref")
        assert [r["source_ref"] for r in rows] == [
            f"settle:{MID}:NO:0", f"settle:{MID}:YES:0"]
        by_ref = {r["source_ref"]: r["pnl"] for r in rows}
        assert by_ref[f"settle:{MID}:YES:0"] == pytest.approx(24.104)
        assert by_ref[f"settle:{MID}:NO:0"] == pytest.approx(-23.696)
        # No in-DB ancestry -> the explicit venue bucket, never ''.
        assert all(r["strategy_source"] == VENUE_STRATEGY for r in rows)

        # -- Cycle 2: the venue still lists both tokens pre-redemption -----
        reconciled2 = await reconciler.reconcile()
        assert len(reconciled2) == 2
        unsettled2 = await _run_mirror_cycle(db, syncer, reconciler, reconciled2)
        assert unsettled2 == [], "settled keys must keep both legs out"
        assert await tracker.settle_via_venue("0xPROXY") == []

        count = await db.fetchone(
            "SELECT COUNT(*) AS n, SUM(pnl) AS s FROM pnl_ledger "
            "WHERE kind = 'settlement'")
        assert count["n"] == 2, "second cycle must book nothing new"
        assert count["s"] == pytest.approx(24.104 - 23.696)
        ds = await db.fetchone(
            "SELECT total_pnl, trades_count, wins, losses FROM daily_stats")
        assert ds["total_pnl"] == pytest.approx(0.408)
        assert (ds["trades_count"], ds["wins"], ds["losses"]) == (2, 1, 1)
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# (b) The projection and the mirror produce the IDENTICAL side for any
#     outcome label and for outcomeIndex 0/1.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("index", [-1, 0, 1])
@pytest.mark.parametrize("outcome", [
    "Yes", "No", "NO", "Welsh Fire", "Manchester Super Giants", ""])
def test_projection_and_mirror_map_to_identical_token(outcome, index):
    rp = ReconciledPosition(
        market_id="m1", condition_id=COND, token_id="tok", outcome=outcome,
        question="q", size=1.0, avg_cost=0.5, current_price=0.5,
        outcome_index=index)
    projected = PositionReconciler(None, None).to_live_positions([rp])[0].token
    mirrored = reconciled_token(rp)
    assert projected == mirrored
    if index == 0:
        assert mirrored is TokenType.YES
    elif index == 1:
        assert mirrored is TokenType.NO
    else:
        # Index absent: the SHARED normalizer, never the ad-hoc ternary
        # (which mapped every team name to NO).
        assert mirrored is TokenType.from_str(outcome)


def test_bot_mirror_and_settled_filter_use_the_unified_mapper():
    """Pin bot.py's cost_basis mirror AND its settled-key filter to
    reconciled_token — a regression to per-site normalizers is the bug."""
    import auramaur.bot as bot_mod
    src = inspect.getsource(bot_mod)
    assert src.count("reconciled_token(rp).value") >= 2
    assert "TokenType.from_str(rp.outcome)" not in src


# ---------------------------------------------------------------------------
# (c) A both-sides holding stays TWO cost_basis rows with per-asset basis.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_both_sides_holding_keeps_two_rows_with_per_asset_basis(
        tmp_path, monkeypatch):
    vps = [
        _vp(WINNER["asset"], WINNER["outcome"], 0,
            WINNER["size"], WINNER["avg"], 0.55),
        _vp(LOSER["asset"], LOSER["outcome"], 1,
            LOSER["size"], LOSER["avg"], 0.45),
    ]
    db, reconciler, syncer, _tracker = await _setup(tmp_path, monkeypatch, vps)
    try:
        reconciled = await reconciler.reconcile()
        await _run_mirror_cycle(db, syncer, reconciler, reconciled)
        rows = await db.fetchall(
            "SELECT token, token_id, size, avg_cost FROM cost_basis "
            "WHERE market_id = ? ORDER BY token", (MID,))
        assert len(rows) == 2, "the PK must not swallow the second leg"
        no, yes = rows
        assert (yes["token"], yes["token_id"]) == ("YES", WINNER["asset"])
        assert yes["size"] == pytest.approx(WINNER["size"])
        assert yes["avg_cost"] == pytest.approx(WINNER["avg"])
        assert (no["token"], no["token_id"]) == ("NO", LOSER["asset"])
        assert no["size"] == pytest.approx(LOSER["size"])
        assert no["avg_cost"] == pytest.approx(LOSER["avg"])
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# (d) Class B: settle under the stub id, migrate ids, sweep under the real
#     id -> no second booking.
# ---------------------------------------------------------------------------

STUB_COND = "0x" + "d" * 62
STUB_ID = STUB_COND[:16]
REAL_ID = "777001"


async def _class_b_db(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    for mid in (STUB_ID, REAL_ID):
        await db.execute(
            """INSERT INTO markets (id, exchange, condition_id, question,
                 last_updated)
               VALUES (?, 'polymarket', ?, 'Stub?', datetime('now'))""",
            (mid, STUB_COND))
    # Settlement booked while the market was still the stub id.
    await db.execute(
        """INSERT INTO pnl_ledger (market_id, kind, token, qty, pnl, is_paper,
             source_ref)
           VALUES (?, 'settlement', 'YES', 10, 6.0, 0, ?)""",
        (STUB_ID, f"settle:{STUB_ID}:YES:0"))
    await db.commit()
    return db


@pytest.mark.asyncio
async def test_repair_migrates_ledger_and_sweep_does_not_rebook(
        tmp_path, monkeypatch):
    db = await _class_b_db(tmp_path)
    try:
        # A fill ref embeds no market id — only its market_id column migrates.
        await db.execute(
            """INSERT INTO pnl_ledger (market_id, kind, token, qty, pnl,
                 is_paper, source_ref)
               VALUES (?, 'sell', 'YES', 1, 0.5, 0, 'fill:42')""", (STUB_ID,))
        # The orphaned live cost_basis row that triggers the repair.
        await db.execute(
            """INSERT INTO cost_basis (market_id, token, token_id, size,
                 avg_cost, total_cost, is_paper)
               VALUES (?, 'YES', 'tok-d', 0, 0.4, 4.0, 0)""", (STUB_ID,))
        await db.commit()

        reconciler = PositionReconciler(_exchange(), db)
        repaired = await reconciler.repair_orphaned_ids([ReconciledPosition(
            market_id=REAL_ID, condition_id=STUB_COND, token_id="tok-d",
            outcome="Yes", question="Stub?", size=10.0)])
        assert repaired == 1

        refs = {r["source_ref"]: r["market_id"] for r in await db.fetchall(
            "SELECT source_ref, market_id FROM pnl_ledger")}
        assert refs == {f"settle:{REAL_ID}:YES:0": REAL_ID, "fill:42": REAL_ID}
        # The market_id-keyed settled-key skip sees the settlement again.
        syncer = PositionSyncer(
            SimpleNamespace(is_live=True), db, None, None, None)
        assert (REAL_ID, "YES") in await syncer._settled_keys(0)

        # The wallet still holds the token; the leg resurrects under the
        # real id and the sweep runs again — it must drain, not re-book.
        await db.execute(
            "UPDATE cost_basis SET size = 10 WHERE market_id = ?", (REAL_ID,))
        await db.commit()

        async def fake_redeemable(_proxy):
            return [_redeemable("tok-d", is_winner=True, outcome="Yes",
                                size=10.0, avg=0.4, cond=STUB_COND)]
        monkeypatch.setattr(
            redeemer_mod, "fetch_redeemable_positions", fake_redeemable)
        tracker = ResolutionTracker(db, None, {}, proxy_address="0xPROXY")
        settled = await tracker.settle_via_venue("0xPROXY")

        assert len(settled) == 1 and settled[0]["correction"] is False
        n = await db.fetchone(
            "SELECT COUNT(*) AS n FROM pnl_ledger WHERE kind = 'settlement'")
        assert n["n"] == 1, "the migrated ref must block a second booking"
        cb = await db.fetchone(
            "SELECT size FROM cost_basis WHERE market_id = ?", (REAL_ID,))
        assert cb["size"] == pytest.approx(0.0)
        ds = await db.fetchone("SELECT COUNT(*) AS n FROM daily_stats")
        assert ds["n"] == 0, "an already-booked leg must not touch daily_stats"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_unmigrated_stub_settlement_caught_by_condition_join(
        tmp_path, monkeypatch):
    """Belt-and-suspenders: a stub-ref settlement the repair never touched
    (its orphan cost_basis row was already gone, so the rename never fired)
    must still block a re-book — the prior check joins markets on
    condition_id."""
    db = await _class_b_db(tmp_path)
    try:
        # The mirror already recreated the holding under the REAL id; the
        # ledger still holds only the stub-ref row (the historical shape).
        await db.execute(
            """INSERT INTO cost_basis (market_id, token, token_id, size,
                 avg_cost, total_cost, is_paper)
               VALUES (?, 'YES', 'tok-d', 10, 0.4, 4.0, 0)""", (REAL_ID,))
        await db.commit()

        async def fake_redeemable(_proxy):
            return [_redeemable("tok-d", is_winner=True, outcome="Yes",
                                size=10.0, avg=0.4, cond=STUB_COND)]
        monkeypatch.setattr(
            redeemer_mod, "fetch_redeemable_positions", fake_redeemable)
        tracker = ResolutionTracker(db, None, {}, proxy_address="0xPROXY")
        settled = await tracker.settle_via_venue("0xPROXY")

        assert len(settled) == 1 and settled[0]["correction"] is False
        rows = await db.fetchall(
            "SELECT source_ref FROM pnl_ledger WHERE kind = 'settlement'")
        assert [r["source_ref"] for r in rows] == [f"settle:{STUB_ID}:YES:0"]
        cb = await db.fetchone(
            "SELECT size FROM cost_basis WHERE market_id = ?", (REAL_ID,))
        assert cb["size"] == pytest.approx(0.0), "the leg must still drain"
        ds = await db.fetchone("SELECT COUNT(*) AS n FROM daily_stats")
        assert ds["n"] == 0
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# (e) Ancestry-free settlements book venue_unattributed, never ''.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ancestry_free_settlement_books_venue_unattributed(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    try:
        await db.execute(
            """INSERT INTO markets (id, exchange, condition_id, question,
                 category, last_updated)
               VALUES ('m1', 'polymarket', '0xc', 'Q?', 'sports',
                       datetime('now'))""")
        await db.commit()
        # No trades/signals ancestry at all.
        await record_ledger_event(
            db, market_id="m1", kind="settlement", token="YES", qty=10,
            pnl=5.0, fees=0, is_paper=False, source_ref="settle:m1:YES:0")
        # Non-settlement kinds keep '' (a missing entry row must stay visible).
        await record_ledger_event(
            db, market_id="m1", kind="sell", token="YES", qty=1,
            pnl=0.5, fees=0, is_paper=False, source_ref="fill:9")
        rows = {r["source_ref"]: r["strategy_source"] for r in await db.fetchall(
            "SELECT source_ref, strategy_source FROM pnl_ledger")}
        assert rows["settle:m1:YES:0"] == VENUE_STRATEGY
        assert rows["fill:9"] == ""
    finally:
        await db.close()
