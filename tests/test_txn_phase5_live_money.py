"""Phase 5 of the txn migration (#353): the live-money entry batches.

Per-writer pattern (docs/plans/txn-migration-plan.md): crash the SECOND
statement of the batch and assert zero rows from ANY statement of the
batch (atomicity), assert the writer's distinct owner label on a clean
call, prove the gateway network call completes BEFORE the span begins
(tight-span ordering), and prove the connection accepts unrelated writes
after the injected failure (no strand — the 2026-07-25 wedge class).

cross_venue_arb additionally carries the duplicate-exposure regression:
a crash anywhere in its batch must leave the verdict's traded_at and the
leg rows CONSISTENT (all absent), so a retry re-enters the pair as a
whole instead of duplicating a leg.

Pillars are built via ``object.__new__`` with only the attributes the
tested path reads (house precedent: test_txn_phase3_signal_sweep.py);
gateway/risk/calibration are minimal stubs that log their boundary into
the same event list the transaction spy writes, so ordering is a plain
index comparison.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from auramaur.db.database import Database
from auramaur.exchange.models import Market, OrderSide
from tests.txn_helpers import failing_on, span_owners, transaction_spy


@contextmanager
def _failing_on_nth(db, needle: str, n: int):
    """Like txn_helpers.failing_on, but only the n-th matching statement
    raises — needed where a batch's statements share one SQL shape (the
    two arb legs' identical INSERTs)."""
    original = db.execute
    seen = 0

    async def _wrapped(sql: str, params: tuple = ()):
        nonlocal seen
        if needle in sql:
            seen += 1
            if seen >= n:
                raise sqlite3.OperationalError(f"injected: {needle} #{seen}")
        return await original(sql, params)

    db.execute = _wrapped
    try:
        yield
    finally:
        db.execute = original


# ---------------------------------------------------------------------------
# Minimal stubs — only what the entry paths read
# ---------------------------------------------------------------------------


def _market(mid: str, yes: float = 0.5, exchange: str = "polymarket") -> Market:
    return Market(id=mid, question=f"Q {mid}?", outcome_yes_price=yes,
                  outcome_no_price=round(1.0 - yes, 4), exchange=exchange,
                  category="other", volume=1000.0, liquidity=1000.0)


def _order(mid: str, side: OrderSide = OrderSide.BUY) -> SimpleNamespace:
    return SimpleNamespace(
        market_id=mid, exchange="polymarket", token_id=f"tok-{mid}",
        token=SimpleNamespace(value="YES"), side=side, size=20.0, price=0.5)


def _submit_result(mid: str, side: OrderSide = OrderSide.BUY) -> SimpleNamespace:
    return SimpleNamespace(
        status="paper", reason="", order=_order(mid, side),
        result=SimpleNamespace(filled_size=20.0, filled_price=0.5,
                               is_paper=True))


class _Risk:
    async def evaluate(self, signal, market):
        return SimpleNamespace(approved=True, position_size=10.0, reason="")


class _Gateway:
    """Single-order gateway stub; logs the network boundary."""

    def __init__(self, events: list) -> None:
        self._events = events

    async def submit(self, intent):
        self._events.append(("gateway", "submit"))
        return _submit_result(intent.market.id)


class _PairedGateway:
    def __init__(self, events: list) -> None:
        self._events = events

    async def submit_paired(self, intent_a, intent_b, **_kwargs):
        self._events.append(("gateway", "submit_paired"))
        return (_submit_result(intent_a.market.id, OrderSide.SELL),
                _submit_result(intent_b.market.id, OrderSide.BUY))


class _Calibration:
    def __init__(self, events: list) -> None:
        self._events = events

    async def record_prediction(self, market_id, prob, category):
        self._events.append(("calibration", "record"))


# ---------------------------------------------------------------------------
# agent_trader — portfolio upsert + thesis INSERT (owner agent_trader.entry)
# ---------------------------------------------------------------------------

_AGENT_CFG = SimpleNamespace(min_edge_pts=5.0, stake_usd=10.0, paper=True)


def _decision(mid: str) -> dict:
    return {"market_id": mid, "prob_yes": 0.8,
            "thesis": "the crowd misprices the fine print"}


async def _agent_pillar(db, events: list):
    from auramaur.strategy.agent_trader import _THESES_TABLE, AgentTraderPillar

    await db.execute(_THESES_TABLE)
    pillar = object.__new__(AgentTraderPillar)
    pillar._db = db
    pillar._exchange_name = "polymarket"
    pillar._cell_suffix = ""
    pillar._risk = _Risk()
    pillar._gateway = _Gateway(events)
    pillar._calibration = _Calibration(events)
    return pillar


@pytest.mark.asyncio
async def test_agent_trader_entry_batch_is_atomic_under_mid_batch_crash():
    db = Database(":memory:")
    await db.connect()
    try:
        events: list = []
        pillar = await _agent_pillar(db, events)
        with failing_on(db, "INSERT INTO agent_trader_theses"):
            with pytest.raises(sqlite3.OperationalError, match="injected"):
                await pillar._try_enter(
                    "opus", _market("ag-m1"), _decision("ag-m1"), _AGENT_CFG)
        # The batch: portfolio upsert (1st) + thesis INSERT (2nd, crashed).
        for table in ("portfolio", "agent_trader_theses"):
            row = await db.fetchone(f"SELECT COUNT(*) AS n FROM {table}")
            assert row["n"] == 0, f"partial write survived in {table}"
        # The signals row is a SEPARATE, earlier span (agent_trader.signal,
        # phase 3) — it commits before submit and survives by design.
        row = await db.fetchone("SELECT COUNT(*) AS n FROM signals")
        assert row["n"] == 1
        # No strand: the connection accepts unrelated writes immediately.
        await db.execute(
            "INSERT OR IGNORE INTO markets (id, question) VALUES ('x','q')")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_agent_trader_entry_owner_ordering_and_calibration_outside():
    db = Database(":memory:")
    await db.connect()
    try:
        events: list = []
        pillar = await _agent_pillar(db, events)
        with transaction_spy(db, events):
            entered = await pillar._try_enter(
                "opus", _market("ag-m2"), _decision("ag-m2"), _AGENT_CFG)
        assert entered is True
        assert "agent_trader.entry" in span_owners(events)
        assert events.count(("agent_trader.entry", "begin")) == 1
        # The gateway submit completes BEFORE the entry span begins, and
        # calibration fires only after the span has ended.
        assert (events.index(("gateway", "submit"))
                < events.index(("agent_trader.entry", "begin")))
        assert (events.index(("agent_trader.entry", "end"))
                < events.index(("calibration", "record")))
        for table in ("portfolio", "agent_trader_theses"):
            row = await db.fetchone(f"SELECT COUNT(*) AS n FROM {table}")
            assert row["n"] == 1, table
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# cross_venue_arb — leg signals x2 + traded_at (owner cross_venue_arb.entry)
# ---------------------------------------------------------------------------


async def _cross_pillar(db, events: list):
    from auramaur.strategy.cross_venue_arb import CrossVenueArbPillar

    pillar = object.__new__(CrossVenueArbPillar)
    pillar._db = db
    pillar._settings = SimpleNamespace(
        cross_venue_arb=SimpleNamespace(stake_usd=10.0, paper=True))
    pillar._exchanges = {"polymarket": object(), "kalshi": object()}
    pillar._risk = _Risk()
    pillar._gateway = _PairedGateway(events)
    await pillar._ensure_schema()
    await db.execute(
        "INSERT INTO cross_venue_verdicts (poly_id, kalshi_id, orientation,"
        " confidence) VALUES ('pa', 'kb', 'same', 0.9)")
    return pillar


async def _enter_cross_pair(pillar):
    return await pillar._enter_pair(
        _market("pa", yes=0.4), _market("kb", yes=0.6, exchange="kalshi"),
        0.2, OrderSide.BUY, OrderSide.SELL, "same", 0.9)


@pytest.mark.asyncio
async def test_cross_venue_entry_batch_is_atomic_under_mid_batch_crash():
    db = Database(":memory:")
    await db.connect()
    try:
        events: list = []
        pillar = await _cross_pillar(db, events)
        # The two legs share one SQL shape — crash the SECOND statement of
        # the batch (leg B's signals row) via the nth-match injector.
        with _failing_on_nth(db, "INSERT INTO signals", 2):
            with pytest.raises(sqlite3.OperationalError, match="injected"):
                await _enter_cross_pair(pillar)
        row = await db.fetchone("SELECT COUNT(*) AS n FROM signals")
        assert row["n"] == 0, "leg A's row survived the crashed batch"
        row = await db.fetchone(
            "SELECT traded_at FROM cross_venue_verdicts "
            "WHERE poly_id = 'pa' AND kalshi_id = 'kb'")
        assert row["traded_at"] is None
        # No strand.
        await db.execute(
            "INSERT OR IGNORE INTO markets (id, question) VALUES ('x','q')")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_cross_venue_crash_anywhere_leaves_pair_reenterable_whole():
    """The duplicate-exposure regression: pre-span, a crash after the legs
    but before traded_at left _already_traded false — the pair could be
    RE-ENTERED next cycle on top of live fills. Now leg rows and traded_at
    are consistent (all absent) after a crash at ANY statement, so the
    retry re-enters the pair as a WHOLE rather than duplicating a leg."""
    db = Database(":memory:")
    await db.connect()
    try:
        events: list = []
        pillar = await _cross_pillar(db, events)
        # Crash the LAST statement — exactly the old partial-write window.
        with failing_on(db, "UPDATE cross_venue_verdicts"):
            with pytest.raises(sqlite3.OperationalError, match="injected"):
                await _enter_cross_pair(pillar)
        row = await db.fetchone("SELECT COUNT(*) AS n FROM signals")
        assert row["n"] == 0, "leg rows survived without their traded_at"
        assert await pillar._already_traded("pa", "kb") is False
        # Fault cleared: the retry books BOTH legs and retires the pair.
        assert await _enter_cross_pair(pillar) is True
        row = await db.fetchone("SELECT COUNT(*) AS n FROM signals")
        assert row["n"] == 2
        assert await pillar._already_traded("pa", "kb") is True
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_cross_venue_entry_owner_and_post_submit_ordering():
    db = Database(":memory:")
    await db.connect()
    try:
        events: list = []
        pillar = await _cross_pillar(db, events)
        with transaction_spy(db, events):
            assert await _enter_cross_pair(pillar) is True
        assert "cross_venue_arb.entry" in span_owners(events)
        assert events.count(("cross_venue_arb.entry", "begin")) == 1
        assert (events.index(("gateway", "submit_paired"))
                < events.index(("cross_venue_arb.entry", "begin")))
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# entailment_arb — _record_leg x2 + traded_at (owners: entailment_arb.entry
# for the paired batch, entailment_arb.leg for the standalone helper)
# ---------------------------------------------------------------------------


def _entailment_pillar(db, events: list):
    from auramaur.strategy.entailment_arb import EntailmentArbPillar

    pillar = object.__new__(EntailmentArbPillar)
    pillar._db = db
    pillar._settings = SimpleNamespace(
        entailment_arb=SimpleNamespace(stake_usd=10.0, paper=True,
                                       min_gap=0.05))
    pillar._exchanges = {"polymarket": object()}
    pillar._risk = _Risk()
    pillar._gateway = _PairedGateway(events)
    return pillar


@pytest.mark.asyncio
async def test_entailment_record_leg_is_atomic_and_carries_its_owner():
    db = Database(":memory:")
    await db.connect()
    try:
        pillar = _entailment_pillar(db, [])
        mid = "ent-m1"
        # Crash the SECOND statement of the helper's markets+signals+
        # portfolio batch: the markets stub must not survive alone.
        with failing_on(db, "INSERT INTO signals"):
            with pytest.raises(sqlite3.OperationalError, match="injected"):
                await pillar._record_leg(
                    _market(mid), _order(mid), _submit_result(mid).result,
                    "above 71000 => above 70200", 0.3)
        for table in ("signals", "portfolio"):
            row = await db.fetchone(f"SELECT COUNT(*) AS n FROM {table}")
            assert row["n"] == 0, f"partial write survived in {table}"
        row = await db.fetchone(
            "SELECT COUNT(*) AS n FROM markets WHERE id = ?", (mid,))
        assert row["n"] == 0, "markets stub survived the crashed batch"
        # No strand.
        await db.execute(
            "INSERT OR IGNORE INTO markets (id, question) VALUES ('x','q')")
        # Clean call: all three rows land under the distinct owner.
        events: list = []
        with transaction_spy(db, events):
            await pillar._record_leg(
                _market(mid), _order(mid), _submit_result(mid).result,
                "above 71000 => above 70200", 0.3)
        assert span_owners(events) == ["entailment_arb.leg"]
        assert events.count(("entailment_arb.leg", "end")) == 1
        for table in ("signals", "portfolio"):
            row = await db.fetchone(f"SELECT COUNT(*) AS n FROM {table}")
            assert row["n"] == 1, table
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_entailment_traded_at_folds_into_the_entry_span():
    """The paired call site opens an outer entailment_arb.entry span that
    _record_leg's own span JOINS (same-task re-entrancy), so both legs'
    markets/signals/portfolio rows AND traded_at are one atomic unit: a
    crash on the LAST statement must also discard the helper's writes."""
    db = Database(":memory:")
    await db.connect()
    try:
        events: list = []
        pillar = _entailment_pillar(db, events)
        await db.execute(
            "INSERT INTO entailment_verdicts (market_id_a, market_id_b,"
            " direction, confidence) VALUES ('ent-a', 'ent-b',"
            " 'a_implies_b', 1.0)")
        implier = _market("ent-a", yes=0.7)
        implied = _market("ent-b", yes=0.4)
        with failing_on(db, "UPDATE entailment_verdicts"):
            with pytest.raises(sqlite3.OperationalError, match="injected"):
                await pillar._enter_pair(
                    implier, implied, 0.3, "ladder", 1.0, False)
        for table in ("signals", "portfolio"):
            row = await db.fetchone(f"SELECT COUNT(*) AS n FROM {table}")
            assert row["n"] == 0, (
                f"{table}: joined leg writes survived the outer rollback")
        row = await db.fetchone(
            "SELECT COUNT(*) AS n FROM markets WHERE id IN"
            " ('ent-a', 'ent-b')")
        assert row["n"] == 0
        # No strand.
        await db.execute(
            "INSERT OR IGNORE INTO markets (id, question) VALUES ('x','q')")
        # Clean call: submit_paired precedes the FIRST span begin (the
        # joined helper spans also surface as same-owner events). Clear the
        # crashed attempt's events so the indices below measure THIS call.
        events.clear()
        with transaction_spy(db, events):
            assert await pillar._enter_pair(
                implier, implied, 0.3, "ladder", 1.0, False) is True
        assert "entailment_arb.entry" in span_owners(events)
        # The joined helper spans surface under their own owner but commit
        # nothing themselves — the fold is proven by the crash above.
        assert "entailment_arb.leg" in span_owners(events)
        assert (events.index(("gateway", "submit_paired"))
                < events.index(("entailment_arb.entry", "begin")))
        row = await db.fetchone("SELECT COUNT(*) AS n FROM signals")
        assert row["n"] == 2
        row = await db.fetchone("SELECT COUNT(*) AS n FROM portfolio")
        assert row["n"] == 2
        row = await db.fetchone(
            "SELECT traded_at FROM entailment_verdicts "
            "WHERE market_id_a = 'ent-a' AND market_id_b = 'ent-b'")
        assert row["traded_at"] is not None
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# resolution_lens — markets+signals+portfolio (owner resolution_lens.entry)
# ---------------------------------------------------------------------------


def _lens_pillar(db, events: list):
    from auramaur.strategy.resolution_lens import ResolutionLensPillar

    pillar = object.__new__(ResolutionLensPillar)
    pillar._db = db
    pillar._exchange_name = "polymarket"
    pillar._source_tag = "resolution_lens"
    pillar._settings = SimpleNamespace(
        resolution_lens=SimpleNamespace(
            min_entry_price=0.6, high_conf_gap_score=0.75, stake_usd=10.0,
            paper=True))
    pillar._risk = _Risk()
    pillar._gateway = _Gateway(events)
    pillar._calibration = _Calibration(events)
    return pillar


def _lens_signal(mid: str):
    from auramaur.exchange.models import Confidence, Signal

    return Signal(market_id=mid, market_question=f"Q {mid}?",
                  claude_prob=0.85, claude_confidence=Confidence.HIGH,
                  market_prob=0.7, edge=15.0,
                  evidence_summary="Resolution lens (gap 0.80): fine print",
                  recommended_side=OrderSide.BUY,
                  strategy_source="resolution_lens")


@pytest.mark.asyncio
async def test_lens_record_position_batch_is_atomic_under_mid_batch_crash():
    db = Database(":memory:")
    await db.connect()
    try:
        pillar = _lens_pillar(db, [])
        mid = "rl-m1"
        with failing_on(db, "INSERT INTO signals"):
            with pytest.raises(sqlite3.OperationalError, match="injected"):
                await pillar._record_position(
                    _lens_signal(mid), _market(mid, yes=0.7), _order(mid),
                    _submit_result(mid).result)
        for table in ("signals", "portfolio"):
            row = await db.fetchone(f"SELECT COUNT(*) AS n FROM {table}")
            assert row["n"] == 0, f"partial write survived in {table}"
        row = await db.fetchone(
            "SELECT COUNT(*) AS n FROM markets WHERE id = ?", (mid,))
        assert row["n"] == 0, "markets stub survived the crashed batch"
        # No strand.
        await db.execute(
            "INSERT OR IGNORE INTO markets (id, question) VALUES ('x','q')")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_lens_entry_owner_ordering_and_calibration_outside():
    """Through the real _enter path: the gateway submit completes before
    the resolution_lens.entry span begins, calibration fires only after
    the span ends, and all three rows land under the distinct owner."""
    db = Database(":memory:")
    await db.connect()
    try:
        events: list = []
        pillar = _lens_pillar(db, events)
        m = _market("rl-m2", yes=0.7)
        with transaction_spy(db, events):
            entered = await pillar._enter(
                m, 0.85, 0.8, "criteria require an official announcement",
                0.15)
        assert entered is True
        assert "resolution_lens.entry" in span_owners(events)
        assert events.count(("resolution_lens.entry", "begin")) == 1
        assert (events.index(("gateway", "submit"))
                < events.index(("resolution_lens.entry", "begin")))
        assert (events.index(("resolution_lens.entry", "end"))
                < events.index(("calibration", "record")))
        for table in ("signals", "portfolio"):
            row = await db.fetchone(f"SELECT COUNT(*) AS n FROM {table}")
            assert row["n"] == 1, table
        row = await db.fetchone(
            "SELECT COUNT(*) AS n FROM markets WHERE id = 'rl-m2'")
        assert row["n"] == 1
    finally:
        await db.close()
