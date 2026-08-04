"""Phase 3 of the txn migration (#353): the non-money bookkeeping sweep.

The per-writer pattern (docs/plans/txn-migration-plan.md): crash the
SECOND statement of the batch and assert zero rows from ANY statement of
the batch (atomicity), assert the writer's distinct owner label on a
clean call, and prove the connection accepts unrelated writes after the
injected failure (no strand — the 2026-07-25 wedge class).

The eight ``_persist_signal`` helpers are byte-similar, so one
parametrized test covers all eight. Instances come from
``object.__new__`` because the helper touches only ``self._db`` (plus
``_tag``/``_exchange_name`` where noted) — house precedent:
tests/test_experiment_agent_trader.py, tests/test_experiment_platform.py.
"""

from __future__ import annotations

import importlib
import inspect
import sqlite3
from contextlib import contextmanager

import pytest

from auramaur.db.database import Database
from auramaur.exchange.models import Confidence, Market, OrderSide, Signal
from tests.txn_helpers import failing_on, span_owners, transaction_spy

# (module, class, owner label, extra attrs _persist_signal reads)
PILLARS = [
    ("auramaur.strategy.bias_harvest", "BiasHarvestPillar",
     "bias_harvest.signal", {}),
    ("auramaur.strategy.econ_indicator", "EconIndicatorPillar",
     "econ_indicator.signal", {}),
    ("auramaur.strategy.informed_flow_pillar", "InformedFlowPillar",
     "informed_flow.signal", {}),
    ("auramaur.strategy.long_horizon", "LongHorizonPillar",
     "long_horizon.signal", {"_tag": "long_horizon"}),
    ("auramaur.strategy.vol_anchor", "VolAnchorPillar",
     "vol_anchor.signal", {}),
    ("auramaur.strategy.weather_temp", "WeatherTempPillar",
     "weather_temp.signal", {}),
    ("auramaur.strategy.agent_trader", "AgentTraderPillar",
     "agent_trader.signal", {"_exchange_name": "polymarket"}),
    ("auramaur.strategy.term_structure", "TermStructurePillar",
     "term_structure.signal", {}),
]


def _signal(mid: str) -> Signal:
    return Signal(market_id=mid, claude_prob=0.6,
                  claude_confidence=Confidence.LOW, market_prob=0.5,
                  edge=10.0, recommended_side=OrderSide.BUY,
                  strategy_source="txn3_test")


def _market(mid: str) -> Market:
    return Market(id=mid, question=f"Q {mid}?", outcome_yes_price=0.5,
                  outcome_no_price=0.5, exchange="polymarket")


def _instance(module: str, cls_name: str, db, extra: dict):
    cls = getattr(importlib.import_module(module), cls_name)
    pillar = object.__new__(cls)
    pillar._db = db
    for key, value in extra.items():
        setattr(pillar, key, value)
    return pillar


@contextmanager
def _failing_on_nth(db, needle: str, n: int):
    """Like txn_helpers.failing_on, but only the n-th matching statement
    raises — needed for batches whose statements share one SQL shape
    (the _read_family curve loop)."""
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
# The eight mirror _persist_signal helpers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("module,cls_name,owner,extra", PILLARS,
                         ids=[p[2] for p in PILLARS])
async def test_persist_signal_is_atomic_and_carries_its_owner(
        module, cls_name, owner, extra):
    db = Database(":memory:")
    await db.connect()
    try:
        pillar = _instance(module, cls_name, db, extra)
        mid = "txn3-m1"
        # (a) crash the SECOND statement: the markets stub must not
        # survive alone.
        with failing_on(db, "INSERT INTO signals"):
            with pytest.raises(sqlite3.OperationalError, match="injected"):
                await pillar._persist_signal(_signal(mid), _market(mid))
        row = await db.fetchone(
            "SELECT COUNT(*) AS n FROM markets WHERE id = ?", (mid,))
        assert row["n"] == 0, "markets stub survived the crashed batch"
        row = await db.fetchone("SELECT COUNT(*) AS n FROM signals")
        assert row["n"] == 0
        # No strand: the connection accepts unrelated writes immediately.
        await db.execute(
            "INSERT OR IGNORE INTO markets (id, question) VALUES ('other','q')")
        # (b) a clean call lands both rows under the distinct owner.
        events: list = []
        with transaction_spy(db, events):
            await pillar._persist_signal(_signal(mid), _market(mid))
        assert span_owners(events) == [owner]
        assert events.count((owner, "end")) == 1
        row = await db.fetchone(
            "SELECT COUNT(*) AS n FROM signals WHERE market_id = ?", (mid,))
        assert row["n"] == 1
        row = await db.fetchone(
            "SELECT COUNT(*) AS n FROM markets WHERE id = ?", (mid,))
        assert row["n"] == 1
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# engine_cycle: the two inline markets+signals pairs
# ---------------------------------------------------------------------------


def test_engine_cycle_signal_pairs_are_spanned_and_risk_stays_outside():
    """Source-inspection (house precedent above in this directory:
    test_engine_cycle.py) — the full-cycle harness is heavy, and the pair
    is byte-similar to the pillar helpers proven atomic above. Asserts:
    the span opens before both statements of the pair, the risk gate sits
    OUTSIDE the with-block (tight-span rule), and the dead commit is
    gone."""
    from auramaur.strategy.engine_cycle import CycleOrchestrationMixin

    for method in (CycleOrchestrationMixin._execute_candidates,
                   CycleOrchestrationMixin._run_cycle_strategic):
        src = inspect.getsource(method)
        begin = src.index('transaction(owner="engine_cycle.signal")')
        markets = src.index("INSERT OR IGNORE INTO markets")
        signals = src.index("INSERT INTO signals")
        gate = src.index("risk_manager.evaluate(")
        assert begin < markets < signals < gate, method.__name__
        # The with-block closes before the risk gate: the evaluate line
        # sits at the SAME indent as the `async with`, not inside it.
        lines = src.splitlines()
        with_line = next(ln for ln in lines
                         if 'transaction(owner="engine_cycle.signal")' in ln)
        gate_line = next(ln for ln in lines if "risk_manager.evaluate(" in ln)
        with_indent = len(with_line) - len(with_line.lstrip())
        gate_indent = len(gate_line) - len(gate_line.lstrip())
        assert gate_indent <= with_indent, (
            f"{method.__name__}: risk evaluation is inside the signal span")
        # The dead trailing commit is gone from the method.
        assert ".commit()" not in src, method.__name__


# ---------------------------------------------------------------------------
# term_structure._read_family: one LLM read -> N curve rows, atomically
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_family_curve_batch_is_atomic(tmp_path):
    from tests.test_term_structure import _ladder, _pillar as _ts_pillar

    reply = ('{"thesis": "t", "curve": [{"market_id": "a", "prob": 0.40},'
             '{"market_id": "b", "prob": 0.70},'
             '{"market_id": "c", "prob": 0.72}]}')
    pillar, db, _ = await _ts_pillar(tmp_path, _ladder(), reply)
    try:
        await pillar._ensure_schema()
        cfg = pillar._settings.term_structure
        # Crash the SECOND curve insert: the first row must not survive
        # alone (a partial curve poisons the strike-set cache comparison).
        with _failing_on_nth(db, "INSERT INTO term_structure_curves", 2):
            with pytest.raises(sqlite3.OperationalError, match="injected"):
                await pillar._read_family("event", _ladder(), cfg)
        row = await db.fetchone(
            "SELECT COUNT(*) AS n FROM term_structure_curves")
        assert row["n"] == 0, "partial curve survived the crashed batch"
        # No strand.
        await db.execute(
            "INSERT OR IGNORE INTO markets (id, question) VALUES ('m','q')")
        # Clean read: the LLM call precedes the span (tight-span ordering),
        # the owner label is distinct, and all N rows land.
        events: list = []

        async def _model(prompt, cfg):
            events.append(("llm", "call"))
            return reply

        pillar._call_model = _model
        with transaction_spy(db, events):
            out = await pillar._read_family("event", _ladder(), cfg)
        assert out is not None
        assert "term_structure.curve" in span_owners(events)
        assert (events.index(("llm", "call"))
                < events.index(("term_structure.curve", "begin")))
        rows = await db.fetchall(
            "SELECT market_id FROM term_structure_curves")
        assert {r["market_id"] for r in rows} == {"a", "b", "c"}
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# agent_trader._ensure_schema: the skipped-forever backfill regression
# ---------------------------------------------------------------------------

_PRE_CELL_THESES = """
CREATE TABLE agent_trader_theses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_alias TEXT NOT NULL,
    market_id TEXT NOT NULL,
    question TEXT DEFAULT '',
    token TEXT DEFAULT '',
    prob REAL,
    market_prob REAL,
    thesis TEXT DEFAULT '',
    stake REAL DEFAULT 0,
    entered INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now'))
)
"""


@pytest.mark.asyncio
async def test_ensure_schema_backfill_lands_on_the_run_after_a_crash():
    """A pre-`cell` table with rows, plus a crash between the ALTER and
    the UPDATE backfill: previously the next run's ALTER raised
    duplicate-column into a bare except that ALSO covered the backfill,
    so the rows stayed at cell='' forever. The backfill is unconditional
    now — it must land on the re-run."""
    from auramaur.strategy.agent_trader import AgentTraderPillar

    db = Database(":memory:")
    await db.connect()
    try:
        # The historical shape: a table from a pre-`cell` binary, with a row.
        await db.execute(_PRE_CELL_THESES)
        await db.execute(
            "INSERT INTO agent_trader_theses (model_alias, market_id)"
            " VALUES ('opus', 'm1')")
        pillar = object.__new__(AgentTraderPillar)
        pillar._db = db
        pillar._schema_ready = False
        # Run 1: the ALTER lands (each statement individually durable under
        # autocommit), then the backfill UPDATE crashes.
        with failing_on(db, "UPDATE agent_trader_theses"):
            with pytest.raises(sqlite3.OperationalError, match="injected"):
                await pillar._ensure_schema()
        row = await db.fetchone(
            "SELECT cell FROM agent_trader_theses WHERE market_id = 'm1'")
        assert row["cell"] == ""          # the crash-between state
        assert pillar._schema_ready is False
        # Run 2 (fault cleared): the ALTER raises duplicate-column, and the
        # backfill must land anyway.
        await pillar._ensure_schema()
        row = await db.fetchone(
            "SELECT cell FROM agent_trader_theses WHERE market_id = 'm1'")
        assert row["cell"] == "agent_trader_opus"
        assert pillar._schema_ready is True
        # Idempotent on a third run (backfill only touches cell='').
        await db.execute(
            "UPDATE agent_trader_theses SET cell = 'kept' WHERE market_id='m1'")
        pillar._schema_ready = False
        await pillar._ensure_schema()
        row = await db.fetchone(
            "SELECT cell FROM agent_trader_theses WHERE market_id = 'm1'")
        assert row["cell"] == "kept"
    finally:
        await db.close()
