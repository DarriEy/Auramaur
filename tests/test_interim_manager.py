"""Interim manager: charter rules, delegation, sunset, and ladder humility."""

import asyncio
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, MagicMock


from auramaur.db.database import Database
from auramaur.exchange.models import Market
from auramaur.strategy.interim_manager import InterimManagerPillar
from config.settings import Settings

THESIS = ("Resolution fine print prices NO events as YES; venue mid ignores "
          "the settlement source's published revision schedule.")


def _market(mid="M1", price=0.40, category="economics"):
    return Market(id=mid, question="Will X happen?", description="",
                  category=category, outcome_yes_price=price,
                  outcome_no_price=1 - price, volume=1000, liquidity=500,
                  exchange="kalshi")


async def _pillar(tmp_proposals=(), ladder_cells=(), risk_approves=True):
    db = Database(":memory:")
    await db.connect()
    for p in tmp_proposals:
        await db.execute(
            """INSERT INTO manager_proposals
               (venue, market_id, side, fair_prob, stake_usd, thesis, status)
               VALUES (?, ?, ?, ?, ?, ?, 'pending')""", p)
    await db.commit()

    settings = Settings()
    settings.interim_manager.enabled = True

    discovery = MagicMock()
    discovery.get_market = AsyncMock(return_value=_market())
    exchange = MagicMock()
    risk = MagicMock()
    risk.evaluate = AsyncMock(return_value=SimpleNamespace(
        approved=risk_approves, position_size=10.0, reason="",
        force_paper=True))
    calibration = MagicMock()
    calibration.record_prediction = AsyncMock()

    pillar = InterimManagerPillar(
        db=db, settings=settings,
        discoveries={"kalshi": discovery}, exchanges={"kalshi": exchange},
        risk_manager=risk, pnl_tracker=None, calibration=calibration)
    pillar._ladder = MagicMock()
    pillar._ladder.report = AsyncMock(return_value=list(ladder_cells))

    order = SimpleNamespace(price=0.41, market_id="M1",
                            token=SimpleNamespace(value="YES"))
    result = SimpleNamespace(is_paper=True, filled_size=10.0, filled_price=0.41)
    gateway = MagicMock()
    gateway.submit = AsyncMock(return_value=SimpleNamespace(
        status="paper", order=order, result=result, reason=""))
    pillar._gateways = {"kalshi": gateway}
    return pillar, db, gateway, risk


PROPOSAL = ("kalshi", "M1", "BUY", 0.55, 10.0, THESIS)


def test_executes_valid_proposal_paper_forced():
    async def run():
        pillar, db, gateway, risk = await _pillar(tmp_proposals=[PROPOSAL])
        assert await pillar.run_once() == 1
        intent = gateway.submit.await_args.args[0]
        assert intent.signal.strategy_source == "interim_manager"
        assert intent.force_paper is True
        row = await db.fetchone("SELECT status FROM manager_proposals")
        assert row["status"] == "executed"
        await db.close()
    asyncio.run(run())


def test_thin_thesis_is_skipped_by_charter():
    async def run():
        pillar, db, gateway, _ = await _pillar(
            tmp_proposals=[("kalshi", "M1", "BUY", 0.55, 10.0, "just vibes")])
        assert await pillar.run_once() == 0
        gateway.submit.assert_not_awaited()
        row = await db.fetchone("SELECT status, reason FROM manager_proposals")
        assert row["status"] == "skipped"
        assert "mechanism" in row["reason"]
        await db.close()
    asyncio.run(run())


def test_delegates_category_to_graduated_strategy_cell():
    async def run():
        pillar, db, gateway, _ = await _pillar(
            tmp_proposals=[PROPOSAL],
            ladder_cells=[{"strategy": "econ_indicator",
                           "category": "economics", "status": "probation"}])
        assert await pillar.run_once() == 0
        gateway.submit.assert_not_awaited()
        row = await db.fetchone("SELECT status, reason FROM manager_proposals")
        assert row["status"] == "skipped"
        assert "delegated to econ_indicator" in row["reason"]
        await db.close()
    asyncio.run(run())


def test_exempt_strategies_do_not_trigger_delegation_or_sunset():
    async def run():
        pillar, db, gateway, _ = await _pillar(
            tmp_proposals=[PROPOSAL],
            ladder_cells=[{"strategy": "arbitrage", "category": "economics",
                           "status": "live"},
                          {"strategy": "market_maker", "category": "economics",
                           "status": "live"}])
        assert await pillar.run_once() == 1
        await db.close()
    asyncio.run(run())


def test_sunset_expires_queue_once_enough_cells_graduate():
    async def run():
        cells = [{"strategy": f"s{i}", "category": f"c{i}", "status": "live"}
                 for i in range(3)]
        pillar, db, gateway, _ = await _pillar(
            tmp_proposals=[PROPOSAL], ladder_cells=cells)
        assert await pillar.run_once() == 0
        gateway.submit.assert_not_awaited()
        row = await db.fetchone("SELECT status, reason FROM manager_proposals")
        assert row["status"] == "expired"
        assert "sunset" in row["reason"]
        await db.close()
    asyncio.run(run())


def test_risk_rejection_is_recorded_not_silently_dropped():
    async def run():
        pillar, db, gateway, risk = await _pillar(
            tmp_proposals=[PROPOSAL], risk_approves=False)
        assert await pillar.run_once() == 0
        row = await db.fetchone("SELECT status, reason FROM manager_proposals")
        assert row["status"] == "skipped"
        assert row["reason"].startswith("risk:")
        await db.close()
    asyncio.run(run())


def test_disabled_by_default_in_tracked_config():
    s = Settings()
    assert s.interim_manager.enabled is False
    assert s.interim_manager.paper is True
    assert "interim_manager" not in set(s.graduation.exempt_strategies)


# ---- v2: the generalized decision rule ----

async def _propose_row(db, **overrides):
    fields = dict(venue="kalshi", market_id="M1", side="BUY", fair_prob=0.55,
                  stake_usd=10.0, thesis=THESIS, status="pending",
                  thesis_class="forecast_divergence")
    fields.update(overrides)
    cols = ", ".join(fields)
    marks = ", ".join("?" for _ in fields)
    await db.execute(
        f"INSERT INTO manager_proposals ({cols}) VALUES ({marks})",
        tuple(fields.values()))
    await db.commit()


def test_max_entry_price_gate_blocks_expensive_fills():
    async def run():
        pillar, db, gateway, _ = await _pillar()
        await _propose_row(db, max_entry_price=0.35)  # market mid is 0.40
        assert await pillar.run_once() == 0
        gateway.submit.assert_not_awaited()
        row = await db.fetchone("SELECT status, reason FROM manager_proposals")
        assert row["status"] == "skipped"
        assert "above limit" in row["reason"]
        await db.close()
    asyncio.run(run())


def test_robust_edge_gate_rejects_thin_edges():
    async def run():
        pillar, db, gateway, _ = await _pillar()
        # fair 0.47 vs mid 0.40: gross 0.07 − default buffers ≈ below 0.05 gate
        await _propose_row(db, fair_prob=0.47)
        assert await pillar.run_once() == 0
        gateway.submit.assert_not_awaited()
        row = await db.fetchone(
            "SELECT status, reason, robust_edge FROM manager_proposals")
        assert row["status"] == "skipped"
        assert "robust edge" in row["reason"]
        assert row["robust_edge"] is not None
        await db.close()
    asyncio.run(run())


def test_wide_robust_edge_executes_and_records_decision_price():
    async def run():
        pillar, db, gateway, _ = await _pillar()
        # fair 0.72 vs mid 0.40 with a tight CI: survives every haircut
        await _propose_row(db, fair_prob=0.72, confidence_lo=0.70,
                           confidence_hi=0.74)
        assert await pillar.run_once() == 1
        row = await db.fetchone(
            "SELECT status, robust_edge, decision_price FROM manager_proposals")
        assert row["status"] == "executed"
        assert row["robust_edge"] > 0.05
        assert row["decision_price"] == pytest.approx(0.40)
        await db.close()
    asyncio.run(run())


def test_thesis_sunset_expires_before_execution():
    async def run():
        pillar, db, gateway, _ = await _pillar()
        await _propose_row(db, fair_prob=0.72, sunset_at="2020-01-01T00:00:00+00:00")
        assert await pillar.run_once() == 0
        gateway.submit.assert_not_awaited()
        row = await db.fetchone("SELECT status, reason FROM manager_proposals")
        assert row["status"] == "expired"
        assert "sunset" in row["reason"]
        await db.close()
    asyncio.run(run())


def test_ci_half_width_is_the_uncertainty_haircut():
    async def run():
        pillar, db, gateway, _ = await _pillar()
        # A modest edge with a sloppy CI (±0.15) must die where the same
        # edge with a tight CI would survive — the haircut is the CI width.
        await _propose_row(db, fair_prob=0.55, confidence_lo=0.40,
                           confidence_hi=0.70)
        assert await pillar.run_once() == 0
        row = await db.fetchone("SELECT status, reason FROM manager_proposals")
        assert row["status"] == "skipped" and "robust edge" in row["reason"]
        await db.close()
    asyncio.run(run())


def test_startup_migrates_an_existing_v31_database(tmp_path):
    """The v32 rollout crashed production: the base DDL indexed a column the
    pre-migration table lacked. Startup against a v31 DB must succeed."""
    import sqlite3 as sq

    db_file = tmp_path / "v31.db"
    conn = sq.connect(db_file)
    conn.executescript(
        """CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
           INSERT INTO schema_version (version) VALUES (31);
           CREATE TABLE manager_proposals (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               venue TEXT NOT NULL, market_id TEXT NOT NULL, side TEXT NOT NULL,
               fair_prob REAL NOT NULL, stake_usd REAL NOT NULL,
               thesis TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'pending',
               reason TEXT NOT NULL DEFAULT '',
               created_at TEXT NOT NULL DEFAULT (datetime('now')),
               decided_at TEXT);""")
    conn.commit()
    conn.close()

    async def run():
        db = Database(str(db_file))
        await db.connect()  # crashed before the fix
        row = await db.fetchone("SELECT version FROM schema_version")
        assert row["version"] >= 32
        cols = {r["name"] for r in await db.fetchall(
            "SELECT name FROM pragma_table_info('manager_proposals')")}
        assert "thesis_class" in cols and "robust_edge" in cols
        idx = await db.fetchone(
            "SELECT 1 AS x FROM sqlite_master WHERE type='index' "
            "AND name='idx_manager_proposals_class'")
        assert idx is not None
        await db.close()
    asyncio.run(run())
