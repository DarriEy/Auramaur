"""Tests for the graduation ladder (risk/graduation.py).

Locks in:
  1. Ladder states: live (record positive), demoted (live negative),
     probation (paper positive, half size), paper_negative, unproven.
  2. observe mode computes but never enforces; off mode is a no-op.
  3. Exempt strategies bypass the ladder.
  4. The trailing window excludes stale events.
  5. RiskManager.evaluate integration: force_paper set, probation
     multiplier applied to position_size, restriction-only.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from auramaur.db.database import Database
from auramaur.risk.graduation import GraduationLadder
from config.settings import GraduationConfig


def _settings(mode="enforce", min_events=5, **kw):
    s = MagicMock()
    s.graduation = GraduationConfig(mode=mode, min_events=min_events, **kw)
    return s


async def _seed(db, strategy, category, *, n, pnl_each, is_paper, days_ago=0):
    for i in range(n):
        await db.execute(
            """INSERT INTO pnl_ledger (market_id, venue, category, strategy_source,
               kind, token, qty, pnl, fees, is_paper, source_ref, realized_at)
               VALUES (?, 'polymarket', ?, ?, 'sell', 'YES', 1, ?, 0, ?,
                       ?, datetime('now', ?))""",
            (f"m-{strategy}-{category}-{is_paper}-{i}", category, strategy,
             pnl_each, is_paper,
             f"ref-{strategy}-{category}-{is_paper}-{i}", f"-{days_ago} days"),
        )
    await db.commit()


def test_ladder_states():
    async def run():
        db = Database(":memory:")
        await db.connect()
        ladder = GraduationLadder(db, _settings())

        # live positive -> live full
        await _seed(db, "s_live", "tech", n=5, pnl_each=1.0, is_paper=0)
        d = await ladder.decide("s_live", "tech")
        assert (d.force_paper, d.size_multiplier, d.status) == (False, 1.0, "live")

        # live negative -> demoted (paper-forced, full size for paper learning)
        await _seed(db, "s_bad", "tech", n=5, pnl_each=-1.0, is_paper=0)
        d = await ladder.decide("s_bad", "tech")
        assert d.force_paper is True and d.status == "demoted"
        assert d.size_multiplier == 1.0

        # paper positive -> probation at half size
        await _seed(db, "s_paper", "tech", n=5, pnl_each=1.0, is_paper=1)
        d = await ladder.decide("s_paper", "tech")
        assert d.force_paper is False and d.status == "probation"
        assert d.size_multiplier == 0.5

        # paper negative -> stays paper
        await _seed(db, "s_pneg", "tech", n=5, pnl_each=-1.0, is_paper=1)
        d = await ladder.decide("s_pneg", "tech")
        assert d.force_paper is True and d.status == "paper_negative"

        # nothing -> unproven
        d = await ladder.decide("s_new", "tech")
        assert d.force_paper is True and d.status == "unproven"

        # live wins over paper when both have >= min_events
        await _seed(db, "s_mixed", "tech", n=5, pnl_each=-1.0, is_paper=0)
        await _seed(db, "s_mixed", "tech", n=5, pnl_each=1.0, is_paper=1)
        d = await ladder.decide("s_mixed", "tech")
        assert d.status == "demoted"  # live record outranks paper

        await db.close()

    asyncio.run(run())


def test_observe_and_off_modes_do_not_enforce():
    async def run():
        db = Database(":memory:")
        await db.connect()

        ladder = GraduationLadder(db, _settings(mode="observe"))
        d = await ladder.decide("anything", "tech")  # unproven cell
        assert d.force_paper is False and d.size_multiplier == 1.0
        assert d.status == "observe:unproven"

        ladder = GraduationLadder(db, _settings(mode="off"))
        d = await ladder.decide("anything", "tech")
        assert d.force_paper is False and d.status == "live"
        await db.close()

    asyncio.run(run())


def test_exempt_strategies_bypass():
    async def run():
        db = Database(":memory:")
        await db.connect()
        ladder = GraduationLadder(db, _settings())
        for strat in ("arbitrage", "market_maker", "order_monitor"):
            d = await ladder.decide(strat, "tech")
            assert d.force_paper is False and d.status == "exempt"
        await db.close()

    asyncio.run(run())


def test_window_excludes_stale_events():
    async def run():
        db = Database(":memory:")
        await db.connect()
        ladder = GraduationLadder(db, _settings(window_days=30))
        # A glorious record... 100 days ago.
        await _seed(db, "s_old", "tech", n=10, pnl_each=5.0, is_paper=0, days_ago=100)
        d = await ladder.decide("s_old", "tech")
        assert d.status == "unproven"  # decayed out of the window
        await db.close()

    asyncio.run(run())


def test_risk_manager_integration():
    """evaluate() sets force_paper and applies the probation multiplier."""
    from tests.test_risk_manager import (
        _make_market, _make_settings, _make_signal, _mock_portfolio,
    )

    async def run():
        from auramaur.risk.checks import CheckResult
        from auramaur.risk.manager import RiskManager

        db = Database(":memory:")
        await db.connect()

        settings = _make_settings()
        settings.graduation = GraduationConfig(mode="enforce", min_events=5)

        with patch("auramaur.risk.manager.check_kill_switch") as mock_kill:
            mock_kill.return_value = CheckResult(
                name="kill_switch", passed=True, reason="", value=False)

            manager = RiskManager(settings, db)
            manager.portfolio = _mock_portfolio()
            signal = _make_signal(edge=10.0, claude_prob=0.60, market_prob=0.50)
            market = _make_market(category="tech")

            # Unproven cell: approved but paper-forced at full size.
            d = await manager.evaluate(signal, market, available_cash=500.0)
            assert d.approved is True
            assert d.force_paper is True
            assert d.graduation_status == "unproven"
            base_size = d.position_size
            assert base_size > 0

            # Probation cell: live with the multiplier applied.
            await _seed(db, "llm", "tech", n=5, pnl_each=1.0, is_paper=1)
            manager.graduation._cache.clear()
            d2 = await manager.evaluate(signal, market, available_cash=500.0)
            assert d2.force_paper is False
            assert d2.graduation_status == "probation"
            assert abs(d2.position_size - base_size * 0.5) < 1e-6

            # Live-positive cell: untouched.
            await _seed(db, "llm", "tech", n=5, pnl_each=1.0, is_paper=0)
            manager.graduation._cache.clear()
            d3 = await manager.evaluate(signal, market, available_cash=500.0)
            assert d3.force_paper is False
            assert d3.graduation_status == "live"
            assert abs(d3.position_size - base_size) < 1e-6

        await db.close()

    asyncio.run(run())


def test_bias_harvest_honors_force_paper():
    from tests.test_bias_harvest import _exchange, _market, _pillar, _risk, _settings as _bh_settings

    async def run():
        from unittest.mock import PropertyMock

        from config.settings import Settings

        db = Database(":memory:")
        await db.connect()
        # paper=False so only graduation's force_paper controls the flag.
        settings = _bh_settings(paper=False)
        ex = _exchange()
        pillar, _ = _pillar(db, settings, [_market()], exchange=ex,
                            risk=_risk(force_paper=True))
        with patch.object(type(settings), "is_live",
                          new_callable=PropertyMock, return_value=True):
            assert isinstance(settings, Settings)
            await pillar.run_once()
        assert ex.prepare_order.call_args[0][3] is False  # paper-forced
        await db.close()

    asyncio.run(run())


async def _seed_paper_positions(db, n, offset=0):
    for i in range(offset, offset + n):
        await db.execute(
            "INSERT INTO portfolio (market_id, exchange, side, size, avg_price, "
            "current_price, is_paper) VALUES (?, 'polymarket', 'BUY', 5, 0.5, 0.5, 1)",
            (f"pp-{i}",),
        )
    await db.commit()


def test_unproven_spray_cap():
    """When the open paper book is already at the breadth cap, an UNPROVEN cell
    returns size x0 (skip) so exploration concentrates instead of spraying.
    Proven/probation/exempt cells are unaffected."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        ladder = GraduationLadder(db, _settings(max_unproven_positions=10))

        # Under the cap: a fresh (unproven) cell still explores at full paper size.
        await _seed_paper_positions(db, 5)
        d = await ladder.decide("s_new", "tech")
        assert (d.force_paper, d.size_multiplier, d.status) == (True, 1.0, "unproven")

        # Cross the cap -> new unproven entries are skipped (x0).
        ladder2 = GraduationLadder(db, _settings(max_unproven_positions=10))
        await _seed_paper_positions(db, 8, offset=5)   # now 13 >= 10
        d2 = await ladder2.decide("s_new2", "tech")
        assert d2.size_multiplier == 0.0 and d2.status == "unproven_capped"

        # A PROVEN (live-positive) cell is NOT capped — restriction targets spray.
        await _seed(db, "s_live", "tech", n=5, pnl_each=1.0, is_paper=0)
        ladder3 = GraduationLadder(db, _settings(max_unproven_positions=10))
        d3 = await ladder3.decide("s_live", "tech")
        assert (d3.force_paper, d3.size_multiplier, d3.status) == (False, 1.0, "live")
        await db.close()

    asyncio.run(run())


def test_unproven_spray_cap_disabled_when_zero():
    async def run():
        db = Database(":memory:")
        await db.connect()
        ladder = GraduationLadder(db, _settings(max_unproven_positions=0))
        await _seed_paper_positions(db, 50)
        d = await ladder.decide("s_new", "tech")
        assert d.status == "unproven" and d.size_multiplier == 1.0  # cap off
        await db.close()

    asyncio.run(run())
