"""Accounting that decides capital must measure what actually happened.

Five independent ways it did not: a token-blind join double-counting both
sides of a market, a fee subtracted twice (in a display panel, in the
paper->live promotion gate, and in the Kraken desk's realised P&L), a settle
that dropped the winning leg, paired-arb decision ids thrown away, and a Brier
gate scored against a coin instead of the instrument's drift.

Claims about BEHAVIOUR are tested by running the behaviour against the real
SQLite schema. Source-text assertions appear only where the property really is
about the source — "no join anywhere in this module is unscoped" is such a
property, and it guards joins not yet written; "this query returns two rows,
not four" is not, and a string test standing in for one proves nothing. The
fee test in the first draft of this file was exactly that mistake: it split on
``risk_free_benchmark(``, which matches the ``def`` line 127 lines earlier, so
it inspected the function SIGNATURE and passed identically before and after
the fix it was supposed to be pinning.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

from auramaur.db.database import Database
from auramaur.evaluation.etf_calibration import COIN, Forecast, clearance
from auramaur.monitoring.attribution import PerformanceAttributor


def _bench_settings():
    """Enough of Settings for risk_free_benchmark; no config file needed."""
    return SimpleNamespace(benchmark=SimpleNamespace(
        risk_free_annual_rate=0.04, book_capital_usd=1000.0))


async def _both_sided_market(db, *, market_id="mkt-both", is_paper=1,
                             size=100.0, avg=0.40, mark=0.50):
    """A market held on BOTH sides — structurally normal for entailment_arb's
    both-or-nothing legs and for paired arbitrage. Two portfolio rows and two
    cost_basis rows, keyed (market_id, is_paper, token) exactly as the schema
    keys them."""
    await db.execute(
        """INSERT INTO markets (id, question, category, last_updated)
           VALUES (?, 'Both sides?', 'crypto', datetime('now'))""",
        (market_id,))
    for token in ("YES", "NO"):
        await db.execute(
            """INSERT INTO portfolio (market_id, exchange, side, size, avg_price,
                                      current_price, token, category, is_paper,
                                      updated_at)
               VALUES (?, 'polymarket', 'BUY', ?, ?, ?, ?, 'crypto', ?,
                       datetime('now'))""",
            (market_id, size, avg, mark, token, is_paper))
        await db.execute(
            """INSERT INTO cost_basis (market_id, token, token_id, size, avg_cost,
                                       total_cost, realized_pnl, is_paper)
               VALUES (?, ?, ?, ?, ?, ?, 0, ?)""",
            (market_id, token, f"tok-{token}", size, avg, avg * size, is_paper))
    await db.commit()


# --- token scoping ---------------------------------------------------------

def test_every_attribution_cost_basis_join_is_token_scoped():
    """A source-level property gets a source-level test: no join added LATER
    may be unscoped either. What those joins actually return is pinned by the
    DB-backed tests below."""
    from auramaur.monitoring import attribution
    src = inspect.getsource(attribution)
    joins = src.count("JOIN cost_basis cb") + src.count("JOIN portfolio p ON cb.market_id")
    scoped = src.count("cb.token = p.token") + src.count("p.token = cb.token")
    assert joins > 0
    assert scoped == joins, f"{joins - scoped} join(s) still unscoped by token"


def test_category_summary_does_not_fan_out_a_both_sided_market():
    """portfolio and cost_basis are BOTH keyed (market_id, is_paper, token).
    Joining on two of the three fans each portfolio row out per cost_basis
    token: 4 positions and $160 exposure where the truth is 2 and $80."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            await _both_sided_market(db)
            rows = await PerformanceAttributor(db).get_category_summary(
                is_live=False)
            crypto = [r for r in rows if r["category"] == "crypto"]
            assert len(crypto) == 1, rows
            row = crypto[0]
            # Two legs, not four. $80 committed, not $160.
            assert row["positions"] == 2, row
            assert abs(row["exposure"] - 80.0) < 1e-9, row
            # (0.50 mark - 0.40 cost) * 100 per leg = $20 together, not $40.
            assert abs(row["unrealized_pnl"] - 20.0) < 1e-9, row
        finally:
            await db.close()
    asyncio.run(run())


def test_category_summary_realized_does_not_double_count_both_sides():
    """The sold_rows arm joins the other direction (cost_basis -> portfolio)
    and fanned out the same way: $28 reported for $14 earned."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            await _both_sided_market(db)
            # Each leg realized $7 by selling; $14 together.
            await db.execute("UPDATE cost_basis SET realized_pnl = 7.0 "
                             "WHERE market_id = 'mkt-both'")
            await db.commit()
            rows = await PerformanceAttributor(db).get_category_summary(
                is_live=False)
            crypto = [r for r in rows if r["category"] == "crypto"][0]
            assert abs(crypto["realized_pnl"] - 14.0) < 1e-9, crypto
        finally:
            await db.close()
    asyncio.run(run())


def test_venue_summary_does_not_fan_out_a_both_sided_market():
    """The same join, grouped by exchange instead of by category."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            await _both_sided_market(db)
            rows = await PerformanceAttributor(db).get_venue_summary(
                is_live=False)
            poly = [r for r in rows if r["venue"] == "polymarket"]
            assert len(poly) == 1, rows
            assert poly[0]["positions"] == 2, poly[0]
            assert abs(poly[0]["exposure"] - 80.0) < 1e-9, poly[0]
            assert abs(poly[0]["unrealized_pnl"] - 20.0) < 1e-9, poly[0]
        finally:
            await db.close()
    asyncio.run(run())


def test_settlement_deletes_only_the_leg_it_settled():
    """With token_scope=None the SELECT reads ONE row via fetchone, but the
    DELETE removed EVERY token row for the market. The unsettled side vanished
    from the portfolio while its cost_basis row survived at size > 0 — so the
    winning leg never settled and never could."""
    async def run():
        from auramaur.strategy.resolution_tracker import ResolutionTracker
        db = Database(":memory:")
        await db.connect()
        try:
            await _both_sided_market(db, is_paper=0)
            tracker = ResolutionTracker(db=db, calibration=AsyncMock(),
                                        discoveries={})
            # Unscoped call: the caller names no token, so ONE leg settles.
            await tracker._settle_position("mkt-both", outcome=True)

            survivors = await db.fetchall(
                "SELECT token FROM portfolio WHERE market_id = 'mkt-both'")
            assert len(survivors) == 1, (
                "settling one leg must delete one leg, got "
                f"{[dict(r) for r in survivors]}")
            other = survivors[0]["token"].upper()

            # The surviving leg must still be settleable, not orphaned.
            cb = await db.fetchone(
                "SELECT size FROM cost_basis WHERE market_id='mkt-both' "
                "AND UPPER(token) = ?", (other,))
            assert cb["size"] > 0, "the unsettled leg must keep its cost basis"

            # A second pass picks it up instead of finding nothing.
            await tracker._settle_position("mkt-both", outcome=True)
            left = await db.fetchall(
                "SELECT token FROM portfolio WHERE market_id = 'mkt-both'")
            assert len(left) == 0, [dict(r) for r in left]
            booked = await db.fetchall(
                "SELECT token FROM pnl_ledger WHERE market_id = 'mkt-both'")
            assert {r["token"].upper() for r in booked} == {"YES", "NO"}, (
                f"both legs must reach the ledger, got {[dict(r) for r in booked]}")
        finally:
            await db.close()
    asyncio.run(run())


# --- fees ------------------------------------------------------------------

def test_benchmark_does_not_subtract_fees_twice():
    """pnl_ledger.pnl is ALREADY net of fees at every writer (pnl.py books
    ``(price - avg_cost) * size - fill.fee``); the fees column is the
    breakdown of what has already been deducted. Subtracting it again printed
    "net of fees -$15.00 ... behind cash" in bold red two lines under a
    +$40.00 header."""
    async def run():
        from auramaur.monitoring.ledger_report import gather_ledger_report
        db = Database(":memory:")
        await db.connect()
        try:
            # A book at +$40 realized, on which $55 of fees were ALREADY
            # charged before pnl was written. Two rows so the span is > 1 day
            # and the benchmark is computable.
            await db.execute(
                """INSERT INTO pnl_ledger (market_id, venue, category,
                       strategy_source, kind, token, qty, pnl, fees, is_paper,
                       source_ref, realized_at)
                   VALUES ('fee-mkt', 'polymarket', 'crypto', 'fee_test',
                           'sell', 'YES', 1, 40.0, 55.0, 1, 'fee-ref-1',
                           datetime('now', '-30 days'))""")
            await db.execute(
                """INSERT INTO pnl_ledger (market_id, venue, category,
                       strategy_source, kind, token, qty, pnl, fees, is_paper,
                       source_ref, realized_at)
                   VALUES ('fee-mkt-2', 'polymarket', 'crypto', 'fee_test',
                           'sell', 'YES', 1, 0.0, 0.0, 1, 'fee-ref-2',
                           datetime('now'))""")
            await db.commit()

            state = await gather_ledger_report(db, is_paper=True,
                                               settings=_bench_settings())
            assert abs(state["total"]["pnl"] - 40.0) < 1e-9, state["total"]
            assert abs(state["total"]["fees"] - 55.0) < 1e-9, state["total"]
            # THE assertion: the benchmark's "net of fees" is +$40, not -$15.
            assert abs(state["benchmark"]["net_pnl"] - 40.0) < 1e-9, \
                state["benchmark"]
            assert state["benchmark"]["available"] is True
            assert state["benchmark"]["annualised"] > 0, \
                "a profitable book must not annualise negative"
        finally:
            await db.close()
    asyncio.run(run())


def test_kraken_desk_realised_pnl_does_not_subtract_fees_twice():
    """kraken_directional is the only book in the live ledger carrying
    non-zero fees, so this is the one site where the double-count had a dollar
    effect rather than a $0 one: -$43.98 reported against a true -$36.53."""
    async def run():
        from auramaur.treasury.kraken_pillar import KrakenPillar
        db = Database(":memory:")
        await db.connect()
        try:
            await db.execute(
                """INSERT INTO pnl_ledger (market_id, venue, category,
                       strategy_source, kind, token, qty, pnl, fees, is_paper,
                       source_ref)
                   VALUES ('XBTUSD', 'kraken', 'crypto', 'kraken_directional',
                           'sell', 'YES', 1, -36.5336, 7.4414, 0, 'k-ref-1')""")
            await db.commit()
            pillar = KrakenPillar.__new__(KrakenPillar)
            pillar._db = db
            got = await pillar._realised_pnl_usd()
            assert abs(got - (-36.5336)) < 1e-9, \
                f"double-counted fees read -43.9750; got {got}"
        finally:
            await db.close()
    asyncio.run(run())


# --- decision ids ----------------------------------------------------------

def test_paired_submission_plumbs_decision_ids():
    """Discarding them left order.decision_id unset, so the mark_fill block in
    _place_and_record never ran and every paired-arb snapshot stayed filled=0
    forever — and require_executable_fills then filtered out the whole holdout
    cohort."""
    from auramaur.broker import execution_gateway
    src = inspect.getsource(execution_gateway.ExecutionGateway.submit_paired)
    assert "decision_id_a = await self._capture_decision" in src
    assert "decision_id_b = await self._capture_decision" in src
    assert "decision_id=decision_id_a" in src
    assert "decision_id=decision_id_b" in src


# --- Brier benchmark -------------------------------------------------------

def _flat_forecaster(n: int, up_rate: float, reference):
    """A forecaster with ZERO information: it answers the base rate itself."""
    outcomes = [1 if i < round(n * up_rate) else 0 for i in range(n)]
    return [Forecast(up_rate, "MEDIUM", o, reference) for o in outcomes]


def test_a_coin_benchmark_hands_out_free_edge_of_exactly_the_drift_gap():
    """E[Brier] of a constant forecast r under base rate q is (r-q)^2 + q(1-q),
    so scoring against 0.5 instead of the instrument's own ~0.56 drift adds
    exactly (0.5-q)^2 to every forecast's edge, information or not.

    The SIZE of that constant is the whole argument, so it is pinned here
    rather than asserted in prose — and it is small: 0.0036 does NOT clear the
    gate at a few hundred resolutions. A flat forecaster needs over a
    thousand. Anything claiming ~400 is wrong."""
    view = clearance(_flat_forecaster(400, 0.56, reference=COIN),
                     min_resolved=100)
    assert abs(view.brier_edge - 0.0036) < 1e-9, view       # (0.5 - 0.56)^2
    assert view.cleared is False
    assert -0.00225 < view.brier_edge_lo < -0.00224, view
    # It does clear eventually — the defect is real, just slower than claimed.
    assert clearance(_flat_forecaster(1100, 0.56, reference=COIN),
                     min_resolved=100).cleared is True


def test_zero_skill_arm_no_longer_clears_against_a_coin():
    """With no recorded benchmark the gate refuses, rather than substituting
    one that is easier to beat."""
    result = clearance(_flat_forecaster(1100, 0.56, reference=None),
                       min_resolved=100)
    assert result.cleared is False
    assert "benchmark" in result.reason


def test_zero_skill_arm_scored_against_its_own_drift_does_not_clear():
    """The correct benchmark gives it exactly nothing, at any sample size."""
    for n in (400, 1100, 3000):
        result = clearance(_flat_forecaster(n, 0.56, reference=0.56),
                           min_resolved=100)
        assert result.cleared is False, n
        assert abs(result.brier_edge) < 1e-12, (n, result)


def test_a_genuinely_skilled_arm_still_clears():
    """The gate must not be merely shut — a real edge has to pass it."""
    forecasts = []
    for i in range(400):
        outcome = 1 if i % 2 == 0 else 0
        forecasts.append(Forecast(0.9 if outcome else 0.1, "HIGH", outcome, 0.56))
    result = clearance(forecasts, min_resolved=100)
    assert result.cleared is True


def test_unscoreable_forecasts_do_not_silence_the_kill_signal():
    """`auramaur ibkr-calibration` re-scores the same forecasts against a coin
    and prints it as a KILL-ONLY diagnostic, because the gate's honest refusal
    would otherwise retire the early kill signal along with the trade
    permission. The asymmetry is what licenses that: the coin is the EASIER
    benchmark, so failing it is conclusive while passing it proves nothing."""
    unscoreable = _flat_forecaster(200, 0.56, reference=None)
    gate = clearance(unscoreable, min_resolved=100)
    assert gate.cleared is False and gate.resolved == 0

    diagnostic = clearance([replace(f, reference=COIN) for f in unscoreable],
                           min_resolved=2)
    assert diagnostic.resolved == 200, "the diagnostic must still see the rows"
    assert diagnostic.brier_edge_lo > gate.brier_edge_lo
