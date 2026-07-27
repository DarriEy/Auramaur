"""Paper-only IBKR ETF book behavior and safety tests."""

import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from auramaur.db.database import Database
from auramaur.bot import run_ibkr_etf_arms_once
from auramaur.exchange.paper import PaperTrader
from auramaur.exchange.ibkr_equity import EquityQuote
from auramaur.strategy.ibkr_etf_paper import IBKRETFPaperPillar
from config.settings import Settings


class QuotesOnlyClient:
    def __init__(self, bid=99.9, ask=100.0):
        self.bid, self.ask = bid, ask

    async def get_quote(self, symbol):
        return EquityQuote(self.bid, self.ask, time.time())

    async def get_adjusted_daily_closes(self, symbol):
        return [(f"session-{day:03d}", 75 + day * 0.2) for day in range(121)]


class Aggregator:
    async def gather(self, *args, **kwargs):
        return []


class Analyzer:
    def __init__(self, probability=0.70, confidence="HIGH"):
        self.probability = probability
        self.confidence = confidence

    async def analyze(self, *args):
        return SimpleNamespace(probability=self.probability,
                               confidence=self.confidence, skipped_reason=None)


async def _pillar(db, client=None, analyzer=None):
    settings = Settings()
    settings.ibkr.etf_paper_enabled = True
    settings.ibkr.etf_symbols = ["SPY"]
    pillar = IBKRETFPaperPillar(
        settings, client or QuotesOnlyClient(), db, Aggregator(),
        analyzer or Analyzer(), model_alias="luna")
    pillar.market_open = lambda: True
    return pillar


@pytest.mark.asyncio
async def test_bullish_view_opens_paper_position_at_ask():
    db = Database(":memory:")
    await db.connect()
    pillar = await _pillar(db)
    await pillar.run_once()

    fill = await db.fetchone("SELECT * FROM ibkr_etf_fills")
    assert fill["model_alias"] == "luna"
    assert fill["symbol"] == "SPY"
    assert fill["side"] == "BUY"
    assert fill["price"] == pytest.approx(100.02)
    pos = await db.fetchone("SELECT * FROM ibkr_etf_positions")
    assert pos["model_alias"] == "luna"
    # The arm now routes its simulated fill through ExecutionGateway, so the
    # standard rails ARE written (that is the whole point — without them the
    # book has no pnl_ledger record and no decision snapshot, and can never
    # graduate). These assertions used to require the rails to be EMPTY; that
    # encoded audit finding #7 (2026-07-26: fills by raw SQL, no record_fill)
    # as if it were the contract. What must stay true is the SCOPING.
    recorded = await db.fetchone("SELECT * FROM fills")
    assert recorded["market_id"] == "ibkr:luna:SPY"
    assert recorded["is_paper"] == 1
    mirrored = await db.fetchone("SELECT * FROM trades")
    assert mirrored["exchange"] == "ibkr"
    assert mirrored["strategy_source"] == "ibkr_etf_luna"
    assert mirrored["is_paper"] == 1
    basis = await db.fetchone("SELECT * FROM cost_basis")
    assert basis["market_id"] == "ibkr:luna:SPY" and basis["is_paper"] == 1
    # Still no portfolio row: that table feeds the paper-breadth cap, the
    # drawdown fallback and the exit monitor, none of which manage this book.
    assert await db.fetchone("SELECT * FROM portfolio") is None
    await db.close()


@pytest.mark.asyncio
async def test_entry_captures_a_decision_and_registers_a_graduation_clock():
    """Prospective evidence is one of the ladder's two inputs, and it was
    permanently empty for this book: 17 registered clocks on 2026-07-27, not
    one of them an IBKR arm."""
    db = Database(":memory:")
    await db.connect()
    pillar = await _pillar(db)
    await pillar.run_once()

    snap = await db.fetchone("SELECT * FROM decision_snapshots")
    assert snap is not None, "no decision captured — no graduation clock"
    assert snap["strategy_source"] == "ibkr_etf_luna"
    assert snap["venue"] == "ibkr"
    assert snap["market_id"] == "ibkr:luna:SPY"
    assert snap["is_paper"] == 1
    # A price book asserts no probability edge: reference and fair must agree,
    # or the ladder's Brier bar scores a forecast the strategy never made.
    assert snap["fair_probability"] == snap["reference_price"]

    experiment = await db.fetchone(
        "SELECT * FROM strategy_experiments WHERE strategy_source = 'ibkr_etf_luna'")
    assert experiment is not None, "clock never started"
    # The frozen version must actually freeze settings.ibkr, not an empty dict.
    assert '"etf_max_entry_usd"' in experiment["config_json"]
    await db.close()


@pytest.mark.asyncio
async def test_round_trip_books_net_pnl_where_the_ladder_reads_it():
    db = Database(":memory:")
    await db.connect()
    analyzer = Analyzer(0.70)
    pillar = await _pillar(db, analyzer=analyzer)
    await pillar.run_once()
    analyzer.probability = 0.30
    pillar._views.clear()
    await pillar.run_once()

    rows = await db.fetchall(
        "SELECT kind, pnl, is_paper, venue, strategy_source FROM pnl_ledger "
        "ORDER BY id")
    assert [r["kind"] for r in rows] == ["commission", "sell", "commission"]
    assert all(r["is_paper"] == 1 for r in rows)
    assert all(r["venue"] == "ibkr" for r in rows)
    assert all(r["strategy_source"] == "ibkr_etf_luna" for r in rows)
    # Net of BOTH commissions, matching the venue-native ibkr_etf_ledger. A
    # gross-only ledger would flatter the arm: the entire realized IBKR record
    # to date is commission.
    native = await db.fetchall("SELECT pnl FROM ibkr_etf_ledger")
    assert sum(r["pnl"] for r in rows) == pytest.approx(
        sum(r["pnl"] for r in native))
    await db.close()


@pytest.mark.asyncio
async def test_bearish_refresh_closes_at_bid_and_attributes_ledger():
    db = Database(":memory:")
    await db.connect()
    analyzer = Analyzer(0.70)
    pillar = await _pillar(db, analyzer=analyzer)
    await pillar.run_once()
    analyzer.probability = 0.30
    pillar._views.clear()
    await pillar.run_once()

    fills = await db.fetchall("SELECT side, price FROM ibkr_etf_fills ORDER BY id")
    assert [(r["side"], r["price"]) for r in fills] == [
        ("BUY", pytest.approx(100.02)), ("SELL", pytest.approx(99.88002))]
    ledger = await db.fetchall(
        "SELECT kind, pnl FROM ibkr_etf_ledger ORDER BY id")
    assert [row["kind"] for row in ledger] == ["commission", "commission", "trade"]
    assert sum(row["pnl"] for row in ledger) < -2.0
    assert await db.fetchone("SELECT * FROM ibkr_etf_positions") is None
    await db.close()


@pytest.mark.asyncio
async def test_wide_spread_blocks_entry_before_analysis():
    db = Database(":memory:")
    await db.connect()
    pillar = await _pillar(db, client=QuotesOnlyClient(99.0, 101.0))
    await pillar.run_once()
    assert await db.fetchone("SELECT * FROM ibkr_etf_fills") is None
    await db.close()


def test_default_profile_is_small_readonly_paper_book():
    settings = Settings()
    assert {"SPY", "QQQ", "IWM", "TLT", "GLD", "VEA"}.issubset(
        settings.ibkr.etf_symbols)
    assert settings.ibkr.etf_paper_enabled is True
    assert settings.ibkr.etf_paper_budget_usd == 5_000.0
    assert settings.ibkr.etf_max_entry_usd == 250.0
    assert settings.ibkr.etf_max_deployment_pct == 50.0
    assert settings.ibkr.etf_max_positions == 4
    assert settings.ibkr.etf_max_signal_refreshes_per_cycle == 4
    assert [(m.alias, m.model, m.effort) for m in settings.ibkr.etf_models] == [
        ("luna", "gpt-5.6-luna", "low"),
        ("terra", "gpt-5.6-terra", "medium"),
        ("sol", "gpt-5.6-sol", "high"),
    ]


@pytest.mark.asyncio
async def test_model_cells_hold_independent_positions():
    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    settings.ibkr.etf_paper_enabled = True
    settings.ibkr.etf_symbols = ["SPY"]
    luna = IBKRETFPaperPillar(settings, QuotesOnlyClient(), db, Aggregator(),
                              Analyzer(0.70), model_alias="luna")
    sol = IBKRETFPaperPillar(settings, QuotesOnlyClient(), db, Aggregator(),
                             Analyzer(0.70), model_alias="sol")
    luna.market_open = sol.market_open = lambda: True
    await luna.run_once()
    await sol.run_once()
    rows = await db.fetchall(
        "SELECT model_alias, symbol FROM ibkr_etf_positions ORDER BY model_alias")
    assert [(r["model_alias"], r["symbol"]) for r in rows] == [
        ("luna", "SPY"), ("sol", "SPY")]
    await db.close()


@pytest.mark.asyncio
async def test_position_count_caps_broad_bullish_universe():
    db = Database(":memory:")
    await db.connect()
    pillar = await _pillar(db)
    pillar._s.ibkr.etf_symbols = ["SPY", "QQQ", "IWM"]
    pillar._s.ibkr.etf_max_positions = 1
    pillar._s.ibkr.etf_max_signal_refreshes_per_cycle = 3
    await pillar.run_once()
    row = await db.fetchone(
        "SELECT COUNT(*) AS n FROM ibkr_etf_positions")
    assert row["n"] == 1
    await db.close()


@pytest.mark.asyncio
async def test_asset_class_cap_reduces_entry_size():
    db = Database(":memory:")
    await db.connect()
    pillar = await _pillar(db)
    pillar._s.ibkr.etf_max_asset_class_pct = 2.0  # $100 of $5k
    await pillar.run_once()
    row = await db.fetchone(
        "SELECT quantity, avg_cost FROM ibkr_etf_positions WHERE model_alias='luna'")
    assert row["quantity"] * row["avg_cost"] + 1.0 <= 100.0
    await db.close()


@pytest.mark.asyncio
async def test_daily_loss_limit_blocks_entries_but_not_loop():
    db = Database(":memory:")
    await db.connect()
    await db.execute(
        """INSERT INTO ibkr_etf_ledger
           (model_alias, kind, pnl, source_ref, realized_at)
           VALUES ('luna', 'trade', -150, 'test:daily-loss', datetime('now'))""")
    await db.commit()
    pillar = await _pillar(db)
    await pillar.run_once()
    assert await db.fetchone("SELECT * FROM ibkr_etf_fills") is None
    await db.close()


@pytest.mark.asyncio
async def test_etf_fill_cannot_move_shared_paper_wallet():
    """The load-bearing isolation guard, now that the arm DOES write the
    shared rails.

    PaperTrader spendable cash is ``initial + SUM(pnl_ledger) -
    SUM(open cost_basis)`` over every is_paper=1 row. Left unscoped, an ETF
    arm's open position would subtract broker exposure from the
    prediction-market book's cash and stop it entering — a change to what
    trades produced purely by a bookkeeping move. The IBKR books are funded by
    their own paper budgets.
    """
    db = Database(":memory:")
    await db.connect()
    wallet = PaperTrader(db, initial_balance=1_000.0)
    before = await wallet._compute_balance()
    pillar = await _pillar(db)
    await pillar.run_once()
    assert await wallet._compute_balance() == before

    # The rows exist — they simply must not be in the wallet's scope.
    open_cost = await db.fetchone(
        "SELECT COALESCE(SUM(size * avg_cost), 0) AS c FROM cost_basis "
        "WHERE is_paper = 1 AND size > 0")
    assert open_cost["c"] > 200.0
    ledger = await db.fetchone(
        "SELECT COUNT(*) AS n FROM pnl_ledger WHERE is_paper = 1")
    assert ledger["n"] > 0
    assert all(r["market_id"].startswith("ibkr:") for r in await db.fetchall(
        "SELECT market_id FROM cost_basis UNION ALL "
        "SELECT market_id FROM pnl_ledger"))
    await db.close()


@pytest.mark.asyncio
async def test_etf_paper_pnl_stays_out_of_the_live_gates():
    """No IBKR paper row may reach the live daily-loss gate or the drawdown
    latch (the 2026-07-22 phantom-peak failure mode)."""
    from auramaur.risk.portfolio import PortfolioTracker

    db = Database(":memory:")
    await db.connect()
    analyzer = Analyzer(0.70)
    pillar = await _pillar(db, analyzer=analyzer)
    await pillar.run_once()
    analyzer.probability = 0.30
    pillar._views.clear()
    await pillar.run_once()

    live_rows = await db.fetchone(
        "SELECT COUNT(*) AS n FROM pnl_ledger WHERE is_paper = 0")
    assert live_rows["n"] == 0

    tracker = PortfolioTracker(db, Settings())
    assert await tracker.get_daily_pnl() == 0.0
    # The drawdown latch is fed by venue cash + synced positions; the arm must
    # contribute no portfolio row for it to observe.
    assert await db.fetchone("SELECT * FROM portfolio") is None
    assert await tracker.get_drawdown() == 0.0
    await db.close()


@pytest.mark.asyncio
async def test_arm_stays_paper_even_when_the_global_live_gates_are_open():
    """The arm is structurally paper-only (DeploymentMode.PAPER_ONLY).

    Routing through the gateway must not hand it the global live flag: the
    intent carries force_paper, so is_live never reaches the order, and the
    simulator refuses a cleared dry_run as the second lock.
    """
    db = Database(":memory:")
    await db.connect()
    pillar = await _pillar(db)
    pillar._s.auramaur_live = True
    pillar._s.execution.live = True
    assert pillar._s.is_live is True
    await pillar.run_once()

    fill = await db.fetchone("SELECT is_paper FROM fills")
    assert fill["is_paper"] == 1
    assert (await db.fetchone("SELECT is_paper FROM trades"))["is_paper"] == 1
    assert (await db.fetchone(
        "SELECT COUNT(*) AS n FROM pnl_ledger WHERE is_paper = 0"))["n"] == 0
    await db.close()


@pytest.mark.asyncio
async def test_symbol_outside_the_manifest_still_fills_the_arms_own_books():
    """Booking is evidence, not the money path.

    etf_symbols is operator-editable; a symbol with no InstrumentSpec cannot
    be minted into a gateway market_id. That must cost ladder evidence, not
    break the cycle or silently drop the arm's own accounting.
    """
    db = Database(":memory:")
    await db.connect()
    pillar = await _pillar(db)
    pillar._s.ibkr.etf_symbols = ["NOTAMANIFESTSYMBOL"]
    pillar._INSTRUMENTS = dict(pillar._INSTRUMENTS)
    pillar._INSTRUMENTS["NOTAMANIFESTSYMBOL"] = ("unknown", "us_broad")
    assert await pillar.run_once() == 1

    native = await db.fetchone("SELECT symbol, side FROM ibkr_etf_fills")
    assert native["symbol"] == "NOTAMANIFESTSYMBOL" and native["side"] == "BUY"
    assert await db.fetchone("SELECT * FROM fills") is None
    assert await db.fetchone("SELECT * FROM pnl_ledger") is None
    await db.close()


@pytest.mark.asyncio
async def test_restart_restores_cursor_view_and_cooldown():
    db = Database(":memory:")
    await db.connect()
    now = time.time()
    await db.execute(
        "INSERT INTO ibkr_etf_state (model_alias, refresh_cursor) VALUES ('luna', 7)")
    await db.execute(
        "INSERT INTO ibkr_etf_cooldowns VALUES ('luna', 'SPY', ?)", (now + 3600,))
    await db.execute(
        """INSERT INTO ibkr_etf_forecasts
           (model_alias, model, symbol, probability, confidence, reference_price,
            opened_session_date, horizon_sessions, due_at)
           VALUES ('luna','gpt','SPY',.66,'HIGH',100,date('now'),5,datetime('now','+7 days'))""")
    await db.commit()
    pillar = await _pillar(db)
    await pillar._ensure_state()
    assert pillar._refresh_cursor == 7
    assert pillar._views["SPY"][1:] == (0.66, "HIGH")
    assert pillar._cooldown["SPY"] > now
    await db.close()


@pytest.mark.asyncio
async def test_forecast_resolves_on_exact_nth_adjusted_session():
    db = Database(":memory:")
    await db.connect()
    client = QuotesOnlyClient()
    client.get_adjusted_daily_closes = AsyncMock(return_value=[
        ("2026-07-10", 99.0), ("2026-07-13", 100.0), ("2026-07-14", 101.0),
        ("2026-07-15", 102.0), ("2026-07-16", 103.0),
        ("2026-07-17", 104.0), ("2026-07-20", 105.0),
    ])
    pillar = await _pillar(db, client=client)
    await db.execute(
        """INSERT INTO ibkr_etf_forecasts
           (model_alias, model, symbol, probability, confidence, reference_price,
            opened_session_date, horizon_sessions, due_at)
           VALUES ('luna','gpt','SPY',.6,'HIGH',100,'2026-07-10',5,datetime('now'))""")
    await db.commit()
    await pillar._resolve_forecasts("SPY")
    row = await db.fetchone(
        "SELECT reference_price, final_price, actual_outcome, last_session_date "
        "FROM ibkr_etf_forecasts")
    assert row["reference_price"] == 99.0
    assert row["final_price"] == 104.0
    assert row["actual_outcome"] == 1
    assert row["last_session_date"] == "2026-07-17"
    await db.close()


@pytest.mark.asyncio
async def test_one_model_failure_does_not_stop_other_arms():
    calls = []

    class Arm:
        def __init__(self, alias, fail=False):
            self.model_alias, self.fail = alias, fail

        async def run_once(self):
            calls.append(self.model_alias)
            if self.fail:
                raise RuntimeError("model failed")

    await run_ibkr_etf_arms_once([Arm("luna", True), Arm("terra"), Arm("sol")])
    assert calls == ["luna", "terra", "sol"]


@pytest.mark.asyncio
async def test_wide_spread_still_exits_a_held_position_but_blocks_entry():
    """A widening spread must never disable a held position's stop-loss.

    The spread gate sat above the held-position branch, so any widening past
    etf_max_spread_bps — an auction, a thin tape, a stress print — silently
    disabled stop_loss / take_profit / trailing_stop / llm_bearish on a
    position already at risk, precisely when the exit matters most. It bounds
    ENTRY quality only.
    """
    db = Database(":memory:")
    await db.connect()
    try:
        # Enter at a tight spread.
        pillar = await _pillar(db, client=QuotesOnlyClient(bid=99.9, ask=100.0))
        await pillar.run_once()
        assert await db.fetchone("SELECT * FROM ibkr_etf_positions") is not None

        # Price collapses AND the spread blows out well past the 20bps cap.
        cfg = pillar._s.ibkr
        crashed = 100.0 * (1 - (cfg.etf_stop_loss_pct + 1) / 100.0)
        wide = QuotesOnlyClient(bid=crashed, ask=crashed * 1.05)  # ~500 bps
        spread_bps = (wide.ask - wide.bid) / ((wide.ask + wide.bid) / 2) * 10_000
        assert spread_bps > cfg.etf_max_spread_bps

        pillar._client = wide
        pillar._cooldown.clear()
        await pillar.run_once()

        sells = await db.fetchall(
            "SELECT * FROM ibkr_etf_fills WHERE side = 'SELL'")
        assert sells, "held position must still be able to exit at a wide spread"
        assert await db.fetchone("SELECT * FROM ibkr_etf_positions") is None
    finally:
        await db.close()


class DeterministicAnalyzer:
    """Shape of MomentumETFAnalyzer: exposes analyze_symbol, not analyze."""

    model = "deterministic_momentum_v1"

    async def analyze_symbol(self, client, symbol, as_of=None):
        return SimpleNamespace(probability=0.70, confidence="HIGH",
                               thesis="momentum", key_risks=())


@pytest.mark.asyncio
async def test_control_arm_records_a_scoreable_forecast():
    """The deterministic control must be scored on the same terms as the LLM arms.

    It used to return from _view before the forecast INSERT, so the
    intelligence-cap A/B ran with a control that wrote zero rows: 282
    forecasts on 2026-07-27, none of them the control. A control that is
    never scored cannot be compared against.
    """
    db = Database(":memory:")
    await db.connect()
    pillar = await _pillar(db, analyzer=DeterministicAnalyzer())
    pillar._alias = "momentum_control"
    await pillar.run_once()

    rows = await db.fetchall(
        "SELECT model_alias, probability, confidence, horizon_sessions "
        "FROM ibkr_etf_forecasts WHERE model_alias='momentum_control'")
    assert rows, "control arm recorded no forecast"
    assert rows[0]["probability"] == pytest.approx(0.70)
    assert rows[0]["confidence"] == "HIGH"
    assert rows[0]["horizon_sessions"] > 0


@pytest.mark.asyncio
async def test_per_arm_threshold_overrides_the_global_gate():
    """A calibrated arm must be able to carry its own entry thresholds.

    The global defaults (0.62 / MEDIUM) were set for a control that returns
    0.70/HIGH by construction. The OpenAI arms are instructed to sit near
    0.50/LOW on weak evidence, so they produced 0.43-0.56 / MEDIUM_LOW and the
    shared gate rejected 100% of 282 forecasts.
    """
    db = Database(":memory:")
    await db.connect()
    pillar = await _pillar(db, analyzer=Analyzer(probability=0.55,
                                                 confidence="MEDIUM_LOW"))
    # Global gate: no entry, and the arm must say which gate bound.
    assert await pillar.run_once() == 0

    db2 = Database(":memory:")
    await db2.connect()
    relaxed = await _pillar(db2, analyzer=Analyzer(probability=0.55,
                                                   confidence="MEDIUM_LOW"))
    relaxed._s.ibkr.etf_arm_min_prob = {"luna": 0.53}
    relaxed._s.ibkr.etf_arm_min_confidence = {"luna": "MEDIUM_LOW"}
    assert relaxed._min_prob == 0.53
    assert await relaxed.run_once() == 1, (
        "per-arm override did not reach entry_proposal, which re-checks both "
        "thresholds and would veto with the global values")

    # A non-overridden arm keeps the global gate.
    relaxed._alias = "terra"
    assert relaxed._min_prob == relaxed._s.ibkr.etf_min_prob


@pytest.mark.asyncio
async def test_benchmark_is_the_drift_not_a_coin_flip():
    """A 0.5 reference would pay an arm for knowing equities rise.

    brier_edge = (reference - outcome)^2 - (fair - outcome)^2. Against 0.5, an
    arm that always answers 0.55 collects a positive edge on any up-drifting
    series without forecasting anything. Against the instrument's own trailing
    up-rate it collects nothing, which is correct.
    """
    from auramaur.risk.ibkr_math import horizon_up_rate

    rising = [100 * 1.002 ** i for i in range(200)]
    drift = horizon_up_rate(rising, 5)
    assert drift == 1.0                      # every 5-session window was up
    assert horizon_up_rate([100.0] * 50, 5) == 0.0
    assert horizon_up_rate([100.0, 101.0], 5) is None   # too little history

    # A constant 0.55 forecast on an always-up series.
    fair, outcome = 0.55, 1
    edge_vs_coin = (0.5 - outcome) ** 2 - (fair - outcome) ** 2
    edge_vs_drift = (drift - outcome) ** 2 - (fair - outcome) ** 2
    assert edge_vs_coin > 0        # rewarded for nothing
    assert edge_vs_drift < 0       # correctly penalised


@pytest.mark.asyncio
async def test_event_family_advances_weekly_so_the_book_can_reach_thirty():
    """market_id as the family caps the book at 28 families forever, two short
    of min_paired_forecasts. Overlapping same-week forecasts must still share
    one family — they share four of five sessions."""
    from auramaur.exchange.ibkr_instruments import BY_KEY
    from auramaur.exchange.ibkr_intent import instrument_event_family

    spec = BY_KEY["SPY"]
    monday = instrument_event_family(spec, "luna", "2026-07-27")
    friday = instrument_event_family(spec, "luna", "2026-07-31")
    next_week = instrument_event_family(spec, "luna", "2026-08-03")
    assert monday == friday          # overlapping windows are one family
    assert monday != next_week       # a fresh window is new evidence
    # 28 symbols x weekly buckets clears 30 within two weeks.
    weekly = {instrument_event_family(BY_KEY[s], "luna", d)
              for s in list(BY_KEY)[:28] for d in ("2026-07-27", "2026-08-03")}
    assert len(weekly) >= 30
    # Arms never share a family.
    assert instrument_event_family(spec, "sol", "2026-07-27") != monday


@pytest.mark.asyncio
async def test_entry_captures_a_real_forecast_against_a_real_benchmark():
    from datetime import date, timedelta

    db = Database(":memory:")
    await db.connect()
    client = QuotesOnlyClient()
    # Real ISO session dates: completed_closes compares them as strings against
    # the as_of date, so the synthetic "session-NNN" ids the other fixtures use
    # sort ABOVE any 20xx date and silently filter the whole history away.
    start = date(2026, 7, 27) - timedelta(days=260)
    bars, day, i = [], start, 0
    while len(bars) < 200:
        if day.weekday() < 5:
            bars.append((day.isoformat(), 75 + i * 0.2))
            i += 1
        day += timedelta(days=1)
    client.get_adjusted_daily_closes = AsyncMock(return_value=bars)
    pillar = await _pillar(db, client=client)
    await pillar.run_once()

    snap = await db.fetchone("SELECT * FROM decision_snapshots")
    assert snap is not None
    # The arm's own probability, not a copy of the reference: a zero edge is
    # what the ladder reads as "no evidence".
    assert snap["fair_probability"] == pytest.approx(0.70)
    assert snap["reference_price"] != pytest.approx(snap["fair_probability"])
    # The family is week-bucketed, not the bare market_id.
    assert snap["event_family"].startswith("ibkr:luna:SPY:")
    assert snap["event_family"] != snap["market_id"]
    # A real BBO was recorded, so the fill can be judged executable rather than
    # stamped 'synthetic' and dropped by require_executable_fills.
    book = await db.fetchone("SELECT * FROM orderbook_snapshots")
    assert book["best_bid"] > 0 and book["best_ask"] > 0
    assert snap["fill_evidence"] in ("book_cross", "venue_fill")
    await db.close()


@pytest.mark.asyncio
async def test_resolved_forecast_publishes_the_outcome_the_ladder_joins_on():
    """_prospective_stats INNER JOINs market_outcomes; without a row the
    decision is invisible to the ladder forever."""
    db = Database(":memory:")
    await db.connect()
    client = QuotesOnlyClient()
    client.get_adjusted_daily_closes = AsyncMock(return_value=[
        ("2026-07-10", 99.0), ("2026-07-13", 100.0), ("2026-07-14", 101.0),
        ("2026-07-15", 102.0), ("2026-07-16", 103.0),
        ("2026-07-17", 104.0), ("2026-07-20", 105.0),
    ])
    pillar = await _pillar(db, client=client)
    await db.execute(
        """INSERT INTO ibkr_etf_forecasts
           (model_alias, model, symbol, probability, confidence,
            reference_price, opened_session_date, horizon_sessions, due_at)
           VALUES ('luna','m','SPY',0.7,'HIGH',100.0,'2026-07-13',5,
                   datetime('now'))""")
    await db.commit()
    await pillar._resolve_forecasts("SPY")

    row = await db.fetchone("SELECT * FROM ibkr_etf_forecasts")
    assert row["actual_outcome"] == 1
    outcome = await db.fetchone("SELECT * FROM market_outcomes")
    assert outcome is not None, "resolved forecast never reached the ladder"
    assert outcome["venue"] == "ibkr"
    assert outcome["outcome"] == 1
    # Keyed by FAMILY, not the symbol-scoped market_id. market_outcomes is
    # UNIQUE(venue, market_id): keying by symbol would let the first resolved
    # forecast own the only row (luna, SPY) can ever have, and every later
    # week would be scored against that one stale outcome.
    assert outcome["event_family"].startswith("ibkr:luna:SPY:2026W")
    assert outcome["market_id"] == outcome["event_family"]
    assert outcome["event_key"] == f"ibkr:{outcome['event_family']}"
    # Consequence, recorded honestly: nothing joins today, because the ladder
    # joins on d.market_id. The outcomes bank; counting them needs a decision.
    assert outcome["market_id"] != "ibkr:luna:SPY"
    await db.close()


@pytest.mark.asyncio
async def test_forecast_never_resolves_against_a_session_still_trading():
    """reqHistoricalData returns TODAY'S PARTIAL BAR.

    Measured 2026-07-27: a "1 M" request at 14:29 ET included 2026-07-27. So
    the horizon's final session counts while it is still trading, and the
    forecast resolves against whatever the price was when the cycle ran rather
    than the close it asks about. actual_outcome IS NULL filters the row out
    afterwards, so the wrong value is permanent.
    """
    db = Database(":memory:")
    await db.connect()
    today = datetime.now(timezone.utc).astimezone(
        ZoneInfo("America/New_York")).date()
    days = [(today - timedelta(days=d)).isoformat() for d in range(9, -1, -1)]
    client = QuotesOnlyClient()
    # Final bar is TODAY, in progress and gapped down.
    client.get_adjusted_daily_closes = AsyncMock(return_value=(
        [(d, 100.0 + i) for i, d in enumerate(days[:-1])] + [(days[-1], 1.0)]))
    pillar = await _pillar(db, client=client)
    await db.execute(
        """INSERT INTO ibkr_etf_forecasts
           (model_alias, model, symbol, probability, confidence,
            reference_price, opened_session_date, horizon_sessions, due_at)
           VALUES ('luna','m','SPY',0.7,'HIGH',100.0,?,5,datetime('now'))""",
        (days[4],))
    await db.commit()
    await pillar._resolve_forecasts("SPY")

    row = await db.fetchone("SELECT * FROM ibkr_etf_forecasts")
    # Only 4 COMPLETED sessions follow days[4]; the 5th is today, still open.
    assert row["actual_outcome"] is None, "resolved against a live session"
    assert row["last_session_date"] != days[-1]
    await db.close()
