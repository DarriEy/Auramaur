from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from auramaur.db.database import Database
from auramaur.exchange.models import Market, OrderBook, OrderBookLevel, OrderSide
from auramaur.monitoring.maker_observatory import (
    MakerObservatory,
    compute_maker_features,
    maker_observatory_feature_report,
    maker_observatory_summary,
    maker_promotion_blockers,
    maker_quote_coverage,
)
from auramaur.strategy.order_flow import OrderFlowTracker


def _book(bid=.40, ask=.44):
    # Deliberately worst-first, matching the CLOB response shape.
    return OrderBook(
        bids=[OrderBookLevel(price=.10, size=100),
              OrderBookLevel(price=bid, size=30)],
        asks=[OrderBookLevel(price=.90, size=100),
              OrderBookLevel(price=ask, size=10)],
    )


def _market():
    return Market(id="m1", question="fixture", clob_token_yes="yes-1",
                  clob_token_no="no-1")


def test_features_are_ordering_agnostic_and_depth_weighted():
    features = compute_maker_features(_book())

    assert features.best_bid == .40
    assert features.best_ask == .44
    assert features.midpoint == pytest.approx(.42)
    assert features.microprice == pytest.approx(.43)
    assert features.bid_depth == 130
    assert features.ask_depth == 110


def test_signed_flow_is_time_decayed_and_directional(monkeypatch):
    tracker = OrderFlowTracker()
    now = datetime(2026, 8, 5, tzinfo=timezone.utc)
    monkeypatch.setattr("auramaur.strategy.order_flow.datetime", type(
        "Clock", (), {"now": staticmethod(lambda tz: now)}))
    tracker.record_trade("m1", OrderSide.BUY, 10)
    tracker.record_trade("m1", OrderSide.SELL, 4)

    assert tracker.signed_flow(("m1",), now=now) == pytest.approx(6)
    assert tracker.signed_flow(("m1",), now=now + timedelta(seconds=60)) == pytest.approx(3)


@pytest.mark.asyncio
async def test_fill_markouts_are_restart_safe_and_mode_separated(tmp_path):
    db = Database(str(tmp_path / "observatory.db"))
    await db.connect()
    try:
        t0 = datetime(2026, 8, 5, tzinfo=timezone.utc)
        observatory = MakerObservatory(
            db, horizons=(30, 60), retention_days=45,
            max_mark_lateness_seconds=5, holdout_days=0)
        observation_id = await observatory.observe(_market(), _book(), observed_at=t0)
        await observatory.record_observed_fill(
            observation_id=observation_id, order_id="paper-1", market_id="m1",
            side="bid", price=.41, size=10, is_paper=True, filled_at=t0,
            fill_evidence="synthetic",
        )

        # A new instance models a restart. Later books fill each due horizon
        # exactly once; the earliest post-horizon book is the executable mark.
        restarted = MakerObservatory(
            db, horizons=(30, 60), retention_days=45,
            max_mark_lateness_seconds=5, holdout_days=0)
        await restarted.observe(_market(), _book(.42, .44),
                                observed_at=t0 + timedelta(seconds=31))
        await restarted.observe(_market(), _book(.44, .46),
                                observed_at=t0 + timedelta(seconds=61))
        marks = await db.fetchall(
            "SELECT horizon_seconds,markout FROM maker_observatory_markouts "
            "ORDER BY horizon_seconds")

        assert [row["horizon_seconds"] for row in marks] == [30, 60]
        assert [row["markout"] for row in marks] == pytest.approx([.02, .04])
        summary = await maker_observatory_summary(db, days=3650)
        assert [(row["horizon_seconds"], row["is_paper"]) for row in summary] == [
            (30, 1), (60, 1),
        ]
        assert [row["valid_marks"] for row in summary] == [1, 1]
        assert [row["credible_holdout_marks"] for row in summary] == [0, 0]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_invalid_books_do_not_create_observations(tmp_path):
    db = Database(str(tmp_path / "observatory.db"))
    await db.connect()
    observatory = MakerObservatory(db)

    with pytest.raises(ValueError, match="two-sided"):
        await observatory.observe(_market(), OrderBook(bids=[], asks=[]))
    assert (await db.fetchone("SELECT COUNT(*) n FROM maker_observations"))["n"] == 0
    await db.close()


@pytest.mark.asyncio
async def test_late_marks_are_retained_but_never_count_as_valid(tmp_path):
    db = Database(str(tmp_path / "late.db"))
    await db.connect()
    try:
        now = datetime.now(timezone.utc)
        observatory = MakerObservatory(
            db, horizons=(30,), max_mark_lateness_seconds=5, holdout_days=0)
        observation_id = await observatory.observe(
            _market(), _book(), observed_at=now - timedelta(seconds=100))
        await observatory.record_observed_fill(
            observation_id=observation_id, order_id="live-late",
            market_id="m1", side="bid", price=.41, size=5, is_paper=False,
            fill_evidence="venue_fill",
            filled_at=now - timedelta(seconds=100),
        )
        await observatory.observe(_market(), _book(), observed_at=now)

        mark = await db.fetchone(
            "SELECT lateness_seconds,is_valid FROM maker_observatory_markouts")
        assert mark["lateness_seconds"] == pytest.approx(70, abs=1)
        assert mark["is_valid"] == 0
        summary = await maker_observatory_summary(db, days=1)
        assert summary[0]["late_marks"] == 1
        assert summary[0]["credible_holdout_marks"] == 0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_feature_report_uses_warmup_threshold_on_holdout_only(tmp_path):
    db = Database(str(tmp_path / "features.db"))
    await db.connect()
    try:
        now = datetime.now(timezone.utc)
        observatory = MakerObservatory(
            db, horizons=(30,), max_mark_lateness_seconds=5, holdout_days=0)
        for index, (imbalance_book, mark_mid) in enumerate((
            (_book(.40, .44), .43),
            (OrderBook(
                bids=[OrderBookLevel(price=.40, size=5)],
                asks=[OrderBookLevel(price=.44, size=50)]), .39),
            (_book(.40, .44), .45),
        )):
            fill_at = now - timedelta(minutes=3-index)
            observation_id = await observatory.observe(
                _market(), imbalance_book, observed_at=fill_at)
            await db.execute(
                "UPDATE maker_observations SET is_holdout=? WHERE id=?",
                (int(index > 0), observation_id))
            await observatory.record_observed_fill(
                observation_id=observation_id, order_id=f"live-{index}",
                market_id="m1", side="bid", price=.41, size=5, is_paper=False,
                fill_evidence="venue_fill", filled_at=fill_at)
            await observatory.observe(
                _market(), _book(mark_mid - .01, mark_mid + .01),
                observed_at=fill_at + timedelta(seconds=31))

        report = await maker_observatory_feature_report(db, days=1)
        imbalance = next(row for row in report
                         if row["feature"] == "depth_imbalance")
        assert imbalance["warmup_n"] == 1
        assert imbalance["holdout_n"] == 2
        assert imbalance["effect"] is not None
    finally:
        await db.close()


async def _observatory(db, **kwargs):
    """An observatory whose warmup/holdout boundary the test controls."""
    kwargs.setdefault("horizons", (30,))
    kwargs.setdefault("max_mark_lateness_seconds", 45)
    observatory = MakerObservatory(db, **kwargs)
    await observatory._register()
    return observatory


async def _set_holdout_boundary(db, observatory, boundary: datetime) -> None:
    """Move the registered holdout start so `observe` classifies against it.

    is_holdout is computed by a SQL CASE against strategy_experiments; setting
    the boundary here (rather than UPDATE-ing is_holdout afterwards) keeps that
    production expression the thing under test.
    """
    await db.execute(
        "UPDATE strategy_experiments SET holdout_starts_at=? "
        "WHERE strategy_version=?",
        (boundary.strftime("%Y-%m-%d %H:%M:%S"), observatory.strategy_version),
    )


async def _fill(observatory, db, *, market_id="m1", side="bid", price=.41,
                evidence="venue_fill", is_paper=False, at, order_id,
                mark_book=None, size=5.0):
    """Observe, book a fill against that observation, then mark it out."""
    observation_id = await observatory.observe(
        _market(), _book(), observed_at=at)
    await observatory.record_observed_fill(
        observation_id=observation_id, order_id=order_id, market_id=market_id,
        side=side, price=price, size=size, is_paper=is_paper,
        fill_evidence=evidence, filled_at=at)
    # A later observation is what triggers the due markout.
    await observatory.observe(_market(), mark_book or _book(.44, .46),
                              observed_at=at + timedelta(seconds=31))
    return observation_id


@pytest.mark.asyncio
async def test_synthetic_paper_fills_are_never_credible_evidence(tmp_path):
    """The PR's core claim, isolated from the holdout and validity filters.

    Both fills below are post-warmup and marked on time; the ONLY difference is
    fill_evidence. If synthetic fills ever became credible, this would notice —
    the previous assertion of `credible_holdout_marks == 0` could not, because
    its fixture was also pre-holdout.
    """
    db = Database(str(tmp_path / "credible.db"))
    await db.connect()
    try:
        now = datetime.now(timezone.utc)
        observatory = await _observatory(db, holdout_days=0)
        await _set_holdout_boundary(db, observatory, now - timedelta(days=1))

        await _fill(observatory, db, at=now - timedelta(seconds=300),
                    order_id="synthetic-1", evidence="synthetic", is_paper=True)
        await _fill(observatory, db, at=now - timedelta(seconds=200),
                    order_id="venue-1", evidence="venue_fill", is_paper=False)

        summary = {row["is_paper"]: row
                   for row in await maker_observatory_summary(db, days=1)}
        # Both were observed in the holdout window and marked on time...
        assert summary[1]["valid_marks"] == 1
        assert summary[0]["valid_marks"] == 1
        # ...but only the venue fill is admissible evidence.
        assert summary[1]["credible_holdout_marks"] == 0
        assert summary[0]["credible_holdout_marks"] == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_warmup_fills_are_excluded_from_holdout_evidence(tmp_path):
    """Exercises the is_holdout CASE, holding evidence and validity constant."""
    db = Database(str(tmp_path / "warmup.db"))
    await db.connect()
    try:
        now = datetime.now(timezone.utc)
        observatory = await _observatory(db, holdout_days=0)
        await _set_holdout_boundary(db, observatory, now - timedelta(seconds=250))

        await _fill(observatory, db, at=now - timedelta(seconds=400),
                    order_id="warmup-1")   # before the boundary
        await _fill(observatory, db, at=now - timedelta(seconds=200),
                    order_id="holdout-1")  # after it

        flags = [row["is_holdout"] for row in await db.fetchall(
            "SELECT is_holdout FROM maker_observations ORDER BY observed_at")]
        assert flags[0] == 0 and flags[-1] == 1

        summary = await maker_observatory_summary(db, days=1)
        assert [row["valid_marks"] for row in summary] == [2]
        assert [row["credible_holdout_marks"] for row in summary] == [1]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_ask_fills_mark_out_from_the_seller_side(tmp_path):
    """A maker ask is a SALE: a falling midpoint is favourable, so markout > 0.

    The bid side is symmetric and already covered; without this the sign
    convention on half of every quote pair was unasserted.
    """
    db = Database(str(tmp_path / "ask.db"))
    await db.connect()
    try:
        t0 = datetime(2026, 8, 5, tzinfo=timezone.utc)
        observatory = await _observatory(db, holdout_days=0)
        observation_id = await observatory.observe(
            _market(), _book(), observed_at=t0)
        await observatory.record_observed_fill(
            observation_id=observation_id, order_id="ask-1", market_id="m1",
            side="ask", price=.43, size=5, is_paper=False,
            fill_evidence="venue_fill", filled_at=t0)
        # Midpoint falls to .40 — good for a seller.
        await observatory.observe(_market(), _book(.39, .41),
                                  observed_at=t0 + timedelta(seconds=31))

        mark = await db.fetchone("SELECT markout FROM maker_observatory_markouts")
        assert mark["markout"] == pytest.approx(.03)

        # And the mirror image: a rising midpoint hurts the seller.
        observation_id = await observatory.observe(
            _market(), _book(), observed_at=t0 + timedelta(seconds=100))
        await observatory.record_observed_fill(
            observation_id=observation_id, order_id="ask-2", market_id="m1",
            side="ask", price=.43, size=5, is_paper=False,
            fill_evidence="venue_fill", filled_at=t0 + timedelta(seconds=100))
        await observatory.observe(_market(), _book(.49, .51),
                                  observed_at=t0 + timedelta(seconds=131))
        marks = await db.fetchall(
            "SELECT m.markout FROM maker_observatory_markouts m "
            "JOIN maker_observatory_fills f ON f.id=m.fill_id "
            "WHERE f.order_id='ask-2'")
        assert marks[0]["markout"] == pytest.approx(-.07)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_feature_report_and_blockers_ignore_paper_fills(tmp_path):
    """Paper evidence must not reach either the scorecard or the promotion gate."""
    db = Database(str(tmp_path / "paperonly.db"))
    await db.connect()
    try:
        now = datetime.now(timezone.utc)
        observatory = await _observatory(db, holdout_days=0)
        await _set_holdout_boundary(db, observatory, now - timedelta(days=1))
        for index in range(3):
            await _fill(observatory, db, at=now - timedelta(seconds=400 - index * 50),
                        order_id=f"paper-{index}", evidence="synthetic",
                        is_paper=True)

        report = await maker_observatory_feature_report(db, days=1)
        assert report == [], "paper fills leaked into the frozen-threshold report"

        summary = await maker_observatory_summary(db, days=1)
        assert summary and all(row["is_paper"] for row in summary)
        assert maker_promotion_blockers(summary) == ["no live fills"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_promotion_blockers_name_every_unmet_bar():
    """The gate must report the real numbers, not a bare pass/fail."""
    rows = [{"horizon_seconds": 30, "is_paper": 0, "credible_holdout_marks": 4,
             "markets": 1, "completeness": .5, "ci_low": -.01, "ci_high": .02}]
    blockers = maker_promotion_blockers(
        rows, min_fills=100, min_markets=5, min_completeness=.95)

    assert blockers == [
        "30s: 4/100 credible holdout marks",
        "30s: 1/5 markets",
        "30s: 50.0%/95.0% valid-mark completeness",
        "30s: mean markout lower CI is not positive",
    ]
    passing = [{"horizon_seconds": 30, "is_paper": 0,
                "credible_holdout_marks": 100, "markets": 5,
                "completeness": .99, "ci_low": .001, "ci_high": .01}]
    assert maker_promotion_blockers(passing) == []


@pytest.mark.asyncio
async def test_completeness_counts_marks_that_were_due_but_never_taken(tmp_path):
    """A fill nobody came back to mark must drag completeness down, not be hidden."""
    db = Database(str(tmp_path / "gaps.db"))
    await db.connect()
    try:
        now = datetime.now(timezone.utc)
        observatory = await _observatory(db, holdout_days=0)
        await _set_holdout_boundary(db, observatory, now - timedelta(days=1))
        await _fill(observatory, db, at=now - timedelta(seconds=300),
                    order_id="marked-1")
        # A due fill whose market was never observed again: due, never marked.
        observation_id = await observatory.observe(
            _market(), _book(), observed_at=now - timedelta(seconds=200))
        await observatory.record_observed_fill(
            observation_id=observation_id, order_id="unmarked-1",
            market_id="m1", side="bid", price=.41, size=5, is_paper=False,
            fill_evidence="venue_fill", filled_at=now - timedelta(seconds=200))

        row = (await maker_observatory_summary(db, days=1))[0]
        assert row["due_marks"] == 2
        assert row["valid_marks"] == 1
        assert row["completeness"] == pytest.approx(.5)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_confidence_interval_is_computed_from_the_marks(tmp_path):
    """The CI must move with the data and stay undefined below two clusters."""
    db = Database(str(tmp_path / "ci.db"))
    await db.connect()
    try:
        now = datetime.now(timezone.utc)
        observatory = await _observatory(db, holdout_days=0)
        await _set_holdout_boundary(db, observatory, now - timedelta(days=1))
        await _fill(observatory, db, at=now - timedelta(seconds=300),
                    order_id="one-market")

        single = (await maker_observatory_summary(db, days=1))[0]
        assert single["markets"] == 1
        assert single["ci_low"] is None, "a single market cannot bound a mean"

        # Second market, both marked at a strictly positive markout.
        await db.execute(
            "INSERT INTO maker_observatory_fills"
            " (order_id,observation_id,market_id,side,fill_price,fill_size,"
            "  is_paper,fill_evidence,filled_at)"
            " SELECT 'm2-fill',observation_id,'m2',side,fill_price,fill_size,"
            "        is_paper,fill_evidence,filled_at"
            " FROM maker_observatory_fills WHERE order_id='one-market'")
        await db.execute(
            "INSERT INTO maker_observatory_markouts"
            " (fill_id,horizon_seconds,target_at,midpoint,markout,marked_at,"
            "  lateness_seconds,is_valid)"
            " SELECT id,30,filled_at,.44,.03,filled_at,0,1"
            " FROM maker_observatory_fills WHERE order_id='m2-fill'")

        both = (await maker_observatory_summary(db, days=1))[0]
        assert both["markets"] == 2
        assert both["ci_low"] is not None and both["ci_high"] is not None
        assert both["ci_low"] <= both["mean_markout"] <= both["ci_high"]
        assert both["ci_low"] > 0, "two positive marks must bound above zero"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_prune_drops_expired_rows_and_keeps_current_ones(tmp_path):
    db = Database(str(tmp_path / "prune.db"))
    await db.connect()
    try:
        now = datetime.now(timezone.utc)
        observatory = await _observatory(db, retention_days=7, holdout_days=0)
        await _fill(observatory, db, at=now - timedelta(seconds=300),
                    order_id="fresh-1")
        stale = now - timedelta(days=30)
        await observatory.observe(_market(), _book(), observed_at=stale)
        await observatory.record_observed_fill(
            observation_id=1, order_id="stale-1", market_id="m1", side="bid",
            price=.41, size=5, is_paper=False, fill_evidence="venue_fill",
            filled_at=stale)

        async def counts():
            totals = []
            for table in ("maker_observations", "maker_observatory_fills",
                          "maker_observatory_markouts"):
                totals.append(
                    (await db.fetchone(f"SELECT COUNT(*) n FROM {table}"))["n"])
            return tuple(totals)

        assert await counts() == (3, 2, 1)
        await observatory.prune()
        observations, fills, markouts = await counts()
        assert fills == 1, "the 30-day-old fill outlived 7-day retention"
        assert observations == 2 and markouts == 1, "current rows were pruned"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_quote_coverage_reports_sampled_presence(tmp_path):
    db = Database(str(tmp_path / "coverage.db"))
    await db.connect()
    try:
        now = datetime.now(timezone.utc)
        observatory = await _observatory(db, holdout_days=0)
        active = SimpleNamespace(placed_at=now - timedelta(seconds=10),
                                 bid_price=.40, ask_price=.44, size=10.0)
        quote = SimpleNamespace(bid_price=.41, ask_price=.44, size=10.0)
        await observatory.observe(_market(), _book(), observed_at=now)
        await observatory.observe(_market(), _book(), quote=quote,
                                  active_quote=active,
                                  observed_at=now - timedelta(seconds=5))

        coverage = await maker_quote_coverage(db, days=1)
        assert coverage["samples"] == 2
        assert coverage["active_samples"] == 1
        assert coverage["changed_samples"] == 1
        assert coverage["sampled_quote_coverage"] == pytest.approx(.5)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_registration_is_issued_once_not_on_every_observation(tmp_path):
    """observe() runs on the quoting path; re-registering spends the shared
    Database serializer to re-learn a constant."""
    db = Database(str(tmp_path / "register.db"))
    await db.connect()
    statements: list[str] = []
    original = db.execute

    async def counting(sql, params=()):
        statements.append(" ".join(sql.split()))
        return await original(sql, params)

    db.execute = counting
    try:
        observatory = MakerObservatory(db, horizons=(30, 60, 300),
                                       holdout_days=0)
        now = datetime.now(timezone.utc)
        for index in range(5):
            await observatory.observe(
                _market(), _book(), observed_at=now + timedelta(seconds=index))

        registrations = [sql for sql in statements
                         if sql.startswith("INSERT OR IGNORE INTO "
                                           "strategy_experiments")]
        horizons = [sql for sql in statements
                    if "maker_observatory_horizons" in sql]
        assert len(registrations) == 1, registrations
        assert len(horizons) == 3, horizons
        assert (await db.fetchone(
            "SELECT COUNT(*) n FROM maker_observatory_horizons"))["n"] == 3
    finally:
        db.execute = original
        await db.close()


def _frozen_market_maker_version(settings) -> str:
    """market_maker's strategy_version, hashed exactly as the gateway does.

    Mirrors ExecutionGateway._capture_decision: the strategy's own settings
    section plus the three risk knobs, canonically serialized.
    """
    import hashlib
    import json

    contract = {
        "strategy_source": "market_maker",
        "strategy_config": settings.market_maker.model_dump(mode="json"),
        "risk": {
            "min_edge_pct": settings.risk.min_edge_pct,
            "max_spread_pct": settings.risk.max_spread_pct,
            "confidence_floor": settings.risk.confidence_floor,
        },
    }
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_observatory_config_cannot_reversion_the_strategy_it_observes():
    """Retuning the instrument must not restart the observed holdout clock.

    The nine knobs live in their own top-level `maker_observatory:` section
    precisely because `settings.market_maker` is hashed into market_maker's
    strategy_version, and _prospective_stats joins on the LATEST version — so a
    knob parked in that section would silently discard the strategy's accrued
    prospective evidence every time the observatory was adjusted.
    """
    from config.settings import Settings

    settings = Settings()
    assert hasattr(settings, "maker_observatory")
    leaked = [name for name in settings.market_maker.model_dump()
              if "observatory" in name]
    assert leaked == [], f"observatory knobs inside the hashed section: {leaked}"

    before = _frozen_market_maker_version(settings)
    for field, value in (("review_days", 30), ("min_fills", 250),
                         ("retention_days", 90), ("enabled", False),
                         ("horizons_seconds", (15, 45)),
                         ("max_mark_lateness_seconds", 90)):
        settings.maker_observatory = settings.maker_observatory.model_copy(
            update={field: value})
        assert _frozen_market_maker_version(settings) == before, field

    # Control: a knob that genuinely belongs to the strategy still re-versions
    # it, so the assertion above is about placement, not a dead hash.
    settings.market_maker = settings.market_maker.model_copy(
        update={"quote_size": settings.market_maker.quote_size + 1})
    assert _frozen_market_maker_version(settings) != before


def _quoting_maker(db, *, observatory_enabled=True):
    from auramaur.strategy.market_maker import MarketMaker

    settings = SimpleNamespace(
        is_live=False,
        market_maker=SimpleNamespace(
            min_spread_bps=40, max_spread_bps=1500, quote_size=10.0,
            max_inventory=50.0, max_markets=5, refresh_seconds=30,
            op_timeout_seconds=15.0, paper=True, cash_reserve_usd=0.0),
        maker_observatory=SimpleNamespace(
            enabled=observatory_enabled, horizons_seconds=(30,),
            retention_days=45, max_mark_lateness_seconds=45, holdout_days=0),
        risk=SimpleNamespace(blocked_categories=[], allowed_categories_live=[]),
    )
    book = OrderBook(bids=[OrderBookLevel(price=.40, size=100)],
                     asks=[OrderBookLevel(price=.46, size=100)])

    async def get_order_book(_token):
        return book

    maker = MarketMaker(settings=settings,
                        exchange=SimpleNamespace(get_order_book=get_order_book),
                        db=db)
    return maker, book


@pytest.mark.asyncio
async def test_quoting_observes_the_book_without_the_observatory_gating_quotes(tmp_path):
    """The shadow-only claim, at the call site.

    Two halves: the quoting path really does record an observation (otherwise
    the instrument measures nothing), and an observatory that throws still
    leaves the quote fully formed and placed.
    """
    db = Database(str(tmp_path / "quoting.db"))
    await db.connect()
    try:
        maker, book = _quoting_maker(db)
        placed: list = []

        async def fake_place(quote, is_live):
            placed.append((quote, is_live))
            return {"success": True}

        maker._place_two_sided = fake_place
        market = _market()

        result, skip = await maker._quote_market(market)

        assert skip is None and result is not None, (skip, result)
        assert len(placed) == 1, "the observatory suppressed a quote"
        rows = await db.fetchall(
            "SELECT market_id,best_bid,best_ask,quote_bid FROM maker_observations")
        assert len(rows) == 1, "the quoting path recorded no observation"
        assert rows[0]["best_bid"] == pytest.approx(.40)
        assert rows[0]["best_ask"] == pytest.approx(.46)
        assert rows[0]["quote_bid"] is not None

        # A broken instrument must not cost a quote.
        async def exploding(*args, **kwargs):
            raise RuntimeError("observatory is down")

        maker._observatory.observe = exploding
        maker._active_quotes.clear()
        result, skip = await maker._quote_market(market)

        assert skip is None and result is not None, (skip, result)
        assert len(placed) == 2, "a failing observatory blocked the quote"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_ask_leg_fills_are_recorded_in_yes_terms(tmp_path):
    """The ask leg BUYS NO, so its venue fill price is a NO price.

    The observatory marks out against the YES-book midpoint, so the ask leg's
    price must be converted (1 - p_NO) on the way in. Recording the raw NO
    price would invert the markout on every ask fill — a sign error on half of
    every quote pair, which is worse than not measuring at all.
    """
    from auramaur.strategy.market_maker import MMQuote

    db = Database(str(tmp_path / "askleg.db"))
    await db.connect()
    try:
        maker, _ = _quoting_maker(db)
        quote = MMQuote(market_id="m1", token_yes_id="yes-1", token_no_id="no-1",
                        bid_price=.40, ask_price=.46, size=10.0, spread_bps=600)
        assert quote.no_leg_price == pytest.approx(.54)

        def result(order_id, filled_price):
            return SimpleNamespace(order_id=order_id, status="filled",
                                   filled_price=filled_price, filled_size=10.0,
                                   is_paper=False)

        async def place_quote_pair(bid_order, ask_order, *, exchange):
            # The ask leg really is priced in NO terms on the wire.
            assert ask_order.price == pytest.approx(.54)
            return result("bid-1", .40), result("ask-1", .55)

        maker._ensure_gateway = lambda: SimpleNamespace(
            place_quote_pair=place_quote_pair)

        await maker._place_two_sided(quote, is_live=True)

        rows = {row["order_id"]: row for row in await db.fetchall(
            "SELECT order_id,side,fill_price FROM maker_observatory_fills")}
        assert rows["bid-1"]["fill_price"] == pytest.approx(.40)
        # Filled at .55 in NO terms == sold YES at .45.
        assert rows["ask-1"]["side"] == "ask"
        assert rows["ask-1"]["fill_price"] == pytest.approx(.45)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_retention_runs_on_the_first_cycle_after_a_restart(tmp_path):
    """A cycle COUNTER would never reach its 24h period on a stack that
    restarts more often than daily, leaving retention permanently unrun."""
    db = Database(str(tmp_path / "retention.db"))
    await db.connect()
    try:
        maker, _ = _quoting_maker(db)
        pruned: list[int] = []

        async def counting_prune():
            pruned.append(1)

        maker._observatory.prune = counting_prune
        await maker.run_cycle([])
        assert pruned == [1], "retention never ran after startup"

        await maker.run_cycle([])
        await maker.run_cycle([])
        assert pruned == [1], "retention ran again inside its 24h period"
    finally:
        await db.close()
