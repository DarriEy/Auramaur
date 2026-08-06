from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from auramaur.db.database import Database
from auramaur.exchange.models import Market, OrderBook, OrderBookLevel, OrderSide
from auramaur.monitoring.maker_observatory import (
    DUE_FILL_SCAN_SQL,
    MARK_BOOK_SQL,
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


def _market(market_id="m1"):
    return Market(id=market_id, question="fixture",
                  clob_token_yes=f"yes-{market_id}",
                  clob_token_no=f"no-{market_id}")


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
        await restarted.resolve_markouts(now=t0 + timedelta(seconds=61))
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
        await observatory.resolve_markouts(now=now)

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
            await observatory.resolve_markouts(
                now=fill_at + timedelta(seconds=31))

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
    # A later observation supplies the mark; the offline resolver takes it.
    await observatory.observe(_market(), mark_book or _book(.44, .46),
                              observed_at=at + timedelta(seconds=31))
    await observatory.resolve_markouts(now=at + timedelta(seconds=31))
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
        await observatory.resolve_markouts(now=t0 + timedelta(seconds=31))

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
        await observatory.resolve_markouts(now=t0 + timedelta(seconds=131))
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
            retention_days=45, max_mark_lateness_seconds=45, holdout_days=0,
            resolve_interval_seconds=60.0, resolve_batch_fills=500),
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


# --------------------------------------------------------------------------
# Markout resolution belongs off the quoting path.
# --------------------------------------------------------------------------


async def _tracing_db(tmp_path, name):
    """A Database that records the SQL every call issues."""
    db = Database(str(tmp_path / name))
    await db.connect()
    statements: list[str] = []
    for method in ("execute", "fetchall", "fetchone"):
        original = getattr(db, method)

        def wrap(original=original):
            async def traced(sql, params=()):
                statements.append(" ".join(str(sql).split()))
                return await original(sql, params)
            return traced

        setattr(db, method, wrap())
    return db, statements


@pytest.mark.asyncio
async def test_observe_never_resolves_markouts_on_the_quoting_path(tmp_path):
    """Reachability, not latency: the scan must not be callable from a quote.

    `_mark_due` used to run inside `observe()`, inside `_quote_market`, inside
    the per-market `op_timeout` window, holding the process-wide Database
    serializer — 93% of observe()'s cost, growing with retained history. A
    market maker that takes seconds to requote gets picked off, so leaving it
    there meant the instrument would cause the adverse selection it measures.

    A fill overdue by an hour must therefore survive any number of
    observations unmarked, and observe() must not so much as READ the fills or
    markouts tables — those reads are the cost.
    """
    db, statements = await _tracing_db(tmp_path, "quotepath.db")
    try:
        t0 = datetime(2026, 8, 6, tzinfo=timezone.utc)
        observatory = MakerObservatory(db, horizons=(30,), holdout_days=0,
                                       max_mark_lateness_seconds=45)
        observation_id = await observatory.observe(
            _market(), _book(), observed_at=t0)
        await observatory.record_observed_fill(
            observation_id=observation_id, order_id="overdue-1", market_id="m1",
            side="bid", price=.41, size=5, is_paper=False,
            fill_evidence="venue_fill", filled_at=t0)

        statements.clear()
        for index in range(3):
            await observatory.observe(
                _market(), _book(.44, .46),
                observed_at=t0 + timedelta(hours=1, seconds=index))

        touched = [sql for sql in statements
                   if "maker_observatory_fills" in sql
                   or "maker_observatory_markouts" in sql]
        assert touched == [], f"observe() read markout state: {touched}"
        assert (await db.fetchone(
            "SELECT COUNT(*) n FROM maker_observatory_markouts"))["n"] == 0, \
            "the quoting path resolved a markout"

        # The resolver still takes the mark, late and visibly invalid.
        assert await observatory.resolve_markouts(
            now=t0 + timedelta(hours=1, seconds=3)) == 1
        mark = await db.fetchone(
            "SELECT lateness_seconds,is_valid FROM maker_observatory_markouts")
        assert mark["is_valid"] == 0
        assert mark["lateness_seconds"] == pytest.approx(3570, abs=1)
    finally:
        await db.close()


async def _replay(db, *, resolve_each: bool):
    """Write one identical history; resolve every cycle, or once at the end.

    The gap between the 200s and 400s books is deliberate: the second fill's
    marks can only be taken from a book far past their horizon, so the run
    exercises the late/invalid branch as well as the timely one.
    """
    t0 = datetime(2026, 8, 6, tzinfo=timezone.utc)
    observatory = MakerObservatory(db, horizons=(30, 60), holdout_days=0,
                                   max_mark_lateness_seconds=45)
    script = (
        (0, (.40, .44), ("bid", .41)),
        (31, (.42, .46), None),
        (62, (.44, .48), None),
        (200, (.30, .34), ("ask", .43)),
        (400, (.30, .34), None),
    )
    for offset, (bid, ask), fill in script:
        at = t0 + timedelta(seconds=offset)
        observation_id = await observatory.observe(
            _market(), _book(bid, ask), observed_at=at)
        if fill is not None:
            side, price = fill
            await observatory.record_observed_fill(
                observation_id=observation_id, order_id=f"fill-{offset}",
                market_id="m1", side=side, price=price, size=5,
                is_paper=False, fill_evidence="venue_fill", filled_at=at)
        if resolve_each:
            await observatory.resolve_markouts(now=at)
    if not resolve_each:
        await observatory.resolve_markouts(now=t0 + timedelta(seconds=400))
    return await db.fetchall(
        """SELECT f.order_id,m.horizon_seconds,m.target_at,m.midpoint,m.markout,
                  m.marked_at,m.lateness_seconds,m.is_valid
             FROM maker_observatory_markouts m
             JOIN maker_observatory_fills f ON f.id=m.fill_id
            ORDER BY f.order_id,m.horizon_seconds""")


@pytest.mark.asyncio
async def test_offline_resolution_reproduces_the_inline_marks(tmp_path):
    """Moving WHEN the scan runs must not change WHAT it concludes.

    A mark is taken from the first observation at or after `filled_at +
    horizon` — precisely the book the old inline scan used, because that scan
    ran on the observation that first found the fill due. So resolving every
    cycle and resolving once, 400 seconds later, must write identical rows:
    same target, same midpoint, same markout, same lateness, same validity.
    Only the wall-clock moment of the INSERT is allowed to differ, and nothing
    records that.
    """
    eager = Database(str(tmp_path / "eager.db"))
    lazy = Database(str(tmp_path / "lazy.db"))
    await eager.connect()
    await lazy.connect()
    try:
        each = [tuple(row) for row in await _replay(eager, resolve_each=True)]
        once = [tuple(row) for row in await _replay(lazy, resolve_each=False)]

        assert each == once, "resolver cadence changed the marks"
        # And the marks are the right ones, so the agreement above is not two
        # runs being identically wrong.
        by_key = {(row[0], row[1]): row for row in each}
        assert set(by_key) == {("fill-0", 30), ("fill-0", 60),
                               ("fill-200", 30), ("fill-200", 60)}
        # Timely: the 31s book (mid .44) and the 62s book (mid .46).
        assert by_key[("fill-0", 30)][4] == pytest.approx(.03)
        assert by_key[("fill-0", 60)][4] == pytest.approx(.05)
        assert [by_key[("fill-0", h)][7] for h in (30, 60)] == [1, 1]
        # Late: both marks come from the 400s book (mid .32) on an ask at .43.
        assert by_key[("fill-200", 30)][4] == pytest.approx(.11)
        assert by_key[("fill-200", 30)][6] == pytest.approx(170)
        assert by_key[("fill-200", 60)][6] == pytest.approx(140)
        assert [by_key[("fill-200", h)][7] for h in (30, 60)] == [0, 0], \
            "a late mark was recorded as valid"
    finally:
        await eager.close()
        await lazy.close()


@pytest.mark.asyncio
async def test_an_interrupted_pass_leaves_marks_unresolved_never_fabricated(tmp_path):
    """Crash safety: a torn pass must resume, not invent or lose a mark."""
    db = Database(str(tmp_path / "interrupted.db"))
    await db.connect()
    try:
        t0 = datetime(2026, 8, 6, tzinfo=timezone.utc)
        observatory = MakerObservatory(db, horizons=(30,), holdout_days=0,
                                       max_mark_lateness_seconds=45)
        for index in range(2):
            at = t0 + timedelta(seconds=index)
            observation_id = await observatory.observe(
                _market(), _book(), observed_at=at)
            await observatory.record_observed_fill(
                observation_id=observation_id, order_id=f"fill-{index}",
                market_id="m1", side="bid", price=.41, size=5, is_paper=False,
                fill_evidence="venue_fill", filled_at=at)
        await observatory.observe(_market(), _book(.44, .46),
                                  observed_at=t0 + timedelta(seconds=31))
        later = t0 + timedelta(seconds=40)

        lookups = {"n": 0}
        original = db.fetchone

        async def failing(sql, params=()):
            if "maker_observations" in sql:
                lookups["n"] += 1
                if lookups["n"] == 2:
                    raise RuntimeError("resolver killed mid-pass")
            return await original(sql, params)

        db.fetchone = failing
        with pytest.raises(RuntimeError):
            await observatory.resolve_markouts(now=later)
        db.fetchone = original

        async def state():
            marks = await db.fetchall(
                "SELECT f.order_id FROM maker_observatory_markouts m "
                "JOIN maker_observatory_fills f ON f.id=m.fill_id "
                "ORDER BY f.order_id")
            pending = await db.fetchall(
                "SELECT order_id FROM maker_observatory_fills "
                "WHERE marks_pending=1 ORDER BY order_id")
            return ([row["order_id"] for row in marks],
                    [row["order_id"] for row in pending])

        marked, pending = await state()
        assert marked == ["fill-0"], "the interrupted fill was marked anyway"
        assert pending == ["fill-1"], "the unresolved fill stopped being pending"

        # The next pass finishes the job, and a third writes nothing new.
        assert await observatory.resolve_markouts(now=later) == 1
        marked, pending = await state()
        assert marked == ["fill-0", "fill-1"] and pending == []
        assert await observatory.resolve_markouts(now=later) == 0
        assert (await state())[0] == ["fill-0", "fill-1"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_partly_marked_fill_stays_pending_until_every_horizon_is_taken(tmp_path):
    """`marks_pending` is the resolver's only memory; retiring a fill early
    would silently drop the horizons it still owes.

    Here the 30s book exists and the 60s book does not yet. The fill must keep
    its place in the pending index and collect the second mark when the book
    arrives — not be written off as done.
    """
    db = Database(str(tmp_path / "partial.db"))
    await db.connect()
    try:
        t0 = datetime(2026, 8, 6, tzinfo=timezone.utc)
        observatory = MakerObservatory(db, horizons=(30, 60), holdout_days=0,
                                       max_mark_lateness_seconds=45)
        observation_id = await observatory.observe(
            _market(), _book(), observed_at=t0)
        await observatory.record_observed_fill(
            observation_id=observation_id, order_id="partial-1", market_id="m1",
            side="bid", price=.41, size=5, is_paper=False,
            fill_evidence="venue_fill", filled_at=t0)
        await observatory.observe(_market(), _book(.44, .46),
                                  observed_at=t0 + timedelta(seconds=31))

        async def pending():
            return (await db.fetchone(
                "SELECT marks_pending p FROM maker_observatory_fills"))["p"]

        assert await observatory.resolve_markouts(
            now=t0 + timedelta(seconds=70)) == 1
        horizons = [row["horizon_seconds"] for row in await db.fetchall(
            "SELECT horizon_seconds FROM maker_observatory_markouts")]
        assert horizons == [30]
        assert await pending() == 1, "a fill still owing a 60s mark was retired"

        # The 60s book arrives; the second mark is taken and only now is the
        # fill retired from the pending index.
        await observatory.observe(_market(), _book(.48, .50),
                                  observed_at=t0 + timedelta(seconds=65))
        assert await observatory.resolve_markouts(
            now=t0 + timedelta(seconds=70)) == 1
        marks = await db.fetchall(
            "SELECT horizon_seconds,markout,is_valid FROM "
            "maker_observatory_markouts ORDER BY horizon_seconds")
        assert [row["horizon_seconds"] for row in marks] == [30, 60]
        assert [row["markout"] for row in marks] == pytest.approx([.04, .08])
        assert [row["is_valid"] for row in marks] == [1, 1]
        assert await pending() == 0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_the_due_fill_scan_seeks_an_index_instead_of_scanning_history(tmp_path):
    """The predicate must stay sargable, and the test must EXPLAIN the SHIPPED
    SQL — hence the module constants rather than a copy pasted here.

    The shape this replaced, `unixepoch(?) - unixepoch(filled_at) >= ?`, wraps
    the column in a function: SQLite could seek to the market but then had to
    walk every retained fill in it (259k per market at 45-day retention) and
    probe the markouts index once per row, on every call.
    """
    db = Database(str(tmp_path / "plan.db"))
    await db.connect()
    try:
        due = await db.fetchall(
            f"EXPLAIN QUERY PLAN {DUE_FILL_SCAN_SQL}",
            ("2026-08-06T00:00:00.000000+00:00", 30, 500))
        plan = " ".join(str(row["detail"]) for row in due)
        assert "idx_maker_fills_pending" in plan, plan
        # The liveness guard must also seek, not walk a market's observations.
        assert "idx_maker_obs_market_time" in plan, plan
        assert "SCAN" not in plan, f"full scan of the fill history: {plan}"
        assert "TEMP B-TREE" not in plan, f"the sort is not index-served: {plan}"

        book = await db.fetchall(
            f"EXPLAIN QUERY PLAN {MARK_BOOK_SQL}",
            ("m1", "2026-08-06T00:00:00.000000+00:00"))
        plan = " ".join(str(row["detail"]) for row in book)
        assert "idx_maker_obs_market_time" in plan, plan
        assert "SCAN" not in plan, f"full scan of the observations: {plan}"
        assert "TEMP B-TREE" not in plan, plan
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_resolver_pass_is_bounded_and_takes_the_oldest_first(tmp_path):
    """One pass must not be able to hold the shared serializer indefinitely.

    A backlog (long outage, clock jump) is drained across passes, oldest
    first, because the oldest marks are the ones closest to going permanently
    invalid.
    """
    db = Database(str(tmp_path / "batch.db"))
    await db.connect()
    try:
        t0 = datetime(2026, 8, 6, tzinfo=timezone.utc)
        observatory = MakerObservatory(db, horizons=(30,), holdout_days=0,
                                       max_mark_lateness_seconds=45,
                                       resolve_batch_fills=2)
        for index in range(5):
            at = t0 + timedelta(seconds=index)
            observation_id = await observatory.observe(
                _market(), _book(), observed_at=at)
            await observatory.record_observed_fill(
                observation_id=observation_id, order_id=f"fill-{index}",
                market_id="m1", side="bid", price=.41, size=5, is_paper=False,
                fill_evidence="venue_fill", filled_at=at)
        await observatory.observe(_market(), _book(.44, .46),
                                  observed_at=t0 + timedelta(seconds=60))
        later = t0 + timedelta(seconds=90)

        assert await observatory.resolve_markouts(now=later) == 2
        marked = [row["order_id"] for row in await db.fetchall(
            "SELECT f.order_id FROM maker_observatory_markouts m "
            "JOIN maker_observatory_fills f ON f.id=m.fill_id "
            "ORDER BY f.filled_at")]
        assert marked == ["fill-0", "fill-1"], marked

        assert await observatory.resolve_markouts(now=later) == 2
        assert await observatory.resolve_markouts(now=later) == 1
        assert await observatory.resolve_markouts(now=later) == 0
        assert (await db.fetchone(
            "SELECT COUNT(*) n FROM maker_observatory_markouts"))["n"] == 5
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_dead_market_cannot_starve_newer_fills_out_of_the_batch(tmp_path):
    """The batch is oldest-first, so unmarkable fills must not hold its head.

    A market that leaves the maker's five and is never observed again strands
    fills that can never be marked at any horizon. Without the liveness guard
    they would sit at the front of the queue until retention pruned them 45
    days later, silently starving every newer mark — completeness would rot
    while the resolver looked busy.
    """
    db = Database(str(tmp_path / "starve.db"))
    await db.connect()
    try:
        t0 = datetime(2026, 8, 6, tzinfo=timezone.utc)
        observatory = MakerObservatory(db, horizons=(30,), holdout_days=0,
                                       max_mark_lateness_seconds=45,
                                       resolve_batch_fills=2)
        # Three older fills on a market that is never observed again.
        for index in range(3):
            at = t0 + timedelta(seconds=index)
            observation_id = await observatory.observe(
                _market("gone"), _book(), observed_at=at)
            await observatory.record_observed_fill(
                observation_id=observation_id, order_id=f"stranded-{index}",
                market_id="gone", side="bid", price=.41, size=5,
                is_paper=False, fill_evidence="venue_fill", filled_at=at)
        # A newer, markable fill on a market that keeps being observed.
        live_at = t0 + timedelta(seconds=100)
        observation_id = await observatory.observe(
            _market(), _book(), observed_at=live_at)
        await observatory.record_observed_fill(
            observation_id=observation_id, order_id="live-1", market_id="m1",
            side="bid", price=.41, size=5, is_paper=False,
            fill_evidence="venue_fill", filled_at=live_at)
        await observatory.observe(_market(), _book(.44, .46),
                                  observed_at=live_at + timedelta(seconds=31))

        assert await observatory.resolve_markouts(
            now=live_at + timedelta(seconds=40)) == 1
        marked = [row["order_id"] for row in await db.fetchall(
            "SELECT f.order_id FROM maker_observatory_markouts m "
            "JOIN maker_observatory_fills f ON f.id=m.fill_id")]
        assert marked == ["live-1"], marked
        # The stranded fills are still pending, not written off — if their
        # market returns they take their (late, invalid) marks as usual.
        stranded = [row["order_id"] for row in await db.fetchall(
            "SELECT order_id FROM maker_observatory_fills "
            "WHERE marks_pending=1 ORDER BY order_id")]
        assert stranded == ["stranded-0", "stranded-1", "stranded-2"]

        await observatory.observe(_market("gone"), _book(.44, .46),
                                  observed_at=live_at + timedelta(seconds=60))
        assert await observatory.resolve_markouts(
            now=live_at + timedelta(seconds=70)) == 2  # batch of 2, oldest first
        late = await db.fetchall(
            "SELECT f.order_id,m.is_valid FROM maker_observatory_markouts m "
            "JOIN maker_observatory_fills f ON f.id=m.fill_id "
            "WHERE f.market_id='gone' ORDER BY f.order_id")
        assert [row["order_id"] for row in late] == ["stranded-0", "stranded-1"]
        assert [row["is_valid"] for row in late] == [0, 0]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_quoting_records_fills_but_only_the_maker_timer_marks_them(tmp_path):
    """At the call site: `_quote_market` observes, and never marks out.

    `MarketMaker.resolve_markouts` is the entry point the bot's own
    `maker_observatory` task drives, so the growing scan runs on a timer
    instead of between deciding a quote and placing it.
    """
    db = Database(str(tmp_path / "makertimer.db"))
    await db.connect()
    try:
        maker, _ = _quoting_maker(db)

        async def fake_place(quote, is_live):
            return {"success": True}

        maker._place_two_sided = fake_place
        market = _market()
        t0 = datetime.now(timezone.utc) - timedelta(seconds=300)

        observation_id = await maker._observatory.observe(
            market, _book(), observed_at=t0)
        await maker._observatory.record_observed_fill(
            observation_id=observation_id, order_id="due-1", market_id="m1",
            side="bid", price=.41, size=5, is_paper=False,
            fill_evidence="venue_fill", filled_at=t0)

        for _ in range(3):
            maker._active_quotes.clear()
            await maker._quote_market(market)
        assert (await db.fetchone(
            "SELECT COUNT(*) n FROM maker_observatory_markouts"))["n"] == 0, \
            "quoting resolved a markout"

        assert await maker.resolve_markouts() == 1
        assert (await db.fetchone(
            "SELECT COUNT(*) n FROM maker_observatory_markouts"))["n"] == 1

        # With the instrument off, the entry point is inert rather than absent.
        off, _ = _quoting_maker(db, observatory_enabled=False)
        assert off._observatory is None
        assert await off.resolve_markouts() == 0
    finally:
        await db.close()


def _observatory_task_stub(maker, *, enabled=True, interval=17.0):
    return SimpleNamespace(
        _components=SimpleNamespace(market_maker=maker),
        settings=SimpleNamespace(maker_observatory=SimpleNamespace(
            enabled=enabled, resolve_interval_seconds=interval)),
        _running=True,
    )


@pytest.mark.asyncio
async def test_the_bot_drives_resolution_from_its_own_task(monkeypatch):
    """Without this wiring nothing would ever mark out.

    Moving the scan off the quoting path only helps if something else runs it,
    and an observatory that records fills and never marks them would look
    installed while measuring nothing. Also pins that the task does NOT
    heartbeat: `check_strategy_data_delivery` fails closed on any heartbeat
    whose strategy has no registered data contract, and this task consumes no
    market data.
    """
    import auramaur.bot as bot_module
    from auramaur.bot import AuramaurBot

    beats: list = []

    async def spy_beat(*args, **kwargs):
        beats.append(args)

    monkeypatch.setattr("auramaur.monitoring.heartbeat.beat", spy_beat)

    calls: list[int] = []
    sleeps: list[float] = []

    class Maker:
        async def resolve_markouts(self):
            calls.append(len(calls))
            return 2

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 3:
            stub._running = False

    monkeypatch.setattr(bot_module.asyncio, "sleep", fake_sleep)
    stub = _observatory_task_stub(Maker())
    await AuramaurBot._task_maker_observatory(stub)

    assert calls == [0, 1, 2], "the resolver was not driven on its own timer"
    assert sleeps == [17.0, 17.0, 17.0], sleeps
    assert beats == [], "the observatory task heartbeat has no data contract"

    # A failing pass must not end the task; measurement never kills the loop.
    class Exploding:
        async def resolve_markouts(self):
            calls.append(-1)
            raise RuntimeError("resolver blew up")

    sleeps.clear()
    calls.clear()
    stub = _observatory_task_stub(Exploding())
    await AuramaurBot._task_maker_observatory(stub)
    assert calls == [-1, -1, -1]

    # Disabled or absent instrument: the task retires instead of spinning.
    sleeps.clear()
    calls.clear()
    await AuramaurBot._task_maker_observatory(
        _observatory_task_stub(Maker(), enabled=False))
    await AuramaurBot._task_maker_observatory(_observatory_task_stub(None))
    assert calls == [] and sleeps == []


def test_the_observatory_task_is_registered_and_classified():
    """A task the bot never starts is a silent no-op, and an unclassified
    `_task_*` is how a trading task once escaped the strategy registry."""
    import inspect

    from auramaur.bot import AuramaurBot
    from auramaur.strategy.registry import NON_STRATEGY_TASKS, STRATEGY_BY_TASK

    assert "_task_maker_observatory" in NON_STRATEGY_TASKS
    assert "_task_maker_observatory" not in STRATEGY_BY_TASK
    source = inspect.getsource(AuramaurBot.run)
    assert "self._task_maker_observatory()" in source, \
        "the resolver task is defined but never started"


# --------------------------------------------------------------------------
# signed_flow must be able to say "no feed".
# --------------------------------------------------------------------------


def test_signed_flow_reports_an_absent_feed_as_none_not_zero(monkeypatch):
    """0.0 and "no data" are different facts and must stay different values."""
    tracker = OrderFlowTracker()
    now = datetime(2026, 8, 6, tzinfo=timezone.utc)
    monkeypatch.setattr("auramaur.strategy.order_flow.datetime", type(
        "Clock", (), {"now": staticmethod(lambda tz: now)}))

    assert tracker.signed_flow(("m1", "", "yes-m1"), now=now) is None

    # A market the feed HAS reached, whose flow genuinely nets to zero, is a
    # measurement of 0.0 — not a gap.
    tracker.record_trade("m1", OrderSide.BUY, 10)
    tracker.record_trade("m1", OrderSide.SELL, 10)
    balanced = tracker.signed_flow(("m1", "", "yes-m1"), now=now)
    assert balanced is not None and balanced == pytest.approx(0.0)

    # Any of the three equivalent feed keys counts as coverage.
    tracker.record_trade("cond-2", OrderSide.BUY, 4)
    assert tracker.signed_flow(("m2", "cond-2", "yes-m2"),
                               now=now) == pytest.approx(4)


@pytest.mark.asyncio
async def test_a_market_with_no_feed_records_null_flow_and_reads_as_uncovered(tmp_path):
    """The 2026-07-29 qwen3 failure mode, pinned at the column.

    OrderFlowTracker is fed only by the websocket price monitor's `on_trade`
    for the first 20 discovered markets; the maker picks its five by spread and
    usually is not in that set. Writing 0.0 there would assert balanced flow
    about a market no feed ever reached — a metric incapable of moving, looking
    healthy forever.
    """
    db = Database(str(tmp_path / "flow.db"))
    await db.connect()
    try:
        tracker = OrderFlowTracker()
        now = datetime.now(timezone.utc)
        tracker.record_trade("m1", OrderSide.BUY, 10)
        tracker.record_trade("m1", OrderSide.SELL, 10)
        observatory = MakerObservatory(db, flow_tracker=tracker, horizons=(30,),
                                       holdout_days=0)
        await observatory.observe(_market("m1"), _book(), observed_at=now)
        await observatory.observe(_market("m2"), _book(), observed_at=now)

        flows = {row["market_id"]: row["signed_flow"] for row in await db.fetchall(
            "SELECT market_id,signed_flow FROM maker_observations")}
        assert flows["m1"] == pytest.approx(0.0, abs=1e-3), \
            "a covered, genuinely balanced market must record a number"
        assert flows["m2"] is None, "no feed was recorded as balanced flow"

        coverage = await maker_quote_coverage(db, days=1)
        assert coverage["samples"] == 2
        assert coverage["flow_samples"] == 1, "NULL flow counted as data"
        assert coverage["flow_coverage"] == pytest.approx(.5)

        # No tracker at all is also absence, not balance.
        untracked = MakerObservatory(db, horizons=(30,), holdout_days=0)
        await untracked.observe(_market("m3"), _book(), observed_at=now)
        assert (await db.fetchone(
            "SELECT signed_flow FROM maker_observations WHERE market_id='m3'")
        )["signed_flow"] is None
    finally:
        await db.close()


def test_the_cli_report_shows_flow_coverage_rather_than_implying_balance(
        tmp_path, monkeypatch):
    """The operator-facing surface has to say how much flow was measured.

    A scorecard that prints a signed_flow effect without saying it was
    computed over a fraction of the marks lets a reader take an absent result
    for a negative one.

    Synchronous on purpose: the command owns its own `asyncio.run`.
    """
    import asyncio

    from click.testing import CliRunner

    from auramaur.cli._base import main

    path = tmp_path / "cli.db"

    async def seed():
        db = Database(str(path))
        await db.connect()
        try:
            now = datetime.now(timezone.utc)
            observatory = await _observatory(db, holdout_days=0)
            await _set_holdout_boundary(db, observatory, now - timedelta(days=1))
            await _fill(observatory, db, at=now - timedelta(seconds=300),
                        order_id="cli-1")
            # One observation carries flow data; the rest do not.
            await db.execute("UPDATE maker_observations SET signed_flow=1.5 "
                             "WHERE id=(SELECT MIN(id) FROM maker_observations)")
        finally:
            await db.close()

    asyncio.run(seed())
    monkeypatch.setattr("auramaur.web.db.runtime_db_path", lambda: path)
    result = CliRunner().invoke(main, ["maker-observatory", "--days", "1"])

    assert result.exit_code == 0, result.output
    assert "Trade-feed coverage" in result.output
    assert "signed_flow is not fully covered" in result.output
    assert "NOT balanced flow" in result.output


@pytest.mark.asyncio
async def test_feature_report_excludes_null_flow_and_surfaces_its_coverage(tmp_path):
    """An unmeasured feature must not read as a measured null result."""
    db = Database(str(tmp_path / "flowreport.db"))
    await db.connect()
    try:
        now = datetime.now(timezone.utc)
        observatory = await _observatory(db, holdout_days=0)
        await _set_holdout_boundary(db, observatory, now - timedelta(seconds=350))
        for index in range(4):
            await _fill(observatory, db, order_id=f"live-{index}",
                        at=now - timedelta(seconds=400 - index * 40))
        # Exactly one warmup and one holdout observation carry flow data; the
        # other two are NULL because no feed reached that market.
        for order_id, value in (("live-1", 5.0), ("live-3", 9.0)):
            await db.execute(
                "UPDATE maker_observations SET signed_flow=? WHERE id="
                "(SELECT observation_id FROM maker_observatory_fills "
                " WHERE order_id=?)", (value, order_id))

        report = {row["feature"]: row
                  for row in await maker_observatory_feature_report(db, days=1)}
        flow = report["signed_flow"]
        assert flow["marks_n"] == 4
        assert flow["covered_n"] == 2, "NULL flow was counted as data"
        assert flow["coverage"] == pytest.approx(.5)
        assert flow["warmup_n"] == 1 and flow["holdout_n"] == 1
        assert flow["threshold"] == pytest.approx(5.0), \
            "a NULL was folded into the warmup median"

        # A feature that WAS recorded everywhere reports full coverage, so the
        # number above is measuring something rather than being a constant.
        imbalance = report["depth_imbalance"]
        assert imbalance["marks_n"] == 4 and imbalance["covered_n"] == 4
        assert imbalance["coverage"] == pytest.approx(1.0)
        assert imbalance["warmup_n"] == 2 and imbalance["holdout_n"] == 2
    finally:
        await db.close()
