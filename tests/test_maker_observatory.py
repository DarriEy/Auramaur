from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from auramaur.db.database import Database
from auramaur.exchange.models import Market, OrderBook, OrderBookLevel, OrderSide
from auramaur.monitoring.maker_observatory import (
    MakerObservatory,
    compute_maker_features,
    maker_observatory_feature_report,
    maker_observatory_summary,
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
        await observatory.record_fill(
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
        await observatory.record_fill(
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
            await observatory.record_fill(
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
