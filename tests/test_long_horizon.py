"""Tests for the long-horizon favorite-underpricing pillar.

Locks in:
  1. calibrated_fair — the pure slope-correction core (favorites underpriced,
     symmetry, slope=1 is identity).
  2. Eligibility — only moderate favorites on active, liquid, LONG-dated,
     non-politics Polymarket markets enter.
  3. PAPER-FORCING — paper=true builds the order with is_live=False.
  4. The favored side maps to the right BUY/SELL signal + fair P(YES).
  5. min_edge / short-horizon / politics exclusions skip.
  6. A fill writes the full rails (signal, trade, portfolio is_paper, calibration).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

from auramaur.broker.pnl import PnLTracker
from auramaur.db.database import Database
from auramaur.exchange.models import (
    Market,
    Order,
    OrderResult,
    OrderSide,
    OrderType,
    TokenType,
)
from auramaur.strategy.long_horizon import LongHorizonPillar, calibrated_fair
from config.settings import Settings


def _market(mid="m1", yes=0.70, liquidity=5000.0, category="tech",
            active=True, days_out=60.0, exchange="polymarket") -> Market:
    return Market(
        id=mid, exchange=exchange, question=f"q-{mid}", category=category,
        active=active, outcome_yes_price=yes, outcome_no_price=round(1 - yes, 2),
        liquidity=liquidity, volume=10000.0,
        end_date=datetime.now(timezone.utc) + timedelta(days=days_out),
        clob_token_yes="tok-yes", clob_token_no="tok-no",
    )


def _settings(**overrides) -> Settings:
    s = Settings()
    s.long_horizon.enabled = True
    s.long_horizon.paper = True
    for k, v in overrides.items():
        setattr(s.long_horizon, k, v)
    return s


def _exchange(filled=True):
    ex = MagicMock()

    def prepare_order(signal, market, size, is_live):
        token = TokenType.NO if signal.recommended_side == OrderSide.SELL else TokenType.YES
        price = market.outcome_yes_price if token == TokenType.YES else 1 - market.outcome_yes_price
        return Order(
            market_id=market.id, token_id="tok", side=OrderSide.BUY, token=token,
            size=round(size / price, 2), price=round(price, 2),
            order_type=OrderType.LIMIT, dry_run=not is_live,
        )

    ex.prepare_order = MagicMock(side_effect=prepare_order)
    ex.place_order = AsyncMock(side_effect=lambda order: OrderResult(
        order_id="ord-1", market_id=order.market_id,
        status="paper" if order.dry_run else "filled",
        filled_size=order.size if filled else 0, filled_price=order.price,
        is_paper=order.dry_run,
    ))
    return ex


def _risk(approved=True, size=8.0, force_paper=False):
    rm = MagicMock()
    decision = MagicMock()
    decision.approved = approved
    decision.position_size = size if approved else 0.0
    decision.reason = "" if approved else "blocked"
    decision.force_paper = force_paper
    rm.evaluate = AsyncMock(return_value=decision)
    return rm


def _pillar(db, settings, markets, exchange=None, risk=None):
    discovery = MagicMock()
    discovery.get_markets = AsyncMock(return_value=markets)
    calibration = MagicMock()
    calibration.record_prediction = AsyncMock()
    return LongHorizonPillar(
        db=db, settings=settings, discovery=discovery,
        exchange=exchange or _exchange(), risk_manager=risk or _risk(),
        pnl_tracker=PnLTracker(db, settings), calibration=calibration,
    ), calibration


# ----------------------------------------------------------------------
# calibrated_fair — the pure core
# ----------------------------------------------------------------------

def test_calibrated_fair_lifts_favorites_and_is_identity_at_slope_one():
    # slope 1.0 -> identity
    assert abs(calibrated_fair(0.70, 1.0) - 0.70) < 1e-9
    # slope > 1 lifts a favorite (the documented underpricing): 0.70 -> ~0.75
    f = calibrated_fair(0.70, 1.32)
    assert 0.74 < f < 0.76
    # symmetry: a longshot is pushed DOWN by the same logit factor
    assert abs((1 - calibrated_fair(0.30, 1.32)) - calibrated_fair(0.70, 1.32)) < 1e-9
    # a 50/50 is unmoved
    assert abs(calibrated_fair(0.50, 1.5) - 0.50) < 1e-9


# ----------------------------------------------------------------------
# Entry behaviour
# ----------------------------------------------------------------------

def test_enters_long_dated_favorite_and_writes_rails():
    async def run():
        db = Database(":memory:")
        await db.connect()
        settings = _settings(slope=1.32)
        ex = _exchange()
        pillar, calibration = _pillar(db, settings, [_market(yes=0.70)], exchange=ex)
        assert await pillar.run_once() == 1

        # paper-forced: prepare_order saw is_live=False
        assert ex.prepare_order.call_args[0][3] is False

        sig = await db.fetchone(
            "SELECT * FROM signals WHERE strategy_source='long_horizon'")
        assert sig is not None and sig["action"] == "BUY"
        # claude_prob is the slope-corrected fair (~0.75), above the 0.70 price.
        assert sig["claude_prob"] > 0.74 and sig["market_prob"] == 0.70

        trade = await db.fetchone(
            "SELECT * FROM trades WHERE strategy_source='long_horizon'")
        assert trade is not None and trade["is_paper"] == 1
        pos = await db.fetchone("SELECT * FROM portfolio WHERE market_id='m1'")
        assert pos is not None and pos["is_paper"] == 1 and pos["token"] == "YES"
        calibration.record_prediction.assert_awaited_once()
        await db.close()

    asyncio.run(run())


def test_favored_no_side_enters_as_sell():
    """YES at 0.30 -> favored NO at 0.70 -> SELL signal (buys NO)."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        pillar, _ = _pillar(db, _settings(slope=1.32), [_market(yes=0.30)])
        assert await pillar.run_once() == 1
        sig = await db.fetchone(
            "SELECT * FROM signals WHERE strategy_source='long_horizon'")
        assert sig["action"] == "SELL"
        pos = await db.fetchone("SELECT token FROM portfolio WHERE market_id='m1'")
        assert pos["token"] == "NO"
        await db.close()

    asyncio.run(run())


def test_skips_short_horizon_politics_thin_and_subedge():
    async def run():
        db = Database(":memory:")
        await db.connect()
        markets = [
            _market("short", yes=0.70, days_out=10.0),        # < 30d horizon
            _market("toolong", yes=0.70, days_out=400.0),     # > max horizon
            _market("thin", yes=0.70, liquidity=100.0),       # < min_liquidity
            _market("pol", yes=0.70, category="politics_us"),  # excluded category
            _market("coinflip", yes=0.52),                    # below band_lo
            _market("kalshi", yes=0.70, exchange="kalshi"),   # poly-only v1
        ]
        pillar, _ = _pillar(db, _settings(slope=1.32), markets)
        assert await pillar.run_once() == 0
        await db.close()

    asyncio.run(run())


def test_min_edge_blocks_when_slope_correction_too_small():
    """A barely-favored market at a low slope yields a sub-min_edge correction."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        # 0.58 favorite at slope 1.05: fair ~0.585 -> edge ~0.005 < min_edge 0.03.
        pillar, _ = _pillar(db, _settings(slope=1.05, min_edge=0.03),
                            [_market(yes=0.58)])
        assert await pillar.run_once() == 0
        # No signal persisted (gated before persist).
        assert await db.fetchone("SELECT 1 FROM signals") is None
        await db.close()

    asyncio.run(run())


def test_one_entry_per_market_and_skips_held():
    async def run():
        db = Database(":memory:")
        await db.connect()
        settings = _settings(slope=1.32)
        pillar, _ = _pillar(db, settings, [_market(yes=0.70)])
        assert await pillar.run_once() == 1
        assert await pillar.run_once() == 0  # signal exists -> no re-entry

        await db.execute(
            "INSERT INTO portfolio (market_id, exchange, side, size, avg_price, "
            "current_price, is_paper) VALUES ('held','polymarket','BUY',5,0.5,0.5,0)")
        await db.commit()
        pillar2, _ = _pillar(db, settings, [_market("held", yes=0.70)])
        assert await pillar2.run_once() == 0  # held by another strategy -> skip
        await db.close()

    asyncio.run(run())


def test_risk_rejection_places_no_order():
    async def run():
        db = Database(":memory:")
        await db.connect()
        ex = _exchange()
        pillar, _ = _pillar(db, _settings(slope=1.32), [_market(yes=0.70)],
                            exchange=ex, risk=_risk(approved=False))
        assert await pillar.run_once() == 0
        ex.place_order.assert_not_awaited()
        await db.close()

    asyncio.run(run())


def test_medium_horizon_now_enters_after_widening():
    """A ~20-day favorite now enters (min_days widened 30->14 to accrue data);
    it would have been skipped under the old >30d floor."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        pillar, _ = _pillar(db, _settings(slope=1.32), [_market(yes=0.70, days_out=20.0)])
        assert await pillar.run_once() == 1
        await db.close()

    asyncio.run(run())


def test_scan_long_dated_paginates_and_uses_date_window():
    """The scan queries Gamma's resolution-date window (not the top-100-by-volume
    page) and paginates until a page is empty — the fix for the 100-row cap."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        from unittest.mock import AsyncMock, MagicMock
        # page 0 -> one market; page 1 -> empty (stops).
        m = _market(yes=0.70)
        disc = MagicMock()
        disc.get_markets = AsyncMock(side_effect=[[m], []])
        cal = MagicMock(); cal.record_prediction = AsyncMock()
        from auramaur.strategy.long_horizon import LongHorizonPillar
        from auramaur.broker.pnl import PnLTracker
        pillar = LongHorizonPillar(db=db, settings=_settings(), discovery=disc,
                                   exchange=_exchange(), risk_manager=_risk(),
                                   pnl_tracker=PnLTracker(db, _settings()), calibration=cal)
        got = await pillar._scan_long_dated(_settings().long_horizon)
        assert got == [m]                                  # one page, then empty -> stop
        # the date window kwargs were passed
        kw = disc.get_markets.call_args_list[0].kwargs
        assert "end_date_min" in kw and "end_date_max" in kw and kw["order"] == "liquidity"
        await db.close()
    asyncio.run(run())


# ----------------------------------------------------------------------
# Kalshi instance + decay harvest (2026-07)
# ----------------------------------------------------------------------


def _kalshi_pillar(db, settings, markets, exchange=None, risk=None):
    discovery = MagicMock()
    discovery.get_markets = AsyncMock(return_value=markets)
    calibration = MagicMock()
    calibration.record_prediction = AsyncMock()
    return LongHorizonPillar(
        db=db, settings=settings, discovery=discovery,
        exchange=exchange or _exchange(), risk_manager=risk or _risk(),
        pnl_tracker=PnLTracker(db, settings), calibration=calibration,
        venue="kalshi",
    )


async def test_kalshi_instance_owns_tag_venue_and_admits_politics_intl():
    """The Kalshi instance writes long_horizon_kalshi signals and a kalshi
    portfolio row, ADMITS politics_intl (the live-book evidence category),
    and rejects Polymarket markets (each instance owns exactly its venue)."""
    db = Database(":memory:")
    await db.connect()
    try:
        settings = _settings()
        markets = [
            _market("KX1", yes=0.75, category="politics_intl", exchange="kalshi"),
            _market("p1", yes=0.75, category="tech", exchange="polymarket"),
        ]
        pillar = _kalshi_pillar(db, settings, markets)
        entered = await pillar.run_once()
        assert entered == 1
        sig = await db.fetchone("SELECT market_id, strategy_source FROM signals")
        assert sig["market_id"] == "KX1"
        assert sig["strategy_source"] == "long_horizon_kalshi"
        port = await db.fetchone(
            "SELECT exchange FROM portfolio WHERE market_id='KX1'")
        assert port["exchange"] == "kalshi"
    finally:
        await db.close()


async def test_poly_instance_still_excludes_politics():
    db = Database(":memory:")
    await db.connect()
    try:
        settings = _settings()
        pillar, _ = _pillar(db, settings, [
            _market("p2", yes=0.75, category="politics_intl")])
        assert await pillar.run_once() == 0
    finally:
        await db.close()


async def _seed_position(db, *, market_id, tag, venue, token, size, avg,
                         yes_price, fresh=True):
    await db.execute(
        """INSERT INTO markets (id, exchange, question, category, active,
           outcome_yes_price, outcome_no_price, last_updated, created_at)
           VALUES (?, ?, 'q?', 'tech', 1, ?, ?, ?, datetime('now'))""",
        (market_id, venue, yes_price, round(1 - yes_price, 2),
         "now" if fresh else "2026-01-01 00:00:00"))
    if fresh:
        await db.execute(
            "UPDATE markets SET last_updated = datetime('now') WHERE id = ?",
            (market_id,))
    await db.execute(
        """INSERT INTO signals (market_id, claude_prob, claude_confidence,
           market_prob, edge, evidence_summary, action, strategy_source)
           VALUES (?, 0.8, 'MEDIUM', 0.7, 8.0, 'e', 'BUY', ?)""",
        (market_id, tag))
    await db.execute(
        """INSERT INTO portfolio (market_id, exchange, side, size, avg_price,
           current_price, unrealized_pnl, category, token, token_id, is_paper,
           updated_at)
           VALUES (?, ?, 'BUY', ?, ?, ?, 0, 'tech', ?, 'tok', 1, datetime('now'))""",
        (market_id, venue, size, avg, avg, token))
    # The BUY's cost basis — what the harvest's SELL realizes against.
    await db.execute(
        """INSERT INTO cost_basis (market_id, token, token_id, size, avg_cost,
           total_cost, realized_pnl, is_paper, updated_at)
           VALUES (?, ?, 'tok', ?, ?, ?, 0, 1, datetime('now'))""",
        (market_id, token, size, avg, size * avg))
    await db.commit()


async def test_decay_harvest_exits_captured_position_and_realizes():
    """A NO position entered at 0.66 with the market now at YES=0.06 (mark
    0.94) has captured (0.94-0.66)/(1-0.66) = 82% >= 60% -> exit fires, and
    the sell realizes a ledger event attributed to the ENTRY cell (the
    token-scoped #268 lookup) — graduation events on a weeks clock."""
    db = Database(":memory:")
    await db.connect()
    try:
        settings = _settings()
        pillar = _kalshi_pillar(db, settings, [])
        await _seed_position(db, market_id="KXH", tag="long_horizon_kalshi",
                             venue="kalshi", token="NO", size=30, avg=0.66,
                             yes_price=0.06)
        exited = await pillar._harvest_decay(settings.long_horizon)
        assert exited == 1
        row = await db.fetchone(
            "SELECT strategy_source, pnl FROM pnl_ledger WHERE market_id='KXH'")
        assert row is not None
        assert row["pnl"] > 0
    finally:
        await db.close()


async def test_decay_harvest_skips_below_floor_and_stale_marks():
    db = Database(":memory:")
    await db.connect()
    try:
        settings = _settings()
        pillar = _kalshi_pillar(db, settings, [])
        # Captured only 26% (< 60%) -> hold.
        await _seed_position(db, market_id="KXLOW", tag="long_horizon_kalshi",
                             venue="kalshi", token="NO", size=30, avg=0.66,
                             yes_price=0.25)
        # Fully captured but the mark is STALE (frozen row) -> skip, not exit.
        await _seed_position(db, market_id="KXSTALE", tag="long_horizon_kalshi",
                             venue="kalshi", token="NO", size=30, avg=0.66,
                             yes_price=0.02, fresh=False)
        assert await pillar._harvest_decay(settings.long_horizon) == 0
        assert await db.fetchone("SELECT 1 FROM pnl_ledger") is None
    finally:
        await db.close()
