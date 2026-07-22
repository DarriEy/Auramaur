"""Tests for position exit strategy (stop-loss, profit target, edge erosion, time decay)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from auramaur.exchange.models import ExitReason, OrderSide
from auramaur.db.database import Database
from auramaur.risk.portfolio import PortfolioTracker


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.fetchall = AsyncMock(return_value=[])
    db.fetchone = AsyncMock(return_value=None)
    return db


@pytest.fixture
def settings():
    s = MagicMock()
    s.execution.stop_loss_pct = 30.0
    s.execution.profit_target_pct = 50.0
    s.execution.edge_erosion_min_pct = 2.0
    s.execution.time_decay_hours = 12.0
    # Off by default so existing exit tests are unaffected; opted in per-test.
    s.execution.free_winners_enabled = False
    s.execution.free_winners_max_upside_pct = 5.0
    s.execution.free_winners_min_hours = 48.0
    s.execution.dust_sweep_enabled = False
    s.execution.dust_max_notional = 1.0
    s.execution.dust_min_age_hours = 24.0
    return s


def _make_position(market_id: str, avg_price: float, size: float = 10.0,
                   side: OrderSide = OrderSide.BUY) -> dict:
    """Return a DB row dict for a position."""
    return {
        "market_id": market_id,
        "side": side.value,
        "size": size,
        "avg_price": avg_price,
        "current_price": avg_price,
        "category": "test",
    }


def _make_market(market_id: str, yes_price: float, end_date: datetime | None = None,
                 active: bool = True, no_price: float | None = None,
                 clob_yes: str = "", clob_no: str = ""):
    m = MagicMock()
    m.id = market_id
    m.outcome_yes_price = yes_price
    m.outcome_no_price = no_price if no_price is not None else round(1.0 - yes_price, 4)
    m.end_date = end_date
    m.active = active
    m.clob_token_yes = clob_yes
    m.clob_token_no = clob_no
    return m


@pytest.mark.asyncio
async def test_unmarkable_position_warns_and_throttles(mock_db, settings, monkeypatch):
    """A position whose market discovery no longer returns must be surfaced
    (it freezes marks and exit checks — the 2026-07-22 stale-paper-marks rot),
    but only once per throttle window, not every monitor tick."""
    from unittest.mock import patch

    mock_db.fetchall = AsyncMock(return_value=[
        _make_position("gone1", avg_price=0.50),
    ])
    gamma = AsyncMock()
    gamma.get_market = AsyncMock(return_value=None)  # dropped from discovery

    tracker = PortfolioTracker(db=mock_db)
    with patch("auramaur.risk.portfolio.log") as mock_log:
        exits = await tracker.check_exits(settings, gamma)
        assert exits == []
        warns = [c for c in mock_log.warning.call_args_list
                 if c.args and c.args[0] == "check_exits.unmarkable_positions"]
        assert len(warns) == 1
        assert warns[0].kwargs["count"] == 1
        assert warns[0].kwargs["sample"] == ["gone1"]

        # Second run inside the throttle window: no new warning.
        await tracker.check_exits(settings, gamma)
        warns = [c for c in mock_log.warning.call_args_list
                 if c.args and c.args[0] == "check_exits.unmarkable_positions"]
        assert len(warns) == 1


@pytest.mark.asyncio
async def test_stop_loss_triggered(mock_db, settings):
    """Position at -35% → STOP_LOSS."""
    mock_db.fetchall = AsyncMock(return_value=[
        _make_position("m1", avg_price=0.60),
    ])
    gamma = AsyncMock()
    # Current price dropped to 0.39 → loss = (0.39-0.60)*10 = -2.10, cost = 6.0, pct = -35%
    gamma.get_market = AsyncMock(return_value=_make_market("m1", 0.39))

    tracker = PortfolioTracker(db=mock_db)
    exits = await tracker.check_exits(settings, gamma)
    assert len(exits) == 1
    assert exits[0][1] == ExitReason.STOP_LOSS


@pytest.mark.asyncio
async def test_profit_target_triggered(mock_db, settings):
    """Position at +55% → PROFIT_TARGET."""
    mock_db.fetchall = AsyncMock(return_value=[
        _make_position("m1", avg_price=0.40),
    ])
    gamma = AsyncMock()
    # Current price rose to 0.62 → gain = (0.62-0.40)*10 = 2.20, cost = 4.0, pct = +55%
    gamma.get_market = AsyncMock(return_value=_make_market("m1", 0.62))

    tracker = PortfolioTracker(db=mock_db)
    exits = await tracker.check_exits(settings, gamma)
    assert len(exits) == 1
    assert exits[0][1] == ExitReason.PROFIT_TARGET


@pytest.mark.asyncio
async def test_edge_erosion_triggered(mock_db, settings):
    """Price converged close to 1.0 → EDGE_EROSION."""
    mock_db.fetchall = AsyncMock(return_value=[
        _make_position("m1", avg_price=0.90),
    ])
    gamma = AsyncMock()
    # Current price at 0.99 → remaining edge = (1.0-0.99)*100 = 1.0% < 2.0%
    # PnL = (0.99-0.90)*10 = 0.90, cost = 9.0, pct = +10% (not triggering stop/profit)
    gamma.get_market = AsyncMock(return_value=_make_market("m1", 0.99))

    tracker = PortfolioTracker(db=mock_db)
    exits = await tracker.check_exits(settings, gamma)
    assert len(exits) == 1
    assert exits[0][1] == ExitReason.EDGE_EROSION


@pytest.mark.asyncio
async def test_time_decay_triggered(mock_db, settings):
    """6h to resolution + small edge → TIME_DECAY."""
    mock_db.fetchall = AsyncMock(return_value=[
        _make_position("m1", avg_price=0.90),
    ])
    gamma = AsyncMock()
    # Current price at 0.96 → remaining edge = 4.0% (< 5.0%)
    # End date 6h from now (< 12h)
    # PnL pct = +6.7% (not triggering stop/profit)
    end_date = datetime.now(timezone.utc) + timedelta(hours=6)
    gamma.get_market = AsyncMock(return_value=_make_market("m1", 0.96, end_date=end_date))

    tracker = PortfolioTracker(db=mock_db)
    exits = await tracker.check_exits(settings, gamma)
    assert len(exits) == 1
    assert exits[0][1] == ExitReason.TIME_DECAY


@pytest.mark.asyncio
async def test_free_winners_far_dated(mock_db, settings):
    """Near-certain winner (4% upside) far from resolution → CAPITAL_EFFICIENCY."""
    settings.execution.free_winners_enabled = True
    mock_db.fetchall = AsyncMock(return_value=[
        _make_position("m1", avg_price=0.90),
    ])
    gamma = AsyncMock()
    # 0.96 → remaining upside 4% (>2% edge-erosion floor, <5% free-winners cap);
    # PnL +6.7% (below profit target). Resolves in ~8 days (>48h).
    end_date = datetime.now(timezone.utc) + timedelta(hours=200)
    gamma.get_market = AsyncMock(return_value=_make_market("m1", 0.96, end_date=end_date))

    tracker = PortfolioTracker(db=mock_db)
    exits = await tracker.check_exits(settings, gamma)
    assert len(exits) == 1
    assert exits[0][1] == ExitReason.CAPITAL_EFFICIENCY


@pytest.mark.asyncio
async def test_free_winners_holds_when_resolution_near(mock_db, settings):
    """Same near-winner but resolving soon (<48h) is held, not freed early."""
    settings.execution.free_winners_enabled = True
    mock_db.fetchall = AsyncMock(return_value=[
        _make_position("m1", avg_price=0.90),
    ])
    gamma = AsyncMock()
    # 0.96, resolves in 24h: free-winners needs >48h, and time-decay needs <=12h,
    # so neither fires -> held.
    end_date = datetime.now(timezone.utc) + timedelta(hours=24)
    gamma.get_market = AsyncMock(return_value=_make_market("m1", 0.96, end_date=end_date))

    tracker = PortfolioTracker(db=mock_db)
    exits = await tracker.check_exits(settings, gamma)
    assert exits == []


@pytest.mark.asyncio
async def test_dust_sweep_old_small_position(mock_db, settings):
    """A small (<$1), sellable, old (>24h) position → DUST_CLEANUP."""
    settings.execution.dust_sweep_enabled = True
    mock_db.fetchall = AsyncMock(return_value=[
        _make_position("m1", avg_price=0.10, size=8.0),  # value $0.80, 8 tokens
    ])
    # First fill 100h ago → older than the 24h guard.
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat()
    mock_db.fetchone = AsyncMock(return_value={"first_entry": old_ts})
    gamma = AsyncMock()
    gamma.get_market = AsyncMock(return_value=_make_market("m1", 0.10))  # flat → no PnL exit

    tracker = PortfolioTracker(db=mock_db)
    exits = await tracker.check_exits(settings, gamma)
    assert len(exits) == 1
    assert exits[0][1] == ExitReason.DUST_CLEANUP


@pytest.mark.asyncio
async def test_dust_sweep_skips_fresh_position(mock_db, settings):
    """A small position that was just opened (<24h) is NOT swept (age guard)."""
    settings.execution.dust_sweep_enabled = True
    mock_db.fetchall = AsyncMock(return_value=[
        _make_position("m1", avg_price=0.10, size=8.0),
    ])
    fresh_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    mock_db.fetchone = AsyncMock(return_value={"first_entry": fresh_ts})
    gamma = AsyncMock()
    gamma.get_market = AsyncMock(return_value=_make_market("m1", 0.10))

    tracker = PortfolioTracker(db=mock_db)
    exits = await tracker.check_exits(settings, gamma)
    assert exits == []


@pytest.mark.asyncio
async def test_dust_sweep_skips_sub_minimum_size(mock_db, settings):
    """A <$1 position too small to sell (poly needs >=5 tokens) is left alone."""
    settings.execution.dust_sweep_enabled = True
    mock_db.fetchall = AsyncMock(return_value=[
        _make_position("m1", avg_price=0.20, size=2.0),  # value $0.40, only 2 tokens
    ])
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat()
    mock_db.fetchone = AsyncMock(return_value={"first_entry": old_ts})
    gamma = AsyncMock()
    gamma.get_market = AsyncMock(return_value=_make_market("m1", 0.20))

    tracker = PortfolioTracker(db=mock_db)
    exits = await tracker.check_exits(settings, gamma)
    assert exits == []


@pytest.mark.asyncio
async def test_no_exit_healthy_position(mock_db, settings):
    """All thresholds safe → empty list."""
    mock_db.fetchall = AsyncMock(return_value=[
        _make_position("m1", avg_price=0.50),
    ])
    gamma = AsyncMock()
    # Current price at 0.55 → PnL = +10%, remaining edge = 45%, no end date
    gamma.get_market = AsyncMock(return_value=_make_market("m1", 0.55))

    tracker = PortfolioTracker(db=mock_db)
    exits = await tracker.check_exits(settings, gamma)
    assert len(exits) == 0


@pytest.mark.asyncio
async def test_binary_exits_skipped_for_ibkr_options(settings):
    """IBKR options are priced in premium, not 0-1 probability. A price that
    would trip edge-erosion for a binary venue must NOT exit an IBKR option;
    only the P&L-ratio exits apply (here PnL is +4%, so it holds)."""
    db = Database(":memory:")
    await db.connect()
    try:
        settings.is_live = True
        # current 0.99 vs avg 0.95 -> +4.2% PnL (no stop/profit); for a binary
        # venue remaining_upside = 1% would fire EDGE_EROSION.
        await db.execute(
            """INSERT INTO portfolio
               (market_id, exchange, side, size, avg_price, current_price,
                category, token, token_id, is_paper)
               VALUES ('IB:AAPL:200:20260418:C', 'ibkr', 'BUY', 10, 0.95, 0.99,
                       'tech', 'YES', '123:buy_call:C:200:20260418', 0)"""
        )
        await db.commit()

        gamma = AsyncMock()
        # Discovery would report a binary 0.99; for IBKR the refresh is skipped,
        # so this must not leak in.
        gamma.get_market = AsyncMock(return_value=_make_market("IB:AAPL:200:20260418:C", 0.99))

        tracker = PortfolioTracker(db=db, settings=settings)
        exits = await tracker.check_exits(settings, gamma, exchange="ibkr")
        assert exits == []  # no edge-erosion / time-decay for options
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_token_id_overrides_yes_label(settings):
    """A position whose outcome label defaulted to YES but whose held token id
    is the market's NO-slot token must be marked at the NO price. A low-priced
    held side marked at its complement's high price is a phantom gain that
    fires PROFIT_TARGET every cycle."""
    db = Database(":memory:")
    await db.connect()
    try:
        settings.is_live = True
        await db.execute(
            """INSERT INTO portfolio
               (market_id, exchange, side, size, avg_price, current_price,
                category, token, token_id, is_paper)
               VALUES ('obama', 'polymarket', 'BUY', 100, 0.126, 0.895,
                       'politics_us', 'YES', 'tok_something', 0)"""
        )
        await db.commit()

        gamma = AsyncMock()
        gamma.get_market = AsyncMock(return_value=_make_market(
            "obama", 0.895, no_price=0.105,
            clob_yes="tok_nothing", clob_no="tok_something",
        ))

        tracker = PortfolioTracker(db=db, settings=settings)
        exits = await tracker.check_exits(settings, gamma, exchange="polymarket")

        # Real P&L is (0.105-0.126)*100 = -$2.10 (-17%): no exit fires.
        assert exits == []
        row = await db.fetchone("SELECT current_price FROM portfolio WHERE market_id='obama'")
        assert row["current_price"] == pytest.approx(0.105)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_unresolved_token_keeps_stored_mark(settings):
    """When the market's CLOB tokens are known but the held token matches
    neither, the stored mark (set by the syncer off the token's own book)
    must not be clobbered with the label outcome's price."""
    db = Database(":memory:")
    await db.connect()
    try:
        settings.is_live = True
        await db.execute(
            """INSERT INTO portfolio
               (market_id, exchange, side, size, avg_price, current_price,
                category, token, token_id, is_paper)
               VALUES ('m1', 'polymarket', 'BUY', 100, 0.126, 0.10,
                       'test', 'YES', 'tok_unknown', 0)"""
        )
        await db.commit()

        gamma = AsyncMock()
        gamma.get_market = AsyncMock(return_value=_make_market(
            "m1", 0.895, no_price=0.105,
            clob_yes="tok_a", clob_no="tok_b",
        ))

        tracker = PortfolioTracker(db=db, settings=settings)
        exits = await tracker.check_exits(settings, gamma, exchange="polymarket")

        assert exits == []
        row = await db.fetchone("SELECT current_price FROM portfolio WHERE market_id='m1'")
        assert row["current_price"] == pytest.approx(0.10)  # untouched
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_remark_does_not_clobber_sibling_token(settings):
    """A market held on BOTH sides has a NO row and a YES row. Re-marking one
    leg must update ONLY that leg — the per-token write must not stamp the NO
    row with the YES price (and vice versa), which would invert a winner into a
    phantom loser in the persisted mark."""
    db = Database(":memory:")
    await db.connect()
    try:
        settings.is_live = True
        # Hold NO (cost 0.66, the real winner) and a small YES leg, same market.
        await db.execute(
            """INSERT INTO portfolio
               (market_id, exchange, side, size, avg_price, current_price,
                category, token, token_id, is_paper)
               VALUES ('mkt', 'polymarket', 'BUY', 60, 0.66, 0.66,
                       'tech', 'NO', 'tok_no', 0)"""
        )
        await db.execute(
            """INSERT INTO portfolio
               (market_id, exchange, side, size, avg_price, current_price,
                category, token, token_id, is_paper)
               VALUES ('mkt', 'polymarket', 'BUY', 2, 0.19, 0.19,
                       'tech', 'YES', 'tok_yes', 0)"""
        )
        await db.commit()

        # Live: NO=0.805 (the held winner), YES=0.195.
        gamma = AsyncMock()
        gamma.get_market = AsyncMock(return_value=_make_market(
            "mkt", 0.195, no_price=0.805,
            clob_yes="tok_yes", clob_no="tok_no",
        ))

        tracker = PortfolioTracker(db=db, settings=settings)
        await tracker.check_exits(settings, gamma, exchange="polymarket")

        no_row = await db.fetchone(
            "SELECT current_price FROM portfolio WHERE market_id='mkt' AND token='NO'")
        yes_row = await db.fetchone(
            "SELECT current_price FROM portfolio WHERE market_id='mkt' AND token='YES'")
        # Each leg marked at its OWN side's live price — no cross-clobber.
        assert no_row["current_price"] == pytest.approx(0.805)
        assert yes_row["current_price"] == pytest.approx(0.195)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_label_used_when_market_tokens_unknown(settings):
    """No CLOB token ids on the market (e.g. Kalshi) — the YES/NO label keeps
    driving the mark, as before."""
    db = Database(":memory:")
    await db.connect()
    try:
        settings.is_live = True
        await db.execute(
            """INSERT INTO portfolio
               (market_id, exchange, side, size, avg_price, current_price,
                category, token, token_id, is_paper)
               VALUES ('m1', 'polymarket', 'BUY', 10, 0.50, 0.50,
                       'test', 'NO', 'tok_x', 0)"""
        )
        await db.commit()

        gamma = AsyncMock()
        gamma.get_market = AsyncMock(return_value=_make_market("m1", 0.45, no_price=0.55))

        tracker = PortfolioTracker(db=db, settings=settings)
        exits = await tracker.check_exits(settings, gamma, exchange="polymarket")

        assert exits == []
        row = await db.fetchone("SELECT current_price FROM portfolio WHERE market_id='m1'")
        assert row["current_price"] == pytest.approx(0.55)  # NO-side price
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_live_exits_ignore_paper_positions(settings):
    """Live exit checks must not act on stale paper rows."""
    db = Database(":memory:")
    await db.connect()
    try:
        settings.is_live = True
        await db.execute(
            """INSERT INTO portfolio
               (market_id, exchange, side, size, avg_price, current_price,
                category, token, token_id, is_paper)
               VALUES
               ('live_ok', 'polymarket', 'BUY', 10, 0.50, 0.50, 'test', 'YES', 'yes1', 0),
               ('paper_exit', 'polymarket', 'BUY', 10, 0.50, 0.50, 'test', 'YES', 'yes2', 1)"""
        )
        await db.commit()

        gamma = AsyncMock()
        gamma.get_market = AsyncMock(return_value=_make_market("live_ok", 0.55))

        tracker = PortfolioTracker(db=db, settings=settings)
        positions = await tracker.get_positions(exchange="polymarket")
        exits = await tracker.check_exits(settings, gamma, exchange="polymarket")

        assert [p.market_id for p in positions] == ["live_ok"]
        assert exits == []
        gamma.get_market.assert_awaited_once_with("live_ok")
    finally:
        await db.close()
