"""Tests for bot-level pending order monitoring."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from auramaur.components import Components
import pytest

from auramaur.bot import AuramaurBot
from auramaur.exchange.models import Order, OrderResult, OrderSide, TokenType


@pytest.mark.asyncio
async def test_order_monitor_records_live_fill_once():
    settings = MagicMock()
    settings.execution.limit_order_ttl_seconds = 60

    bot = AuramaurBot(settings=settings)
    bot._running = True

    order = Order(
        market_id="m1",
        exchange="polymarket",
        token_id="tok_yes",
        token=TokenType.YES,
        side=OrderSide.BUY,
        size=10,
        price=0.50,
        dry_run=False,
    )
    exchange = SimpleNamespace(
        _live_pending={"live-1": order},
        get_order_status=AsyncMock(
            return_value=OrderResult(
                order_id="live-1",
                market_id="m1",
                status="filled",
                filled_size=10,
                filled_price=0.52,
                is_paper=False,
            )
        ),
    )
    paper = SimpleNamespace(
        pending_orders=[],
        check_fills=AsyncMock(return_value=[]),
        cancel_expired=AsyncMock(return_value=0),
    )
    pnl_tracker = AsyncMock()
    db_cursor = MagicMock()
    db_cursor.rowcount = 1
    db = AsyncMock()
    db.execute = AsyncMock(return_value=db_cursor)
    db.commit = AsyncMock()

    bot._components = Components({
        "paper": paper,
        "exchange": exchange,
        "discovery": AsyncMock(),
        "pnl_tracker": pnl_tracker,
        "db": db,
    })

    async def stop_after_loop(_seconds):
        bot._running = False

    with patch("asyncio.sleep", new=AsyncMock(side_effect=stop_after_loop)):
        await bot._task_order_monitor()

    pnl_tracker.record_fill.assert_awaited_once()
    fill = pnl_tracker.record_fill.await_args.args[0]
    assert fill.order_id == "live-1"
    assert fill.size == 10
    assert fill.price == 0.52
    assert fill.is_paper is False
    assert "live-1" not in exchange._live_pending
    db.execute.assert_awaited()
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_order_monitor_cancelled_order_writes_no_phantom_trade():
    """A cancelled live order with no pre-written trades row (rowcount==0, i.e.
    placed outside the gateway) must NOT insert a fallback trades row — it never
    executed. Regression: the market maker cancels/expires the bulk of its
    post-only quotes, and the unconditional fallback INSERT was fabricating a
    phantom 'BUY' at the full quoted size for each one (~1.6k/week), corrupting
    strategy attribution. Only an actual fill may insert a fallback row."""
    settings = MagicMock()
    settings.execution.limit_order_ttl_seconds = 60

    bot = AuramaurBot(settings=settings)
    bot._running = True

    order = Order(
        market_id="m1",
        exchange="polymarket",
        token_id="tok_yes",
        token=TokenType.YES,
        side=OrderSide.BUY,
        size=10,
        price=0.50,
        dry_run=False,
    )
    exchange = SimpleNamespace(
        _live_pending={"live-1": order},
        get_order_status=AsyncMock(
            return_value=OrderResult(
                order_id="live-1",
                market_id="m1",
                status="cancelled",   # terminal, but NOT a fill
                filled_size=0,
                filled_price=0.0,
                is_paper=False,
            )
        ),
    )
    paper = SimpleNamespace(
        pending_orders=[],
        check_fills=AsyncMock(return_value=[]),
        cancel_expired=AsyncMock(return_value=0),
    )
    pnl_tracker = AsyncMock()
    db_cursor = MagicMock()
    db_cursor.rowcount = 0   # no pre-existing gateway row to UPDATE
    db = AsyncMock()
    db.execute = AsyncMock(return_value=db_cursor)
    db.commit = AsyncMock()

    bot._components = Components({
        "paper": paper,
        "exchange": exchange,
        "discovery": AsyncMock(),
        "pnl_tracker": pnl_tracker,
        "db": db,
    })

    async def stop_after_loop(_seconds):
        bot._running = False

    with patch("asyncio.sleep", new=AsyncMock(side_effect=stop_after_loop)):
        await bot._task_order_monitor()

    # No fill recorded (it didn't execute) ...
    pnl_tracker.record_fill.assert_not_awaited()
    # ... and crucially, NO INSERT INTO trades for the cancelled order.
    inserts = [
        c for c in db.execute.await_args_list
        if "INSERT INTO trades" in str(c.args[0])
    ]
    assert inserts == []
    assert "live-1" not in exchange._live_pending  # still cleaned up


@pytest.mark.asyncio
async def test_order_monitor_ttl_cancels_stale_live_order():
    """A live limit order still resting past the TTL is cancelled to free balance.

    Live GTC orders never auto-expire on-chain, so without this they'd lock
    collateral forever. Mirrors the paper cancel_expired behaviour.
    """
    settings = MagicMock()
    settings.execution.limit_order_ttl_seconds = 60

    bot = AuramaurBot(settings=settings)
    bot._running = True

    stale_order = Order(
        market_id="m1",
        exchange="polymarket",
        token_id="tok_yes",
        token=TokenType.YES,
        side=OrderSide.BUY,
        size=10,
        price=0.50,
        dry_run=False,
    )
    # Placed 10 minutes ago -> older than the 60s TTL.
    stale_order.created_at = datetime.now(timezone.utc) - timedelta(minutes=10)

    exchange = SimpleNamespace(
        _live_pending={"live-1": stale_order},
        # Still open/unfilled -> non-terminal status, so the TTL branch runs.
        get_order_status=AsyncMock(return_value=OrderResult(
            order_id="live-1", market_id="m1", status="pending",
            filled_size=0, filled_price=0.50, is_paper=False,
        )),
        cancel_order=AsyncMock(return_value=True),
    )
    paper = SimpleNamespace(
        pending_orders=[],
        check_fills=AsyncMock(return_value=[]),
        cancel_expired=AsyncMock(return_value=0),
    )
    bot._components = Components({
        "paper": paper,
        "exchange": exchange,
        "discovery": AsyncMock(),
        "pnl_tracker": AsyncMock(),
        "db": AsyncMock(),
    })

    async def stop_after_loop(_seconds):
        bot._running = False

    with patch("asyncio.sleep", new=AsyncMock(side_effect=stop_after_loop)):
        await bot._task_order_monitor()

    exchange.cancel_order.assert_awaited_once_with("live-1")
    assert "live-1" not in exchange._live_pending


@pytest.mark.asyncio
async def test_order_monitor_keeps_fresh_live_order():
    """A live order younger than the TTL is left resting (not cancelled)."""
    settings = MagicMock()
    settings.execution.limit_order_ttl_seconds = 60

    bot = AuramaurBot(settings=settings)
    bot._running = True

    fresh_order = Order(
        market_id="m1", exchange="polymarket", token_id="tok_yes",
        token=TokenType.YES, side=OrderSide.BUY, size=10, price=0.50, dry_run=False,
    )
    fresh_order.created_at = datetime.now(timezone.utc)  # just placed

    exchange = SimpleNamespace(
        _live_pending={"live-1": fresh_order},
        get_order_status=AsyncMock(return_value=OrderResult(
            order_id="live-1", market_id="m1", status="pending",
            filled_size=0, filled_price=0.50, is_paper=False,
        )),
        cancel_order=AsyncMock(return_value=True),
    )
    paper = SimpleNamespace(
        pending_orders=[],
        check_fills=AsyncMock(return_value=[]),
        cancel_expired=AsyncMock(return_value=0),
    )
    bot._components = Components({
        "paper": paper, "exchange": exchange, "discovery": AsyncMock(),
        "pnl_tracker": AsyncMock(), "db": AsyncMock(),
    })

    async def stop_after_loop(_seconds):
        bot._running = False

    with patch("asyncio.sleep", new=AsyncMock(side_effect=stop_after_loop)):
        await bot._task_order_monitor()

    exchange.cancel_order.assert_not_called()
    assert "live-1" in exchange._live_pending


@pytest.mark.asyncio
async def test_order_monitor_polls_all_live_exchanges():
    settings = MagicMock()
    settings.execution.limit_order_ttl_seconds = 60

    bot = AuramaurBot(settings=settings)
    bot._running = True

    poly_order = Order(
        market_id="poly-m1",
        exchange="polymarket",
        token_id="poly_yes",
        token=TokenType.YES,
        side=OrderSide.BUY,
        size=5,
        price=0.40,
        dry_run=False,
    )
    kalshi_order = Order(
        market_id="KXTEST",
        exchange="kalshi",
        token_id="KXTEST",
        token=TokenType.YES,
        side=OrderSide.BUY,
        size=3,
        price=0.55,
        dry_run=False,
    )

    poly = SimpleNamespace(
        _live_pending={"poly-1": poly_order},
        get_order_status=AsyncMock(return_value=OrderResult(
            order_id="poly-1",
            market_id="poly-m1",
            status="filled",
            filled_size=5,
            filled_price=0.41,
            is_paper=False,
        )),
    )
    kalshi = SimpleNamespace(
        _live_pending={"kalshi-1": kalshi_order},
        get_order_status=AsyncMock(return_value=OrderResult(
            order_id="kalshi-1",
            market_id="KXTEST",
            status="filled",
            filled_size=3,
            filled_price=0.56,
            is_paper=False,
        )),
    )
    paper = SimpleNamespace(
        pending_orders=[],
        check_fills=AsyncMock(return_value=[]),
        cancel_expired=AsyncMock(return_value=0),
    )
    pnl_tracker = AsyncMock()
    db_cursor = MagicMock()
    db_cursor.rowcount = 1
    db = AsyncMock()
    db.execute = AsyncMock(return_value=db_cursor)
    db.commit = AsyncMock()

    bot._components = Components({
        "paper": paper,
        "exchange": poly,
        "exchanges": {"polymarket": poly, "kalshi": kalshi},
        "discovery": AsyncMock(),
        "pnl_tracker": pnl_tracker,
        "db": db,
    })

    async def stop_after_loop(_seconds):
        bot._running = False

    with patch("asyncio.sleep", new=AsyncMock(side_effect=stop_after_loop)):
        await bot._task_order_monitor()

    assert pnl_tracker.record_fill.await_count == 2
    assert poly._live_pending == {}
    assert kalshi._live_pending == {}
    poly.get_order_status.assert_awaited_once_with("poly-1")
    kalshi.get_order_status.assert_awaited_once_with("kalshi-1")


@pytest.mark.asyncio
async def test_ttl_cancel_writes_cancelled_status_to_db():
    """A TTL-cancelled live order must not leave its trades row 'pending'.

    The on-chain cancel releases the collateral, so without the status write
    the DB claims an open order that no longer exists (#94: 26 orphaned rows,
    $194 of phantom pending buys).
    """
    settings = MagicMock()
    settings.execution.limit_order_ttl_seconds = 60

    bot = AuramaurBot(settings=settings)
    bot._running = True

    stale_order = Order(
        market_id="m1",
        exchange="polymarket",
        token_id="tok_yes",
        token=TokenType.YES,
        side=OrderSide.BUY,
        size=10,
        price=0.50,
        dry_run=False,
    )
    stale_order.created_at = datetime.now(timezone.utc) - timedelta(minutes=10)

    exchange = SimpleNamespace(
        _live_pending={"live-1": stale_order},
        get_order_status=AsyncMock(return_value=OrderResult(
            order_id="live-1", market_id="m1", status="pending",
            filled_size=0, filled_price=0.50, is_paper=False,
        )),
        cancel_order=AsyncMock(return_value=True),
    )
    paper = SimpleNamespace(
        pending_orders=[],
        check_fills=AsyncMock(return_value=[]),
        cancel_expired=AsyncMock(return_value=0),
    )
    db = AsyncMock()
    bot._components = Components({
        "paper": paper,
        "exchange": exchange,
        "discovery": AsyncMock(),
        "pnl_tracker": AsyncMock(),
        "db": db,
    })

    async def stop_after_loop(_seconds):
        bot._running = False

    with patch("asyncio.sleep", new=AsyncMock(side_effect=stop_after_loop)):
        await bot._task_order_monitor()

    assert "live-1" not in exchange._live_pending
    update_calls = [
        c for c in db.execute.await_args_list
        if "UPDATE trades SET status = 'cancelled'" in c.args[0]
    ]
    assert len(update_calls) == 1
    assert update_calls[0].args[1] == ("live-1",)
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_ttl_cancel_failure_keeps_order_tracked():
    """A failed cancel (e.g. racing a fill) keeps the order in _live_pending
    so the next status poll resolves it, instead of dropping it untracked."""
    settings = MagicMock()
    settings.execution.limit_order_ttl_seconds = 60

    bot = AuramaurBot(settings=settings)
    bot._running = True

    stale_order = Order(
        market_id="m1",
        exchange="polymarket",
        token_id="tok_yes",
        token=TokenType.YES,
        side=OrderSide.BUY,
        size=10,
        price=0.50,
        dry_run=False,
    )
    stale_order.created_at = datetime.now(timezone.utc) - timedelta(minutes=10)

    exchange = SimpleNamespace(
        _live_pending={"live-1": stale_order},
        get_order_status=AsyncMock(return_value=OrderResult(
            order_id="live-1", market_id="m1", status="pending",
            filled_size=0, filled_price=0.50, is_paper=False,
        )),
        cancel_order=AsyncMock(return_value=False),
    )
    paper = SimpleNamespace(
        pending_orders=[],
        check_fills=AsyncMock(return_value=[]),
        cancel_expired=AsyncMock(return_value=0),
    )
    db = AsyncMock()
    bot._components = Components({
        "paper": paper,
        "exchange": exchange,
        "discovery": AsyncMock(),
        "pnl_tracker": AsyncMock(),
        "db": db,
    })

    async def stop_after_loop(_seconds):
        bot._running = False

    with patch("asyncio.sleep", new=AsyncMock(side_effect=stop_after_loop)):
        await bot._task_order_monitor()

    assert "live-1" in exchange._live_pending
    cancelled_writes = [
        c for c in db.execute.await_args_list
        if "UPDATE trades SET status = 'cancelled'" in str(c.args[0])
    ]
    assert cancelled_writes == []


@pytest.mark.asyncio
async def test_reconcile_orphaned_pending_trades():
    """DB rows stuck 'pending' with no in-memory tracking get resolved via the
    exchange's authoritative order status."""
    settings = MagicMock()
    bot = AuramaurBot(settings=settings)

    exchange = SimpleNamespace(
        _live_pending={"still-tracked": object()},
        get_order_status=AsyncMock(return_value=OrderResult(
            order_id="orphan-1", market_id="m1", status="cancelled",
            filled_size=0, filled_price=0, is_paper=False,
        )),
    )
    db = AsyncMock()
    db.fetchall = AsyncMock(return_value=[
        {"order_id": "still-tracked", "exchange": "polymarket"},
        {"order_id": "PAPER-abc123", "exchange": "polymarket"},
        {"order_id": "orphan-1", "exchange": "polymarket"},
    ])
    bot._components = Components({"db": db})

    await bot._reconcile_orphaned_pending_trades([("polymarket", exchange)])

    # Only the genuine orphan is queried and rewritten
    exchange.get_order_status.assert_awaited_once_with("orphan-1")
    update_calls = [
        c for c in db.execute.await_args_list
        if c.args[0].startswith("UPDATE trades SET status = ?")
    ]
    assert len(update_calls) == 1
    assert update_calls[0].args[1] == ("cancelled", "orphan-1")
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_order_monitor_reconciles_orphans_periodically():
    """Orphaned live orders created mid-session (e.g. a lost cancel during a
    network blip leaving a resting BUY untracked) must be re-pulled into
    _live_pending periodically, not only at startup — otherwise their locked
    collateral is stuck until the next restart (the $27 cash-lock incident)."""
    settings = MagicMock()
    settings.execution.limit_order_ttl_seconds = 60

    bot = AuramaurBot(settings=settings)
    bot._running = True

    exchange = SimpleNamespace(
        _live_pending={},
        reconcile_open_orders=AsyncMock(return_value=0),
        get_order_status=AsyncMock(),
    )
    paper = SimpleNamespace(
        pending_orders=[],
        check_fills=AsyncMock(return_value=[]),
        cancel_expired=AsyncMock(return_value=0),
    )
    bot._components = Components({
        "paper": paper,
        "exchange": exchange,
        "discovery": AsyncMock(),
        "db": AsyncMock(),
    })

    calls = {"n": 0}

    async def stop_after_n(_seconds):
        calls["n"] += 1
        if calls["n"] >= 12:  # let ~12 loop cycles run (reconcile_every=10)
            bot._running = False

    with patch("asyncio.sleep", new=AsyncMock(side_effect=stop_after_n)):
        await bot._task_order_monitor()

    # Startup pass + at least one periodic re-pull (at cycle 10) => >= 2.
    assert exchange.reconcile_open_orders.await_count >= 2
