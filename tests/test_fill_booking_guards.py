"""A resting or refused order is not a position — and a real fill IS one.

Three shapes of one defect, all of them feeding the ledger the graduation
ladder reads to authorize live capital.

1. ``PaperTrader.check_fills`` stamped ``status="filled"`` over an
   ``execute()`` that REFUSED for insufficient paper cash — booking a
   size-0/price-0 fill and marking the decision executable with
   "trade_through" evidence, which ``graduation.credible_fill_evidence``
   accepts.

2. Nine pillar call sites wrote a full-size ``portfolio``/leg record on
   ``status="pending"`` — which is what every live Polymarket ``place_order``
   returns and what maker-priced paper orders always defer to. All of them now
   route through the ONE predicate,
   ``broker.execution_gateway.booked_as_position``, which is the same
   expression the gateway applies before it writes the fill (and which the
   gateway itself now calls), so a portfolio row cannot claim a position the
   gateway's own ``fills``/``cost_basis`` writes never recorded.

3. The mirror hole the guard opened: a deferred paper fill wrote
   ``fills``/``cost_basis`` and no ``portfolio`` row, leaving a holding that
   settles correctly but is invisible to every ``max_open`` count and to the
   risk manager, both of which read FROM ``portfolio``.

EVERY test here is BEHAVIOURAL — it drives the real code path and asserts on
rows in a real (in-memory) database. None of them inspects source text. The
predecessors of these tests used ``inspect.getsource`` and asserted a string
literal, which was proven vacuous: replacing a pillar's guard with a
computed-but-unused local followed by an unconditional record restored the
regression with all of them still green. They also could not survive the
refactor that gave the codebase a single shared predicate.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from auramaur.broker.pnl import PnLTracker
from auramaur.db.database import Database
from auramaur.exchange.models import (
    Market,
    Order,
    OrderBook,
    OrderBookLevel,
    OrderResult,
    OrderSide,
    OrderType,
    TokenType,
)
from auramaur.exchange.paper import PaperTrader
from config.settings import Settings


# ----------------------------------------------------------------------
# Shared stubs. `place_order` is the ONE knob: "pending"/filled_size=0 is
# every live Polymarket placement and every deferred maker-priced paper
# order; "paper"/filled_size=size is a real execution.
# ----------------------------------------------------------------------

RESTING = "resting"
EXECUTED = "executed"


def _place_order_result(order: Order, outcome: str) -> OrderResult:
    if outcome == RESTING:
        return OrderResult(
            order_id=f"PAPER-LMT-{order.market_id}", market_id=order.market_id,
            status="pending", filled_size=0.0, filled_price=0.0,
            is_paper=order.dry_run)
    return OrderResult(
        order_id=f"ord-{order.market_id}", market_id=order.market_id,
        status="paper" if order.dry_run else "filled",
        filled_size=order.size, filled_price=order.price,
        is_paper=order.dry_run)


def _exchange(outcome: str):
    ex = MagicMock()

    def prepare_order(signal, market, size, is_live):
        token = (TokenType.NO if signal.recommended_side == OrderSide.SELL
                 else TokenType.YES)
        price = (market.outcome_yes_price if token == TokenType.YES
                 else 1 - market.outcome_yes_price)
        return Order(
            market_id=market.id, exchange=market.exchange, token_id="tok",
            side=OrderSide.BUY, token=token,
            size=round(size / max(price, 0.01), 2), price=round(price, 2),
            order_type=OrderType.LIMIT, dry_run=not is_live)

    ex.prepare_order = MagicMock(side_effect=prepare_order)
    ex.place_order = AsyncMock(
        side_effect=lambda o: _place_order_result(o, outcome))
    ex.cancel_order = AsyncMock(return_value=True)
    # A book with a capturable maker spread (bias_harvest posts at the bid).
    ex.get_order_book = AsyncMock(return_value=OrderBook(
        bids=[OrderBookLevel(price=0.91, size=200.0)],
        asks=[OrderBookLevel(price=0.93, size=200.0)]))
    return ex


def _risk(size=10.0):
    rm = MagicMock()
    d = MagicMock()
    d.approved = True
    d.position_size = size
    d.reason = ""
    d.force_paper = False
    rm.evaluate = AsyncMock(return_value=d)
    return rm


async def _portfolio_rows(db, market_id: str | None = None) -> list:
    if market_id is None:
        return await db.fetchall("SELECT * FROM portfolio") or []
    return await db.fetchall(
        "SELECT * FROM portfolio WHERE market_id = ?", (market_id,)) or []


# ======================================================================
# 1. PaperTrader — a refusal must never leave as a fill
# ======================================================================

@pytest.mark.asyncio
async def test_refused_trade_through_is_not_reported_as_a_fill():
    """A rested BUY trades through on a book that cannot fund it.
    execute(force=True) still hits the balance gate — `force` bypasses only
    the marketability check."""
    db = Database(":memory:")
    await db.connect()
    trader = PaperTrader(db, initial_balance=5.0)   # far below 100 * 0.82
    order = Order(market_id="m1", exchange="polymarket", token_id="t1",
                  side=OrderSide.BUY, token=TokenType.YES, size=100.0,
                  price=0.82, dry_run=True)
    trader.submit_limit_order(order)
    assert trader.pending_orders, "order should be resting"

    filled = await trader.check_fills({"m1": 0.80})   # trades through

    assert filled == [], "a refusal must not be handed over as a fill"
    # And it must not vanish mid-pass: before the fix it neither filled nor
    # rested. (cancel_expired still clears the queue on the monitor's next
    # line — see the comment in check_fills.)
    assert trader.pending_orders, "the unfilled order must stay resting"
    await db.close()


@pytest.mark.asyncio
async def test_funded_trade_through_still_fills():
    db = Database(":memory:")
    await db.connect()
    trader = PaperTrader(db, initial_balance=1000.0)
    order = Order(market_id="m1", exchange="polymarket", token_id="t1",
                  side=OrderSide.BUY, token=TokenType.YES, size=100.0,
                  price=0.82, dry_run=True)
    trader.submit_limit_order(order)

    filled = await trader.check_fills({"m1": 0.80})

    assert len(filled) == 1
    result, _ = filled[0]
    assert result.status == "filled"
    assert result.filled_size > 0
    assert trader.pending_orders == []
    await db.close()


# ======================================================================
# 2a. bias_harvest — the pillar the PR was opened for
# ======================================================================

def _bias_market(mid="m1", yes=0.92) -> Market:
    return Market(
        id=mid, exchange="polymarket", question=f"q-{mid}", category="tech",
        active=True, outcome_yes_price=yes, outcome_no_price=round(1 - yes, 2),
        liquidity=5000.0, volume=10000.0,
        end_date=datetime.now(timezone.utc) + timedelta(days=10),
        clob_token_yes="tok-yes", clob_token_no="tok-no")


def _bias_pillar(db, outcome: str):
    from auramaur.strategy.bias_harvest import BiasHarvestPillar

    s = Settings()
    s.bias_harvest.enabled = True
    s.bias_harvest.paper = True
    s.bias_harvest.band_lo = 0.80
    s.bias_harvest.band_hi = 0.97
    s.bias_harvest.stake_usd = 10.0
    s.bias_harvest.paper_maker_fill_rate = 1.0
    disc = MagicMock()
    disc.get_markets = AsyncMock(return_value=[_bias_market()])
    cal = MagicMock()
    cal.record_prediction = AsyncMock()
    return BiasHarvestPillar(
        db=db, settings=s, discovery=disc, exchange=_exchange(outcome),
        risk_manager=_risk(), pnl_tracker=PnLTracker(db, s), calibration=cal)


@pytest.mark.asyncio
async def test_bias_harvest_resting_order_writes_no_portfolio_row():
    db = Database(":memory:")
    await db.connect()
    try:
        pillar = _bias_pillar(db, RESTING)
        await pillar.run_once()
        # The order WAS placed (the trades-mirror proves the path ran)…
        assert await db.fetchone(
            "SELECT 1 FROM trades WHERE strategy_source LIKE 'bias_harvest%'")
        # …and booked nothing as a holding.
        assert await _portfolio_rows(db) == []
        assert await db.fetchall("SELECT * FROM fills") == []
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_bias_harvest_executed_order_writes_the_portfolio_row():
    db = Database(":memory:")
    await db.connect()
    try:
        pillar = _bias_pillar(db, EXECUTED)
        assert await pillar.run_once() == 1
        rows = await _portfolio_rows(db, "m1")
        assert len(rows) == 1
        row = rows[0]
        assert row["is_paper"] == 1
        assert row["token"] == "YES"
        # Size is the FILLED size, not a fallback to the requested size.
        fill = await db.fetchone("SELECT size, price FROM fills WHERE market_id='m1'")
        assert abs(row["size"] - fill["size"]) < 1e-9
        assert abs(row["avg_price"] - fill["price"]) < 1e-9
    finally:
        await db.close()


# ======================================================================
# 2b. weather_temp — one of the five sites this PR newly swept
# ======================================================================

def _weather_pillar(db, outcome: str):
    from auramaur.strategy.weather_temp import WeatherTempPillar

    s = Settings()
    s.weather_temp.enabled = True
    s.weather_temp.paper = True
    s.weather_temp.min_edge = 0.10
    market = Market(
        id="w1", exchange="polymarket",
        question="Will the highest temperature in Tokyo be 23°C on June 20?",
        outcome_yes_price=0.27, outcome_no_price=0.73, liquidity=4000.0,
        volume=4000.0, category="weather",
        end_date=datetime.now(timezone.utc) + timedelta(days=1),
        clob_token_yes="ty", clob_token_no="tn")
    disc = MagicMock()
    disc.get_markets = AsyncMock(return_value=[market])
    weather = MagicMock()
    # ~0 ensemble members in [22.5, 23.5) vs a market asking 0.27.
    weather.daily_ensemble = AsyncMock(
        return_value=[26, 27, 28, 25, 29, 24, 30, 26, 27, 28])
    cal = MagicMock()
    cal.record_prediction = AsyncMock()
    return WeatherTempPillar(
        db=db, settings=s, discovery=disc, exchange=_exchange(outcome),
        risk_manager=_risk(), pnl_tracker=PnLTracker(db, s),
        calibration=cal, weather=weather)


@pytest.mark.asyncio
async def test_weather_temp_resting_order_writes_no_portfolio_row():
    db = Database(":memory:")
    await db.connect()
    try:
        pillar = _weather_pillar(db, RESTING)
        assert await pillar.run_once() == 1        # the ENTRY still happened
        assert await db.fetchone(
            "SELECT 1 FROM signals WHERE strategy_source='weather_temp'")
        assert await _portfolio_rows(db) == []
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_weather_temp_executed_order_writes_the_portfolio_row():
    db = Database(":memory:")
    await db.connect()
    try:
        pillar = _weather_pillar(db, EXECUTED)
        assert await pillar.run_once() == 1
        rows = await _portfolio_rows(db, "w1")
        assert len(rows) == 1
        fill = await db.fetchone("SELECT size, price FROM fills WHERE market_id='w1'")
        assert abs(rows[0]["size"] - fill["size"]) < 1e-9
    finally:
        await db.close()


# ======================================================================
# 2c. cross_venue_arb — the leg path (its _record_leg writes `signals`,
#     not `portfolio`, so the phantom it produced was a traded-leg record)
# ======================================================================

def _arb_pillar(db, outcome: str):
    from auramaur.strategy.cross_venue_arb import CrossVenueArbPillar

    s = Settings()
    s.cross_venue_arb.enabled = True
    s.cross_venue_arb.paper = True
    q = "Will the Fed cut rates at the July meeting?"

    def mk(mid, venue, yes):
        return Market(
            id=mid, exchange=venue, question=q, active=True,
            outcome_yes_price=yes, outcome_no_price=round(1 - yes, 2),
            liquidity=5000.0, volume=5000.0, spread=0.01, category="economics",
            end_date=datetime.now(timezone.utc) + timedelta(days=20),
            clob_token_yes="ty", clob_token_no="tn")

    poly, kalshi = mk("p1", "polymarket", 0.40), mk("k1", "kalshi", 0.55)
    disc = MagicMock()
    disc.get_markets = AsyncMock(return_value=[poly])
    kdisc = MagicMock()
    kdisc.get_markets = AsyncMock(return_value=[kalshi])
    analyzer = MagicMock()
    analyzer._call_llm = AsyncMock(return_value=(
        '{"orientation": "same", "confidence": 0.95, '
        '"counterexample": "none found"}'))
    exchanges = {"polymarket": _exchange(outcome), "kalshi": _exchange(outcome)}
    return CrossVenueArbPillar(
        db=db, settings=s, discovery=disc,
        exchange=exchanges["polymarket"], risk_manager=_risk(),
        pnl_tracker=PnLTracker(db, s), analyzer=analyzer,
        kalshi_discovery=kdisc, exchanges=exchanges)


@pytest.mark.asyncio
async def test_cross_venue_resting_legs_record_no_leg_rows():
    db = Database(":memory:")
    await db.connect()
    try:
        pillar = _arb_pillar(db, RESTING)
        assert await pillar.run_once() == 1        # both legs WERE placed
        assert await db.fetchall(
            "SELECT * FROM signals WHERE strategy_source='cross_venue_arb'") == []
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_cross_venue_executed_legs_record_both():
    db = Database(":memory:")
    await db.connect()
    try:
        pillar = _arb_pillar(db, EXECUTED)
        assert await pillar.run_once() == 1
        rows = await db.fetchall(
            "SELECT market_id FROM signals WHERE strategy_source='cross_venue_arb'")
        assert sorted(r["market_id"] for r in rows) == ["k1", "p1"]
    finally:
        await db.close()


# ======================================================================
# 2d. oddlot_tender — the raw-OrderResult call shape of the same predicate
# ======================================================================

def _oddlot_pillar(db, share_order_status: str):
    from auramaur.data_sources.edgar import TenderFiling
    from auramaur.strategy.oddlot_tender import OddLotTenderPillar

    s = Settings()
    s.oddlot_tender.enabled = True
    s.oddlot_tender.paper = True
    s.oddlot_tender.min_premium_pct = 2.0
    s.oddlot_tender.llm_min_confidence = 0.8
    s.ibkr.enabled = True
    filing = TenderFiling(
        accession="0001-26-000001", cik="1234567",
        company="Acme Corp (ACME) (CIK 1234567)", form="SC TO-I",
        filed_at="2026-06-09", primary_doc="scto.htm")
    edgar = MagicMock()
    edgar.recent_tender_filings = AsyncMock(return_value=[filing])
    edgar.fetch_document = AsyncMock(
        return_value="This tender offer includes odd lot priority ...")
    analyzer = MagicMock()
    analyzer._call_llm = AsyncMock(return_value=(
        '{"odd_lot_priority": true, "requires_record_date_holding": false, '
        '"tender_price": 20.0, "tender_price_high": 20.0, '
        '"expiration": "2026-07-15", "conditions": "none material", '
        '"confidence": 0.95}'))
    equity = MagicMock()
    equity.get_price = AsyncMock(return_value=19.0)   # 5.3% premium

    async def place(sym, side, qty, limit_price, dry_run):
        # The LIVE path: ibkr_equity returns pending/filled_size=0 for EVERY
        # share order. The PAPER path returns "paper" with the full quantity.
        if share_order_status == RESTING:
            return OrderResult(order_id="42", market_id=sym, status="pending",
                               filled_size=0.0, filled_price=limit_price,
                               is_paper=False)
        return OrderResult(order_id="PAPER", market_id=sym, status="paper",
                           filled_size=float(qty), filled_price=limit_price,
                           is_paper=True)

    equity.place_share_order = AsyncMock(side_effect=place)
    return OddLotTenderPillar(
        db=db, settings=s, edgar=edgar, analyzer=analyzer, alerts=None,
        equity_client=equity, pnl_tracker=PnLTracker(db, s))


@pytest.mark.asyncio
async def test_oddlot_resting_share_order_books_nothing():
    db = Database(":memory:")
    await db.connect()
    try:
        pillar = _oddlot_pillar(db, RESTING)
        await pillar.run_once()
        assert await _portfolio_rows(db) == []
        assert await db.fetchall("SELECT * FROM fills") == []
        status = await db.fetchone("SELECT status FROM oddlot_filings")
        assert status["status"] == "order_resting"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_oddlot_paper_share_order_books_the_position():
    db = Database(":memory:")
    await db.connect()
    try:
        pillar = _oddlot_pillar(db, EXECUTED)
        await pillar.run_once()
        rows = await _portfolio_rows(db)
        assert len(rows) == 1 and rows[0]["size"] == 99.0
        status = await db.fetchone("SELECT status FROM oddlot_filings")
        assert status["status"] == "entered"
    finally:
        await db.close()


# ======================================================================
# 3. The order monitor: the paper branch guards, AND a booked deferred
#    fill materializes the portfolio row it used to leave missing.
# ======================================================================

class _Components(dict):
    """Stands in for the bot's component registry (``.get`` + attribute)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:                       # pragma: no cover
            raise AttributeError(name) from exc


def _monitor(db, settings):
    from auramaur.bot_order_monitor import OrderMonitorMixin

    mon = OrderMonitorMixin()
    mon._components = _Components(db=db, pnl_tracker=PnLTracker(db, settings))
    return mon


def _deferred_order(market_id="dm1", size=10.75, price=0.93) -> Order:
    order = Order(market_id=market_id, exchange="polymarket", token_id="tok",
                  side=OrderSide.BUY, token=TokenType.NO, size=size,
                  price=price, order_type=OrderType.LIMIT, dry_run=True)
    order.decision_id = None
    return order


@pytest.mark.asyncio
async def test_order_monitor_paper_branch_drops_a_size_zero_fill():
    db = Database(":memory:")
    await db.connect()
    try:
        mon = _monitor(db, Settings())
        order = _deferred_order()
        refusal = OrderResult(order_id="PAPER-LMT-1", market_id="dm1",
                              status="filled", filled_size=0.0,
                              filled_price=0.0, is_paper=True)
        await mon._record_deferred_paper_fills([(refusal, order)])
        assert await db.fetchall("SELECT * FROM fills") == []
        assert await _portfolio_rows(db) == []
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_order_monitor_books_a_deferred_fill_and_its_portfolio_row():
    """The mirror-image hole the entry guard opened.

    record_fill writes fills + cost_basis and nothing else, and position sync
    is mode-scoped (it does not maintain paper rows in a live bot), so the
    holding used to exist in cost_basis with NO portfolio row: it settled
    correctly but was invisible to RiskManager and to every pillar's
    _open_position_count, which read FROM portfolio.
    """
    db = Database(":memory:")
    await db.connect()
    try:
        mon = _monitor(db, Settings())
        order = _deferred_order()
        result = OrderResult(order_id="PAPER-LMT-1", market_id="dm1",
                             status="filled", filled_size=10.75,
                             filled_price=0.93, is_paper=True)
        await mon._record_deferred_paper_fills([(result, order)])

        cb = await db.fetchone(
            "SELECT size, avg_cost FROM cost_basis WHERE market_id='dm1'")
        assert cb is not None and abs(cb["size"] - 10.75) < 1e-9

        rows = await _portfolio_rows(db, "dm1")
        assert len(rows) == 1, "a booked deferred fill must leave a portfolio row"
        row = rows[0]
        # Projected FROM cost_basis, so the two can never disagree about the
        # holding _settle_position will price.
        assert abs(row["size"] - cb["size"]) < 1e-9
        assert abs(row["avg_price"] - cb["avg_cost"]) < 1e-9
        assert row["is_paper"] == 1 and row["token"] == "NO"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_second_deferred_fill_accumulates_rather_than_replacing():
    """Why the row is projected from cost_basis and not from the one fill.

    The portfolio upsert REPLACES size. A fill-shaped write would leave the
    second fill's size on the row while cost_basis carried the sum, and
    _settle_position would then price a different holding depending on which
    branch it took.
    """
    db = Database(":memory:")
    await db.connect()
    try:
        mon = _monitor(db, Settings())
        first = OrderResult(order_id="PAPER-LMT-1", market_id="dm1",
                            status="filled", filled_size=10.0,
                            filled_price=0.90, is_paper=True)
        second = OrderResult(order_id="PAPER-LMT-2", market_id="dm1",
                             status="filled", filled_size=5.0,
                             filled_price=0.80, is_paper=True)
        await mon._record_deferred_paper_fills([(first, _deferred_order())])
        await mon._record_deferred_paper_fills([(second, _deferred_order())])

        cb = await db.fetchone(
            "SELECT size, avg_cost FROM cost_basis WHERE market_id='dm1'")
        rows = await _portfolio_rows(db, "dm1")
        assert len(rows) == 1
        assert abs(cb["size"] - 15.0) < 1e-9
        assert abs(rows[0]["size"] - cb["size"]) < 1e-9
        assert abs(rows[0]["avg_price"] - cb["avg_cost"]) < 1e-9
    finally:
        await db.close()


# ======================================================================
# The predicate itself
# ======================================================================

def test_booked_as_position_matches_the_gateways_own_fill_test():
    from auramaur.broker.execution_gateway import (
        _FILLED_STATUSES,
        ExecutionResult,
        booked_as_position,
    )

    statuses = ("filled", "partial", "pending", "rejected", "paper",
                "cancelled", "expired")
    for status in statuses:
        for size in (0.0, 5.0):
            result = OrderResult(order_id="x", market_id="m", status=status,
                                 filled_size=size)
            expected = status in _FILLED_STATUSES and size > 0
            assert booked_as_position(result) is expected, (status, size)
            # Same answer through the gateway's own wrapper…
            assert booked_as_position(
                ExecutionResult(status=status, result=result)) is expected
    # …and nothing that never reached placement is a position.
    assert booked_as_position(None) is False
    assert booked_as_position(ExecutionResult(status="skipped")) is False

    # A test double of either shape must read the same way the real object
    # does — the pillar suites stub the gateway with MagicMocks, and a
    # predicate that silently answered "not a position" for every one of them
    # would be a guard that only ever fires in tests.
    from unittest.mock import MagicMock as _MM
    double = _MM()
    double.status = "paper"
    double.result = _MM()
    double.result.filled_size = 33.3
    assert booked_as_position(double) is True
    double.result.filled_size = 0.0
    assert booked_as_position(double) is False
    double.status = "pending"
    double.result.filled_size = 33.3
    assert booked_as_position(double) is False


def test_every_position_booking_site_uses_the_shared_predicate():
    """No tenth copy of the predicate. This is a STRUCTURAL check and is not
    a substitute for the behavioural tests above — it exists so a future
    hand-rolled ``status != "pending" and ...`` re-opens the class loudly."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "auramaur"
    gateway = root / "broker" / "execution_gateway.py"
    offenders = []
    for path in root.rglob("*.py"):
        if path == gateway:
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            code = line.split("#", 1)[0]
            if "filled_size" in code and ("_FILLED_STATUSES" in code
                                          or 'status != "pending"' in code
                                          or 'status == "filled"' in code):
                offenders.append(f"{path.relative_to(root)}:{lineno}")
    assert offenders == [], (
        "hand-rolled fill predicate(s) — use "
        f"broker.execution_gateway.booked_as_position: {offenders}")


# ======================================================================
# 4. The gateway's instant-fill mirror: a paper fill that closes the
#    holding must take the portfolio row with it.
#
#    The deferred-fill path got this projection first; the INSTANT path
#    did not, so a paper exit that filled immediately through paper
#    interception zeroed cost_basis and left the portfolio row at full
#    size until resolution cleanup — inflating RiskManager exposure and
#    every max_open count the whole time (observed on the live book the
#    evening the deferred-fill fix deployed). These drive the real
#    consumer path: submit_exit → _place_and_record → _record_result.
# ======================================================================

def _instant_fill_exchange():
    """An exchange whose place_order executes immediately, paper-style."""
    ex = MagicMock()
    ex.place_order = AsyncMock(side_effect=lambda o: OrderResult(
        order_id=f"PAPER-X-{o.market_id}", market_id=o.market_id,
        status="paper" if o.dry_run else "filled",
        filled_size=o.size, filled_price=o.price, is_paper=o.dry_run))
    return ex


def _exit_gateway(db, settings, exchange):
    from auramaur.broker.execution_gateway import ExecutionGateway

    return ExecutionGateway(
        router=None, exchange=exchange, exchange_name="polymarket",
        settings=settings, db=db, pnl_tracker=PnLTracker(db, settings))


async def _seed_paper_holding(db, settings, market_id, token, size, price):
    """A pillar-shaped holding: cost_basis via record_fill + the pillar's own
    portfolio write (exactly the state an instant-fill exit finds)."""
    from auramaur.exchange.models import Fill

    await PnLTracker(db, settings).record_fill(Fill(
        order_id=f"seed-{market_id}", market_id=market_id, token_id="tok",
        side=OrderSide.BUY, token=token, size=size, price=price,
        is_paper=True))
    await db.execute(
        """INSERT INTO portfolio (market_id, exchange, side, size, avg_price,
           token, token_id, is_paper, updated_at)
           VALUES (?, 'polymarket', 'BUY', ?, ?, ?, 'tok', 1, datetime('now'))""",
        (market_id, size, price, token.value))


def _exit_order(market_id, token, size, price, dry_run=True) -> Order:
    return Order(market_id=market_id, exchange="polymarket", token_id="tok",
                 side=OrderSide.SELL, token=token, size=size, price=price,
                 order_type=OrderType.LIMIT, dry_run=dry_run, source="exit")


@pytest.mark.asyncio
async def test_instant_paper_exit_deletes_the_portfolio_row():
    """The full-close regression: cost_basis reaches zero, and the portfolio
    row — which used to stand at full size until the market resolved — goes
    with it."""
    db = Database(":memory:")
    await db.connect()
    try:
        s = Settings()
        gw = _exit_gateway(db, s, _instant_fill_exchange())
        await _seed_paper_holding(db, s, "xm1", TokenType.NO, 10.99, 0.91)

        res = await gw.submit_exit(
            _exit_order("xm1", TokenType.NO, 10.99, 0.96),
            exchange=gw.exchange, exchange_name="polymarket")

        assert res.status == "paper"
        cb = await db.fetchone(
            "SELECT size, realized_pnl FROM cost_basis WHERE market_id='xm1'")
        assert cb is not None and abs(cb["size"]) < 1e-9
        assert cb["realized_pnl"] > 0, "the sell itself must still book P&L"
        assert await _portfolio_rows(db, "xm1") == [], (
            "a closed paper holding must not keep a portfolio row")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_partial_paper_exit_projects_the_remaining_size():
    """A partial close must shrink the row to cost_basis's remainder — the
    same projected-from-cost_basis rule the deferred path follows, for the
    same reason: the upsert REPLACES size, so any fill-shaped write here
    would leave the FULL entry size standing."""
    db = Database(":memory:")
    await db.connect()
    try:
        s = Settings()
        gw = _exit_gateway(db, s, _instant_fill_exchange())
        await _seed_paper_holding(db, s, "xm2", TokenType.YES, 20.0, 0.50)

        await gw.submit_exit(
            _exit_order("xm2", TokenType.YES, 8.0, 0.60),
            exchange=gw.exchange, exchange_name="polymarket")

        cb = await db.fetchone(
            "SELECT size, avg_cost FROM cost_basis WHERE market_id='xm2'")
        rows = await _portfolio_rows(db, "xm2")
        assert abs(cb["size"] - 12.0) < 1e-9
        assert len(rows) == 1
        assert abs(rows[0]["size"] - cb["size"]) < 1e-9
        assert abs(rows[0]["avg_price"] - cb["avg_cost"]) < 1e-9
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_projection_labels_the_row_with_the_gateways_venue():
    """``Order.exchange`` DEFAULTS to "polymarket" — truthy — so any
    ``order.exchange or ...`` fallback mislabels every order whose builder
    left the field alone, and the kalshi long_horizon instance does exactly
    that. The projection must take the gateway's own venue-scoped name."""
    db = Database(":memory:")
    await db.connect()
    try:
        s = Settings()
        gw = _exit_gateway(db, s, _instant_fill_exchange())
        order = Order(market_id="kx1", token_id="tok", side=OrderSide.BUY,
                      token=TokenType.YES, size=10.0, price=0.30,
                      order_type=OrderType.LIMIT, dry_run=True)
        result = OrderResult(order_id="PAPER-K-1", market_id="kx1",
                             status="paper", filled_size=10.0,
                             filled_price=0.30, is_paper=True)

        await gw.record_external_fill(
            order, result, strategy_source="long_horizon_kalshi",
            exchange_name="kalshi")

        rows = await _portfolio_rows(db, "kx1")
        assert len(rows) == 1
        assert rows[0]["exchange"] == "kalshi"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_live_fill_does_not_touch_paper_portfolio_rows():
    """The projection is paper-mode maintenance only. Live portfolio rows
    belong to position sync against the venue; a live fill on a market that
    also carries a paper holding must leave the paper row alone."""
    db = Database(":memory:")
    await db.connect()
    try:
        s = Settings()
        gw = _exit_gateway(db, s, _instant_fill_exchange())
        await _seed_paper_holding(db, s, "xm3", TokenType.YES, 15.0, 0.40)

        await gw.submit_exit(
            _exit_order("xm3", TokenType.YES, 15.0, 0.55, dry_run=False),
            exchange=gw.exchange, exchange_name="polymarket")

        rows = await _portfolio_rows(db, "xm3")
        assert len(rows) == 1 and rows[0]["is_paper"] == 1
        assert abs(rows[0]["size"] - 15.0) < 1e-9, (
            "a live fill must not resize or delete the paper row")
    finally:
        await db.close()
