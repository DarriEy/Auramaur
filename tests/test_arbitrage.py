"""Tests for arbitrage execution."""

from __future__ import annotations

import asyncio

import pytest

from auramaur.db.database import Database
from auramaur.exchange.models import OrderSide
from auramaur.strategy.arbitrage import ArbitrageExecutor
from auramaur.strategy.correlation import CorrelationDetector


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def db(event_loop):
    async def _setup():
        database = Database(db_path=":memory:")
        await database.connect()
        return database
    database = event_loop.run_until_complete(_setup())
    yield database
    event_loop.run_until_complete(database.close())


@pytest.fixture
def executor(db):
    correlator = CorrelationDetector(db=db)
    return ArbitrageExecutor(db=db, correlator=correlator)


def run(coro, loop):
    return loop.run_until_complete(coro)


class TestArbitrageExecutor:
    def test_no_opportunities_returns_empty(self, executor, event_loop):
        result = run(executor.generate_arb_signals(), event_loop)
        assert result == []

    def test_conditional_violation_generates_pair(self, executor, event_loop):
        """Conditional violation should produce buy-B sell-A signals."""
        db = executor._db
        # Market A (primary): 70% — too high if it implies B
        run(db.execute(
            """INSERT INTO markets (id, condition_id, question, active, outcome_yes_price, outcome_no_price, last_updated)
               VALUES ('primary', 'c1', 'Win primary?', 1, 0.70, 0.30, datetime('now'))"""
        ), event_loop)
        # Market B (general): 60% — if A implies B, P(A) should be <= P(B)
        run(db.execute(
            """INSERT INTO markets (id, condition_id, question, active, outcome_yes_price, outcome_no_price, last_updated)
               VALUES ('general', 'c2', 'Win general?', 1, 0.60, 0.40, datetime('now'))"""
        ), event_loop)
        # Conditional relationship
        run(db.execute(
            """INSERT INTO market_relationships
               (market_id_a, market_id_b, relationship_type, strength, description, detected_at)
               VALUES ('primary', 'general', 'conditional', 0.9, 'Primary implies general', datetime('now'))"""
        ), event_loop)
        run(db.commit(), event_loop)

        pairs = run(executor.generate_arb_signals(), event_loop)
        assert len(pairs) == 1
        buy_sig, sell_sig, opp = pairs[0]
        # Should buy B (general, underpriced) and sell A (primary, overpriced)
        assert buy_sig.market_id == "general"
        assert buy_sig.recommended_side == OrderSide.BUY
        assert sell_sig.market_id == "primary"
        assert sell_sig.recommended_side == OrderSide.SELL

    def test_price_divergence_generates_pair(self, executor, event_loop):
        """Same-event divergence should buy cheap and sell expensive."""
        db = executor._db
        run(db.execute(
            """INSERT INTO markets (id, condition_id, question, active, outcome_yes_price, outcome_no_price, last_updated)
               VALUES ('cheap', 'c1', 'Event?', 1, 0.45, 0.55, datetime('now'))"""
        ), event_loop)
        run(db.execute(
            """INSERT INTO markets (id, condition_id, question, active, outcome_yes_price, outcome_no_price, last_updated)
               VALUES ('expensive', 'c2', 'Same event?', 1, 0.55, 0.45, datetime('now'))"""
        ), event_loop)
        run(db.execute(
            """INSERT INTO market_relationships
               (market_id_a, market_id_b, relationship_type, strength, description, detected_at)
               VALUES ('cheap', 'expensive', 'same_event', 0.95, 'Same event', datetime('now'))"""
        ), event_loop)
        run(db.commit(), event_loop)

        pairs = run(executor.generate_arb_signals(), event_loop)
        assert len(pairs) == 1
        buy_sig, sell_sig, _ = pairs[0]
        assert buy_sig.market_id == "cheap"
        assert sell_sig.market_id == "expensive"

    def test_missing_market_skipped(self, executor, event_loop):
        """Opportunities with missing markets should be skipped."""
        db = executor._db
        # Only create one market
        run(db.execute(
            """INSERT INTO markets (id, condition_id, question, active, outcome_yes_price, outcome_no_price, last_updated)
               VALUES ('exists', 'c1', 'Q?', 1, 0.50, 0.50, datetime('now'))"""
        ), event_loop)
        run(db.execute(
            """INSERT INTO market_relationships
               (market_id_a, market_id_b, relationship_type, strength, description, detected_at)
               VALUES ('exists', 'missing', 'same_event', 0.9, 'Related', datetime('now'))"""
        ), event_loop)
        run(db.commit(), event_loop)

        pairs = run(executor.generate_arb_signals(), event_loop)
        assert len(pairs) == 0


# --- Category gate (2026-06-12): exempt books still respect category bans ---

def test_arb_category_gate_filters_blocked_and_non_allowlisted():
    """The arb scanner was caught quoting KBO baseball live — an arb is
    hedged only when both legs fill; a single-leg fill is directional sports
    inventory. Blocked categories are never scanned; with the live allowlist
    set, unknown/'other' markets are skipped too."""
    from auramaur.exchange.models import Market
    from auramaur.strategy.arbitrage_scanner import ArbitrageScanner

    def _mkt(mid, question, category=""):
        return Market(id=mid, question=question, category=category)

    kbo = _mkt("kbo", "KBO: SSG Landers vs. Samsung Lions")
    labeled = _mkt("lab", "Will this happen?", category="sports")
    unknown = _mkt("unk", "Will the gadget ship this quarter?")
    crypto = _mkt("cry", "Will Bitcoin reach $200k?", category="crypto")

    live = ArbitrageScanner(
        discoveries={}, blocked_categories=["sports", "politics_us"],
        allowed_categories_live=["crypto", "tech", "politics_intl"])
    assert [m.id for m in [kbo, labeled, unknown, crypto]
            if live._category_ok(m)] == ["cry"]

    paper = ArbitrageScanner(
        discoveries={}, blocked_categories=["sports", "politics_us"],
        allowed_categories_live=None)
    assert [m.id for m in [kbo, labeled, unknown, crypto]
            if paper._category_ok(m)] == ["unk", "cry"]


@pytest.mark.asyncio
async def test_internal_arb_uses_engine_exchange_attribute():
    """Regression: _execute_internal_arb must read engine.exchange, not the
    non-existent engine._exchange (which AttributeError'd the live internal-arb
    branch). The fake engine deliberately exposes ONLY .exchange, so the old
    code would raise — a MagicMock engine would mask the bug by auto-creating
    ._exchange."""
    from unittest.mock import AsyncMock, MagicMock
    from types import SimpleNamespace

    from auramaur.bot_arb import ArbExecutionMixin
    from auramaur.exchange.models import Market, OrderResult
    from auramaur.strategy.arbitrage_scanner import ArbOpportunity

    class _FakeEngine:
        def __init__(self, exchange):
            self.exchange = exchange  # NOTE: no `_exchange`
        async def _get_available_cash(self):
            return 1000.0

    exchange = MagicMock()
    exchange.place_order = AsyncMock(side_effect=lambda o: OrderResult(
        order_id=f"ord-{o.token_id}", market_id=o.market_id, status="paper",
        filled_size=o.size, filled_price=o.price, is_paper=True))

    mixin = ArbExecutionMixin.__new__(ArbExecutionMixin)
    mixin.settings = MagicMock()
    mixin.settings.is_live = False
    mixin._exit_gateway = MagicMock()
    mixin._exit_gateway.record_external_fill = AsyncMock()
    db = MagicMock()
    db.fetchone = AsyncMock(return_value=None)  # pairing guard: not already held
    alerts = MagicMock()
    alerts.send = AsyncMock()
    mixin._components = {"db": db, "alerts": alerts}

    market = Market(
        id="m1", exchange="polymarket", question="Will X?",
        outcome_yes_price=0.40, outcome_no_price=0.50,
        clob_token_yes="tyes", clob_token_no="tno",
    )
    opp = ArbOpportunity(
        market_a=market, market_b=market, exchange_a="polymarket",
        exchange_b="polymarket", price_a=0.40, price_b=0.50, spread=0.10,
        expected_profit_pct=10.0, question="Will X?", arb_type="internal",
    )
    decision = SimpleNamespace(approved=True, position_size=20.0, reason="")
    risk = MagicMock()
    risk.evaluate = AsyncMock(return_value=decision)
    engines = {"polymarket": _FakeEngine(exchange)}

    await mixin._execute_internal_arb(opp, risk, engines)

    # Both legs placed through engine.exchange — proves the attribute resolves.
    assert exchange.place_order.await_count == 2
    assert mixin._exit_gateway.record_external_fill.await_count == 2
