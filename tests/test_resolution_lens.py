"""Tests for the resolution-language lens and the name-the-gap gate.

Locks in:
  Lens:
    1. Lexical triggers fire on the documented win shapes (announce,
       permanent, literal counting) and not on plain questions.
    2. The LLM verdict gates entries: low gap_score / small edge /
       mechanism 'none' do not trade; verdicts cache (one call per market).
    3. Lens signals carry mispricing_reason from birth and respect
       paper-forcing + risk rejection; full rails written on entry.
  Gate:
    4. check_mispricing_named semantics (enabled/applies/threshold/none).
    5. RiskManager integration: unexplained llm divergence is blocked when
       enabled; named reason passes; non-llm sources are exempt; the
       auditor is called lazily and its verdict is attached.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

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
from auramaur.nlp.gap_audit import GapAuditor
from auramaur.risk.checks import check_mispricing_named
from auramaur.strategy.resolution_lens import ResolutionLensPillar, has_lens_trigger
from config.settings import Settings


def _market(mid="m1", question="Will Iran announce a permanent peace deal by July 31?",
            yes=0.30, liquidity=5000.0, spread=0.01, days_out=20.0,
            description="Resolves YES only if an official written agreement explicitly "
                        "described as permanent is announced by both governments before "
                        "the deadline.") -> Market:
    return Market(
        id=mid, exchange="polymarket", question=question, active=True,
        outcome_yes_price=yes, outcome_no_price=round(1 - yes, 2),
        liquidity=liquidity, volume=liquidity, spread=spread,
        description=description, category="politics_intl",
        end_date=datetime.now(timezone.utc) + timedelta(days=days_out),
        clob_token_yes="ty", clob_token_no="tn",
    )


def _settings(**overrides) -> Settings:
    s = Settings()
    s.resolution_lens.enabled = True
    s.resolution_lens.paper = True
    s.resolution_lens.min_edge = 0.08
    s.resolution_lens.min_gap_score = 0.4
    for k, v in overrides.items():
        setattr(s.resolution_lens, k, v)
    return s


def _exchange():
    ex = MagicMock()

    def prepare_order(signal, market, size, is_live):
        token = TokenType.NO if signal.recommended_side == OrderSide.SELL else TokenType.YES
        price = market.outcome_yes_price if token == TokenType.YES else 1 - market.outcome_yes_price
        return Order(market_id=market.id, token_id="tok", side=OrderSide.BUY,
                     token=token, size=round(size / max(price, 0.01), 2),
                     price=round(price, 2), order_type=OrderType.LIMIT,
                     dry_run=not is_live)

    ex.prepare_order = MagicMock(side_effect=prepare_order)
    ex.place_order = AsyncMock(side_effect=lambda order: OrderResult(
        order_id=f"ord-{order.market_id}", market_id=order.market_id,
        status="paper" if order.dry_run else "filled",
        filled_size=order.size, filled_price=order.price, is_paper=order.dry_run))
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


def _analyzer(fair=0.10, gap=0.8, mech="'permanent' requires explicit permanence language; "
                                       "crowd prices mere de-escalation"):
    a = MagicMock()
    a._call_llm = AsyncMock(return_value=(
        f'{{"fair_prob": {fair}, "gap_score": {gap}, "mechanism": "{mech}"}}'))
    return a


def _pillar(db, settings, markets, exchange=None, risk=None, analyzer=None):
    discovery = MagicMock()
    discovery.get_markets = AsyncMock(return_value=markets)
    calibration = MagicMock()
    calibration.record_prediction = AsyncMock()
    return ResolutionLensPillar(
        db=db, settings=settings, discovery=discovery,
        exchange=exchange or _exchange(), risk_manager=risk or _risk(),
        pnl_tracker=PnLTracker(db, settings), calibration=calibration,
        analyzer=analyzer if analyzer is not None else _analyzer(),
    )


# ----------------------------------------------------------------------
# Lexical triggers
# ----------------------------------------------------------------------

def test_lens_triggers():
    assert has_lens_trigger("Will Trump announce a blockade by June 30?")
    assert has_lens_trigger("US x Iran permanent peace deal by July?")
    assert has_lens_trigger("Will the video get between 40 and 50 million views?")
    assert has_lens_trigger("Will Claude be the #1 AI model on LMArena?")
    assert has_lens_trigger("Will the Fed officially confirm a pause?")
    assert not has_lens_trigger("Will the Lakers win the NBA Finals?")
    assert not has_lens_trigger("Bitcoin up or down this week?")


# ----------------------------------------------------------------------
# Lens pillar
# ----------------------------------------------------------------------

def test_lens_enters_on_strong_verdict_with_named_reason():
    async def run():
        db = Database(":memory:")
        await db.connect()
        ex = _exchange()
        risk = _risk()
        # Market at 0.30, strict fair 0.10 -> SELL (buy NO), edge 0.20.
        pillar = _pillar(db, _settings(), [_market(yes=0.30)], exchange=ex, risk=risk)
        assert await pillar.run_once() == 1

        signal = risk.evaluate.await_args.args[0]
        assert signal.strategy_source == "resolution_lens"
        assert signal.mispricing_reason.startswith("behavioral: ")
        assert signal.claude_confidence.value == "HIGH"  # gap 0.8 >= 0.7
        assert signal.recommended_side == OrderSide.SELL

        order = ex.place_order.await_args.args[0]
        assert order.dry_run is True  # paper-forced
        assert order.token == TokenType.NO

        # Verdict cached; trades/portfolio/signals rows written.
        assert await db.fetchone("SELECT 1 FROM lens_verdicts WHERE market_id='m1'")
        assert await db.fetchone(
            "SELECT 1 FROM trades WHERE strategy_source='resolution_lens'")
        # One shot per market.
        assert await pillar.run_once() == 0
        await db.close()

    asyncio.run(run())


def test_lens_skips_weak_verdicts_and_caches():
    async def run():
        db = Database(":memory:")
        await db.connect()
        for fair, gap, mech in (
            (0.28, 0.1, "tiny wording nuance"),   # gap_score below floor
            (0.27, 0.8, "real mechanism"),        # |edge| 0.03 below min_edge
            (0.10, 0.8, "none"),                  # mechanism 'none'
        ):
            analyzer = _analyzer(fair=fair, gap=gap, mech=mech)
            ex = _exchange()
            pillar = _pillar(db, _settings(), [_market(mid=f"m-{fair}-{gap}", yes=0.30)],
                             exchange=ex, analyzer=analyzer)
            assert await pillar.run_once() == 0
            ex.place_order.assert_not_awaited()
            analyzer._call_llm.assert_awaited_once()  # verdict computed once...
            assert await pillar.run_once() == 0
            analyzer._call_llm.assert_awaited_once()  # ...then served from cache
        await db.close()

    asyncio.run(run())


def test_lens_no_trigger_no_llm_call():
    async def run():
        db = Database(":memory:")
        await db.connect()
        analyzer = _analyzer()
        pillar = _pillar(db, _settings(),
                         [_market(question="Will the Lakers win the title?")],
                         analyzer=analyzer)
        assert await pillar.run_once() == 0
        analyzer._call_llm.assert_not_awaited()
        await db.close()

    asyncio.run(run())


# ----------------------------------------------------------------------
# Name-the-gap check + RiskManager integration
# ----------------------------------------------------------------------

def test_check_mispricing_named_semantics():
    async def run():
        # disabled -> always passes
        assert (await check_mispricing_named("", 0.2, enabled=False)).passed
        # below threshold -> passes
        assert (await check_mispricing_named("", 0.03, enabled=True,
                                             min_divergence=0.05)).passed
        # doesn't apply (non-llm source) -> passes
        assert (await check_mispricing_named("", 0.2, enabled=True,
                                             applies=False)).passed
        # significant divergence, no reason -> blocked
        assert not (await check_mispricing_named("", 0.2, enabled=True)).passed
        assert not (await check_mispricing_named("none", 0.2, enabled=True)).passed
        # named mechanism -> passes
        assert (await check_mispricing_named(
            "behavioral: crowd prices headline", 0.2, enabled=True)).passed

    asyncio.run(run())


def test_risk_manager_gate_blocks_unexplained_and_calls_auditor():
    from tests.test_risk_manager import (
        _make_market, _make_settings, _make_signal, _mock_portfolio,
    )

    async def run():
        from auramaur.risk.checks import CheckResult
        from auramaur.risk.manager import RiskManager

        db = Database(":memory:")
        await db.connect()
        settings = _make_settings()
        settings.risk.mispricing_gate_enabled = True
        settings.risk.mispricing_min_divergence = 0.05
        settings.risk.mispricing_audit_ttl_hours = 12.0

        with patch("auramaur.risk.manager.check_kill_switch") as mock_kill:
            mock_kill.return_value = CheckResult(
                name="kill_switch", passed=True, reason="", value=False)

            manager = RiskManager(settings, db)
            manager.portfolio = _mock_portfolio()
            market = _make_market()

            # No auditor wired: unexplained 10pt divergence is blocked.
            signal = _make_signal(edge=10.0, claude_prob=0.60, market_prob=0.50)
            d = await manager.evaluate(signal, market, available_cash=500.0)
            assert d.approved is False
            assert any(c.name == "mispricing_named" and not c.passed
                       for c in d.checks)

            # Auditor wired and names a mechanism: passes, reason attached.
            analyzer = _analyzer()
            analyzer._call_llm = AsyncMock(return_value=(
                '{"mechanism": "informational", "reason": "primary source X '
                'not yet digested by the crowd"}'))
            manager.gap_auditor = GapAuditor(db, analyzer, settings)
            signal2 = _make_signal(edge=10.0, claude_prob=0.60, market_prob=0.50)
            d2 = await manager.evaluate(signal2, market, available_cash=500.0)
            assert d2.approved is True
            assert signal2.mispricing_reason.startswith("informational: ")

            # Auditor says none: blocked.
            analyzer._call_llm = AsyncMock(return_value=(
                '{"mechanism": "none", "reason": "market likely right"}'))
            manager.gap_auditor = GapAuditor(db, analyzer, settings)
            signal3 = _make_signal(edge=10.0, claude_prob=0.62, market_prob=0.50)
            signal3.market_id = "other-market"
            d3 = await manager.evaluate(signal3, market, available_cash=500.0)
            assert d3.approved is False

            # Non-llm sources are exempt (pre-named or synthetic probs).
            signal4 = _make_signal(edge=10.0, claude_prob=0.60, market_prob=0.50)
            signal4.strategy_source = "bias_harvest"
            d4 = await manager.evaluate(signal4, market, available_cash=500.0)
            assert not any(c.name == "mispricing_named" and not c.passed
                           for c in d4.checks)

        await db.close()

    asyncio.run(run())


def test_gap_audit_caching():
    async def run():
        db = Database(":memory:")
        await db.connect()
        settings = MagicMock()
        settings.risk.mispricing_audit_ttl_hours = 12.0
        analyzer = MagicMock()
        analyzer._call_llm = AsyncMock(return_value=(
            '{"mechanism": "behavioral", "reason": "headline vs fine print"}'))
        auditor = GapAuditor(db, analyzer, settings)

        signal = MagicMock()
        signal.market_id = "mX"
        signal.claude_prob = 0.60
        signal.market_prob = 0.50
        market = MagicMock()
        market.question = "q"
        market.description = "d"

        r1 = await auditor.audit(signal, market)
        r2 = await auditor.audit(signal, market)
        assert r1 == r2 == "behavioral: headline vs fine print"
        analyzer._call_llm.assert_awaited_once()  # second hit served from cache

        # Material price move re-audits.
        signal.market_prob = 0.60
        await auditor.audit(signal, market)
        assert analyzer._call_llm.await_count == 2
        await db.close()

    asyncio.run(run())
