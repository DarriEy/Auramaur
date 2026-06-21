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
                                       "crowd prices mere de-escalation",
              verify="confirmed", verify_conf=0.9):
    a = MagicMock()

    async def _call(prompt, **kw):
        # The verify pass asks for a {"verdict": ...}; the lens pass for fair_prob.
        if "ADVERSARIALLY CHECK" in prompt or '"verdict"' in prompt:
            return f'{{"verdict": "{verify}", "confidence": {verify_conf}, "why": "x"}}'
        return f'{{"fair_prob": {fair}, "gap_score": {gap}, "mechanism": "{mech}"}}'

    a._call_llm = AsyncMock(side_effect=_call)
    return a


def _analyzer3(fair=0.10, gap=0.8, mech="'permanent' bar", verify="confirmed",
               verify_conf=0.9, grounded=0.10, ground_conf=0.9):
    """Analyzer mock that also answers the Phase 3 GROUND_PROMPT."""
    a = MagicMock()

    async def _call(prompt, **kw):
        if "grounded_prob" in prompt:
            return f'{{"grounded_prob": {grounded}, "confidence": {ground_conf}, "why": "x"}}'
        if "ADVERSARIALLY CHECK" in prompt or '"verdict"' in prompt:
            return f'{{"verdict": "{verify}", "confidence": {verify_conf}, "why": "x"}}'
        return f'{{"fair_prob": {fair}, "gap_score": {gap}, "mechanism": "{mech}"}}'

    a._call_llm = AsyncMock(side_effect=_call)
    return a


def _aggregator_mock(n=2):
    from auramaur.data_sources.base import NewsItem
    agg = MagicMock()
    items = [NewsItem(id=f"e{i}", source="news", title=f"headline {i}",
                      content="evidence body") for i in range(n)]
    agg.gather = AsyncMock(return_value=items)
    return agg


def _pillar3(db, settings, markets, exchange=None, risk=None, analyzer=None,
                 aggregator=None):
    discovery = MagicMock()
    discovery.get_markets = AsyncMock(return_value=markets)
    calibration = MagicMock(); calibration.record_prediction = AsyncMock()
    return ResolutionLensPillar(
        db=db, settings=settings, discovery=discovery,
        exchange=exchange or _exchange(), risk_manager=risk or _risk(),
        pnl_tracker=PnLTracker(db, settings), calibration=calibration,
        analyzer=analyzer if analyzer is not None else _analyzer3(),
        aggregator=aggregator,
    )


def test_phase3_grounding_confirms_gap_then_enters():
    """Evidence-grounded comprehension on a verified gap: when grounding agrees
    the strict bar is unmet (low grounded P(YES)), the edge holds and the lens
    trades the grounded fair; lens_verdicts records grounded_fair."""
    async def run():
        db = Database(":memory:"); await db.connect()
        try:
            with patch("auramaur.nlp.query_decomposer.extract_search_queries",
                       return_value=["q"]), \
                 patch("auramaur.nlp.evidence_ranker.rank_evidence",
                       side_effect=lambda q, items, **kw: items):
                m = _market(yes=0.30)
                pillar = _pillar3(db, _settings(), [m],
                                      analyzer=_analyzer3(fair=0.10, grounded=0.10),
                                      aggregator=_aggregator_mock())
                entered = await pillar.run_once()
                assert entered == 1
                row = await db.fetchone(
                    "SELECT grounded_fair FROM lens_verdicts WHERE market_id=?", (m.id,))
                assert row["grounded_fair"] is not None
                assert abs(row["grounded_fair"] - 0.10) < 1e-6
        finally:
            await db.close()
    asyncio.run(run())


def test_phase3_grounding_kills_edge_skips():
    """The synergy gate: a real fine-print gap that current evidence shows is
    ALREADY priced for reality (grounded fair ≈ market) drops the recomputed edge
    below the floor — the lens skips instead of trading a stale mechanic."""
    async def run():
        db = Database(":memory:"); await db.connect()
        try:
            with patch("auramaur.nlp.query_decomposer.extract_search_queries",
                       return_value=["q"]), \
                 patch("auramaur.nlp.evidence_ranker.rank_evidence",
                       side_effect=lambda q, items, **kw: items):
                m = _market(yes=0.30)
                # criteria fair 0.10 (edge 0.20), but grounding says 0.29 ≈ market
                pillar = _pillar3(db, _settings(), [m],
                                      analyzer=_analyzer3(fair=0.10, grounded=0.29),
                                      aggregator=_aggregator_mock())
                entered = await pillar.run_once()
                assert entered == 0   # grounded edge 0.01 < min_edge 0.08
        finally:
            await db.close()
    asyncio.run(run())


def test_phase3_low_confidence_falls_back_to_criteria_fair():
    """Low grounding confidence is ignored — the lens falls back to the
    Phase 1+2-validated criteria fair rather than trusting weak evidence."""
    async def run():
        db = Database(":memory:"); await db.connect()
        try:
            with patch("auramaur.nlp.query_decomposer.extract_search_queries",
                       return_value=["q"]), \
                 patch("auramaur.nlp.evidence_ranker.rank_evidence",
                       side_effect=lambda q, items, **kw: items):
                m = _market(yes=0.30)
                # grounding would kill the edge (0.29) BUT confidence is low -> ignored
                pillar = _pillar3(db, _settings(), [m],
                                      analyzer=_analyzer3(fair=0.10, grounded=0.29,
                                                          ground_conf=0.2),
                                      aggregator=_aggregator_mock())
                entered = await pillar.run_once()
                assert entered == 1   # criteria fair 0.10 used -> edge 0.20 holds
        finally:
            await db.close()
    asyncio.run(run())


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


def test_eligible_loosens_thresholds_in_paper_mode():
    """A short-dated, lower-liquidity market is eligible while paper-forced
    (venue guards don't apply) but rejected once flipped to live."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        # ~2h to resolution, $400 liquidity: fails live (12h / $1000),
        # passes paper (1h / $250).
        m = _market(days_out=2.0 / 24.0, liquidity=400.0)

        paper = _pillar(db, _settings(paper=True), [m])
        assert paper._eligible(m) is True

        live = _pillar(db, _settings(paper=False), [m])
        assert live._eligible(m) is False

    asyncio.run(run())


# ----------------------------------------------------------------------
# Phase 1 + 2 upgrades
# ----------------------------------------------------------------------

def test_full_criteria_not_truncated_at_800():
    """The lens must feed the FULL criteria (tail kept) — the decisive clause
    lives at the end and the old [:800] cut it off."""
    async def run():
        db = Database(":memory:"); await db.connect()
        try:
            tail = "RESOLVES NO unless an OFFICIAL signed treaty is filed."
            desc = ("filler. " * 200) + tail        # ~1600 chars, clause at end
            analyzer = _analyzer()
            pillar = _pillar(db, _settings(), [_market(yes=0.30)], analyzer=analyzer)
            m = _market(yes=0.30); m.description = desc
            await pillar._verdict(m)
            sent = analyzer._call_llm.await_args_list[0].args[0]
            assert tail in sent                      # tail survived
            assert len(desc) > 800                   # would have been cut before
        finally:
            await db.close()
    asyncio.run(run())


def test_refuted_mechanism_blocks_entry():
    """A strong gap whose mechanism the adversarial pass REFUTES does not trade."""
    async def run():
        db = Database(":memory:"); await db.connect()
        try:
            ex = _exchange()
            analyzer = _analyzer(verify="refuted", verify_conf=0.9)
            pillar = _pillar(db, _settings(), [_market(yes=0.30)], exchange=ex, analyzer=analyzer)
            assert await pillar.run_once() == 0
            ex.place_order.assert_not_awaited()
            # cached as refuted (0) so it isn't re-verified
            row = await db.fetchone("SELECT verified FROM lens_verdicts WHERE market_id='m1'")
            assert int(row["verified"]) == 0
        finally:
            await db.close()
    asyncio.run(run())


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


def test_lexical_candidates_scan_whole_table_not_volume_top():
    """Candidates are selected by fine-print lexical shape across the FULL
    markets table — not the top-N-by-volume scan — so the long-tail markets
    the lens targets are reachable (root cause of zero signals)."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            async def _ins(mid, q, exchange="polymarket", active=1, liq=5000.0,
                           end_days=20):
                await db.execute(
                    "INSERT INTO markets (id, exchange, question, description, "
                    "category, end_date, outcome_yes_price, outcome_no_price, "
                    "liquidity, clob_token_yes, clob_token_no, active, last_updated) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?, datetime('now'))",
                    (mid, exchange, q,
                     "Long fine-print resolution criteria describing the strict "
                     "written bar that must be met for YES to resolve.",
                     "politics_intl",
                     (datetime.now(timezone.utc) + timedelta(days=end_days)).isoformat(),
                     0.30, 0.70, liq, "ty", "tn", active))
            await _ins("hit", "Will Iran officially announce a permanent ceasefire?")
            await _ins("plain", "Will Bitcoin go up tomorrow?")            # no trigger
            await _ins("inactive", "Will X officially confirm it?", active=0)
            await _ins("kalshi", "Will Y officially announce?", exchange="kalshi")
            # resolved-but-active rows (past end_date) are ~87% of the table —
            # the lens can't trade them, so the lexical scan must exclude them.
            await _ins("stale", "Will Z officially announce a permanent deal?", end_days=-5)
            await db.commit()

            pillar = _pillar(db, _settings(), markets=[])
            cands = await pillar._lexical_candidates()
            assert {m.id for m in cands} == {"hit"}
            assert cands[0].description  # fine print loaded from the cache
        finally:
            await db.close()

    asyncio.run(run())
