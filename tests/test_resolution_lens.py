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
# Favorite-discipline entry-price floor (BUY side only)
# ----------------------------------------------------------------------

def test_buy_below_entry_floor_blocked():
    """A BUY whose YES side is below min_entry_price (the near-coin-flip band
    where the weather cell took every loss) does not trade."""
    async def run():
        db = Database(":memory:"); await db.connect()
        try:
            ex = _exchange()
            # fair 0.90 vs market 0.55 -> BUY YES (edge +0.35), but 0.55 < 0.65.
            analyzer = _analyzer(fair=0.90, gap=0.8, mech="modal bin favored")
            pillar = _pillar(db, _settings(min_entry_price=0.65),
                             [_market(yes=0.55)], exchange=ex, analyzer=analyzer)
            assert await pillar.run_once() == 0
            ex.place_order.assert_not_awaited()
        finally:
            await db.close()
    asyncio.run(run())


def test_buy_at_or_above_entry_floor_enters():
    """The same BUY clears once the YES side is itself a market favorite."""
    async def run():
        db = Database(":memory:"); await db.connect()
        try:
            ex = _exchange()
            analyzer = _analyzer(fair=0.90, gap=0.8, mech="modal bin favored")
            pillar = _pillar(db, _settings(min_entry_price=0.65),
                             [_market(yes=0.70)], exchange=ex, analyzer=analyzer)
            assert await pillar.run_once() == 1
            order = ex.place_order.await_args.args[0]
            assert order.token == TokenType.YES
        finally:
            await db.close()
    asyncio.run(run())


def test_sell_longshot_exempt_from_entry_floor():
    """The floor gates BUYs only: a SELL of an overpriced-YES longshot (the
    permanence/announce shape) trades even though the NO side it buys is well
    below the floor — that is the lens's other documented edge."""
    async def run():
        db = Database(":memory:"); await db.connect()
        try:
            ex = _exchange()
            # Default analyzer: fair 0.10 vs market 0.30 -> SELL (buy NO).
            pillar = _pillar(db, _settings(min_entry_price=0.65),
                             [_market(yes=0.30)], exchange=ex)
            assert await pillar.run_once() == 1
            order = ex.place_order.await_args.args[0]
            assert order.token == TokenType.NO
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


def test_known_cached_gap_is_prioritized_over_fresh_under_tight_budget():
    """The 2026-06-24 silence fix. A qualifying gap the lens already found (cached,
    not yet verified) must be processed BEFORE fresh candidates, so its verify
    call lands inside a tight per-cycle LLM budget. Placed LAST behind two fresh
    candidates with a 1-call budget, the cached gap still enters — without
    prioritization the fresh verdicts would eat the budget and starve it (the
    failure mode that left the proven cell silent for six days)."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            ex = _exchange()
            settings = _settings(max_llm_calls_per_cycle=1, max_entries_per_cycle=5)
            fresh1 = _market(mid="fresh1", yes=0.30)
            fresh2 = _market(mid="fresh2", yes=0.30)
            cached = _market(mid="cached", yes=0.30)   # priced for a SELL (fair 0.10)
            # cached LAST in scan order; fresh ones would otherwise consume budget
            pillar = _pillar(db, settings, [fresh1, fresh2, cached], exchange=ex)
            await pillar._ensure_schema()
            # Pre-seed the cached qualifying gap, unverified (-1) so it needs a
            # verify call — exactly the state that was being starved.
            await db.execute(
                "INSERT INTO lens_verdicts (market_id, fair_prob, gap_score, "
                "mechanism, verified) VALUES ('cached', 0.10, 0.8, 'permanence bar', -1)")
            await db.commit()

            entered = await pillar.run_once()
            assert entered == 1                       # the cached gap, not starved
            # it is specifically the cached market that traded
            assert await db.fetchone(
                "SELECT 1 FROM portfolio WHERE market_id='cached'") is not None
            assert await db.fetchone(
                "SELECT 1 FROM portfolio WHERE market_id='fresh1'") is None
        finally:
            await db.close()

    asyncio.run(run())


def test_live_cell_gap_beats_higher_gap_paper_cell_for_the_slot():
    """gap_score is NOT edge quality. The proven LIVE cell (weather) must get the
    scarce per-cycle entry slot before a HIGHER-gap PAPER cell (politics, which
    the lens loses on) — otherwise paper exploration starves the one edge that
    earns real money."""
    async def run():
        from types import SimpleNamespace
        db = Database(":memory:")
        await db.connect()
        try:
            ex = _exchange()
            settings = _settings(paper=False, max_entries_per_cycle=1)
            risk = _risk()

            async def _decide(strategy, category):   # weather trades live; rest paper
                return SimpleNamespace(force_paper=(category != "weather"))
            risk.graduation = SimpleNamespace(decide=_decide)

            wx = _market(mid="wx", yes=0.30); wx.category = "weather"
            pol = _market(mid="pol", yes=0.30); pol.category = "politics_intl"
            # politics has the HIGHER gap and is placed FIRST in scan order
            pillar = _pillar(db, settings, [pol, wx], exchange=ex, risk=risk)
            await pillar._ensure_schema()
            await db.execute("INSERT INTO lens_verdicts (market_id, fair_prob, "
                             "gap_score, mechanism, verified) VALUES ('pol',0.10,0.9,'bar',1)")
            await db.execute("INSERT INTO lens_verdicts (market_id, fair_prob, "
                             "gap_score, mechanism, verified) VALUES ('wx',0.10,0.4,'bar',1)")
            await db.commit()

            entered = await pillar.run_once()
            assert entered == 1
            # the LIVE weather cell took the single slot, not the higher-gap paper one
            assert await db.fetchone(
                "SELECT 1 FROM portfolio WHERE market_id='wx'") is not None
            assert await db.fetchone(
                "SELECT 1 FROM portfolio WHERE market_id='pol'") is None
        finally:
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


def test_kalshi_spike_instance_is_parametrized_and_isolated():
    """The Kalshi measurement-spike instance binds the Kalshi venue, uses the
    distinct 'resolution_lens_kalshi' source (own graduation cells), applies the
    thinner Kalshi liquidity floor, and rejects Polymarket markets — while the
    default Poly instance is unchanged."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        settings = _settings()

        # Kalshi-bound instance.
        kalshi = _pillar(db, settings, [])
        # rebuild with explicit kalshi params (the _pillar helper defaults Poly)
        from auramaur.strategy.resolution_lens import ResolutionLensPillar
        kalshi = ResolutionLensPillar(
            db=db, settings=settings, discovery=MagicMock(),
            exchange=_exchange(), risk_manager=_risk(),
            pnl_tracker=PnLTracker(db, settings), calibration=MagicMock(),
            analyzer=_analyzer(), exchange_name="kalshi",
            source_tag="resolution_lens_kalshi",
        )
        assert kalshi._exchange_name == "kalshi"
        assert kalshi._source_tag == "resolution_lens_kalshi"

        # A Kalshi market passes the exchange filter; a Poly one is rejected.
        k_mkt = _market(mid="k1", liquidity=350.0)
        k_mkt.exchange = "kalshi"
        assert kalshi._eligible(k_mkt) is True          # thin book OK on Kalshi floor
        poly_mkt = _market(mid="p1", liquidity=5000.0)  # wrong venue for this instance
        assert kalshi._eligible(poly_mkt) is False

        # The default Poly instance still rejects Kalshi markets (isolation).
        poly = _pillar(db, settings, [])
        assert poly._exchange_name == "polymarket"
        assert poly._source_tag == "resolution_lens"
        assert poly._eligible(k_mkt) is False

    asyncio.run(run())


# ----------------------------------------------------------------------
# Error-path caching + Claude pinning (2026-07 lens salvage)
# ----------------------------------------------------------------------

def test_verdict_error_not_cached_then_gives_up_after_three():
    """An LLM/parse error must NOT write the permanent neutral verdict (that
    poisoned the cache for every market scanned while calls were failing).
    Retry instead — but after 3 strikes persist the neutral row so one
    confusing market can't eat the per-cycle budget forever."""
    async def run():
        db = Database(":memory:"); await db.connect()
        m = _market()
        a = MagicMock()
        a._call_llm = AsyncMock(side_effect=RuntimeError("boom"))
        p = _pillar(db, _settings(), [m], analyzer=a)
        await p._ensure_schema()

        assert await p._verdict(m) is None
        assert await p._verdict(m) is None
        row = await db.fetchone(
            "SELECT 1 FROM lens_verdicts WHERE market_id = ?", (m.id,))
        assert row is None  # nothing persisted while retrying

        v = await p._verdict(m)  # third strike -> neutral row, permanent
        assert v == (m.outcome_yes_price, 0.0, "none", -1)
        row = await db.fetchone(
            "SELECT gap_score, mechanism FROM lens_verdicts WHERE market_id = ?",
            (m.id,))
        assert row["gap_score"] == 0.0 and row["mechanism"] == "none"

    asyncio.run(run())


def test_verify_infra_error_leaves_verdict_unverified():
    """A verify-pass ERROR is not a refutation: verified must stay -1 (retry
    next cycle), not be persisted as 0 (permanent kill of a real gap)."""
    async def run():
        db = Database(":memory:"); await db.connect()
        m = _market()
        good = _analyzer()
        p = _pillar(db, _settings(), [m], analyzer=good)
        await p._ensure_schema()
        await p._verdict(m)  # seed a cached gap (verified=-1)

        p._analyzer = MagicMock()
        p._analyzer._call_llm = AsyncMock(side_effect=RuntimeError("budget"))
        assert await p._verify_mechanism(m, 0.10, "mech") is False
        row = await db.fetchone(
            "SELECT verified FROM lens_verdicts WHERE market_id = ?", (m.id,))
        assert row["verified"] == -1  # NOT 0

    asyncio.run(run())


def test_lens_calls_pin_to_claude():
    """All three lens LLM calls must pass pin_claude=True — the fine-print
    read is the edge; routing it to another model is what silenced the cell
    (2026-06-25 -> 07-02)."""
    async def run():
        db = Database(":memory:"); await db.connect()
        m = _market()
        a = _analyzer3()
        p = _pillar3(db, _settings(), [m], analyzer=a,
                     aggregator=_aggregator_mock())
        await p._ensure_schema()
        await p._verdict(m)
        await p._verify_mechanism(m, 0.10, "mech")
        await p._ground(m, 0.10, "mech")
        assert a._call_llm.await_count == 3
        for call in a._call_llm.await_args_list:
            assert call.kwargs.get("pin_claude") is True

    asyncio.run(run())


def test_pin_claude_bypasses_router_and_uses_reserve():
    """ClaudeAnalyzer._call_llm(pin_claude=True) must skip Gemini routing and
    spend against the full budget; unpinned callers stop at budget - reserve."""
    async def run():
        from auramaur.nlp.analyzer import ClaudeAnalyzer
        from auramaur.nlp.errors import BudgetExhausted

        s = Settings()
        s.gemini.enabled = True
        s.gemini.off_hours_utc = list(range(24))  # router would pick Gemini
        s.nlp.daily_claude_call_budget = 100
        s.nlp.claude_reserve_for_pinned = 25

        a = ClaudeAnalyzer(settings=s)
        with patch.object(a, "_call_claude_cli", new=AsyncMock(return_value="ok")) as cli, \
             patch("auramaur.nlp.llm_router.call_gemini", new=AsyncMock(return_value="gem")):
            s.gemini_api_key = "k"
            assert await a._call_llm("p", pin_claude=True) == "ok"
            assert cli.await_args.kwargs.get("reserved") is True
            assert await a._call_llm("p") == "gem"  # unpinned -> routed

        # Budget math: unpinned stops 25 early, pinned runs to the cap.
        with patch("auramaur.nlp.call_budget.calls_today", return_value=80):
            try:
                await a._call_claude_cli("p")
                raise AssertionError("unpinned should be exhausted at 80/100")
            except BudgetExhausted:
                pass
            with patch.object(
                    asyncio, "create_subprocess_exec",
                    side_effect=AssertionError("stop before real CLI")):
                try:
                    await a._call_claude_cli("p", reserved=True)
                except AssertionError:
                    pass  # got PAST the budget gate (reserved slice)

    asyncio.run(run())


def test_budget_exhaustion_never_strikes_or_caches():
    """BudgetExhausted is a GLOBAL condition, not the market's fault: it must
    not count toward the 3-strike give-up, no matter how many cycles it lasts
    — a day-long budget outage was permanently neutral-caching every scanned
    market (5 markets poisoned on 2026-07-02 before this fix)."""
    async def run():
        from auramaur.nlp.errors import BudgetExhausted

        db = Database(":memory:"); await db.connect()
        m = _market()
        a = MagicMock()
        a._call_llm = AsyncMock(side_effect=BudgetExhausted("daily cap"))
        p = _pillar(db, _settings(), [m], analyzer=a)
        await p._ensure_schema()

        for _ in range(10):  # many cycles of exhaustion
            assert await p._verdict(m) is None
        row = await db.fetchone(
            "SELECT 1 FROM lens_verdicts WHERE market_id = ?", (m.id,))
        assert row is None                       # nothing cached, ever
        assert p._verdict_failures.get(m.id, 0) == 0  # no strikes accrued

        # Budget returns -> the very next call succeeds normally.
        p._analyzer = _analyzer()
        v = await p._verdict(m)
        assert v is not None and v[1] == 0.8

    asyncio.run(run())


def test_kalshi_instance_merges_close_window_slice():
    """Kalshi's generic discovery surfaces mostly ultra-long-dated novelty
    markets, so the DB-backed lexical scan starves the Kalshi lens (funnel
    audit: zero candidates passed all gates). When the venue's discovery
    exposes get_markets_by_close_window, the lens must merge that near-dated
    tradeable slice (trigger-filtered, deduped) into its scan — and a
    discovery without the method (Polymarket) stays a no-op."""
    async def run():
        from auramaur.strategy.resolution_lens import ResolutionLensPillar

        db = Database(":memory:")
        await db.connect()
        settings = _settings()
        calibration = MagicMock(); calibration.record_prediction = AsyncMock()

        in_window = _market(
            mid="KXTEST-26AUG-T5", liquidity=350.0, days_out=20.0,
            question="Will the BLS officially confirm CPI above 5 percent?")
        in_window.exchange = "kalshi"
        no_trigger = _market(mid="KXNOPE", liquidity=350.0, days_out=20.0,
                             question="Higher temperature in NYC?")
        no_trigger.exchange = "kalshi"

        discovery = MagicMock()
        discovery.get_markets = AsyncMock(return_value=[])
        discovery.get_markets_by_close_window = AsyncMock(
            return_value=[in_window, no_trigger])

        pillar = ResolutionLensPillar(
            db=db, settings=settings, discovery=discovery,
            exchange=_exchange(), risk_manager=_risk(),
            pnl_tracker=PnLTracker(db, settings), calibration=calibration,
            analyzer=_analyzer(), exchange_name="kalshi",
            source_tag="resolution_lens_kalshi",
        )
        entered = await pillar.run_once()

        # The close-window fetch was used with the lens's resolution window...
        assert discovery.get_markets_by_close_window.await_count == 1
        args = discovery.get_markets_by_close_window.await_args
        assert args.args[1] > args.args[0] > 0  # (min_ts, max_ts) sane
        # ...the trigger-bearing in-window market traded; the no-trigger one
        # was filtered before any LLM call.
        assert entered == 1
        sig = await db.fetchone(
            "SELECT market_id FROM signals WHERE strategy_source='resolution_lens_kalshi'")
        assert sig is not None and sig["market_id"] == "KXTEST-26AUG-T5"
        row = await db.fetchone(
            "SELECT 1 FROM lens_verdicts WHERE market_id='KXNOPE'")
        assert row is None

        # A discovery WITHOUT the method degrades to the plain scan (no-op).
        class _NoWindow:
            async def get_markets(self, limit=300):
                return []
        poly = ResolutionLensPillar(
            db=db, settings=settings, discovery=_NoWindow(),
            exchange=_exchange(), risk_manager=_risk(),
            pnl_tracker=PnLTracker(db, settings), calibration=calibration,
            analyzer=_analyzer(),
        )
        assert await poly._close_window_candidates() == []

    asyncio.run(run())
