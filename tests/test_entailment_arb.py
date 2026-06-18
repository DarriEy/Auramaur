"""Tests for entailment arbitrage (strategy/entailment_arb.py).

Locks in:
  1. Ladder detection: threshold families (above/below) and Top-N families
     produce mathematically-directed (implier => implied) pairs; unrelated
     questions never pair.
  2. Dead-book guard: illiquid / wide-spread / non-Polymarket markets are
     never believed.
  3. Violations below min_gap are ignored; one shot per pair.
  4. Both-or-nothing legs: a risk rejection on either leg places nothing.
  5. Paper-forcing (config and graduation force_paper).
  6. LLM verification: stored direction is never trusted; verdicts are
     cached; low confidence blocks the trade.
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
from auramaur.strategy.entailment_arb import (
    EntailmentArbPillar,
    kalshi_ladder_pairs,
    ladder_pairs,
    parse_kalshi_ladder,
    parse_threshold,
    parse_topn,
)
from config.settings import Settings


def _kx(ticker, yes, question="What will CPI YoY be?") -> Market:
    return Market(
        id=ticker, exchange="kalshi", ticker=ticker, question=question,
        description="Above X%", active=True,
        outcome_yes_price=yes, outcome_no_price=round(1 - yes, 2),
        liquidity=5000.0, volume=5000.0, spread=0.01,
        end_date=datetime.now(timezone.utc) + timedelta(days=5),
    )


def test_parse_kalshi_ladder():
    assert parse_kalshi_ladder("KXCPIYOY-26NOV-T4.5") == (("kxthr", "KXCPIYOY-26NOV"), 4.5)
    assert parse_kalshi_ladder("KXPAYROLLS-26NOV-T90000") == (("kxthr", "KXPAYROLLS-26NOV"), 90000.0)
    # categorical Fed strikes are NOT monotonic ladders -> excluded
    assert parse_kalshi_ladder("KXFEDDECISION-28JAN-H26") is None
    assert parse_kalshi_ladder("KXFEDDECISION-28JAN-C25") is None
    assert parse_kalshi_ladder("KXSOMETHING") is None


def test_kalshi_ladder_pairs_direction_and_no_cross_series():
    markets = [
        _kx("KXCPIYOY-26NOV-T4.5", 0.30),
        _kx("KXCPIYOY-26NOV-T4.4", 0.40),
        _kx("KXCPIYOY-26NOV-T4.6", 0.20),
        # different series, same "Above %" subtitle — must NOT pair with CPI
        _kx("KXU3-26NOV-T4.5", 0.30, question="What will unemployment be?"),
        # different PERIOD — must not pair with the NOV CPI bins
        _kx("KXCPIYOY-26DEC-T4.5", 0.30),
    ]
    pairs = kalshi_ladder_pairs(markets)
    fams = {(a.ticker, b.ticker) for a, b, _why in pairs}
    # 3 NOV-CPI bins -> 3 ordered pairs, all implier=higher-strike => implied=lower
    assert ("KXCPIYOY-26NOV-T4.6", "KXCPIYOY-26NOV-T4.5") in fams
    assert ("KXCPIYOY-26NOV-T4.6", "KXCPIYOY-26NOV-T4.4") in fams
    assert ("KXCPIYOY-26NOV-T4.5", "KXCPIYOY-26NOV-T4.4") in fams
    assert len(pairs) == 3  # no cross-series / cross-period contamination
    # every implier is the higher strike (so P(implier) <= P(implied) must hold)
    for impl, imp, _ in pairs:
        assert float(impl.ticker.split("-T")[1]) > float(imp.ticker.split("-T")[1])


def test_kalshi_ladder_ignores_non_kalshi():
    poly = _market("p1", "BTC above 70000 on Friday?", 0.5)
    assert kalshi_ladder_pairs([poly]) == []


def _market(mid, question, yes, liquidity=5000.0, spread=0.01,
            exchange="polymarket", days_out=5.0, active=True) -> Market:
    return Market(
        id=mid, exchange=exchange, question=question, active=active,
        outcome_yes_price=yes, outcome_no_price=round(1 - yes, 2),
        liquidity=liquidity, volume=liquidity, spread=spread,
        end_date=datetime.now(timezone.utc) + timedelta(days=days_out),
        clob_token_yes="ty", clob_token_no="tn", category="crypto",
    )


def _settings(**overrides) -> Settings:
    s = Settings()
    s.entailment_arb.enabled = True
    s.entailment_arb.paper = True
    s.entailment_arb.min_gap = 0.04
    s.entailment_arb.llm_enabled = False
    for k, v in overrides.items():
        setattr(s.entailment_arb, k, v)
    return s


def _exchange():
    ex = MagicMock()

    def prepare_order(signal, market, size, is_live):
        token = TokenType.NO if signal.recommended_side == OrderSide.SELL else TokenType.YES
        price = market.outcome_yes_price if token == TokenType.YES else 1 - market.outcome_yes_price
        return Order(
            market_id=market.id, token_id="tok", side=OrderSide.BUY,
            token=token, size=round(size / max(price, 0.01), 2),
            price=round(price, 2), order_type=OrderType.LIMIT,
            dry_run=not is_live,
        )

    ex.prepare_order = MagicMock(side_effect=prepare_order)
    ex.place_order = AsyncMock(side_effect=lambda order: OrderResult(
        order_id=f"ord-{order.market_id}", market_id=order.market_id,
        status="paper" if order.dry_run else "filled",
        filled_size=order.size, filled_price=order.price,
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


def _pillar(db, settings, markets, exchange=None, risk=None, analyzer=None):
    discovery = MagicMock()
    discovery.get_markets = AsyncMock(return_value=markets)
    return EntailmentArbPillar(
        db=db, settings=settings, discovery=discovery,
        exchange=exchange or _exchange(), risk_manager=risk or _risk(),
        pnl_tracker=PnLTracker(db, settings), analyzer=analyzer,
    )


# ----------------------------------------------------------------------
# Ladder parsing
# ----------------------------------------------------------------------

def test_threshold_and_topn_parsing():
    key1, d1, v1 = parse_threshold("Bitcoin above 70,200 on March 20, 10PM ET?")
    key2, d2, v2 = parse_threshold("Bitcoin above 71,000 on March 20, 10PM ET?")
    assert key1 == key2 and d1 == "above" and (v1, v2) == (70200.0, 71000.0)
    # different date -> different family
    key3, _, _ = parse_threshold("Bitcoin above 70,200 on March 21, 10PM ET?")
    assert key3 != key1

    k1, n1 = parse_topn("Will Mac Meissner finish in the Top 10 at the Valero?")
    k2, n2 = parse_topn("Will Mac Meissner finish in the Top 20 at the Valero?")
    assert k1 == k2 and (n1, n2) == (10, 20)
    k3, _ = parse_topn("Will Aaron Rai finish in the Top 20 at the Valero?")
    assert k3 != k1


def test_ladder_pairs_direction():
    above_lo = _market("lo", "Bitcoin above 70,200 on March 20, 10PM ET?", 0.50)
    above_hi = _market("hi", "Bitcoin above 71,400 on March 20, 10PM ET?", 0.60)
    pairs = ladder_pairs([above_lo, above_hi])
    assert len(pairs) == 1
    implier, implied, why = pairs[0]
    assert implier.id == "hi" and implied.id == "lo"  # above(hi) => above(lo)

    below_lo = _market("blo", "Will CPI be below 2.5 in May?", 0.30)
    below_hi = _market("bhi", "Will CPI be below 3.5 in May?", 0.20)
    pairs = ladder_pairs([below_lo, below_hi])
    implier, implied, _ = pairs[0]
    assert implier.id == "blo" and implied.id == "bhi"  # below(lo) => below(hi)

    top5 = _market("t5", "Will X finish in the Top 5 at the Open?", 0.30)
    top20 = _market("t20", "Will X finish in the Top 20 at the Open?", 0.20)
    pairs = ladder_pairs([top5, top20])
    implier, implied, _ = pairs[0]
    assert implier.id == "t5" and implied.id == "t20"  # Top 5 => Top 20

    # Unrelated questions never pair.
    assert ladder_pairs([above_lo, top5]) == []


# ----------------------------------------------------------------------
# Pillar behavior
# ----------------------------------------------------------------------

def _violating_ladder():
    # above(hi) => above(lo), but P(hi)=0.60 > P(lo)=0.50: 10c violation.
    return [
        _market("lo", "Bitcoin above 70,200 on March 20, 10PM ET?", 0.50),
        _market("hi", "Bitcoin above 71,400 on March 20, 10PM ET?", 0.60),
    ]


def test_enters_violating_pair_both_legs_paper():
    async def run():
        db = Database(":memory:")
        await db.connect()
        ex = _exchange()
        pillar = _pillar(db, _settings(), _violating_ladder(), exchange=ex)
        assert await pillar.run_once() == 1

        # Both legs placed, both dry-run (paper-forced).
        assert ex.place_order.await_count == 2
        orders = [c.args[0] for c in ex.place_order.await_args_list]
        assert all(o.dry_run for o in orders)
        # NO on the implier (hi), YES on the implied (lo).
        by_mid = {o.market_id: o for o in orders}
        assert by_mid["hi"].token == TokenType.NO
        assert by_mid["lo"].token == TokenType.YES

        trades = await db.fetchall(
            "SELECT market_id FROM trades WHERE strategy_source='entailment_arb'")
        assert {t["market_id"] for t in trades} == {"hi", "lo"}

        # One shot per pair: second scan does nothing.
        assert await pillar.run_once() == 0
        await db.close()

    asyncio.run(run())


def test_below_min_gap_ignored():
    async def run():
        db = Database(":memory:")
        await db.connect()
        markets = [
            _market("lo", "Bitcoin above 70,200 on March 20, 10PM ET?", 0.50),
            _market("hi", "Bitcoin above 71,400 on March 20, 10PM ET?", 0.52),
        ]
        pillar = _pillar(db, _settings(min_gap=0.04), markets)
        assert await pillar.run_once() == 0
        await db.close()

    asyncio.run(run())


def test_dead_book_guard():
    async def run():
        db = Database(":memory:")
        await db.connect()
        # Violation exists but the implier book is junk in various ways.
        for bad in (
            _market("hi", "Bitcoin above 71,400 on March 20, 10PM ET?", 0.60,
                    liquidity=10.0),                       # illiquid
            _market("hi", "Bitcoin above 71,400 on March 20, 10PM ET?", 0.60,
                    spread=0.40),                          # dead-book spread
            _market("hi", "Bitcoin above 71,400 on March 20, 10PM ET?", 0.60,
                    exchange="kalshi"),                    # wrong venue
            _market("hi", "Bitcoin above 71,400 on March 20, 10PM ET?", 0.60,
                    days_out=0.01),                        # resolving too soon
        ):
            markets = [_market("lo", "Bitcoin above 70,200 on March 20, 10PM ET?", 0.50), bad]
            pillar = _pillar(db, _settings(), markets)
            assert await pillar.run_once() == 0
        await db.close()

    asyncio.run(run())


def test_risk_rejection_on_either_leg_places_nothing():
    async def run():
        db = Database(":memory:")
        await db.connect()
        ex = _exchange()
        pillar = _pillar(db, _settings(), _violating_ladder(), exchange=ex,
                         risk=_risk(approved=False))
        assert await pillar.run_once() == 0
        ex.place_order.assert_not_awaited()
        await db.close()

    asyncio.run(run())


def test_graduation_force_paper_honored():
    async def run():
        db = Database(":memory:")
        await db.connect()
        ex = _exchange()
        settings = _settings(paper=False)  # only graduation forces paper here
        pillar = _pillar(db, settings, _violating_ladder(), exchange=ex,
                         risk=_risk(force_paper=True))
        await pillar.run_once()
        # is_live arg to prepare_order must be False on both legs.
        assert all(c.args[3] is False for c in ex.prepare_order.call_args_list)
        await db.close()

    asyncio.run(run())


def test_llm_verification_gates_fuzzy_pairs():
    async def run():
        db = Database(":memory:")
        await db.connect()
        a = _market("ma", "Will Pezeshkian be out by December 31?", 0.60)
        b = _market("mb", "Will Khorasani be head of state at year end?", 0.40)
        await db.execute(
            "INSERT INTO market_relationships (market_id_a, market_id_b, "
            "relationship_type, strength) VALUES ('ma', 'mb', 'conditional', 0.9)")
        await db.commit()

        # Analyzer says: entailment confirmed a=>b with high confidence.
        analyzer = MagicMock()
        analyzer._call_llm = AsyncMock(return_value=(
            '{"direction": "a_implies_b", "confidence": 0.95, '
            '"counterexample": "none found"}'))
        ex = _exchange()
        pillar = _pillar(db, _settings(llm_enabled=True), [a, b],
                         exchange=ex, analyzer=analyzer)
        assert await pillar.run_once() == 1
        analyzer._call_llm.assert_awaited_once()

        # Verdict is cached: a new pillar over the same pair does not re-call
        # (and does not re-trade — pair already traded).
        analyzer2 = MagicMock()
        analyzer2._call_llm = AsyncMock()
        pillar2 = _pillar(db, _settings(llm_enabled=True), [a, b],
                          analyzer=analyzer2)
        assert await pillar2.run_once() == 0
        analyzer2._call_llm.assert_not_awaited()
        await db.close()

    asyncio.run(run())


def test_conditional_pairs_fetched_outside_scan_window():
    """Pre-computed conditional pairs are niche/low-volume and rarely land in
    the top-N-by-volume scan window; _conditional_pairs must fetch each leg
    directly rather than intersect with by_id, or the strategy stays silent
    (root cause of entailment_arb emitting zero signals)."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            a = _market("ca", "Will A happen?", 0.30)
            b = _market("cb", "Will B happen?", 0.50)
            await db.execute(
                "INSERT INTO market_relationships (market_id_a, market_id_b, "
                "relationship_type, strength) VALUES ('ca', 'cb', 'conditional', 0.9)")
            await db.commit()

            # Scan window (get_markets) returns NEITHER leg — both must be
            # fetched directly via get_market.
            pillar = _pillar(db, _settings(), markets=[])
            pillar._discovery.get_market = AsyncMock(
                side_effect=lambda mid: {"ca": a, "cb": b}.get(mid))

            pairs = await pillar._conditional_pairs(by_id={})
            assert len(pairs) == 1
            assert {pairs[0][0].id, pairs[0][1].id} == {"ca", "cb"}
            assert pillar._discovery.get_market.await_count == 2
        finally:
            await db.close()

    asyncio.run(run())


def test_llm_low_confidence_blocks():
    async def run():
        db = Database(":memory:")
        await db.connect()
        a = _market("ma", "Will Pezeshkian be out by December 31?", 0.60)
        b = _market("mb", "Will Khorasani be head of state at year end?", 0.40)
        await db.execute(
            "INSERT INTO market_relationships (market_id_a, market_id_b, "
            "relationship_type, strength) VALUES ('ma', 'mb', 'conditional', 0.9)")
        await db.commit()

        analyzer = MagicMock()
        analyzer._call_llm = AsyncMock(return_value=(
            '{"direction": "a_implies_b", "confidence": 0.55, '
            '"counterexample": "Pezeshkian could be out via a successor other '
            'than Khorasani"}'))
        ex = _exchange()
        pillar = _pillar(db, _settings(llm_enabled=True), [a, b],
                         exchange=ex, analyzer=analyzer)
        assert await pillar.run_once() == 0
        ex.place_order.assert_not_awaited()
        await db.close()

    asyncio.run(run())
