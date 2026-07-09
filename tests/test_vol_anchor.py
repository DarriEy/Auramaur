"""Vol-anchor pillar: GBM formula pins, question parsing, the sigma blend,
and the standard-rails entry path (risk gate, claim rule, edge floor,
fail-closed data)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from auramaur.db.database import Database
from auramaur.exchange.models import Market
from auramaur.strategy.vol_anchor import (
    VolAnchorPillar,
    blended_sigma,
    detect_asset,
    parse_threshold,
    terminal_above_prob,
    touch_prob,
)


# ---------------------------------------------------------------------------
# Formulas — pinned to the 2026-07-09 ETH verification numbers
# ---------------------------------------------------------------------------


def test_touch_prob_pinned_to_eth_case():
    """ETH spot 1734.54, barrier 3000, T=0.48y — martingale convention.
    sigma 0.65 -> ~0.168, sigma 0.70 -> ~0.193 (the depth agent's 0.16-0.19)."""
    assert touch_prob(1734.54, 3000, 0.65, 0.48) == pytest.approx(0.168, abs=0.01)
    assert touch_prob(1734.54, 3000, 0.70, 0.48) == pytest.approx(0.193, abs=0.01)


def test_touch_prob_monotone_in_sigma_and_t():
    p = [touch_prob(100, 150, s, 0.5) for s in (0.4, 0.6, 0.8)]
    assert p[0] < p[1] < p[2]
    q = [touch_prob(100, 150, 0.6, t) for t in (0.1, 0.5, 1.0)]
    assert q[0] < q[1] < q[2]


def test_touch_prob_downward_barrier_symmetric_shape():
    down = touch_prob(100, 70, 0.6, 0.5)
    assert 0.0 < down < 1.0
    # Already through the barrier region: degenerate inputs return sane values.
    assert touch_prob(100, 100, 0.6, 0.5) == 1.0
    assert touch_prob(0, 100, 0.6, 0.5) == 0.0


def test_terminal_above_pinned_to_eth_case():
    """ETH 4-day: spot 1734.54 > 1600, T=4/365. sigma 0.55 -> ~0.92
    (matches crowd 0.92-0.93 at implied sigma ~52-55)."""
    assert terminal_above_prob(1734.54, 1600, 0.55, 4 / 365) == \
        pytest.approx(0.92, abs=0.015)
    # Touch dominates terminal for the same upward strike.
    assert touch_prob(100, 120, 0.6, 0.3) > terminal_above_prob(100, 120, 0.6, 0.3)


def test_blended_sigma_term_structure():
    """Short horizon ~ the tape; long horizon ~ the anchor — the term
    structure the crowd flattens."""
    near = blended_sigma(0.52, 0.70, 4 / 365, 0.25)
    far = blended_sigma(0.52, 0.70, 0.48, 0.25)
    assert near == pytest.approx(0.52, abs=0.01)
    assert far > 0.66  # mostly anchor
    assert near < far


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_threshold_variants():
    kind, strike, dl = parse_threshold("Will Bitcoin reach $130,000 by December 31, 2026?")
    assert (kind, strike) == ("touch_up", 130000.0)
    assert dl == datetime(2026, 12, 31, tzinfo=timezone.utc)

    kind, strike, _ = parse_threshold("Will Ethereum be above $1,600 on July 13?")
    assert (kind, strike) == ("above", 1600.0)

    kind, strike, _ = parse_threshold("Will the price of Bitcoin be below $60,000 on July 20, 2026?")
    assert (kind, strike) == ("below", 60000.0)

    kind, strike, _ = parse_threshold("Will Solana dip to $80 by August 15, 2026?")
    assert (kind, strike) == ("touch_down", 80.0)

    assert parse_threshold("Will Bitcoin be volatile this month?") is None
    assert parse_threshold("") is None


def test_detect_asset():
    assert detect_asset("Will Bitcoin reach $130,000 by...") == "bitcoin"
    assert detect_asset("Will the price of Ethereum be above...") == "ethereum"
    assert detect_asset("Will XRP dip to $1 by...") == "ripple"
    assert detect_asset("Will WTI Crude Oil hit (HIGH) $80 in July?") is None
    assert detect_asset("Will Tether depeg?") is None


# ---------------------------------------------------------------------------
# Pillar wiring
# ---------------------------------------------------------------------------


def _future(days: int) -> str:
    d = datetime.now(timezone.utc) + timedelta(days=days)
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def _market(mid: str, question: str, yes: float) -> Market:
    return Market(id=mid, question=question, outcome_yes_price=yes,
                  outcome_no_price=round(1 - yes, 2), liquidity=5000.0,
                  volume=10000.0, active=True, exchange="polymarket")


def _settings():
    s = MagicMock()
    cfg = s.vol_anchor
    cfg.enabled = True
    cfg.paper = True
    cfg.interval_seconds = 3600
    cfg.scan_limit = 100
    cfg.min_liquidity = 1000.0
    cfg.min_days = 1.0
    cfg.max_days = 240.0
    cfg.min_edge_pts = 8.0
    cfg.stake_usd = 10.0
    cfg.max_entries_per_cycle = 3
    cfg.realized_window_days = 30
    cfg.tau_years = 0.25
    cfg.long_run_vol = {"bitcoin": 0.50, "ethereum": 0.70}
    cfg.exclude_categories = []
    s.risk.blocked_categories = []
    return s


async def _pillar(tmp_path, markets, spot=1734.54, realized=0.52):
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    discovery = MagicMock()
    discovery.get_markets = AsyncMock(return_value=markets)
    risk = MagicMock()
    decision = MagicMock()
    decision.approved = True
    decision.position_size = 8.0
    decision.reason = ""
    decision.force_paper = False
    risk.evaluate = AsyncMock(return_value=decision)
    calibration = MagicMock()
    calibration.record_prediction = AsyncMock()
    pillar = VolAnchorPillar(
        db=db, settings=_settings(), discovery=discovery, exchange=MagicMock(),
        risk_manager=risk, pnl_tracker=MagicMock(), calibration=calibration)
    pillar._asset_state = AsyncMock(return_value=(spot, realized))

    result = MagicMock()
    result.status = "paper"
    result.reason = ""
    order = MagicMock()
    order.token.value = "YES"
    order.token_id = "tok"
    order.price = 0.13
    order.size = 76.9
    fill = MagicMock()
    fill.is_paper = True
    fill.filled_size = 76.9
    fill.filled_price = 0.13

    def _submit(intent):
        order.market_id = intent.market.id
        result.order = order
        result.result = fill
        return result

    pillar._gateway = MagicMock()
    pillar._gateway.submit = AsyncMock(side_effect=_submit)
    return pillar, db, risk


@pytest.mark.asyncio
async def test_enters_underpriced_long_dated_touch(tmp_path):
    """The ETH shape: calm tape (realized 0.52), anchor 0.70 -> blended sigma
    ~0.67 at ~6mo -> fair ~0.18 vs crowd 0.09 -> ~9 pt edge -> entry.
    (Crowd 0.10 is edge 7.97 — correctly under the 8.0 floor; see
    test_small_edge_skipped for the near-fair case.)"""
    m = _market("m1", f"Will Ethereum reach $3,000 by {_future(175)}?", 0.09)
    pillar, db, risk = await _pillar(tmp_path, [m])
    try:
        assert await pillar.run_once() == 1
        risk.evaluate.assert_awaited_once()
        sig = await db.fetchone(
            "SELECT strategy_source, claude_prob, market_prob FROM signals")
        assert sig["strategy_source"] == "vol_anchor"
        assert sig["claude_prob"] > sig["market_prob"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_small_edge_skipped(tmp_path):
    """Crowd already near fair -> below the 8pt floor -> no entry."""
    m = _market("m1", f"Will Ethereum reach $3,000 by {_future(175)}?", 0.16)
    pillar, db, _ = await _pillar(tmp_path, [m])
    try:
        assert await pillar.run_once() == 0
        pillar._gateway.submit.assert_not_awaited()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_no_data_fails_closed(tmp_path):
    m = _market("m1", f"Will Ethereum reach $3,000 by {_future(175)}?", 0.10)
    pillar, db, _ = await _pillar(tmp_path, [m])
    pillar._asset_state = AsyncMock(return_value=None)
    try:
        assert await pillar.run_once() == 0
        pillar._gateway.submit.assert_not_awaited()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_claimed_market_skipped(tmp_path):
    m = _market("m1", f"Will Ethereum reach $3,000 by {_future(175)}?", 0.10)
    pillar, db, _ = await _pillar(tmp_path, [m])
    try:
        await db.execute(
            """INSERT INTO trades (market_id, timestamp, side, size, price,
               is_paper, order_id, status, exchange, strategy_source)
               VALUES ('m1', datetime('now'), 'BUY', 10, 0.1, 1, 'x', 'paper',
                       'polymarket', 'llm')""")
        await db.commit()
        assert await pillar.run_once() == 0
        pillar._gateway.submit.assert_not_awaited()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_non_crypto_and_unparseable_ignored(tmp_path):
    ms = [
        _market("m1", f"Will WTI Crude Oil hit (HIGH) $80 by {_future(20)}?", 0.15),
        _market("m2", "Will Bitcoin be volatile in July?", 0.50),
    ]
    pillar, db, _ = await _pillar(tmp_path, ms)
    try:
        assert await pillar.run_once() == 0
        pillar._asset_state.assert_not_awaited()  # nothing to price at all
    finally:
        await db.close()
