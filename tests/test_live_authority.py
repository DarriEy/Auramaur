from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from config.settings import GraduationConfig, LiveAuthorityGrant
from auramaur.db.database import Database
from auramaur.risk.graduation import GraduationLadder


def _grant(**overrides):
    values = {
        "venues": ["polymarket"],
        "categories": ["politics_us"],
        "max_stake_usd": 12.0,
        "max_open_notional_usd": 50.0,
        "granted_at": "2026-08-01",
        "review_by": "2099-12-31",
        "evidence_basis": "bounded operator trial",
        "stop_loss_usd": 40.0,
        "review_after_settlements": 25,
    }
    values.update(overrides)
    return LiveAuthorityGrant(**values)


def _ladder(grant=None, *, exempt=None, row=None):
    cfg = GraduationConfig(
        mode="enforce",
        exempt_strategies=exempt or [],
        live_authority={"llm": [grant]} if grant else {},
    )
    settings = SimpleNamespace(
        graduation=cfg,
        risk=SimpleNamespace(category_gate_exempt_strategies=["arbitrage"]),
    )
    db = SimpleNamespace(fetchone=AsyncMock(return_value=row or {
        "pnl": 0.0, "realizations": 0, "notional": 0.0, "positions": 0,
    }))
    return GraduationLadder(db, settings)


@pytest.mark.asyncio
async def test_matching_grant_is_scoped_and_carries_stake_cap():
    ladder = _ladder(_grant())
    decision = await ladder.decide("llm", "politics_us", "POLYMARKET")
    assert decision.status == "operator_grant"
    assert decision.force_paper is False
    assert decision.authority == "operator_grant"
    assert decision.max_stake_usd == 12.0


def test_directional_exemption_is_rejected_at_config_load():
    with pytest.raises(ValidationError, match="live_authority"):
        GraduationConfig(exempt_strategies=["llm"])

@pytest.mark.asyncio
async def test_structural_exemption_survives():
    ladder = _ladder(exempt=["arbitrage"])
    decision = await ladder.decide("arbitrage", "sports", "polymarket")
    assert decision.status == "exempt"
    assert decision.force_paper is False


@pytest.mark.asyncio
async def test_grant_fails_closed_on_loss_or_review_count():
    loss = _ladder(_grant(), row={"pnl": -40.0, "realizations": 1, "notional": 0.0})
    assert (await loss.decide(
        "llm", "politics_us", "polymarket")).status == "grant_loss_limit"

    review = _ladder(_grant(), row={"pnl": 5.0, "realizations": 25, "notional": 0.0})
    assert (await review.decide(
        "llm", "politics_us", "polymarket")).status == "grant_review_due"


@pytest.mark.asyncio
async def test_expired_grant_fails_closed_without_ledger_lookup():
    ladder = _ladder(_grant(review_by="2026-08-02"))
    decision = await ladder.decide("llm", "politics_us", "polymarket")
    assert decision.status == "grant_expired"
    assert decision.force_paper is True
    ladder._db.fetchone.assert_not_awaited()


@pytest.mark.asyncio
async def test_grant_fails_closed_when_evidence_store_is_unavailable():
    ladder = _ladder(_grant())
    ladder._db.fetchone.side_effect = RuntimeError("database unavailable")
    decision = await ladder.decide("llm", "politics_us", "polymarket")
    assert decision.status == "grant_evidence_unavailable"
    assert decision.force_paper is True
    assert decision.max_stake_usd == 12.0



@pytest.mark.asyncio
async def test_off_keeps_grant_bounds_and_observe_does_not_apply_them():
    off = _ladder(_grant(), row={
        "pnl": -40.0, "realizations": 1, "notional": 0.0})
    off._settings.graduation.mode = "off"
    assert (await off.decide(
        "llm", "politics_us", "polymarket")).status == "grant_loss_limit"

    observe = _ladder(_grant(), row={
        "pnl": -40.0, "realizations": 1, "notional": 0.0})
    observe._settings.graduation.mode = "observe"
    observe._compute = AsyncMock(return_value=GraduationLadder.__dict__.get(
        "_unused", None))
    # Use an explicit evidence decision; observe must never query grant SQL.
    from auramaur.risk.graduation import CellDecision
    observe._compute = AsyncMock(return_value=CellDecision(
        True, 0.0, "unproven", "no evidence"))
    decision = await observe.decide("llm", "politics_us", "polymarket")
    assert decision.status == "observe:unproven"
    assert decision.force_paper is False
    observe._db.fetchone.assert_not_awaited()


@pytest.mark.asyncio
async def test_venue_is_required_when_strategy_has_grants():
    decision = await _ladder(_grant()).decide("llm", "politics_us")
    assert decision.status == "grant_venue_required"
    assert decision.force_paper is True


@pytest.mark.asyncio
async def test_real_db_counts_sell_and_blank_label_loss_fail_closed():
    db = Database(":memory:")
    await db.connect()
    try:
        await db.execute(
            """INSERT INTO pnl_ledger
               (market_id,venue,category,strategy_source,kind,token,qty,pnl,
                is_paper,source_ref,realized_at)
               VALUES ('m1','polymarket','','llm','sell','YES',1,-5,0,
                       'test:sell','2026-08-02T00:00:00')"""
        )
        cfg = GraduationConfig(
            mode="enforce",
            live_authority={"llm": [_grant(
                stop_loss_usd=4, review_after_settlements=1)]},
        )
        settings = SimpleNamespace(
            graduation=cfg,
            risk=SimpleNamespace(
                category_gate_exempt_strategies=["arbitrage"],
                live_categories_only={"llm": ["politics_us"]},
                live_venues_only={"llm": ["polymarket"]},
                allowed_categories_live=["politics_us"],
                allowed_categories_live_extra={},
            ),
        )
        decision = await GraduationLadder(db, settings).decide(
            "llm", "politics_us", "polymarket")
        assert decision.status == "grant_loss_limit"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_grant_blocks_when_open_notional_is_exhausted():
    ladder = _ladder(_grant(max_open_notional_usd=50), row={
        "pnl": 0.0, "realizations": 0, "notional": 50.0})
    decision = await ladder.decide("llm", "politics_us", "polymarket")
    assert decision.status == "grant_open_limit"
    assert decision.force_paper is True
    assert decision.max_stake_usd == 0.0


def test_startup_crosscheck_rejects_inert_scope():
    ladder = _ladder(_grant(categories=["typo_category"]))
    ladder._settings.risk.live_categories_only = {"llm": ["politics_us"]}
    ladder._settings.risk.live_venues_only = {"llm": ["polymarket"]}
    ladder._settings.risk.allowed_categories_live = ["politics_us"]
    ladder._settings.risk.allowed_categories_live_extra = {}
    ladder._settings.graduation.min_markets_overrides = {"llm": 1}
    assert ladder.authority_crosscheck() == [
        "llm grant categories are not live-eligible: ['typo_category']"
    ]

def test_grant_rejects_invalid_review_window_and_empty_scope():
    with pytest.raises(ValidationError):
        _grant(review_by="2026-08-01")
    with pytest.raises(ValidationError):
        _grant(venues=[])
