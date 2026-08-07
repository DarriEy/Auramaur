from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from config.settings import GraduationConfig, LiveAuthorityGrant
from auramaur.risk.graduation import CellDecision, GraduationLadder


def _grant(**overrides):
    values = {
        "venues": ["polymarket"],
        "categories": ["politics_us"],
        "max_stake_usd": 12.0,
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
        "pnl": 0.0, "settlements": 0,
    }))
    return GraduationLadder(db, settings)


@pytest.mark.asyncio
async def test_matching_grant_is_scoped_and_carries_stake_cap():
    ladder = _ladder(_grant(), exempt=["llm"])
    decision = await ladder.decide("llm", "politics_us", "POLYMARKET")
    assert decision.status == "operator_grant"
    assert decision.force_paper is False
    assert decision.authority == "operator_grant"
    assert decision.max_stake_usd == 12.0


@pytest.mark.asyncio
async def test_directional_exemption_without_matching_grant_is_ignored():
    ladder = _ladder(_grant(), exempt=["llm"])
    ladder._compute = AsyncMock(return_value=CellDecision(
        True, 1.0, "unproven", "no evidence"))
    decision = await ladder.decide("llm", "politics_us", "kalshi")
    assert decision.force_paper is True
    assert decision.status == "unproven"


@pytest.mark.asyncio
async def test_structural_exemption_survives():
    ladder = _ladder(exempt=["arbitrage"])
    decision = await ladder.decide("arbitrage", "sports", "polymarket")
    assert decision.status == "exempt"
    assert decision.force_paper is False


@pytest.mark.asyncio
async def test_grant_fails_closed_on_loss_or_review_count():
    loss = _ladder(_grant(), row={"pnl": -40.0, "settlements": 1})
    assert (await loss.decide(
        "llm", "politics_us", "polymarket")).status == "grant_loss_limit"

    review = _ladder(_grant(), row={"pnl": 5.0, "settlements": 25})
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


def test_grant_rejects_invalid_review_window_and_empty_scope():
    with pytest.raises(ValidationError):
        _grant(review_by="2026-08-01")
    with pytest.raises(ValidationError):
        _grant(venues=[])
