from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from auramaur.experiments.strategies.oddlot_tender import (
    OddLotTenderDisposition,
    OddLotTenderInputs,
    OddLotTenderRejection,
    OddLotTenderRules,
    assess_oddlot_tender,
)


def _inputs(**changes) -> OddLotTenderInputs:
    values = dict(
        accession="0001", company="Acme Corp", ticker="ACME", form="SC TO-I",
        filed_at="2026-06-09", odd_lot_priority=True,
        requires_record_date_holding=False, tender_price=20.0,
        tender_price_high=22.0, expiration="2026-07-15",
        conditions="none material", confidence=0.95,
    )
    values.update(changes)
    return OddLotTenderInputs(**values)


def _rules(**changes) -> OddLotTenderRules:
    values = dict(min_confidence=0.8, min_premium_pct=2.0,
                  max_position_usd=2500.0)
    values.update(changes)
    return OddLotTenderRules(**values)


def test_alert_contract_preserves_manual_submission_and_expiration():
    proposal = assess_oddlot_tender(_inputs(), _rules()).proposal
    assert proposal is not None
    assert proposal.disposition is OddLotTenderDisposition.ALERT_ONLY
    assert proposal.manual_submission_required is True
    assert proposal.expiration == "2026-07-15"
    assert "submit in TWS before expiration" in proposal.alert_message
    assert "$20.00-$22.00" in proposal.alert_message
    with pytest.raises(FrozenInstanceError):
        proposal.expiration = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"odd_lot_priority": False}, OddLotTenderRejection.NO_PRIORITY),
        ({"requires_record_date_holding": True},
         OddLotTenderRejection.RECORD_DATE_REQUIRED),
        ({"confidence": 0.79}, OddLotTenderRejection.LOW_CONFIDENCE),
        ({"tender_price": float("nan")}, OddLotTenderRejection.INVALID_INPUT),
    ],
)
def test_candidate_gates_fail_closed(changes, reason):
    assessment = assess_oddlot_tender(_inputs(**changes), _rules())
    assert assessment.proposal is None
    assert assessment.rejection is reason


def test_entry_proposal_matches_production_limits():
    proposal = assess_oddlot_tender(
        _inputs(entry_enabled=True, market_price=19.0), _rules(),
    ).proposal
    assert proposal is not None and proposal.entry is not None
    assert proposal.disposition is OddLotTenderDisposition.ENTER
    assert proposal.entry.quantity == 99
    assert proposal.entry.limit_price == 19.0
    assert proposal.entry.conservative_payout == 20.0


def test_entry_dispositions_match_legacy_path():
    thin = assess_oddlot_tender(
        _inputs(entry_enabled=True, market_price=19.9), _rules(),
    ).proposal
    missing = assess_oddlot_tender(
        _inputs(entry_enabled=True, market_price=None), _rules(),
    ).proposal
    expensive = assess_oddlot_tender(
        _inputs(entry_enabled=True, market_price=30.0, tender_price=40.0),
        _rules(max_position_usd=25.0),
    ).proposal
    assert thin and thin.disposition is OddLotTenderDisposition.PREMIUM_TOO_THIN
    assert missing and missing.disposition is OddLotTenderDisposition.NO_MARKET_PRICE
    assert expensive and expensive.disposition is OddLotTenderDisposition.TOO_EXPENSIVE


def test_pure_module_has_no_live_execution_imports():
    path = Path("auramaur/experiments/strategies/oddlot_tender.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(
        name.startswith(("auramaur.exchange", "auramaur.broker", "auramaur.strategy"))
        for name in imports
    )
