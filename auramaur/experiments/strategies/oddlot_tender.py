"""Pure proposal formation for manually submitted odd-lot tenders."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class OddLotTenderRejection(str, Enum):
    INVALID_INPUT = "invalid_input"
    NO_PRIORITY = "no_priority"
    RECORD_DATE_REQUIRED = "record_date_required"
    LOW_CONFIDENCE = "low_confidence"


class OddLotTenderDisposition(str, Enum):
    ALERT_ONLY = "alerted"
    NO_MARKET_PRICE = "alerted_no_price"
    PREMIUM_TOO_THIN = "premium_too_thin"
    TOO_EXPENSIVE = "too_expensive"
    ENTER = "enter"


@dataclass(frozen=True)
class OddLotTenderRules:
    min_confidence: float
    min_premium_pct: float
    max_position_usd: float


@dataclass(frozen=True)
class OddLotTenderInputs:
    accession: str
    company: str
    ticker: str
    form: str
    filed_at: str
    odd_lot_priority: bool
    requires_record_date_holding: bool
    tender_price: float
    tender_price_high: float
    expiration: str
    conditions: str
    confidence: float
    entry_enabled: bool = False
    market_price: float | None = None


@dataclass(frozen=True)
class TenderEntryProposal:
    ticker: str
    quantity: int
    limit_price: float
    conservative_payout: float
    premium_pct: float


@dataclass(frozen=True)
class OddLotTenderProposal:
    accession: str
    ticker: str
    alert_message: str
    expiration: str
    manual_submission_required: bool
    disposition: OddLotTenderDisposition
    entry: TenderEntryProposal | None = None


@dataclass(frozen=True)
class OddLotTenderAssessment:
    proposal: OddLotTenderProposal | None
    rejection: OddLotTenderRejection | None = None


def assess_oddlot_tender(
    inputs: OddLotTenderInputs,
    rules: OddLotTenderRules,
) -> OddLotTenderAssessment:
    """Form an alert/manual-tender proposal and optional equity entry."""
    values = (
        inputs.tender_price,
        inputs.tender_price_high,
        inputs.confidence,
        rules.min_confidence,
        rules.min_premium_pct,
        rules.max_position_usd,
    )
    if (
        not inputs.accession
        or not inputs.company
        or any(not math.isfinite(value) for value in values)
        or inputs.tender_price < 0.0
        or inputs.tender_price_high < 0.0
        or not 0.0 <= inputs.confidence <= 1.0
        or not 0.0 <= rules.min_confidence <= 1.0
        or rules.max_position_usd < 0.0
    ):
        return OddLotTenderAssessment(None, OddLotTenderRejection.INVALID_INPUT)
    if not inputs.odd_lot_priority:
        return OddLotTenderAssessment(None, OddLotTenderRejection.NO_PRIORITY)
    if inputs.requires_record_date_holding:
        return OddLotTenderAssessment(None, OddLotTenderRejection.RECORD_DATE_REQUIRED)
    if inputs.confidence < rules.min_confidence:
        return OddLotTenderAssessment(None, OddLotTenderRejection.LOW_CONFIDENCE)

    ticker = inputs.ticker or "?"
    message = (
        f"ODD-LOT TENDER: {inputs.company} [{ticker}] {inputs.form} filed "
        f"{inputs.filed_at} — price ${inputs.tender_price:.2f}"
        + (
            f"-${inputs.tender_price_high:.2f}"
            if inputs.tender_price_high > inputs.tender_price
            else ""
        )
        + f", expires {inputs.expiration or '?'}. "
        f"Conditions: {inputs.conditions or 'none noted'}. "
        "TENDERING IS MANUAL — submit in TWS before expiration."
    )

    disposition = OddLotTenderDisposition.ALERT_ONLY
    entry = None
    if inputs.entry_enabled and ticker != "?":
        price = inputs.market_price
        if price is None or not math.isfinite(price) or price <= 0.0:
            disposition = OddLotTenderDisposition.NO_MARKET_PRICE
        else:
            premium_pct = (inputs.tender_price - price) / price * 100.0
            if premium_pct < rules.min_premium_pct:
                disposition = OddLotTenderDisposition.PREMIUM_TOO_THIN
            else:
                quantity = min(99, int(rules.max_position_usd // price))
                if quantity < 1:
                    disposition = OddLotTenderDisposition.TOO_EXPENSIVE
                else:
                    disposition = OddLotTenderDisposition.ENTER
                    entry = TenderEntryProposal(
                        ticker=ticker,
                        quantity=quantity,
                        limit_price=price,
                        conservative_payout=inputs.tender_price,
                        premium_pct=premium_pct,
                    )

    return OddLotTenderAssessment(OddLotTenderProposal(
        accession=inputs.accession,
        ticker=ticker,
        alert_message=message,
        expiration=inputs.expiration,
        manual_submission_required=True,
        disposition=disposition,
        entry=entry,
    ))
