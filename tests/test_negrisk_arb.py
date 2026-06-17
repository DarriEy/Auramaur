"""Tests for NegRisk multi-outcome (buy-all-NO) arbitrage detection."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from auramaur.exchange.models import Market
from auramaur.strategy.arbitrage_scanner import ArbitrageScanner


def _leg(idx: int, no_price: float, group: str = "evt_1", neg_risk: bool = True,
         category: str = "geopolitics") -> Market:
    """One outcome of a NegRisk event, priced by its NO price.

    Defaults to the geopolitics category (Polymarket taker fee 0) so fee-agnostic
    detection tests keep their original payout math; pass a fee-bearing category
    to exercise the fee.
    """
    return Market(
        id=f"poly_{group}_{idx}",
        exchange="polymarket",
        question=f"Will candidate {idx} win?",
        category=category,
        outcome_yes_price=round(1.0 - no_price, 4),
        outcome_no_price=no_price,
        clob_token_no=f"tok_no_{idx}",
        clob_token_yes=f"tok_yes_{idx}",
        neg_risk=neg_risk,
        neg_risk_market_id=group,
        volume=10000,
        liquidity=5000,
    )


def _scanner(markets: list[Market], min_profit: float = 1.5) -> ArbitrageScanner:
    disc = AsyncMock()
    disc.get_markets = AsyncMock(return_value=markets)
    return ArbitrageScanner(
        discoveries={"polymarket": disc},
        exchange_fees={"polymarket": 0.0},
        min_profit_after_fees_pct=min_profit,
    )


@pytest.mark.asyncio
async def test_negrisk_arb_detected_when_no_legs_underpriced():
    """3 outcomes with NO prices summing to 1.8 < (N-1)=2, net of the taker fee.
    politics category -> 0.04 taker: guaranteed_payout = 2*(1-0.04)=1.92."""
    legs = [_leg(0, 0.6, category="politics"), _leg(1, 0.6, category="politics"),
            _leg(2, 0.6, category="politics")]
    scanner = _scanner(legs)

    opps = await scanner.scan_negrisk()

    assert len(opps) == 1
    opp = opps[0]
    assert opp.n_outcomes == 3
    assert opp.total_no_cost == pytest.approx(1.8)
    assert opp.guaranteed_payout == pytest.approx(1.92)
    # profit 0.12 on cost 1.8 -> ~6.67%
    assert opp.expected_profit_pct == pytest.approx(0.12 / 1.8 * 100, rel=1e-3)


@pytest.mark.asyncio
async def test_no_arb_when_legs_sum_above_payout():
    """NO prices summing to 2.1 > (N-1)=2 has no edge — nothing flagged."""
    legs = [_leg(0, 0.7), _leg(1, 0.7), _leg(2, 0.7)]
    scanner = _scanner(legs)

    opps = await scanner.scan_negrisk()
    assert opps == []


@pytest.mark.asyncio
async def test_fee_erodes_marginal_arb():
    """A thin arb that clears fee-free is rejected once the category taker fee
    applies. 'other' -> 0.05: net payout 2*(1-0.05)=1.9 < cost 1.965."""
    legs = [_leg(0, 0.655, category="other"), _leg(1, 0.655, category="other"),
            _leg(2, 0.655, category="other")]  # sum 1.965
    disc = AsyncMock()
    disc.get_markets = AsyncMock(return_value=legs)
    scanner = ArbitrageScanner(
        discoveries={"polymarket": disc},
        exchange_fees={"polymarket": 0.05},
        min_profit_after_fees_pct=1.5,
    )
    opps = await scanner.scan_negrisk()
    assert opps == []


@pytest.mark.asyncio
async def test_non_negrisk_markets_ignored():
    """Markets without the neg_risk flag are never grouped into a package."""
    legs = [_leg(0, 0.6, neg_risk=False), _leg(1, 0.6, neg_risk=False)]
    scanner = _scanner(legs)
    opps = await scanner.scan_negrisk()
    assert opps == []


@pytest.mark.asyncio
async def test_separate_events_not_cross_grouped():
    """Legs from different events must not be summed into one bogus package."""
    legs = [
        _leg(0, 0.4, group="evt_A"),
        _leg(0, 0.4, group="evt_B"),
    ]
    scanner = _scanner(legs)
    opps = await scanner.scan_negrisk()
    # Each "group" has only 1 leg -> no package possible.
    assert opps == []
