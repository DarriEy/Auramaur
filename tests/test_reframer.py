"""Tests for the reframing parser — converting options to binary questions."""

import pytest
from datetime import datetime, timezone

from auramaur.nlp.reframer import (
    OptionContract,
    ReframeType,
    ReframedMarket,
    TradeMapping,
    reframe_option_as_binary,
    reframe_earnings_binary,
    select_interesting_strikes,
)


def _make_option(
    symbol: str = "AAPL",
    strike: float = 200.0,
    right: str = "C",
    delta: float = 0.45,
    mid_price: float = 5.50,
    underlying_price: float = 195.0,
    implied_vol: float = 0.30,
    volume: int = 500,
    days_to_expiry: int = 30,
) -> OptionContract:
    from datetime import timedelta
    expiry = datetime.now(timezone.utc) + timedelta(days=days_to_expiry)
    return OptionContract(
        symbol=symbol,
        strike=strike,
        expiry=expiry,
        right=right,
        delta=delta,
        mid_price=mid_price,
        bid=mid_price - 0.10,
        ask=mid_price + 0.10,
        implied_vol=implied_vol,
        volume=volume,
        open_interest=1000,
        underlying_price=underlying_price,
        con_id=12345,
    )


class TestReframeCallOption:
    def test_call_creates_price_above_question(self):
        opt = _make_option(right="C", strike=200.0, symbol="AAPL")
        rm = reframe_option_as_binary(opt)

        assert rm.reframe_type == ReframeType.PRICE_ABOVE
        assert "AAPL" in rm.market.question
        assert "above" in rm.market.question.lower()
        assert "$200" in rm.market.question

    def test_call_delta_becomes_yes_price(self):
        opt = _make_option(delta=0.45)
        rm = reframe_option_as_binary(opt)

        assert rm.market.outcome_yes_price == pytest.approx(0.45, abs=0.01)
        assert rm.market.outcome_no_price == pytest.approx(0.55, abs=0.01)

    def test_call_trade_mapping(self):
        opt = _make_option(right="C")
        rm = reframe_option_as_binary(opt)

        assert rm.trade_mapping.buy_yes_action == "buy_call"
        assert rm.trade_mapping.sell_yes_action == "buy_put"

    def test_call_market_fields(self):
        opt = _make_option(right="C", symbol="AAPL")
        rm = reframe_option_as_binary(opt)

        assert rm.market.exchange == "ibkr"
        assert rm.market.ticker == "AAPL"
        assert rm.market.category == "options"
        assert rm.market.active is True
        assert rm.market.id.startswith("IB:AAPL:")


class TestReframePutOption:
    def test_put_creates_price_below_question(self):
        opt = _make_option(right="P", strike=190.0, delta=-0.35)
        rm = reframe_option_as_binary(opt)

        assert rm.reframe_type == ReframeType.PRICE_BELOW
        assert "below" in rm.market.question.lower()
        assert "$190" in rm.market.question

    def test_put_delta_magnitude_becomes_yes_price(self):
        opt = _make_option(right="P", delta=-0.35)
        rm = reframe_option_as_binary(opt)

        assert rm.market.outcome_yes_price == pytest.approx(0.35, abs=0.01)

    def test_put_trade_mapping(self):
        opt = _make_option(right="P")
        rm = reframe_option_as_binary(opt)

        assert rm.trade_mapping.buy_yes_action == "buy_put"
        assert rm.trade_mapping.sell_yes_action == "buy_call"


class TestDeltaClamping:
    def test_very_high_delta_clamped(self):
        opt = _make_option(delta=0.99)
        rm = reframe_option_as_binary(opt)
        assert rm.market.outcome_yes_price == 0.99

    def test_very_low_delta_clamped(self):
        opt = _make_option(delta=0.005)
        rm = reframe_option_as_binary(opt)
        assert rm.market.outcome_yes_price == 0.01


class TestReframeEarnings:
    def test_earnings_binary(self):
        opt = _make_option(strike=195.0, delta=0.50)
        rm = reframe_earnings_binary(
            symbol="AAPL",
            earnings_date=opt.expiry,
            underlying_price=195.0,
            implied_move_pct=5.0,
            atm_call=opt,
        )

        assert rm is not None
        assert rm.reframe_type == ReframeType.EARNINGS_BEAT
        assert "beat earnings" in rm.market.question.lower()
        assert rm.market.exchange == "ibkr"
        assert rm.market.category == "earnings"

    def test_earnings_no_atm_returns_none(self):
        rm = reframe_earnings_binary(
            symbol="AAPL",
            earnings_date=datetime.now(timezone.utc),
            underlying_price=195.0,
            implied_move_pct=5.0,
            atm_call=None,
        )
        assert rm is None


class TestSelectInterestingStrikes:
    def test_filters_extreme_deltas(self):
        options = [
            _make_option(delta=0.02, strike=250),  # Too far OTM
            _make_option(delta=0.98, strike=150),  # Too deep ITM
            _make_option(delta=0.45, strike=200),  # Good
        ]
        selected = select_interesting_strikes(options, 195.0)
        assert len(selected) == 1
        assert selected[0].strike == 200

    def test_filters_bad_expiry(self):
        options = [
            _make_option(days_to_expiry=3, strike=200),   # Too soon
            _make_option(days_to_expiry=120, strike=200),  # Too far
            _make_option(days_to_expiry=30, strike=200),   # Good
        ]
        selected = select_interesting_strikes(options, 195.0)
        assert len(selected) == 1
        assert selected[0].strike == 200

    def test_respects_max_contracts(self):
        options = [
            _make_option(delta=0.3 + i * 0.05, strike=190 + i * 5)
            for i in range(20)
        ]
        selected = select_interesting_strikes(options, 195.0, max_contracts=5)
        assert len(selected) <= 5

    def test_prefers_atm(self):
        options = [
            _make_option(delta=0.15, strike=220),  # Far OTM
            _make_option(delta=0.50, strike=195),  # ATM
            _make_option(delta=0.85, strike=170),  # Deep ITM
        ]
        selected = select_interesting_strikes(options, 195.0, max_contracts=1)
        assert len(selected) == 1
        assert selected[0].delta == pytest.approx(0.50, abs=0.01)


class TestMarketIdDeterministic:
    def test_same_option_same_id(self):
        opt1 = _make_option(symbol="AAPL", strike=200, right="C", days_to_expiry=30)
        opt2 = _make_option(symbol="AAPL", strike=200, right="C", days_to_expiry=30)
        # Same expiry date
        opt2.expiry = opt1.expiry

        rm1 = reframe_option_as_binary(opt1)
        rm2 = reframe_option_as_binary(opt2)

        assert rm1.market.id == rm2.market.id

    def test_different_strike_different_id(self):
        opt1 = _make_option(strike=200)
        opt2 = _make_option(strike=210)
        opt2.expiry = opt1.expiry

        rm1 = reframe_option_as_binary(opt1)
        rm2 = reframe_option_as_binary(opt2)

        assert rm1.market.id != rm2.market.id
