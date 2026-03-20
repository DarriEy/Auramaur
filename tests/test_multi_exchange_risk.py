"""Tests for multi-exchange risk enforcement."""

import pytest

from auramaur.exchange.models import Market, Order, OrderSide


def test_market_default_exchange():
    """Market model should default to polymarket exchange."""
    m = Market(id="x", question="?")
    assert m.exchange == "polymarket"


def test_market_kalshi_exchange():
    """Market model should accept kalshi exchange."""
    m = Market(id="KXTEST", exchange="kalshi", ticker="KXTEST", question="?")
    assert m.exchange == "kalshi"
    assert m.ticker == "KXTEST"
    assert m.condition_id == ""  # Not required for Kalshi


def test_order_default_exchange():
    """Order model should default to polymarket exchange."""
    o = Order(market_id="x", side=OrderSide.BUY, size=10, price=0.5)
    assert o.exchange == "polymarket"


def test_order_kalshi_exchange():
    """Order model should accept kalshi exchange."""
    o = Order(market_id="KXTEST", exchange="kalshi", side=OrderSide.BUY, size=10, price=0.5)
    assert o.exchange == "kalshi"


def test_backward_compat_condition_id():
    """condition_id should default to empty string (was implicitly required)."""
    m = Market(id="test", question="Test?")
    assert m.condition_id == ""


def test_backward_compat_ticker():
    """ticker should default to empty string."""
    m = Market(id="test", question="Test?")
    assert m.ticker == ""
