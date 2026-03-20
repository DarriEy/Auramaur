"""Tests for market category classification."""

from auramaur.strategy.classifier import classify_market


def test_politics_us():
    assert classify_market("Will Trump win the 2024 election?") == "politics_us"


def test_economics():
    assert classify_market("Will the Fed raise interest rates in March?") == "economics"


def test_crypto():
    assert classify_market("Will Bitcoin reach $100k by year end?") == "crypto"


def test_sports():
    assert classify_market("Will the Lakers win the NBA championship?") == "sports"


def test_unknown():
    assert classify_market("Will aliens land on Earth?") == "other"


def test_description_helps():
    cat = classify_market("Will this happen?", description="The unemployment rate and GDP growth")
    assert cat == "economics"
