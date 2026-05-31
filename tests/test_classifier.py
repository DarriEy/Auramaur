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


# --- Substring-collision regressions (word-boundary matching) ---

def test_nflx_ticker_is_not_sports():
    """'nfl' must not match inside the Netflix ticker 'NFLX'."""
    assert classify_market("Will Netflix (NFLX) finish the week above $1000?") != "sports"


def test_eurovision_is_entertainment_not_sports():
    """'Jury Winner' must not pull Eurovision into sports via 'winner'."""
    assert classify_market(
        "Will Australia be the Jury Winner in the 2026 Eurovision Song Contest?"
    ) == "entertainment"


def test_eurovision_top3_not_politics_intl():
    """'eu' must not match inside 'Eurovision' and route to politics_intl."""
    assert classify_market("Will Israel be in the top 3 at Eurovision 2026?") == "entertainment"


def test_warren_is_not_intl_war():
    """'war' must not match inside 'Warren'."""
    assert classify_market("Will Elizabeth Warren win re-election to the Senate?") == "politics_us"


def test_dated_match_still_sports():
    """The 'win on <year>' pattern keeps dated match markets as sports."""
    assert classify_market("Will Poland win on 2026-03-26?") == "sports"
    assert classify_market("Will Liverpool FC win on 2026-04-08?") == "sports"


def test_vs_marker_still_sports():
    assert classify_market("Arkansas Razorbacks vs. Arizona Wildcats") == "sports"
