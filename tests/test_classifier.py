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


def test_election_winner_boilerplate_not_sports():
    """'winner' in resolution boilerplate must not make an election sports."""
    cat = classify_market(
        "Will the Republicans win the Mississippi Senate race in 2026?",
        description="This market will resolve according to the winner of the 2026 "
                    "midterm Mississippi U.S. Senate election, inclusive of any run-offs.",
    )
    assert cat == "politics_us"


def test_primary_winner_boilerplate_not_sports():
    cat = classify_market(
        "Will Phil Weiser win the 2026 Colorado Governor Democratic primary?",
        description="This market will resolve according to the winner of the "
                    "Democratic Primary for Governor of Colorado.",
    )
    assert cat == "politics_us"


def test_foreign_election_is_politics_intl():
    """Generic governance terms (president/election) must not pull a foreign
    election into politics_us — country context routes it to politics_intl."""
    assert classify_market(
        "Will Jordan Bardella win the 2027 French presidential election?"
    ) == "politics_intl"
    assert classify_market("Rodrigo Paz out as President of Bolivia?") == "politics_intl"
    assert classify_market("Will Germany hold a snap election in 2026?") == "politics_intl"


def test_us_election_stays_politics_us():
    """US-marked elections must still classify as politics_us after the split."""
    assert classify_market("Will Trump win the 2024 election?") == "politics_us"
    assert classify_market(
        "Will the Democratic Party win the NY-21 House seat?"
    ) == "politics_us"


def test_unmarked_governance_defaults_us():
    """Governance terms with no country marker default to politics_us (the bulk
    of prediction-market election markets), preserving prior behavior."""
    assert classify_market("Who will win the presidential election?") == "politics_us"


def test_commodity_not_politics():
    """A commodities/price market must not land in any politics bucket."""
    assert classify_market("Will WTI Crude Oil close above $88 on Friday?") not in (
        "politics_us",
        "politics_intl",
    )


def test_ncaa_is_sports():
    assert classify_market("NCAA Tournament: No. 15 seed to pull off an upset?") == "sports"


def test_individual_sports_stay_blocked():
    """Individual-event sports (no team-vs marker) must classify as sports so
    the block keeps them out, not leak into the tradeable 'other' bucket."""
    for q in [
        "Will Scottie Scheffler win the 2026 Masters?",
        "Who will win Wimbledon 2026 men's singles?",
        "Will Max Verstappen win the 2026 Monaco Grand Prix?",
        "Will Novak Djokovic win the US Open?",
        "Will Team USA win gold in the 2026 Olympics?",
        "Will the winner of the 2026 Tour de France be from Slovenia?",
    ]:
        assert classify_market(q) == "sports", q
