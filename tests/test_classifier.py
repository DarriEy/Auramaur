"""Tests for market category classification."""

from auramaur.strategy.classifier import classify_market, classify_tags


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


# --- 2026-06 production mislabels: description boilerplate poisoning ---
# These exact markets were stored as politics_us and traded live (or false-
# blocked) because keyword scoring ran on resolution boilerplate.

def test_tennis_marker_in_description_is_sports():
    """Tennis matchup with no marker in the question — the description names
    the sport. Stored as politics_us in production via 'primary' boilerplate;
    bought live 2026-06-11 ($4.30)."""
    assert classify_market(
        "Libema Open: Magda Linette vs Mia Pohankova",
        description="This market refers to the tennis match between Magda "
                    "Linette and Mia Pohankova in the Libema Open.",
    ) == "sports"


def test_vs_without_period_is_sports():
    assert classify_market("Libema Open: Magda Linette vs Mia Pohankova") == "sports"


def test_american_league_is_not_politics_us():
    """'American League' must not hit a US-politics marker."""
    assert classify_market(
        "Will Seattle Mariners win the 2026 AL West title?",
        description="This market will resolve according to the team that "
                    "wins the 2026 MLB American League West division.",
    ) == "sports"


def test_primary_listing_boilerplate_not_politics():
    """'primary listing' in IPO boilerplate must not classify as politics_us."""
    assert classify_market(
        "SpaceX IPO closing market cap above $1.8T?",
        description="This market will resolve to Yes if the official closing "
                    "price for SpaceX's market capitalization on its first day "
                    "of primary listing exceeds the threshold.",
    ) not in ("politics_us", "politics_intl")


def test_transcription_boilerplate_not_tech():
    """'AI transcription' resolution boilerplate must not out-vote a political
    question — question hits count double."""
    assert classify_market(
        'Will JD Vance say "Crazy" during Michigan visit?',
        description="Resolution will be according to AI transcription services "
                    "of the full remarks.",
    ) == "politics_us"


def test_esports_not_stolen_by_sports_markers():
    """CS/Dota matchups read like sports ('X vs Y') but must stay esports —
    sports is blocked, esports is not."""
    assert classify_market(
        "Counter-Strike: CYBERSHOKE Esports vs Eternal Fire (BO1)") == "esports"
    assert classify_market("Counter-Strike: FURIA vs TYLOO - Map 2 Winner") == "esports"
    assert classify_market(
        "Game 2: Both Teams Beat Roshan?",
        description="This market refers to Game 2 of the Dota 2 series.",
    ) == "esports"


def test_dutch_parliament_not_politics_us():
    """'Dutch House of Representatives' must not hit the US 'house' marker."""
    assert classify_market(
        "Dutch House of Representatives dissolved in 2026?") == "politics_intl"


# --- Venue tag mapping (authoritative layer above keyword scoring) ---

def test_classify_tags_maps_known_labels():
    assert classify_tags(["Tennis", "2026 Predictions"]) == "sports"
    assert classify_tags(["exchange", "Tech", "Crypto"]) == "crypto"
    assert classify_tags(["France", "Politics", "Macron"]) == "politics_intl"
    assert classify_tags(["US Politics", "Featured"]) == "politics_us"


def test_classify_tags_inconclusive():
    """Bare 'Politics' is ambiguous (US? intl?) — fall through to keywords."""
    assert classify_tags(["Politics", "Featured"]) == ""
    assert classify_tags([]) == ""
    assert classify_tags(None) == ""


def test_classify_tags_esports_beats_sports():
    assert classify_tags(["Sports", "Esports", "CS2"]) == "esports"
