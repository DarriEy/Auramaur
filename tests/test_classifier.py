"""Tests for market category classification."""

from auramaur.strategy.classifier import (
    blocked_category_hit,
    classify_market,
    classify_tags,
)


def test_blocked_category_hit_stored_empty_and_mislabeled():
    """The pre-filter helper mirrors the gateway: block on the stored label OR a
    fresh classification, so it catches an empty label (the bypass) AND a
    stale/mislabeled one (stored 'other' that is really sports)."""
    blocked = {"sports", "politics_us"}
    # stored label is blocked
    assert blocked_category_hit(blocked, "Team A vs Team B", "", "sports") == "sports"
    # empty label, but the question classifies into a blocked category
    assert blocked_category_hit(
        blocked, "Will Republicans win the Senate?", "", "") == "politics_us"
    # mislabeled 'other' that is really sports -> caught via fresh classify
    assert blocked_category_hit(
        blocked, "United States vs. Paraguay: USA O/U 2.5", "", "other") == "sports"
    # a genuinely allowed market returns None
    assert blocked_category_hit(blocked, "Will OpenAI release GPT-6?", "", "tech") is None
    # empty blocked set never blocks
    assert blocked_category_hit(set(), "Team A vs Team B", "", "sports") is None


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


def test_subnational_and_byelection_are_politics_intl():
    """Sub-national/Commonwealth markers without a country word used to default
    to politics_us (governance term + no intl marker). Alberta/Quebec/Scotland
    separatist votes and UK by-elections must route to politics_intl."""
    assert classify_market("Will Alberta vote for independence in 2026?") == "politics_intl"
    assert classify_market(
        "Will Andy Burnham win the 2026 Makerfield by-election?") == "politics_intl"
    assert classify_market("Will Scotland vote for independence?") == "politics_intl"
    assert classify_market("Will Quebec hold an independence referendum?") == "politics_intl"


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


def test_entertainment_structural_patterns_2026_06_25_audit():
    """Live-gate audit (2026-06-25): celebrity weddings, movie/TV casting,
    album/box-office and TV-season markets were stored as 'other' and dodged the
    entertainment block. The classifier must now catch them."""
    cases = [
        "Will Brittany Mahomes be a Bridesmaid for the wedding of Travis Kelce and Taylor Swift?",
        "Who will perform the next James Bond Song?",
        "When will The Last of Us Season 3 be released?",
        "Will Johnny Depp be casted in the next Pirates of the Caribbean?",
        "Will Glen Powell be casted in the next Miami Vice?",
        "New Playboi Carti Album before GTA VI?",
    ]
    for q in cases:
        assert classify_market(q) == "entertainment", q


def test_unclassified_matchups_default_to_blocked_without_enumeration():
    """The sustainable fix: a head-to-head match-up with no edge-category signal
    is a live-event outcome and defaults to the blocked 'sports' bucket — we do
    NOT enumerate every game/league/roster. Call of Duty, an invented sport, and
    a tennis fixture all land in a BLOCKED category (sports or esports) with no
    game name in the keyword lists."""
    blocked = {"sports", "esports"}
    assert classify_market(
        "Call of Duty: Los Angeles Thieves vs Carolina Royal Ravens") in blocked
    assert classify_market("Madeupball: Team Alpha vs Team Beta") in blocked
    assert classify_market("Perugia: Pablo Llamas Ruiz vs Michele Ribecai") in blocked


def test_vs_no_longer_steals_edge_matchups():
    """Moving 'vs' out of the eager priority check: edge match-ups now score
    their real category instead of being stolen into sports."""
    assert classify_market("Trump vs Biden 2028 election?") == "politics_us"
    assert classify_market("Bitcoin vs Ethereum market cap by 2027?") == "crypto"


def test_cricket_caught_via_venue_tag():
    """The authoritative path is the VENUE TAG, not question keywords. A bare
    'Will New Zealand win?' is unclassifiable from text, but Polymarket tags it
    'Cricket' -> classify_tags maps that to sports (blocked)."""
    from auramaur.strategy.classifier import classify_tags
    assert classify_tags(["Cricket", "International", "Sports"]) == "sports"


def test_new_patterns_dont_false_block_legit_categories():
    """The new entertainment/sports markers must not steal weather/politics/
    econ/crypto markets."""
    assert classify_market("Will a Category 5 hurricane make landfall this season?") != "entertainment"
    assert classify_market("Will Trump win the 2028 election?") not in ("entertainment", "sports")
    assert classify_market("Will the Fed cut rates in Q3?") == "economics"
    assert classify_market("Will Bitcoin hit $200k?") == "crypto"
