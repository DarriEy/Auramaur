"""Gamma category resolution must trust the venue before the keyword
classifier.

Bare ``/markets/{id}`` objects carry no events/tags — Polymarket's
taxonomy rides only on the parent-event expansion — so the only venue
signal in the bare shape is ``sportsMarketType``. Ignoring it sent "The
Hundred, Women: Manchester Super Giants vs Welsh Fire" to the keyword
fallback, which read "Welsh" as Wales and filed live cricket under
politics_intl (2026-08-05; siblings landed as 'legal' and agent arms
paper-traded them there)."""

from auramaur.exchange.gamma import GammaClient


def _parse(data: dict):
    client = object.__new__(GammaClient)
    return client._parse_market(data)


def test_bare_sports_market_type_wins_over_keyword_fallback():
    # Mirrors the real bare /markets/3126999 response shape: no events,
    # no tags, no category — only sportsMarketType identifies it.
    market = _parse({
        "id": "3126999",
        "question": "The Hundred, Women: Welsh Fire vs Trent Rockets",
        "slug": "crichundredw-wel-tre-2026-07-29",
        "sportsMarketType": "moneyline",
        "outcomePrices": "[\"0.455\", \"0.545\"]",
    })
    assert market is not None
    assert market.category == "sports"


def test_curated_event_tags_still_outrank_sports_market_type():
    # An esports moneyline: tags say esports, sportsMarketType says only
    # "sports" — the more specific curated taxonomy must win.
    market = _parse({
        "id": "42",
        "question": "CS2 Major: FaZe vs Vitality",
        "sportsMarketType": "moneyline",
        "events": [{"id": "7", "tags": [{"label": "Esports"}]}],
    })
    assert market is not None
    assert market.category == "esports"


def test_untagged_non_sports_market_still_falls_through():
    market = _parse({
        "id": "43",
        "question": "Will the ECB cut rates in September?",
    })
    assert market is not None
    assert market.category == ""
