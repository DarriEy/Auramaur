"""Bluesky source: failures must be LOUD and auth must gate correctly.

The 2026-08-04 finding: 29,743 fetches over 14 days returned zero items
because the public appview 403s datacenter traffic and the old code
swallowed it at DEBUG — a source failing 100% of the time looked healthy
in the funnel. These tests pin the two behaviors that prevent a repeat."""

import pytest

from auramaur.data_sources.bluesky import BlueskyFetchError, BlueskySource


def test_auth_gates_on_both_credentials():
    assert not BlueskySource()._authed
    assert not BlueskySource(identifier="me.bsky.social")._authed
    assert not BlueskySource(app_password="xxxx")._authed
    assert BlueskySource(identifier="me.bsky.social",
                         app_password="xxxx")._authed


@pytest.mark.asyncio
async def test_fetch_propagates_endpoint_failure_loudly(monkeypatch):
    """A non-200 must escape fetch() so the aggregator records
    status='error' — never a healthy-looking empty."""
    src = BlueskySource()

    async def boom(session, params, retried=False):
        raise BlueskyFetchError("search returned 403")

    monkeypatch.setattr(src, "_search", boom)
    with pytest.raises(BlueskyFetchError):
        await src.fetch("Trump pardons")


@pytest.mark.asyncio
async def test_fetch_parses_posts_and_filters_low_engagement(monkeypatch):
    src = BlueskySource()
    canned = {"posts": [
        {"record": {"text": "Big political development in the CA-21 race",
                    "createdAt": "2026-08-04T12:00:00Z"},
         "author": {"handle": "reporter.bsky.social"},
         "uri": "at://did:plc:abc/app.bsky.feed.post/xyz",
         "likeCount": 40, "repostCount": 10, "replyCount": 3},
        {"record": {"text": "meh"}, "author": {"handle": "rando"},
         "uri": "at://did:plc:zzz/app.bsky.feed.post/qqq",
         "likeCount": 0, "repostCount": 0, "replyCount": 0},
    ]}

    async def fake(session, params, retried=False):
        return canned

    monkeypatch.setattr(src, "_search", fake)
    items = await src.fetch("CA-21 special election")
    assert len(items) == 1
    assert items[0].source == "bluesky"
    assert "CA-21" in items[0].title
    assert items[0].url.startswith("https://bsky.app/profile/did:plc:abc/")


def _post(text, likes=0, reposts=0, replies=0, created="2026-08-04T12:00:00Z",
          uri="at://did:plc:abc/app.bsky.feed.post/x1"):
    return {"record": {"text": text, "createdAt": created},
            "author": {"handle": "someone.bsky.social"}, "uri": uri,
            "likeCount": likes, "repostCount": reposts, "replyCount": replies}


async def _fetch_with(monkeypatch, posts, q="test query"):
    src = BlueskySource()

    async def fake(session, params, retried=False):
        fake.params = params
        return {"posts": posts}

    monkeypatch.setattr(src, "_search", fake)
    items = await src.fetch(q)
    return items, fake.params


@pytest.mark.asyncio
async def test_noise_gates_2026_08_04(monkeypatch):
    """The day-one noise: drive-by one-liners cleared the old gate on two
    likes; link-only posts carried no content. Short text now needs REAL
    engagement — and terse breaking posts from followed accounts survive."""
    items, _ = await _fetch_with(monkeypatch, [
        _post("Winning is the only thing.", likes=2),          # dropped now
        _post("youtu.be/BuG0hcJHDMI?...", likes=9, replies=3), # link-only
        _post("BREAKING: Smith resigns", likes=500),           # terse + real
        _post("A substantial multi-sentence take on the race that easily "
              "clears the length floor without any engagement at all."),
    ])
    titles = [i.title for i in items]
    assert "BREAKING: Smith resigns" in titles
    assert any(t.startswith("A substantial") for t in titles)
    assert len(items) == 2


@pytest.mark.asyncio
async def test_repost_chains_collapse_to_one_slot(monkeypatch):
    items, _ = await _fetch_with(monkeypatch, [
        _post("Same viral take on the election, word for word repeated.",
              likes=40, uri="at://did:plc:a/app.bsky.feed.post/1"),
        _post("Same viral take on the election, word for word REPEATED.",
              likes=3, uri="at://did:plc:b/app.bsky.feed.post/2"),
    ])
    assert len(items) == 1


@pytest.mark.asyncio
async def test_relevance_blends_freshness_with_engagement(monkeypatch):
    """A five-minute-old breaking post must outrank a nine-day-old post of
    equal engagement — the old engagement-only score priced the breaking
    moment at exactly zero."""
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    fresh = (now - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    stale = (now - timedelta(days=9)).strftime("%Y-%m-%dT%H:%M:%SZ")
    items, params = await _fetch_with(monkeypatch, [
        _post("Fresh substantial reporting on the race, no engagement yet, "
              "posted minutes ago by a correspondent.", created=fresh,
              uri="at://did:plc:a/app.bsky.feed.post/f"),
        _post("Stale substantial reporting on the race from last week with "
              "identical zero engagement to compare.", created=stale,
              uri="at://did:plc:b/app.bsky.feed.post/s"),
    ])
    by_title = {i.title[:5]: i.relevance_score for i in items}
    assert by_title["Fresh"] > by_title["Stale"]
    assert by_title["Fresh"] >= 0.24  # fresh-but-quiet floors near 0.25
    # And the query itself is freshness-bounded at the API.
    assert "since" in params
