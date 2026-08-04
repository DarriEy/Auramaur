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
