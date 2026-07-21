"""Tests for the local-LLM evidence distiller."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from auramaur.db.database import Database
from auramaur.nlp.evidence_distiller import (
    EvidenceDistiller,
    load_distilled_map,
)
from config.settings import LocalDistillerConfig, LocalLLMConfig


def _settings(**distiller_overrides) -> SimpleNamespace:
    cfg = LocalLLMConfig(
        enabled=True,
        distiller=LocalDistillerConfig(enabled=True, **distiller_overrides))
    return SimpleNamespace(local_llm=cfg)


def _client(reply) -> MagicMock:
    client = MagicMock()
    client.generate_json = AsyncMock(return_value=reply)
    return client


async def _seed(db: Database, content_hash: str = "hashA",
                market_id: str = "m1", observed_offset: str = "-1 hours") -> None:
    await db.execute(
        """INSERT OR REPLACE INTO evidence_observations
           (run_id, item_id, source, title, url, content_hash, excerpt,
            observed_at, market_id)
           VALUES ('run1', ?, 'rss', 'Fed holds rates', '', ?,
                   'The Fed held rates steady in June.',
                   datetime('now', ?), ?)""",
        (f"item-{content_hash}", content_hash, observed_offset, market_id))
    await db.execute(
        """INSERT OR REPLACE INTO markets (id, question, last_updated)
           VALUES (?, 'Will the Fed cut rates in September?', datetime('now'))""",
        (market_id,))
    await db.commit()


_GOOD_REPLY = {
    "claims": [{
        "claim": "The Fed held rates steady in June 2026.",
        "entities": ["Federal Reserve"],
        "event_date": "2026-06-17",
        "markets_affected": [
            {"market_id": "m1", "direction": "no"},
            {"market_id": "hallucinated-id", "direction": "yes"},
        ],
    }],
}


async def test_run_once_persists_claims_and_progress():
    db = Database(":memory:")
    await db.connect()
    try:
        await _seed(db)
        distiller = EvidenceDistiller(db, _settings(), _client(_GOOD_REPLY))
        assert await distiller.run_once() == 1
        claim = await db.fetchone(
            "SELECT claim, markets_affected FROM distilled_claims")
        assert "held rates steady" in claim["claim"]
        # Hallucinated market ids are dropped; only offered candidates remain.
        assert "hallucinated-id" not in claim["markets_affected"]
        assert "m1" in claim["markets_affected"]
        progress = await db.fetchone(
            "SELECT status, claims FROM distill_progress WHERE content_hash='hashA'")
        assert progress["status"] == "done"
        assert progress["claims"] == 1
    finally:
        await db.close()


async def test_second_run_skips_processed_items():
    db = Database(":memory:")
    await db.connect()
    try:
        await _seed(db)
        client = _client(_GOOD_REPLY)
        distiller = EvidenceDistiller(db, _settings(), client)
        await distiller.run_once()
        await distiller.run_once()
        assert client.generate_json.await_count == 1
    finally:
        await db.close()


async def test_transport_error_retries_capped():
    db = Database(":memory:")
    await db.connect()
    try:
        await _seed(db)
        client = _client(None)  # generate_json fails open with None
        distiller = EvidenceDistiller(db, _settings(), client)
        for _ in range(5):
            await distiller.run_once()
        # Attempts capped at 3: selected while attempts < 3, then benched.
        assert client.generate_json.await_count == 3
        progress = await db.fetchone(
            "SELECT status, attempts FROM distill_progress WHERE content_hash='hashA'")
        assert progress["status"] == "error"
        assert progress["attempts"] == 3
    finally:
        await db.close()


async def test_claim_clamped_and_empty_reply_marked_empty():
    db = Database(":memory:")
    await db.connect()
    try:
        await _seed(db, content_hash="hashLong", market_id="m1")
        long_reply = {"claims": [{"claim": "x" * 900}]}
        distiller = EvidenceDistiller(db, _settings(), _client(long_reply))
        await distiller.run_once()
        claim = await db.fetchone("SELECT claim FROM distilled_claims")
        assert len(claim["claim"]) <= 300

        await _seed(db, content_hash="hashEmpty", market_id="m1")
        distiller = EvidenceDistiller(db, _settings(), _client({"claims": []}))
        await distiller.run_once()
        progress = await db.fetchone(
            "SELECT status FROM distill_progress WHERE content_hash='hashEmpty'")
        assert progress["status"] == "empty"
    finally:
        await db.close()


async def test_old_evidence_not_selected():
    db = Database(":memory:")
    await db.connect()
    try:
        await _seed(db, observed_offset="-72 hours")
        client = _client(_GOOD_REPLY)
        distiller = EvidenceDistiller(
            db, _settings(max_item_age_hours=24), client)
        assert await distiller.run_once() == 0
        client.generate_json.assert_not_awaited()
    finally:
        await db.close()


async def test_retention_cleanup():
    db = Database(":memory:")
    await db.connect()
    try:
        await db.execute(
            """INSERT INTO distilled_claims
               (content_hash, claim, model, created_at)
               VALUES ('old', 'stale claim', 'qwen3:8b',
                       datetime('now', '-30 days'))""")
        await db.commit()
        distiller = EvidenceDistiller(
            db, _settings(retention_days=14), _client({"claims": []}))
        await distiller.run_once()
        assert await db.fetchone(
            "SELECT 1 FROM distilled_claims WHERE content_hash='old'") is None
    finally:
        await db.close()


async def test_load_distilled_map_budget_and_join():
    db = Database(":memory:")
    await db.connect()
    try:
        await _seed(db)
        distiller = EvidenceDistiller(db, _settings(), _client(_GOOD_REPLY))
        await distiller.run_once()

        full = await load_distilled_map(db, ["m1"], char_budget=600)
        assert "m1" in full
        assert "(no)" in full["m1"]          # per-market direction rendered
        assert "held rates steady" in full["m1"]

        # A tiny budget yields nothing rather than a truncated half-line.
        tiny = await load_distilled_map(db, ["m1"], char_budget=5)
        assert tiny == {}

        # Markets with no joined evidence get no entry.
        other = await load_distilled_map(db, ["m-unknown"], char_budget=600)
        assert other == {}
    finally:
        await db.close()
