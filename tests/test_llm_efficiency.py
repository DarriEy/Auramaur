"""Tests for LLM burn-reduction levers.

Covers the rate-limit-saving changes:
  - Lever 1: effort tiering (config plumbing)
  - Lever 2: edge-gated ensemble (_batch_has_edge)
  - Lever 4: cacheable static system prompt split
  - Lever 5: strategic per-market cache partition + all-cached short-circuit
"""

from __future__ import annotations

import asyncio

import pytest

from auramaur.db.database import Database
from auramaur.exchange.models import Market
from auramaur.nlp.cache import coarse_evidence_digest, make_cache_key
from auramaur.nlp.strategic import (
    STRATEGIC_BATCH_PROMPT,
    STRATEGIC_SYSTEM_PROMPT,
    BatchAnalysisResult,
    StrategicAnalysis,
    StrategicAnalyzer,
)
from config.settings import Settings


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def db(event_loop):
    async def _setup():
        database = Database(db_path=":memory:")
        await database.connect()
        return database

    database = event_loop.run_until_complete(_setup())
    yield database
    event_loop.run_until_complete(database.close())


def run(coro, loop):
    return loop.run_until_complete(coro)


class _Item:
    """Minimal NewsItem-like evidence stub."""

    def __init__(self, title: str):
        self.title = title


def _market(mid: str, price: float) -> Market:
    return Market(
        id=mid,
        exchange="polymarket",
        question=f"Will market {mid} resolve YES?",
        description="x" * 60,
        outcome_yes_price=price,
    )


def _analyzer(db) -> StrategicAnalyzer:
    return StrategicAnalyzer(settings=Settings(), db=db)


# ---------------------------------------------------------------------------
# Lever 1 — effort tiering config
# ---------------------------------------------------------------------------

class TestEffortTiering:
    def test_default_tiers(self):
        nlp = Settings().nlp
        assert nlp.effort_primary == "max"
        assert nlp.effort_adversarial == "medium"
        assert nlp.effort_ensemble_secondary == "high"
        assert nlp.effort_tool_use == "high"

    def test_tool_use_tightened(self):
        nlp = Settings().nlp
        assert nlp.tool_use_max_markets_per_cycle == 2
        assert nlp.tool_use_edge_threshold_pct == 8.0


# ---------------------------------------------------------------------------
# Lever 2 — edge-gated ensemble
# ---------------------------------------------------------------------------

class TestBatchEdgeGate:
    def test_edge_present_triggers(self, db):
        a = _analyzer(db)
        a._settings.llm_ensemble.edge_threshold_pct = 5.0
        markets = [_market("m1", 0.50), _market("m2", 0.50)]
        result = StrategicAnalysis(markets=[
            BatchAnalysisResult(market_id="m1", probability=0.52, confidence="LOW"),
            BatchAnalysisResult(market_id="m2", probability=0.70, confidence="HIGH"),  # 20pt edge
        ])
        assert a._batch_has_edge(result, markets) is True

    def test_no_edge_skips(self, db):
        a = _analyzer(db)
        a._settings.llm_ensemble.edge_threshold_pct = 5.0
        markets = [_market("m1", 0.50), _market("m2", 0.50)]
        result = StrategicAnalysis(markets=[
            BatchAnalysisResult(market_id="m1", probability=0.52, confidence="LOW"),
            BatchAnalysisResult(market_id="m2", probability=0.48, confidence="LOW"),
        ])
        assert a._batch_has_edge(result, markets) is False


# ---------------------------------------------------------------------------
# Lever 4 — cacheable system-prompt split
# ---------------------------------------------------------------------------

class TestPromptSplit:
    def test_system_prompt_is_static(self):
        # Never .format()'d → must carry no template placeholders.
        assert "{world_model}" not in STRATEGIC_SYSTEM_PROMPT
        assert "{markets_block}" not in STRATEGIC_SYSTEM_PROMPT
        assert '"entity_graph"' in STRATEGIC_SYSTEM_PROMPT  # response schema lives here

    def test_user_prompt_carries_dynamic_fields(self):
        out = STRATEGIC_BATCH_PROMPT.format(
            world_model="WM", calibration_feedback="CAL", markets_block="MKTS",
        )
        assert "WM" in out and "CAL" in out and "MKTS" in out


# ---------------------------------------------------------------------------
# Lever 5 — strategic per-market cache
# ---------------------------------------------------------------------------

class TestCoarseDigest:
    def test_reorder_stable(self):
        a = [_Item("Reuters: ceasefire"), _Item("AP: oil steady"), _Item("blog: maybe")]
        b = list(reversed(a))
        assert coarse_evidence_digest(a) == coarse_evidence_digest(b)

    def test_content_change_busts(self):
        a = [_Item("Reuters: ceasefire holds")]
        b = [_Item("Reuters: ceasefire collapses")]
        assert coarse_evidence_digest(a) != coarse_evidence_digest(b)


class TestStrategicCache:
    def test_partition_reuses_fresh_entry(self, db, event_loop):
        a = _analyzer(db)
        market = _market("m1", 0.40)
        evidence = {market.id: [_Item("some headline")]}

        # Prime the cache for this market.
        key = make_cache_key(market.question, coarse_evidence_digest(evidence[market.id]))
        run(a._cache.put(
            key, market.id,
            BatchAnalysisResult(market_id="m1", probability=0.33, confidence="MEDIUM").model_dump(),
            ttl_seconds=900, market_price=0.40,
        ), event_loop)

        cached, to_analyze, keys = run(a._partition_cached([market], evidence), event_loop)
        assert len(cached) == 1
        assert cached[0].probability == 0.33
        assert to_analyze == []
        assert keys[market.id] == key

    def test_all_cached_short_circuits_llm(self, db, event_loop):
        a = _analyzer(db)
        market = _market("m1", 0.40)
        evidence = {market.id: [_Item("some headline")]}
        key = make_cache_key(market.question, coarse_evidence_digest(evidence[market.id]))
        run(a._cache.put(
            key, market.id,
            BatchAnalysisResult(market_id="m1", probability=0.33, confidence="MEDIUM").model_dump(),
            ttl_seconds=900, market_price=0.40,
        ), event_loop)

        # If the LLM is called despite a full cache hit, fail loudly.
        async def _boom(*args, **kwargs):
            raise AssertionError("LLM should not be called when all markets are cached")

        a._call_llm = _boom  # type: ignore[assignment]

        out = run(a.analyze_batch([market], evidence), event_loop)
        assert len(out.markets) == 1
        assert out.markets[0].probability == 0.33

    def test_price_move_invalidates(self, db, event_loop):
        a = _analyzer(db)
        market = _market("m1", 0.40)
        evidence = {market.id: [_Item("some headline")]}
        key = make_cache_key(market.question, coarse_evidence_digest(evidence[market.id]))
        # Cached at price 0.40; market is now 0.40 -> hit. Move price -> miss.
        run(a._cache.put(
            key, market.id,
            BatchAnalysisResult(market_id="m1", probability=0.33, confidence="MEDIUM").model_dump(),
            ttl_seconds=900, market_price=0.40,
        ), event_loop)

        moved = _market("m1", 0.60)  # +50% move
        cached, to_analyze, _ = run(a._partition_cached([moved], {moved.id: evidence["m1"]}), event_loop)
        assert cached == []
        assert len(to_analyze) == 1

    def test_cache_disabled_bypasses(self, db, event_loop):
        a = _analyzer(db)
        a._settings.nlp.strategic_cache_enabled = False
        market = _market("m1", 0.40)
        evidence = {market.id: [_Item("some headline")]}
        key = make_cache_key(market.question, coarse_evidence_digest(evidence[market.id]))
        run(a._cache.put(
            key, market.id,
            BatchAnalysisResult(market_id="m1", probability=0.33, confidence="MEDIUM").model_dump(),
            ttl_seconds=900, market_price=0.40,
        ), event_loop)

        cached, to_analyze, _ = run(a._partition_cached([market], evidence), event_loop)
        assert cached == []  # disabled → no reuse
        assert len(to_analyze) == 1


class TestStrategicParseAndThrottle:
    """Parse robustness (bare array vs object) + batch cadence throttle."""

    def test_parse_accepts_bare_array(self):
        # The model frequently returns a bare array of per-market results;
        # this used to 100% parse_fail and discard the (paid-for) batch.
        raw = '[{"market_id": "m1", "probability": 0.3, "confidence": "MEDIUM"}]'
        out = StrategicAnalyzer._parse_strategic_response(raw)
        assert [r.market_id for r in out.markets] == ["m1"]
        assert out.markets[0].probability == 0.3

    def test_parse_still_accepts_object(self):
        raw = '{"markets": [{"market_id": "m2", "probability": 0.7}], "world_model_update": "x"}'
        out = StrategicAnalyzer._parse_strategic_response(raw)
        assert [r.market_id for r in out.markets] == ["m2"]
        assert out.world_model_update == "x"

    def test_parse_array_in_fences_and_prose(self):
        raw = 'Sure:\n```json\n[{"market_id":"m3","probability":0.5}]\n```\ndone'
        out = StrategicAnalyzer._parse_strategic_response(raw)
        assert [r.market_id for r in out.markets] == ["m3"]

    def test_parse_garbage_returns_empty(self):
        out = StrategicAnalyzer._parse_strategic_response("no json here at all")
        assert out.markets == []

    def test_batch_throttle_skips_llm_within_interval(self, db, event_loop):
        s = Settings()
        s.nlp.strategic_min_interval_seconds = 3600
        s.nlp.skip_second_opinion = True  # isolate to the batch call
        a = StrategicAnalyzer(settings=s, db=db)
        market = _market("m1", 0.40)
        evidence: dict = {}  # no evidence → avoids the compressor stub mismatch
        calls = {"n": 0}

        async def fake_llm(*args, **kwargs):
            calls["n"] += 1
            return '[{"market_id": "m1", "probability": 0.3, "confidence": "MEDIUM"}]'

        a._call_llm = fake_llm  # type: ignore[assignment]

        run(a.analyze_batch_with_adversarial([market], evidence), event_loop)
        assert calls["n"] == 1  # first run hits the LLM

        # Immediate second run is within the interval → served from cache, no call.
        run(a.analyze_batch_with_adversarial([market], evidence), event_loop)
        assert calls["n"] == 1
