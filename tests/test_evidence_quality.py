"""Tests for info-content optimization (signal-per-token of LLM inputs).

Covers:
  - Lever A: global evidence re-ranking (evidence_ranker)
  - Lever B/E: compressor dedup, boilerplate drop, excerpt/directional dedup, negation
  - Lever C: calibration reliability curve
  - Lever D: centralized source authority + relevance fallback chain
  - Lever F: world-model entity summary by importance
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from auramaur.data_sources.base import NewsItem
from auramaur.nlp.evidence_compressor import (
    _deduplicate,
    _extract_directional,
    _extract_facts,
    compress_evidence,
)
from auramaur.nlp.evidence_ranker import rank_evidence
from auramaur.nlp.relevance import relevance_scores
from auramaur.nlp.sources import SOURCE_AUTHORITY, authority
from auramaur.nlp.strategic import EntityRelation, StrategicAnalyzer, WorldModel

_NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _ni(id, source, title, content="", age_h=1.0, url=""):
    return NewsItem(
        id=id, source=source, title=title, content=content, url=url,
        published_at=_NOW - timedelta(hours=age_h),
    )


# ---------------------------------------------------------------------------
# Lever D — source authority
# ---------------------------------------------------------------------------

class TestSourceAuthority:
    def test_known_tiers(self):
        assert authority("reuters") > authority("reddit")
        assert authority("unknown_source") == 1.0

    def test_url_domain_boost(self):
        # Low-tier source name but authoritative URL → takes the higher weight.
        assert authority("rss", "https://www.reuters.com/world/x") == SOURCE_AUTHORITY["reuters"]

    def test_url_does_not_substring_match_paths_or_hostnames(self):
        assert authority("rss", "https://example.com/congress-vote") == SOURCE_AUTHORITY["rss"]
        assert authority("rss", "https://reuters.com.evil.example/story") == SOURCE_AUTHORITY["rss"]

    def test_handles_none(self):
        assert authority(None, None) == 1.0


# ---------------------------------------------------------------------------
# Lever D/E — relevance backends
# ---------------------------------------------------------------------------

class TestRelevance:
    def test_heuristic_ranks_topical_higher(self):
        q = "Will the Federal Reserve cut interest rates?"
        texts = ["Federal Reserve signals interest rate cut", "Celebrity gossip roundup"]
        s = relevance_scores(q, texts, backend="heuristic")
        assert s[0] > s[1]

    def test_tfidf_backend(self):
        q = "Will the Federal Reserve cut interest rates?"
        texts = ["Federal Reserve signals interest rate cut", "Celebrity gossip roundup"]
        s = relevance_scores(q, texts, backend="tfidf")
        assert s[0] > s[1]

    def test_empty_texts(self):
        assert relevance_scores("anything", [], backend="heuristic") == []

    def test_unknown_backend_falls_through(self):
        s = relevance_scores("q about rates", ["rates rates rates", "noise"], backend="bogus")
        assert len(s) == 2


# ---------------------------------------------------------------------------
# Lever A — global evidence ranking
# ---------------------------------------------------------------------------

class TestEvidenceRanker:
    def test_recency_and_authority_beat_noise(self):
        items = [
            _ni("noise", "reddit", "off-topic meme", age_h=1),
            _ni("good", "reuters", "Fed signals interest rate cut soon",
                "The Federal Reserve indicated a rate cut.", age_h=2),
            _ni("stale", "reuters", "Fed rate cut analysis", "old", age_h=2000),
        ]
        ranked = rank_evidence(
            "Will the Fed cut interest rates?", items, top_n=2, backend="heuristic", now=_NOW,
        )
        ids = [r.id for r in ranked]
        assert ids[0] == "good"
        assert "noise" not in ids or "stale" not in ids  # at least one junk dropped

    def test_respects_top_n(self):
        items = [_ni(str(i), "web", f"headline {i} rates", age_h=i + 1) for i in range(10)]
        ranked = rank_evidence("rates", items, top_n=3, backend="heuristic", now=_NOW)
        assert len(ranked) == 3

    def test_empty(self):
        assert rank_evidence("q", [], top_n=5) == []


# ---------------------------------------------------------------------------
# Lever B/E — compressor
# ---------------------------------------------------------------------------

class TestCompressor:
    def test_content_dedup(self):
        items = [
            _ni("1", "reuters", "Senate passes the bill", "The Senate approved the measure 60-40."),
            _ni("2", "ap", "Senate approves measure", "The Senate approved the measure 60-40."),  # same body
        ]
        unique = _deduplicate(items)
        assert len(unique) == 1

    def test_facts_drop_boilerplate(self):
        # "according to" alone should no longer qualify as a fact.
        items = [_ni("1", "web", "Analysts comment",
                     "According to sources, people are talking about things.")]
        facts = _extract_facts(items)
        assert facts == []

    def test_facts_keep_figures(self):
        items = [_ni("1", "reuters", "Deal signed",
                     "The company announced a $5B acquisition on 03/15/2026.")]
        facts = _extract_facts(items)
        assert any("$5B" in f or "acquisition" in f for f in facts)

    def test_directional_negation_flip(self):
        # "will not be approved" should read as NO, not YES.
        items = [_ni("1", "reuters", "Regulators say the merger will not be approved this year",
                     "Officials confirmed the merger will not be approved.", age_h=1)]
        yes, no, used = _extract_directional("Will the merger be approved?", items)
        assert no and not yes
        assert items[0].title in used

    def test_excerpts_skip_directional_titles(self):
        items = [
            _ni("1", "reuters", "Bill approved by wide margin", "The bill was approved.", age_h=1),
            _ni("2", "ap", "Markets react to the vote", "Stocks moved after the decision.", age_h=2),
        ]
        out = compress_evidence("Will the bill be approved?", "criteria", items, max_chars=2000)
        # The approved headline appears as a directional signal, so it should not
        # be repeated verbatim in the KEY EXCERPTS section.
        assert "FOR YES" in out
        excerpt_section = out.split("KEY EXCERPTS:")[-1] if "KEY EXCERPTS:" in out else ""
        assert "Bill approved by wide margin" not in excerpt_section

    def test_no_evidence(self):
        assert compress_evidence("q", "d", []) == "(No evidence available)"


# ---------------------------------------------------------------------------
# Lever C — calibration reliability curve
# ---------------------------------------------------------------------------

class _Row(dict):
    """Sqlite-row-like: supports row['key']."""


def _row(predicted, outcome, question="q"):
    return _Row(predicted_prob=predicted, claude_prob=predicted,
                actual_outcome=outcome, market_id="m", question=question)


class TestCalibrationCurve:
    def test_detects_overconfidence(self):
        # Said ~90% many times but only half resolved YES → overconfident.
        rows = [_row(0.9, i % 2 == 0, f"q{i}") for i in range(10)]
        out = StrategicAnalyzer._calibration_curve(rows)
        assert "Brier" in out
        assert "OVER" in out

    def test_well_calibrated_band(self):
        rows = [_row(0.7, i < 7, f"q{i}") for i in range(10)]  # 70% said, 70% YES
        out = StrategicAnalyzer._calibration_curve(rows)
        assert "well-calibrated" in out

    def test_empty(self):
        assert "No resolved" in StrategicAnalyzer._calibration_curve([])


# ---------------------------------------------------------------------------
# Lever F — entity summary by importance
# ---------------------------------------------------------------------------

class TestWorldModelSummary:
    def test_orders_by_connections(self):
        wm = WorldModel()
        wm.entity_graph = {
            "minor": EntityRelation(state="x", relations=[], market_ids=["a"]),
            "major": EntityRelation(state="y", relations=["r1", "r2"], market_ids=["a", "b", "c", "d"]),
        }
        out = wm.summary()
        # The more-connected entity should be listed first.
        assert out.index("major") < out.index("minor")
