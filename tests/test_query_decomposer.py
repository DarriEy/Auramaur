"""Tests for query decomposition."""

from auramaur.nlp.query_decomposer import extract_search_queries


class TestQueryDecomposer:
    def test_basic_question(self):
        queries = extract_search_queries("Will Trump win the 2024 election?")
        assert len(queries) >= 2
        # Should not start with "Will"
        assert not queries[0].startswith("Will")

    def test_extracts_entities(self):
        queries = extract_search_queries("Will the Federal Reserve cut rates by 50bps?")
        # Should have a query with "Federal Reserve"
        assert any("Federal Reserve" in q for q in queries)

    def test_handles_simple_question(self):
        queries = extract_search_queries("Will it rain tomorrow?")
        assert len(queries) >= 1

    def test_deduplicates(self):
        queries = extract_search_queries("Will Bitcoin reach 100000?")
        lowered = [q.lower() for q in queries]
        assert len(lowered) == len(set(lowered))

    def test_max_three_queries(self):
        queries = extract_search_queries(
            "Will President Biden sign the Infrastructure Investment and Jobs Act before December 2025?"
        )
        assert len(queries) <= 3

    def test_action_word_extraction(self):
        queries = extract_search_queries("Will Congress pass the spending bill?")
        # Should have a focused query like "Congress pass"
        assert any("pass" in q.lower() for q in queries)
