"""Tests for the robust Claude JSON parser — it must recover a JSON value
(object OR array) even when the model trails off into prose, which is what
caused the benign-but-noisy arb_scanner.llm_match_failed errors."""

from __future__ import annotations

import pytest

from auramaur.nlp.analyzer import _extract_json_span, _parse_claude_json


def test_parses_plain_object_and_array():
    assert _parse_claude_json('{"a": 1}') == {"a": 1}
    assert _parse_claude_json("[1, 2, 3]") == [1, 2, 3]


def test_empty_array_followed_by_prose():
    # The exact shape that errored: a valid [] then an explanation.
    raw = '[]\n\nNo matches found. List B consists entirely of "next SecGen" markets.'
    assert _parse_claude_json(raw) == []


def test_array_of_objects_followed_by_prose():
    raw = '[{"id_a": "x", "id_b": "y"}]\nThese two markets describe the same event.'
    assert _parse_claude_json(raw) == [{"id_a": "x", "id_b": "y"}]


def test_object_followed_by_prose():
    raw = '{"probability": 0.7}\nHere is my reasoning: the base rate is high.'
    assert _parse_claude_json(raw) == {"probability": 0.7}


def test_markdown_fenced_json():
    assert _parse_claude_json('```json\n{"a": 2}\n```') == {"a": 2}


def test_brackets_inside_strings_do_not_fool_extraction():
    # A ']' inside a string literal must not end the array early.
    raw = '[{"q": "Will the S&P close in [4000, 4100]?"}] then some prose.'
    assert _parse_claude_json(raw) == [{"q": "Will the S&P close in [4000, 4100]?"}]


def test_extract_span_handles_nesting():
    assert _extract_json_span('xx [1, [2, 3], 4] yy') == "[1, [2, 3], 4]"
    assert _extract_json_span("no json here") is None


def test_unparseable_still_raises():
    with pytest.raises(ValueError):
        _parse_claude_json("totally not json at all")
