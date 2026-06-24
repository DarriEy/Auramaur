"""Confidence enum accepts case-/separator-insensitive strings.

An LLM (Gemini, in conservation mode) returned ``'medium'`` lowercase, and
``Confidence('medium')`` crashed a live trading cycle with
``'medium' is not a valid Confidence``. The ``_missing_`` hook normalizes so
every Confidence(...) call site is protected at once.
"""

from __future__ import annotations

import pytest

from auramaur.exchange.models import Confidence


@pytest.mark.parametrize("value,expected", [
    ("medium", Confidence.MEDIUM),
    ("MEDIUM", Confidence.MEDIUM),
    ("high", Confidence.HIGH),
    ("low", Confidence.LOW),
    ("medium_high", Confidence.MEDIUM_HIGH),
    ("Medium-Low", Confidence.MEDIUM_LOW),
    (" High ", Confidence.HIGH),
    ("MEDIUM HIGH", Confidence.MEDIUM_HIGH),
])
def test_confidence_normalizes(value, expected):
    assert Confidence(value) is expected


def test_canonical_values_unaffected():
    assert Confidence("MEDIUM") is Confidence.MEDIUM
    assert Confidence.HIGH.value == "HIGH"


def test_truly_invalid_still_raises():
    with pytest.raises(ValueError):
        Confidence("definitely-not-a-level")
