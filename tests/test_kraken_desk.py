"""The Kraken desk call.

The failure mode this guards is a stronger model becoming a more expensive way
to lose money: the desk PROPOSES, and arithmetic disposes. Parsing fails closed,
and the cost gate is applied after the model has spoken.
"""

import pytest

from auramaur.strategy.kraken_desk import (
    DeskDecision, parse_desk_response, tradeable,
)

PAIRS = ["XBTUSDC", "ETHUSDC", "SOLUSDC"]


def _reply(rows):
    import json
    return json.dumps(rows)


def test_a_clean_reply_parses():
    text = _reply([
        {"pair": "XBTUSDC", "action": "skip", "probability": 0.48,
         "confidence": "LOW", "reasoning": "downtrend"},
        {"pair": "ETHUSDC", "action": "enter", "probability": 0.62,
         "confidence": "MEDIUM", "reasoning": "momentum turning"},
    ])
    out = parse_desk_response(text, PAIRS)
    assert set(out) == {"XBTUSDC", "ETHUSDC"}
    assert out["ETHUSDC"].action == "enter"
    assert out["ETHUSDC"].probability == pytest.approx(0.62)


def test_prose_and_code_fences_are_tolerated():
    text = ('Here is my read.\n```json\n'
            '[{"pair":"SOLUSDC","action":"hold","probability":0.5,'
            '"confidence":"LOW","reasoning":"flat"}]\n```\nThanks.')
    out = parse_desk_response(text, PAIRS)
    assert out["SOLUSDC"].action == "hold"


def test_a_malformed_entry_is_dropped_not_defaulted():
    """Defaulting a missing probability to 0.5 or an unknown action to 'skip'
    would turn a broken reply into a confident-looking decision."""
    text = _reply([
        {"pair": "XBTUSDC", "action": "enter"},                      # no prob
        {"pair": "ETHUSDC", "action": "yolo", "probability": 0.9},    # bad action
        {"pair": "SOLUSDC", "action": "enter", "probability": 1.4},   # out of range
        {"pair": "DOGEUSD", "action": "enter", "probability": 0.9},   # not asked
        {"pair": "XBTUSDC", "action": "enter", "probability": "abc"},  # not a number
    ])
    assert parse_desk_response(text, PAIRS) == {}


def test_garbage_yields_no_decisions():
    for text in ("", "I could not decide.", "[", "null", "{}", "[1,2,3]"):
        assert parse_desk_response(text, PAIRS) == {}


def test_arithmetic_overrides_the_model():
    """The decisive property. A confident 'enter' below the cost threshold is
    refused, and the refusal names the numbers."""
    ask = DeskDecision("ETHUSDC", "enter", 0.55, "HIGH", "very confident")
    ok, why = tradeable(ask, 0.594)          # ETH at a 3-day horizon
    assert not ok and "0.594" in why and "0.550" in why

    ok, why = tradeable(ask, 0.544)          # the same view at 2 weeks
    assert ok and "0.544" in why


def test_no_horizon_that_pays_is_a_refusal_not_a_licence():
    ask = DeskDecision("XBTUSDC", "enter", 0.99, "HIGH", "certain")
    ok, why = tradeable(ask, None)
    assert not ok and "no probability" in why


def test_only_enter_can_trade():
    for action in ("skip", "hold", "exit"):
        ok, why = tradeable(
            DeskDecision("XBTUSDC", action, 0.99, "HIGH", ""), 0.5)
        assert not ok and action in why
