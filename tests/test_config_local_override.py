"""Tests for the tracked-vs-local config split.

config/settings.py loads the tracked ``defaults.yaml`` and deep-merges an
optional gitignored ``defaults.local.yaml`` on top, so operational and
never-commit values (the live gate, paper headroom, model-budget conservation)
live outside version control while the tracked file stays a paper-safe baseline.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from config.settings import _deep_merge


def test_deep_merge_override_wins_and_nested_dicts_merge():
    base = {"execution": {"live": False, "paper_initial_balance": 111.0},
            "nlp": {"model": "opus", "budget": 80}}
    over = {"execution": {"live": True},
            "nlp": {"budget": 175},
            "new_section": {"x": 1}}
    merged = _deep_merge(base, over)
    # Override wins on the keys it sets...
    assert merged["execution"]["live"] is True
    assert merged["nlp"]["budget"] == 175
    # ...while siblings in the same nested dict are preserved.
    assert merged["execution"]["paper_initial_balance"] == 111.0
    assert merged["nlp"]["model"] == "opus"
    # Entirely new sections are added.
    assert merged["new_section"] == {"x": 1}
    # Inputs are not mutated.
    assert base["execution"]["live"] is False


def test_deep_merge_handles_empty_override():
    base = {"a": {"b": 1}}
    assert _deep_merge(base, {}) == base
    assert _deep_merge(base, None) == base


def test_tracked_defaults_are_paper_safe():
    """The committed defaults.yaml alone (no local overrides) must never go
    live — a fresh clone / CI must default to paper."""
    tracked = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "config" / "defaults.yaml").read_text()
    )
    assert tracked["execution"]["live"] is False
    # analysis_mode is a local conservation knob — it must not be pinned in the
    # tracked file (pydantic default "auto" applies on a fresh clone).
    assert "analysis_mode" not in tracked.get("nlp", {})
    # Gemini conservation mode must not leak into tracked defaults.
    assert tracked["gemini"]["claude_budget_threshold"] == 0.8
    assert len(tracked["gemini"]["off_hours_utc"]) == 6
