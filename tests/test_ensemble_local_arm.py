"""Tests for the local (Ollama) ensemble arm — measurement-only semantics."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from auramaur.db.database import Database
from auramaur.nlp import local_llm
from auramaur.nlp.ensemble_analyzer import EnsembleAnalyzer
from config.settings import LocalEnsembleArmConfig, LocalLLMConfig


def _settings(*, arm_enabled: bool = True, measure_only: bool = True):
    settings = MagicMock()
    settings.llm_ensemble.models = ["opus", "sonnet"]
    settings.llm_ensemble.default_weight = 0.5
    settings.llm_ensemble.min_samples_for_weights = 10
    settings.nlp.daily_claude_call_budget = 175
    settings.local_llm = LocalLLMConfig(
        enabled=True,
        model="qwen3:8b",
        ensemble_arm=LocalEnsembleArmConfig(
            enabled=arm_enabled, measure_only=measure_only))
    return settings


def _market():
    return SimpleNamespace(
        id="m1", question="Will X happen?", description="desc",
        outcome_yes_price=0.5, category="")


def _analyzer(db, settings) -> EnsembleAnalyzer:
    local_llm._client = None  # never reuse a singleton across tests
    analyzer = EnsembleAnalyzer(settings, db)
    local_llm._client = None
    analyzer._call_model = AsyncMock(return_value={"probability": 0.6})
    analyzer._local_client = MagicMock()
    analyzer._local_client.generate_json = AsyncMock(
        return_value={"probability": 0.9})
    return analyzer


async def test_arm_present_only_when_enabled():
    db = Database(":memory:")
    await db.connect()
    try:
        on = EnsembleAnalyzer(_settings(arm_enabled=True), db)
        assert [a.name for a in on._arms] == ["opus", "sonnet", "ollama:qwen3:8b"]
        local_llm._client = None
        off = EnsembleAnalyzer(_settings(arm_enabled=False), db)
        assert [a.name for a in off._arms] == ["opus", "sonnet"]
        assert off._local_client is None
    finally:
        local_llm._client = None
        await db.close()


async def test_measure_only_recorded_but_not_blended():
    db = Database(":memory:")
    await db.connect()
    try:
        analyzer = _analyzer(db, _settings(measure_only=True))
        result = await analyzer.estimate_probability(_market(), [])
        # Blend is claude-arms only: both mocked at 0.6.
        assert abs(result.probability - 0.6) < 1e-9
        # But the local arm still accrued Brier history.
        rows = await db.fetchall(
            "SELECT model, probability FROM ensemble_predictions ORDER BY model")
        models = [r["model"] for r in rows]
        assert models == ["ollama:qwen3:8b", "opus", "sonnet"]
        assert rows[0]["probability"] == 0.9
    finally:
        local_llm._client = None
        await db.close()


async def test_blendable_when_measure_only_off():
    db = Database(":memory:")
    await db.connect()
    try:
        analyzer = _analyzer(db, _settings(measure_only=False))
        result = await analyzer.estimate_probability(_market(), [])
        # Equal default weights over 0.6, 0.6, 0.9.
        assert abs(result.probability - 0.7) < 1e-9
    finally:
        local_llm._client = None
        await db.close()


async def test_local_failure_leaves_claude_arms_intact():
    db = Database(":memory:")
    await db.connect()
    try:
        analyzer = _analyzer(db, _settings(measure_only=True))
        analyzer._local_client.generate_json = AsyncMock(return_value=None)
        result = await analyzer.estimate_probability(_market(), [])
        assert abs(result.probability - 0.6) < 1e-9
        assert result.skipped_reason is None
        rows = await db.fetchall("SELECT model FROM ensemble_predictions")
        assert sorted(r["model"] for r in rows) == ["opus", "sonnet"]
    finally:
        local_llm._client = None
        await db.close()


async def test_local_probability_clamped():
    db = Database(":memory:")
    await db.connect()
    try:
        analyzer = _analyzer(db, _settings(measure_only=True))
        analyzer._local_client.generate_json = AsyncMock(
            return_value={"probability": 1.7})
        await analyzer.estimate_probability(_market(), [])
        row = await db.fetchone(
            "SELECT probability FROM ensemble_predictions "
            "WHERE model = 'ollama:qwen3:8b'")
        assert row["probability"] == 0.99
    finally:
        local_llm._client = None
        await db.close()
