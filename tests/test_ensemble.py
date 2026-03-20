"""Tests for ensemble estimator."""

from __future__ import annotations

import asyncio

import pytest

from auramaur.db.database import Database
from auramaur.nlp.ensemble import EnsembleEstimator, ProbabilitySource


class MockSource:
    """Simple mock probability source for testing."""

    def __init__(self, name: str, prob: float | None):
        self._name = name
        self._prob = prob

    @property
    def name(self) -> str:
        return self._name

    async def estimate(self, question: str, category: str = "") -> float | None:
        return self._prob


class FailingSource:
    """Source that always raises an error."""

    @property
    def name(self) -> str:
        return "failing"

    async def estimate(self, question: str, category: str = "") -> float | None:
        raise RuntimeError("Source unavailable")


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


class TestEnsembleEstimator:
    def test_single_source(self, db, event_loop):
        sources = [MockSource("claude", 0.7)]
        ensemble = EnsembleEstimator(db=db, sources=sources)
        result = run(ensemble.estimate("Will X happen?"), event_loop)
        assert abs(result["probability"] - 0.7) < 0.001

    def test_equal_weight_average(self, db, event_loop):
        sources = [MockSource("claude", 0.6), MockSource("polls", 0.8)]
        ensemble = EnsembleEstimator(db=db, sources=sources)
        result = run(ensemble.estimate("Will X happen?"), event_loop)
        assert abs(result["probability"] - 0.7) < 0.001

    def test_weighted_average(self, db, event_loop):
        sources = [MockSource("claude", 0.6), MockSource("polls", 0.8)]
        ensemble = EnsembleEstimator(db=db, sources=sources)
        ensemble._weights = {"claude": 2.0, "polls": 1.0}
        result = run(ensemble.estimate("Will X happen?"), event_loop)
        # (0.6*2 + 0.8*1) / 3 = 2.0/3 ~ 0.667
        assert abs(result["probability"] - 0.667) < 0.01

    def test_source_returns_none(self, db, event_loop):
        sources = [MockSource("claude", 0.6), MockSource("polls", None)]
        ensemble = EnsembleEstimator(db=db, sources=sources)
        result = run(ensemble.estimate("Will X happen?"), event_loop)
        # Only claude contributes
        assert abs(result["probability"] - 0.6) < 0.001
        assert "polls" not in result["sources"]

    def test_failing_source_handled(self, db, event_loop):
        sources = [MockSource("claude", 0.6), FailingSource()]
        ensemble = EnsembleEstimator(db=db, sources=sources)
        result = run(ensemble.estimate("Will X happen?"), event_loop)
        assert abs(result["probability"] - 0.6) < 0.001

    def test_no_sources(self, db, event_loop):
        ensemble = EnsembleEstimator(db=db, sources=[])
        result = run(ensemble.estimate("Will X happen?"), event_loop)
        assert result["probability"] is None

    def test_all_sources_none(self, db, event_loop):
        sources = [MockSource("a", None), MockSource("b", None)]
        ensemble = EnsembleEstimator(db=db, sources=sources)
        result = run(ensemble.estimate("Will X happen?"), event_loop)
        assert result["probability"] is None

    def test_source_details_returned(self, db, event_loop):
        sources = [MockSource("claude", 0.6), MockSource("polls", 0.8)]
        ensemble = EnsembleEstimator(db=db, sources=sources)
        result = run(ensemble.estimate("Will X happen?"), event_loop)
        assert "claude" in result["sources"]
        assert "polls" in result["sources"]
        assert result["sources"]["claude"] == 0.6
