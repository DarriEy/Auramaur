"""Tests for Platt scaling calibration."""

from __future__ import annotations

import asyncio
import math

import pytest

from auramaur.db.database import Database
from auramaur.nlp.calibration import CalibrationTracker, _logit, _sigmoid


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def db(event_loop):
    """Create an in-memory database for testing."""

    async def _setup():
        database = Database(db_path=":memory:")
        await database.connect()
        return database

    database = event_loop.run_until_complete(_setup())
    yield database
    event_loop.run_until_complete(database.close())


@pytest.fixture
def tracker(db):
    return CalibrationTracker(db=db, min_samples=5)


def run(coro, loop):
    return loop.run_until_complete(coro)


class TestPlattHelpers:
    def test_logit_sigmoid_inverse(self):
        """sigmoid(logit(p)) should return p."""
        for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
            assert abs(_sigmoid(_logit(p)) - p) < 1e-10

    def test_logit_clamps_extremes(self):
        """logit should not blow up on 0 or 1."""
        assert math.isfinite(_logit(0.0))
        assert math.isfinite(_logit(1.0))

    def test_sigmoid_extreme_values(self):
        assert _sigmoid(1000) == 1.0
        assert _sigmoid(-1000) == 0.0


class TestCalibrationTracker:
    def test_adjust_no_params_returns_raw(self, tracker, event_loop):
        """With no fitted params, adjust() should return the raw probability."""
        result = run(tracker.adjust(0.7, "politics"), event_loop)
        assert result == 0.7

    def test_fit_params_insufficient_data(self, tracker, event_loop):
        """fit_params should return None when not enough samples."""
        # Only record 3 predictions (min_samples=5)
        for i in range(3):
            run(
                tracker.record_prediction(f"mkt_{i}", 0.6, "politics"),
                event_loop,
            )
            run(tracker.record_resolution(f"mkt_{i}", True), event_loop)

        result = run(tracker.fit_params(category="politics"), event_loop)
        assert result is None

    def test_fit_params_with_data(self, tracker, event_loop):
        """fit_params should return (a, b) tuple when enough samples."""
        # Create predictions that are systematically too high
        for i in range(10):
            predicted = 0.7 + (i % 3) * 0.05  # 0.70, 0.75, 0.80
            actual = i < 5  # Only 50% resolve YES
            run(tracker.record_prediction(f"mkt_{i}", predicted, ""), event_loop)
            run(tracker.record_resolution(f"mkt_{i}", actual), event_loop)

        result = run(tracker.fit_params(category=None), event_loop)
        assert result is not None
        a, b = result
        assert isinstance(a, float)
        assert isinstance(b, float)

    def test_adjust_with_bias_correction(self, tracker, event_loop):
        """Platt scaling should correct a systematic overconfidence bias."""
        # Predictions are systematically 10-15% too high
        import random

        random.seed(42)
        for i in range(30):
            true_prob = 0.4 + random.random() * 0.2  # true prob 0.4-0.6
            predicted = true_prob + 0.12  # systematically 12% too high
            actual = random.random() < true_prob
            run(tracker.record_prediction(f"mkt_{i}", predicted, ""), event_loop)
            run(tracker.record_resolution(f"mkt_{i}", actual), event_loop)

        run(tracker.fit_params(category=None), event_loop)

        # Now test: a prediction of 0.65 should be adjusted downward
        calibrated = run(tracker.adjust(0.65), event_loop)
        assert calibrated < 0.65, (
            f"Expected calibrated ({calibrated:.3f}) < raw (0.65) for overconfident predictor"
        )

    def test_per_category_calibration(self, tracker, event_loop):
        """Per-category params should be used when available."""
        # Politics: predictions are too high
        for i in range(10):
            run(tracker.record_prediction(f"pol_{i}", 0.8, "politics"), event_loop)
            run(tracker.record_resolution(f"pol_{i}", i < 5), event_loop)

        # Sports: predictions are too low
        for i in range(10):
            run(tracker.record_prediction(f"spt_{i}", 0.3, "sports"), event_loop)
            run(tracker.record_resolution(f"spt_{i}", i < 7), event_loop)

        # Fit both categories
        run(tracker.fit_params(category="politics"), event_loop)
        run(tracker.fit_params(category="sports"), event_loop)

        pol_adjusted = run(tracker.adjust(0.8, "politics"), event_loop)
        spt_adjusted = run(tracker.adjust(0.3, "sports"), event_loop)

        # Politics predictions should be adjusted down from 0.8
        assert pol_adjusted < 0.8
        # Sports predictions should be adjusted up from 0.3
        assert spt_adjusted > 0.3

    def test_category_fallback_to_global(self, tracker, event_loop):
        """Unknown category should fall back to global params."""
        # Fit global params only
        for i in range(10):
            run(tracker.record_prediction(f"mkt_{i}", 0.7, ""), event_loop)
            run(tracker.record_resolution(f"mkt_{i}", i < 4), event_loop)

        run(tracker.fit_params(category=None), event_loop)

        # Query with unknown category should use global params
        result = run(tracker.adjust(0.7, "unknown_category"), event_loop)
        assert result != 0.7  # Should be adjusted (not raw)

    def test_refit_all(self, tracker, event_loop):
        """refit_all should fit global and qualifying categories."""
        # Global data
        for i in range(10):
            run(tracker.record_prediction(f"g_{i}", 0.6, ""), event_loop)
            run(tracker.record_resolution(f"g_{i}", i < 5), event_loop)

        # Category with enough data
        for i in range(8):
            run(tracker.record_prediction(f"p_{i}", 0.7, "politics"), event_loop)
            run(tracker.record_resolution(f"p_{i}", i < 4), event_loop)

        # Category with too little data
        for i in range(3):
            run(tracker.record_prediction(f"s_{i}", 0.5, "sports"), event_loop)
            run(tracker.record_resolution(f"s_{i}", True), event_loop)

        run(tracker.refit_all(), event_loop)

        # Global should be fitted
        global_result = run(tracker.adjust(0.6, ""), event_loop)
        assert global_result != 0.6

        # Politics should be fitted
        pol_result = run(tracker.adjust(0.7, "politics"), event_loop)
        assert pol_result != 0.7

    def test_record_prediction_with_category(self, tracker, event_loop):
        """record_prediction should store category."""
        run(tracker.record_prediction("mkt_1", 0.6, "crypto"), event_loop)

        row = run(
            tracker._db.fetchone(
                "SELECT category FROM calibration WHERE market_id = ?",
                ("mkt_1",),
            ),
            event_loop,
        )
        assert row["category"] == "crypto"

    def test_brier_score(self, tracker, event_loop):
        """Brier score should work with the updated schema."""
        run(tracker.record_prediction("m1", 0.9, ""), event_loop)
        run(tracker.record_resolution("m1", True), event_loop)
        run(tracker.record_prediction("m2", 0.9, ""), event_loop)
        run(tracker.record_resolution("m2", False), event_loop)

        score = run(tracker.get_brier_score(), event_loop)
        assert score is not None
        # (0.9-1)^2 + (0.9-0)^2 / 2 = (0.01 + 0.81) / 2 = 0.41
        assert abs(score - 0.41) < 0.01


class TestOnlineCalibration:
    """Tests for online (continuous) calibration updates."""

    def test_online_update_modifies_params(self, tracker, event_loop):
        """online_update should modify cached Platt scaling parameters."""
        # First fit initial params with batch data
        for i in range(10):
            run(tracker.record_prediction(f"mkt_{i}", 0.7, ""), event_loop)
            run(tracker.record_resolution(f"mkt_{i}", i < 5), event_loop)

        # Clear the online updates that happened during record_resolution
        # by refitting from scratch
        result = run(tracker.fit_params(category=None), event_loop)
        assert result is not None
        old_a, old_b = result

        # Now do an explicit online update
        run(tracker.online_update(0.8, 0.0, ""), event_loop)

        # Params should have changed
        new_params = tracker._params_cache[""]
        new_a, new_b = new_params
        assert (new_a, new_b) != (old_a, old_b), "Online update should modify params"

    def test_online_update_direction(self, tracker, event_loop):
        """Online update after overconfident prediction should shift params."""
        # Fit initial params
        for i in range(10):
            run(tracker.record_prediction(f"mkt_{i}", 0.6, ""), event_loop)
            run(tracker.record_resolution(f"mkt_{i}", i < 6), event_loop)

        run(tracker.fit_params(category=None), event_loop)
        before = run(tracker.adjust(0.9, ""), event_loop)

        # We predicted 0.9 but outcome was NO (0.0) — overconfident
        run(tracker.online_update(0.9, 0.0, ""), event_loop)

        after = run(tracker.adjust(0.9, ""), event_loop)
        # After seeing overconfidence, calibrated prob for 0.9 should decrease
        assert after < before, (
            f"After overconfident miss, calibrated should decrease: {after:.4f} < {before:.4f}"
        )

    def test_record_resolution_triggers_online_update(self, tracker, event_loop):
        """record_resolution should automatically call online_update."""
        # Fit initial params
        for i in range(10):
            run(tracker.record_prediction(f"init_{i}", 0.6, ""), event_loop)
            run(tracker.record_resolution(f"init_{i}", i < 6), event_loop)

        run(tracker.fit_params(category=None), event_loop)
        old_params = tracker._params_cache[""]

        # Record a new prediction + resolution
        run(tracker.record_prediction("new_mkt", 0.85, ""), event_loop)
        run(tracker.record_resolution("new_mkt", False), event_loop)

        # Params should have been updated automatically
        new_params = tracker._params_cache[""]
        assert new_params != old_params, (
            "record_resolution should trigger online_update and change params"
        )

    def test_online_update_persists_to_db(self, tracker, event_loop):
        """Online update should persist new params to database."""
        # Fit initial params
        for i in range(10):
            run(tracker.record_prediction(f"mkt_{i}", 0.5, ""), event_loop)
            run(tracker.record_resolution(f"mkt_{i}", i < 5), event_loop)

        run(tracker.fit_params(category=None), event_loop)

        # Do online update
        run(tracker.online_update(0.8, 1.0, ""), event_loop)
        expected = tracker._params_cache[""]

        # Verify DB has the updated params
        row = run(
            tracker._db.fetchone(
                "SELECT a, b FROM calibration_params WHERE category = ?",
                ("",),
            ),
            event_loop,
        )
        assert row is not None
        assert abs(row["a"] - expected[0]) < 1e-10
        assert abs(row["b"] - expected[1]) < 1e-10

    def test_online_update_category_specific(self, tracker, event_loop):
        """Online update with category should update both category and global."""
        # Fit global and category params
        for i in range(10):
            run(tracker.record_prediction(f"g_{i}", 0.6, ""), event_loop)
            run(tracker.record_resolution(f"g_{i}", i < 6), event_loop)
        for i in range(10):
            run(tracker.record_prediction(f"p_{i}", 0.7, "politics"), event_loop)
            run(tracker.record_resolution(f"p_{i}", i < 4), event_loop)

        run(tracker.fit_params(category=None), event_loop)
        run(tracker.fit_params(category="politics"), event_loop)
        old_global = tracker._params_cache[""]
        old_politics = tracker._params_cache["politics"]

        # Online update with category
        run(tracker.online_update(0.8, 0.0, "politics"), event_loop)

        # Both should have changed
        assert tracker._params_cache[""] != old_global
        assert tracker._params_cache["politics"] != old_politics

    def test_moving_brier_score_empty(self, tracker):
        """Moving Brier score should return None when no recent predictions."""
        assert tracker.get_moving_brier_score() is None

    def test_moving_brier_score_single(self, tracker, event_loop):
        """Moving Brier score should work with a single prediction."""
        # Trigger via online_update to populate the deque
        # Need fitted params first
        for i in range(10):
            run(tracker.record_prediction(f"mkt_{i}", 0.6, ""), event_loop)
            run(tracker.record_resolution(f"mkt_{i}", i < 6), event_loop)
        run(tracker.fit_params(category=None), event_loop)

        # Clear the deque and add one sample manually
        tracker._recent_predictions.clear()
        run(tracker.online_update(0.9, 1.0, ""), event_loop)

        score = tracker.get_moving_brier_score()
        assert score is not None
        # (0.9 - 1.0)^2 = 0.01
        assert abs(score - 0.01) < 1e-10

    def test_moving_brier_score_window(self, tracker, event_loop):
        """Moving Brier score should only track the last N predictions."""
        # Fit params
        for i in range(10):
            run(tracker.record_prediction(f"mkt_{i}", 0.5, ""), event_loop)
            run(tracker.record_resolution(f"mkt_{i}", i < 5), event_loop)
        run(tracker.fit_params(category=None), event_loop)
        tracker._recent_predictions.clear()

        # Add 25 predictions (window is 20)
        for i in range(25):
            run(tracker.online_update(0.5, float(i % 2), ""), event_loop)

        assert len(tracker._recent_predictions) == 20

    def test_no_online_update_without_fitted_params(self, tracker, event_loop):
        """online_update should be a no-op when no params are fitted."""
        # No params fitted yet — online update should not crash
        run(tracker.online_update(0.7, 1.0, ""), event_loop)

        # But it should still update the moving Brier score
        score = tracker.get_moving_brier_score()
        assert score is not None

    def test_online_update_no_resolution_data(self, tracker, event_loop):
        """record_resolution for unknown market should not crash."""
        # Record resolution for a market with no prediction
        run(tracker.record_resolution("nonexistent", True), event_loop)
        # Should not crash — just no online update triggered
