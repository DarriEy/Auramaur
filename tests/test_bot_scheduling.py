"""Focused tests for the bot's extracted adaptive scheduling policy."""

from types import SimpleNamespace
from unittest.mock import patch

from auramaur.bot_scheduling import BotSchedulingMixin


def _scheduler(*, cash: float = 10.0, adaptive: bool = True):
    scheduler = BotSchedulingMixin()
    scheduler._last_known_cash = cash
    scheduler.settings = SimpleNamespace(intervals=SimpleNamespace(
        adaptive_enabled=adaptive,
        quiet_hours_utc=[3],
        peak_hours_utc=[15],
        quiet_multiplier=8.0,
        off_peak_multiplier=4.0,
    ))
    return scheduler


def test_cash_starvation_has_priority_and_slows_interval():
    scheduler = _scheduler(cash=4.99)

    assert scheduler._get_schedule_mode() == "starved"
    assert scheduler._adaptive_interval(10) == 50


def test_disabled_adaptation_keeps_base_interval():
    scheduler = _scheduler(adaptive=False)

    assert scheduler._get_schedule_mode() == ""
    assert scheduler._adaptive_interval(10) == 10


def test_quiet_and_off_peak_multipliers():
    scheduler = _scheduler()

    with patch("auramaur.bot_scheduling.datetime") as clock:
        clock.now.return_value.hour = 3
        assert scheduler._get_schedule_mode() == "quiet"
        assert scheduler._adaptive_interval(10) == 80

        clock.now.return_value.hour = 8
        assert scheduler._get_schedule_mode() == "off_peak"
        assert scheduler._adaptive_interval(10) == 40
