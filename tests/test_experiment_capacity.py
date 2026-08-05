"""Experiment portfolio capacity and time-box retirement guards."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from config.settings import Settings


def test_tracked_defaults_leave_one_experiment_slot():
    settings = Settings()

    assert len(settings.active_paper_trials) == 11
    assert settings.experiment_capacity.max_concurrent_paper_trials == 12
    assert "settlement_arb" not in settings.active_paper_trials


def test_starting_trials_over_capacity_fails_closed():
    with pytest.raises(ValidationError, match="concurrent paper trials exceed capacity"):
        Settings(
            weather_temp={"enabled": True},
            econ_indicator={"enabled": True},
        )


def test_expired_spikes_are_explicitly_wound_down():
    with open("config/defaults.yaml", encoding="utf-8") as stream:
        defaults = yaml.safe_load(stream)

    assert defaults["settlement_arb"]["enabled"] is False
    assert defaults["resolution_lens"]["kalshi_enabled"] is False
