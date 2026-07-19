"""Regression tests for deliberate, paper-only IBKR activation."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_tracked_ibkr_defaults_are_inert_and_paper_only():
    defaults = yaml.safe_load((ROOT / "config" / "defaults.yaml").read_text())
    ibkr = defaults["ibkr"]

    assert ibkr["enabled"] is False
    assert ibkr["environment"] == "paper"
    assert ibkr["readonly"] is True
    assert ibkr["paper_trade"] is True
    assert ibkr["options_enabled"] is False
    assert ibkr["etf_paper_enabled"] is False
    assert ibkr["multiasset_paper_enabled"] is False


def test_compose_ibkr_fallbacks_require_explicit_paper_activation():
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text())
    environment = compose["services"]["auramaur"]["environment"]

    assert environment["IBKR__ENABLED"] == "${IBKR_ENABLED:-false}"
    assert environment["IBKR__ENVIRONMENT"] == "${IBKR_QUOTE_ENVIRONMENT:-paper}"
    assert environment["AURAMAUR_IBKR_ENVIRONMENT"] == "${IBKR_QUOTE_ENVIRONMENT:-paper}"
    assert environment["IBKR__MULTIASSET_PAPER_ENABLED"] == (
        "${IBKR_MULTIASSET_PAPER_ENABLED:-false}"
    )
    assert environment["IBKR__ETF_PAPER_ENABLED"] == "false"
    assert environment["IBKR__READONLY"] == "true"
    assert environment["IBKR__PAPER_TRADE"] == "true"


def test_example_environment_does_not_arm_ibkr():
    values = {}
    for line in (ROOT / ".env.example").read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value

    assert values["IBKR_ENABLED"] == "false"
    assert values["IBKR_MULTIASSET_PAPER_ENABLED"] == "false"
    assert values["IBKR_QUOTE_ENVIRONMENT"] == "paper"
