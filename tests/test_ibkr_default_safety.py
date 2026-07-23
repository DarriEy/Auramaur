"""Regression tests for deliberate, paper-only IBKR activation."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_tracked_ibkr_defaults_enable_readonly_paper_only():
    defaults = yaml.safe_load((ROOT / "config" / "defaults.yaml").read_text())
    ibkr = defaults["ibkr"]

    assert ibkr["enabled"] is True
    assert ibkr["environment"] == "paper"
    assert ibkr["readonly"] is True
    assert ibkr["paper_trade"] is True
    assert ibkr["options_enabled"] is False
    assert ibkr["etf_paper_enabled"] is True
    assert ibkr["multiasset_paper_enabled"] is True


def test_compose_ibkr_fallbacks_enable_readonly_paper_books():
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text())
    environment = compose["services"]["auramaur"]["environment"]

    assert environment["IBKR__ENABLED"] == "${IBKR_ENABLED:-true}"
    assert environment["IBKR__ENVIRONMENT"] == "${IBKR_QUOTE_ENVIRONMENT:-paper}"
    assert environment["AURAMAUR_IBKR_ENVIRONMENT"] == "${IBKR_QUOTE_ENVIRONMENT:-paper}"
    assert environment["IBKR__MULTIASSET_PAPER_ENABLED"] == (
        "${IBKR_MULTIASSET_PAPER_ENABLED:-true}"
    )
    assert environment["IBKR__ETF_PAPER_ENABLED"] == "${IBKR_ETF_PAPER_ENABLED:-true}"
    assert environment["IBKR__READONLY"] == "true"
    assert environment["IBKR__PAPER_TRADE"] == "true"


def test_example_environment_arms_only_ibkr_paper_books():
    values = {}
    for line in (ROOT / ".env.example").read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value

    assert values["IBKR_ENABLED"] == "true"
    assert values["IBKR_MULTIASSET_PAPER_ENABLED"] == "true"
    assert values["IBKR_ETF_PAPER_ENABLED"] == "true"
    assert values["IBKR_QUOTE_ENVIRONMENT"] == "paper"
