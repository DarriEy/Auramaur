"""Tests for settings and three-gate safety."""

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from config.settings import IBKRConfig, Settings


def test_live_gates_present():
    s = Settings()
    # Verify the three-gate safety model exists
    assert hasattr(s, 'auramaur_live')
    assert hasattr(s, 'is_live')
    assert hasattr(s.execution, 'live')


def test_all_gates_closed():
    s = Settings(auramaur_live=False)
    s.execution.live = False
    assert s.is_live is False


def test_env_gate_only():
    s = Settings(auramaur_live=True)
    s.execution.live = False
    assert s.is_live is False


def test_config_gate_only():
    s = Settings(auramaur_live=False)
    s.execution.live = True
    assert s.is_live is False


def test_both_gates_no_kill_switch():
    s = Settings(auramaur_live=True)
    s.execution.live = True
    with patch.object(Path, "exists", return_value=False):
        assert s.is_live is True


def test_kill_switch_overrides():
    s = Settings(auramaur_live=True)
    s.execution.live = True
    with patch.object(Path, "exists", return_value=True):
        assert s.is_live is False


def test_default_risk_params():
    s = Settings()
    assert s.risk.max_drawdown_pct == 15.0
    # max_stake_per_market <= 1.0 is interpreted as a fraction of equity
    # (0.02 = 2%); see RiskManager. defaults.yaml tunes it to 2%.
    assert s.risk.max_stake_per_market == 0.02
    assert s.risk.daily_loss_limit == 200.0
    assert s.risk.max_open_positions == 500
    # Kalshi gets a lower candidate liquidity floor than Polymarket (thinner book)
    assert s.risk.min_liquidity == 1000.0
    assert s.risk.kalshi_min_liquidity == 300.0
    assert s.kelly.fraction == 0.30


# ---------------------------------------------------------------------------
# API intensity presets
# ---------------------------------------------------------------------------

def test_intensity_medium_is_default():
    from config.settings import NLPConfig
    cfg = NLPConfig()
    assert cfg.api_intensity == "medium"
    assert cfg.skip_second_opinion is False
    assert cfg.max_markets_per_cycle == 10
    assert cfg.daily_claude_call_budget == 100


def test_intensity_low():
    from config.settings import NLPConfig
    cfg = NLPConfig(api_intensity="low")
    assert cfg.skip_second_opinion is True
    assert cfg.max_markets_per_cycle == 10
    assert cfg.evidence_per_source == 3
    assert cfg.daily_claude_call_budget == 50


def test_intensity_full_blast():
    from config.settings import NLPConfig
    cfg = NLPConfig(api_intensity="full_blast")
    assert cfg.skip_second_opinion is False
    assert cfg.max_markets_per_cycle == 50
    assert cfg.evidence_per_source == 10
    assert cfg.daily_claude_call_budget == 0  # unlimited


def test_intensity_explicit_override_wins():
    """Explicit values should beat the preset."""
    from config.settings import NLPConfig
    cfg = NLPConfig(api_intensity="full_blast", max_markets_per_cycle=5)
    assert cfg.max_markets_per_cycle == 5  # explicit override
    assert cfg.evidence_per_source == 10  # from preset


def test_intensity_explicit_value_equal_to_medium_default_wins():
    """Regression: an explicit value that HAPPENS to equal the medium default
    must still beat the preset. defaults.yaml sets api_intensity: "low" with
    skip_second_opinion: false — the old heuristic read the explicit false as
    "unset" and flipped it to True, silently disabling the adversarial second
    opinion (readiness divergence criterion starved at 0 samples)."""
    from config.settings import NLPConfig
    cfg = NLPConfig(api_intensity="low", skip_second_opinion=False)
    assert cfg.skip_second_opinion is False  # explicit override kept
    assert cfg.daily_claude_call_budget == 50  # unset field still gets preset


def test_yaml_defaults_keep_second_opinion_enabled():
    """The deployed defaults.yaml pairs api_intensity: "low" with an explicit
    skip_second_opinion: false; the loaded settings must honor it."""
    s = Settings()
    assert s.nlp.skip_second_opinion is False


def test_kalshi_config_defaults():
    s = Settings()
    assert s.kalshi.enabled is True
    assert s.kalshi.environment == "prod"
    assert s.kalshi.api_key == ""
    assert s.kalshi.private_key_path == ""


def test_yaml_defaults_safe():
    """YAML defaults must never drift to unsafe values."""
    import yaml
    from pathlib import Path

    defaults_path = Path(__file__).parent.parent / "config" / "defaults.yaml"
    with open(defaults_path) as f:
        raw = yaml.safe_load(f)

    # execution.live must be explicitly set (not missing)
    assert "live" in raw["execution"], "defaults.yaml must have execution.live"

    # confidence_floor must be LOW, MEDIUM, or HIGH
    assert raw["risk"]["confidence_floor"] in ("LOW", "MEDIUM", "HIGH"), (
        "defaults.yaml confidence_floor must be LOW, MEDIUM, or HIGH"
    )

    # Hard ceilings — these must never be exceeded regardless of tuning
    assert raw["risk"]["max_drawdown_pct"] <= 25.0, (
        "defaults.yaml max_drawdown_pct must be <= 25%"
    )
    assert raw["risk"]["max_stake_per_market"] <= 100.0, (
        "defaults.yaml max_stake_per_market must be <= $100"
    )
    assert raw["risk"]["daily_loss_limit"] <= 500.0, (
        "defaults.yaml daily_loss_limit must be <= $500"
    )
    assert raw["risk"]["max_open_positions"] <= 1000, (
        "defaults.yaml max_open_positions must be <= 1000"
    )
    assert raw["risk"]["category_exposure_cap_pct"] <= 80.0, (
        "defaults.yaml category_exposure_cap_pct must be <= 80%"
    )
    assert raw["kelly"]["fraction"] <= 0.50, (
        "defaults.yaml kelly fraction must be <= 50%"
    )

    # Kraken's experimental directional book must ship as a bounded paper trial.
    kraken = raw["kraken"]
    assert kraken["directional_llm_paper"] is True
    assert kraken["auto_convert"] is False
    assert kraken["directional_liquidate_orphans"] is False
    assert 0 < kraken["directional_budget_usd"] <= 2 * kraken["max_order_usd"]
    assert kraken["directional_pairs"] == ["XBTUSDC", "ETHUSDC", "SOLUSDC"]
    assert raw["coinbase"]["paper_enabled"] is True

    ibkr = raw["ibkr"]
    assert ibkr["etf_paper_enabled"] is True
    assert ibkr["options_enabled"] is False
    assert ibkr["auto_fx_enabled"] is False
    assert {"SPY", "QQQ", "IWM", "TLT", "GLD", "VEA"}.issubset(
        ibkr["etf_symbols"])
    from auramaur.exchange.ibkr_instruments import GLOBAL_ETFS
    assert ibkr["etf_symbols"] == [spec.symbol for spec in GLOBAL_ETFS]
    assert 0 < ibkr["etf_max_entry_usd"] <= ibkr["etf_paper_budget_usd"]
    assert [m["alias"] for m in ibkr["etf_models"]] == ["luna", "terra", "sol"]


def test_hf_token_exported_to_environ(monkeypatch):
    """hf_token from .env/constructor must reach os.environ — huggingface_hub
    reads HF_TOKEN from the environment, not from Settings."""
    import os
    monkeypatch.delenv("HF_TOKEN", raising=False)
    Settings(hf_token="hf_test123")
    assert os.environ.get("HF_TOKEN") == "hf_test123"


def test_hf_token_does_not_override_shell_env(monkeypatch):
    """A token already exported in the shell wins over the .env value."""
    import os
    monkeypatch.setenv("HF_TOKEN", "hf_from_shell")
    Settings(hf_token="hf_from_dotenv")
    assert os.environ.get("HF_TOKEN") == "hf_from_shell"


def test_hf_token_empty_leaves_environ_untouched(monkeypatch):
    import os
    monkeypatch.delenv("HF_TOKEN", raising=False)
    Settings(hf_token="")
    assert "HF_TOKEN" not in os.environ


def test_nested_environment_override_for_portable_broker(monkeypatch):
    monkeypatch.setenv("IBKR__HOST", "ibgateway")
    monkeypatch.setenv("IBKR__PAPER_PORT", "4002")
    settings = Settings()
    assert settings.ibkr.host == "ibgateway"
    assert settings.ibkr.paper_port == 4002


@pytest.mark.parametrize("override", [
    {"etf_symbols": ["SPY", "SPY"]},
    {"etf_paper_budget_usd": 100, "etf_max_entry_usd": 101},
    {"etf_max_asset_class_pct": 81, "etf_max_deployment_pct": 80},
    {"etf_min_prob": .40, "etf_exit_prob": .47},
    {"etf_openai_daily_call_limit": 2},
])
def test_ibkr_etf_config_rejects_invalid_experiments(override):
    with pytest.raises(ValidationError):
        IBKRConfig(**override)


def test_ibkr_etf_config_rejects_duplicate_model_aliases():
    arm = {"alias": "same", "model": "gpt-test", "effort": "low"}
    with pytest.raises(ValidationError, match="aliases must be unique"):
        IBKRConfig(etf_models=[arm, arm])


def test_ibkr_etf_model_effort_is_strict():
    with pytest.raises(ValidationError):
        IBKRConfig(etf_models=[
            {"alias": "test", "model": "gpt-test", "effort": "maximum"}])
