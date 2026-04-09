"""Tests for settings and three-gate safety."""

from pathlib import Path
from unittest.mock import patch

from config.settings import Settings


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
    assert s.risk.max_stake_per_market == 25.0
    assert s.risk.daily_loss_limit == 200.0
    assert s.risk.max_open_positions == 500
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
