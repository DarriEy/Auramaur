from auramaur.subprocess_security import analysis_subprocess_env


def test_analysis_environment_excludes_trading_secrets(monkeypatch):
    monkeypatch.setenv("PATH", "/bin")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "claude-auth")
    monkeypatch.setenv("KRAKEN_API_SECRET", "kraken-secret")
    monkeypatch.setenv("POLYGON_PRIVATE_KEY", "wallet-key")
    monkeypatch.setenv("POLYMARKET_API_KEY", "venue-key")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "webhook")

    child_env = analysis_subprocess_env()

    assert child_env["CLAUDE_CODE_OAUTH_TOKEN"] == "claude-auth"
    assert child_env["PATH"] == "/bin"
    assert "KRAKEN_API_SECRET" not in child_env
    assert "POLYGON_PRIVATE_KEY" not in child_env
    assert "POLYMARKET_API_KEY" not in child_env
    assert "DISCORD_WEBHOOK_URL" not in child_env
