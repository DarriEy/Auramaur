from config.settings import Settings


def test_secrets_are_excluded_from_repr_and_serialization():
    settings = Settings(
        _env_file=None,
        kraken_api_secret="kraken-secret",
        polygon_private_key="wallet-secret",
        discord_webhook_url="webhook-secret",
    )

    rendered = repr(settings)
    dumped = settings.model_dump()

    assert "kraken-secret" not in rendered
    assert "wallet-secret" not in rendered
    assert "webhook-secret" not in rendered
    assert "kraken_api_secret" not in dumped
    assert "polygon_private_key" not in dumped
    assert "discord_webhook_url" not in dumped
