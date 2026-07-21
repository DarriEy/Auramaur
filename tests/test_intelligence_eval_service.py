from datetime import datetime, timezone

from auramaur.db.database import Database
from auramaur.evaluation.service import IntelligenceEvalService
from auramaur.exchange.models import Market
from config.settings import Settings


class Discovery:
    def __init__(self):
        self.market = Market(
            id="m1", question="Will the test pass?", description="Resolves YES if it passes.",
            category="technology", outcome_yes_price=0.55, outcome_no_price=0.45,
            spread=0.02, liquidity=5000, volume=10000,
            end_date=datetime(2026, 8, 1, tzinfo=timezone.utc),
        )

    async def get_markets(self, active=True, limit=100):
        return [self.market]

    async def get_market(self, market_id):
        return self.market


class LocalClient:
    def __init__(self):
        self.calls = []

    async def generate_json(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        assert "evidence_cutoff" in prompt
        assert kwargs["schema"]["additionalProperties"] is False
        return {
            "prob_yes": 0.65, "action": "YES", "confidence": 0.7,
            "thesis": "The frozen rules and snapshot support YES.",
        }


async def test_service_records_paired_treatments_without_trading(tmp_path):
    db = Database(str(tmp_path / "eval-service.db"))
    await db.connect()
    try:
        settings = Settings()
        settings.intelligence_eval.enabled = True
        settings.intelligence_eval.markets_per_cycle = 1
        settings.local_llm.enabled = True
        client = LocalClient()
        service = IntelligenceEvalService(db, settings, Discovery(), client)

        assert await service.run_once() == 3
        episodes = await db.fetchall("SELECT * FROM evaluation_episodes")
        forecasts = await db.fetchall(
            "SELECT f.*, r.arm_name FROM evaluation_forecasts f "
            "JOIN evaluation_runs r ON r.run_id=f.run_id ORDER BY r.arm_name")
        assert len(episodes) == 1
        assert len(forecasts) == 3
        assert {row["episode_hash"] for row in forecasts} == {episodes[0]["episode_hash"]}
        assert {row["arm_name"] for row in forecasts} == {
            "local_single", "local_samples_4", "local_samples_4_critic",
        }
        assert len(client.calls) == 10
        assert len({call[1]["seed"] for call in client.calls}) == 10
        assert await db.fetchone("SELECT 1 FROM trades") is None
        assert await db.fetchone("SELECT 1 FROM portfolio") is None
    finally:
        await db.close()


async def test_service_does_not_duplicate_resolution_detection(tmp_path):
    db = Database(str(tmp_path / "eval-settle.db"))
    await db.connect()
    try:
        settings = Settings()
        settings.intelligence_eval.markets_per_cycle = 1
        discovery, client = Discovery(), LocalClient()
        service = IntelligenceEvalService(db, settings, discovery, client)
        await service.run_once()
        discovery.market.result = "yes"
        discovery.market.active = False
        discovery.market.closed = True
        await service.run_once()
        assert await db.fetchone("SELECT 1 FROM evaluation_outcomes") is None
        assert await db.fetchone("SELECT 1 FROM market_outcomes") is None
    finally:
        await db.close()


def test_intelligence_eval_defaults_off():
    settings = Settings()
    assert settings.intelligence_eval.enabled is False
