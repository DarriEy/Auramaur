import asyncio
from datetime import datetime, timedelta, timezone

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
        attempts = await db.fetchall("SELECT * FROM evaluation_attempts")
        assert len(attempts) == 10
        assert {row["stage"] for row in attempts} == {"sample", "critic"}
        cycle = await db.fetchone("SELECT * FROM evaluation_cycles")
        assert cycle["selected_markets"] == 1
        assert cycle["attempts"] == 10
        assert cycle["failed_attempts"] == 0
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


async def test_unchanged_market_is_skipped_until_reprice(tmp_path):
    db = Database(str(tmp_path / "eval-change.db"))
    await db.connect()
    try:
        settings = Settings()
        settings.intelligence_eval.markets_per_cycle = 1
        settings.intelligence_eval.reevaluate_after_hours = 24
        discovery, client = Discovery(), LocalClient()
        service = IntelligenceEvalService(db, settings, discovery, client)
        assert await service.run_once() == 3
        assert await service.run_once() == 0
        assert len(client.calls) == 10
        discovery.market.outcome_yes_price += settings.intelligence_eval.reprice_threshold
        assert await service.run_once() == 3
        assert len(client.calls) == 20
    finally:
        await db.close()


async def test_multi_market_rotation_family_dedup_and_successive_halving(tmp_path):
    class ManyMarkets:
        def __init__(self):
            now = datetime.now(timezone.utc)
            specs = [
                ("m1", "family-a", "politics", 0.49, 2),
                ("m2", "family-b", "sports", 0.30, 3),
                ("m3", "family-c", "economics", 0.70, 4),
                ("m4", "family-d", "technology", 0.20, 5),
                # Higher-volume duplicate must not consume a second family slot.
                ("m1-duplicate", "family-a", "weather", 0.51, 1),
            ]
            self.markets = []
            for index, (market_id, family, category, price, days) in enumerate(specs):
                self.markets.append(Market(
                    id=market_id, question=f"Question {market_id}?", description="Rules",
                    category=category, outcome_yes_price=price,
                    outcome_no_price=1 - price, spread=0.02, liquidity=5000,
                    volume=10000 - index, end_date=now + timedelta(days=days),
                    neg_risk_market_id=family,
                ))

        async def get_markets(self, active=True, limit=100):
            return self.markets

    class ConcurrentClient(LocalClient):
        def __init__(self):
            super().__init__()
            self.active = self.peak = 0

        async def generate_json(self, prompt, **kwargs):
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.001)
            try:
                return await super().generate_json(prompt, **kwargs)
            finally:
                self.active -= 1

    db = Database(str(tmp_path / "eval-many.db"))
    await db.connect()
    try:
        settings = Settings()
        settings.intelligence_eval.markets_per_cycle = 4
        settings.intelligence_eval.expensive_fraction = 0.25
        settings.intelligence_eval.max_concurrency = 2
        settings.intelligence_eval.market_concurrency = 4
        discovery, client = ManyMarkets(), ConcurrentClient()
        service = IntelligenceEvalService(db, settings, discovery, client)

        # One informative market gets all three arms (10 calls); the other
        # three get one cheap call each.
        assert await service.run_once() == 6
        assert len(client.calls) == 13
        assert client.peak == 2
        episodes = await db.fetchall(
            "SELECT DISTINCT event_family FROM evaluation_episodes")
        assert len(episodes) == 4
        cycle = await db.fetchone("SELECT * FROM evaluation_cycles")
        assert cycle["unique_families"] == 4
        assert cycle["selected_markets"] == 4
    finally:
        await db.close()
